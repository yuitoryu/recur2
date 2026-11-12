from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_recur import RecurConfig


class AttentionResidual(nn.Module):
    """Mix raw branch outputs along depth independently for each token."""

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.norm = nn.RMSNorm(hidden_size, eps=1e-5)

    def forward(self, sources):
        values = torch.stack(sources)
        score_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        keys = self.norm(values.to(score_dtype))
        scores = (keys * self.query.to(score_dtype)).sum(dim=-1)
        weights = scores.softmax(dim=0).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(dim=0)


class DepthHistory:
    """Full branch history or sums of fixed-size blocks across all executions."""

    def __init__(self, embedding, block_size):
        self.completed = [embedding]
        self.partial = None
        self.count = 0
        self.block_size = block_size

    def sources(self):
        return self.completed if self.partial is None else self.completed + [self.partial]

    def append(self, branch):
        if self.block_size is None:
            self.completed.append(branch)
            return
        self.partial = branch if self.partial is None else self.partial + branch
        self.count += 1
        if self.count == self.block_size:
            self.completed.append(self.partial)
            self.partial = None
            self.count = 0


def layer_kwargs(config):
    return dict(
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        activation=config.activation,
        dropout=config.dropout,
        batch_first=True,
        norm_first=True,
    )


class DepthResidualLayer(nn.TransformerEncoderLayer):
    """Share branch weights between repeats, with independent depth queries."""

    def __init__(self, config, repeats):
        super().__init__(**layer_kwargs(config))
        self.attn_res = nn.ModuleList(AttentionResidual(config.hidden_size) for _ in range(repeats))
        self.ffn_res = nn.ModuleList(AttentionResidual(config.hidden_size) for _ in range(repeats))

    def forward(self, history, mask, repeat_index, flash_attention):
        x = self.norm1(self.attn_res[repeat_index](history.sources()))
        attention = temporal_attention(self.self_attn, x, mask, flash_attention)
        history.append(self.dropout1(attention))
        x = self.norm2(self.ffn_res[repeat_index](history.sources()))
        history.append(self._ff_block(x))


class EncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, *, flash_attention=None):
        if flash_attention is None:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal=is_causal)
        x = src + self.dropout1(flash_attention(self.self_attn, self.norm1(src)))
        return x + self._ff_block(self.norm2(x))


def make_encoder(layer, count):
    return nn.TransformerEncoder(layer, num_layers=count, enable_nested_tensor=False)


def run_encoder(encoder, x, mask, flash_attention):
    if flash_attention is None:
        return encoder(x, mask=mask)
    for layer in encoder.layers:
        x = layer(x, flash_attention=flash_attention)
    return x


def temporal_attention(attention, x, mask, flash_attention):
    if flash_attention is not None:
        return flash_attention(attention, x)
    return attention(x, x, x, attn_mask=mask, need_weights=False)[0]


class WeightedBlock(nn.Module):
    def __init__(self, config, count, repeats):
        super().__init__()
        self.repeat = repeats
        width = config.hidden_size
        router_dim = width // 12
        self.encoder = make_encoder(EncoderLayer(**layer_kwargs(config)), count)
        self.w_down = nn.Linear(repeats * width, router_dim, bias=False)
        self.causal_attn = nn.MultiheadAttention(router_dim, num_heads=1, batch_first=True)
        self.w_up = nn.Linear(router_dim, repeats, bias=False)

    def forward(self, x, mask, router_mask, flash_attention):
        loop_outputs = []
        for _ in range(self.repeat):
            x = run_encoder(self.encoder, x, mask, flash_attention)
            loop_outputs.append(x)
        compressed = self.w_down(torch.cat(loop_outputs, dim=-1))
        routed = temporal_attention(self.causal_attn, compressed, router_mask, flash_attention)
        weights = self.w_up(routed).softmax(dim=-1)
        return (torch.stack(loop_outputs, dim=2) * weights.unsqueeze(-1)).sum(dim=2)


def attention_masks(length, device, visible, num_heads):
    causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
    if visible is None or bool(visible.all()):
        return causal, causal
    blocked = causal | ~visible[:, None, :]
    # A padded query can attend to itself so even an all-pad row stays finite.
    diagonal = torch.eye(length, dtype=torch.bool, device=device)
    blocked = blocked & ~(~visible[:, :, None] & diagonal)
    return blocked.repeat_interleave(num_heads, dim=0), blocked


class RecurForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RecurConfig
    accepts_loss_kwargs = False
    _supports_flash_attn = True

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList()
        for count, repeats in config.recur_pattern:
            if config.variant == "block_attnres":
                group = nn.ModuleList(DepthResidualLayer(config, repeats) for _ in range(count))
            elif config.variant == "full_attnres":
                group = make_encoder(DepthResidualLayer(config, repeats), count)
            elif config.variant == "weighted" and repeats > 1:
                group = WeightedBlock(config, count, repeats)
            else:
                group = make_encoder(EncoderLayer(**layer_kwargs(config)), count)
            self.layers.append(group)

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.variant in ("full_attnres", "block_attnres"):
            self.output_attn_res = AttentionResidual(config.hidden_size)
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, AttentionResidual):
            nn.init.zeros_(module.query)

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _flash_self_attention(self, attention, x, attention_mask):
        # Keep the existing packed QKV and output projection parameters.
        batch, length, width = x.shape
        qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        if not qkv.is_cuda or qkv.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FlashAttention-2 requires CUDA FP16/BF16 projections. Set bf16: true or fp16: true.")
        query, key, value = qkv.view(batch, length, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
        output, _ = ALL_ATTENTION_FUNCTIONS["flash_attention_2"](
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=attention.dropout if self.training else 0.0,
            scaling=attention.head_dim ** -0.5,
            is_causal=True,
        )
        return attention.out_proj(output.reshape(batch, length, width))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        return_dict=None,
    ):
        if use_cache:
            raise ValueError("RecurForCausalLM requires use_cache=False")
        # Transformers 4.x passes cache_position even when the full sequence is supplied.
        # Only position_ids determine embedding positions, including for left padding.
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        x = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = x.shape
        if not 1 <= length <= self.config.max_position_embeddings:
            raise ValueError(f"Sequence length must be between 1 and {self.config.max_position_embeddings}")

        visible = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape (batch, sequence)")
            visible = attention_mask.to(device=x.device, dtype=torch.bool)
        if position_ids is None:
            if visible is None:
                position_ids = torch.arange(length, device=x.device)
            else:
                position_ids = (visible.long().cumsum(dim=-1) - 1).clamp_min(0)
        x = x + self.position_embedding(position_ids)

        flash_attention = None
        if self.config._attn_implementation == "flash_attention_2":
            # The Transformers backend handles unpadding for nonempty sequences.
            # Preserve the finite masked-attention path for batches with an empty row.
            if visible is None or bool(visible.all()):
                flash_attention = partial(self._flash_self_attention, attention_mask=None)
            elif bool(visible.any(dim=-1).all()):
                flash_attention = partial(self._flash_self_attention, attention_mask=visible)
        if flash_attention is None:
            mask, router_mask = attention_masks(length, x.device, visible, self.config.num_attention_heads)
        else:
            mask = router_mask = None

        if self.config.variant in ("full_attnres", "block_attnres"):
            block_size = self.config.block_size if self.config.variant == "block_attnres" else None
            history = DepthHistory(x, block_size)
            for group, (_, repeats) in zip(self.layers, self.config.recur_pattern):
                layers = group if self.config.variant == "block_attnres" else group.layers
                for repeat_index in range(repeats):
                    for layer in layers:
                        layer(history, mask, repeat_index, flash_attention)
            x = self.output_attn_res(history.sources())
        else:
            for group, (_, repeats) in zip(self.layers, self.config.recur_pattern):
                if isinstance(group, WeightedBlock):
                    x = group(x, mask, router_mask, flash_attention)
                else:
                    for _ in range(repeats):
                        x = run_encoder(group, x, mask, flash_attention)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            if labels.shape != (batch, length):
                raise ValueError("labels must have shape (batch, sequence)")
            targets = labels[:, 1:].to(logits.device)
            if visible is not None:
                targets = targets.masked_fill(~(visible[:, :-1] & visible[:, 1:]), -100)
            scores = logits[:, :-1, :]
            if scores.dtype in (torch.float16, torch.bfloat16):
                scores = scores.float()
            total = F.cross_entropy(
                scores.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )
            loss = total / targets.ne(-100).sum().clamp_min(1)

        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=loss, logits=logits)
