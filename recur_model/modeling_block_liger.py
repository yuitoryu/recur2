"""Block AttnRes with a loss-only Liger path for full pretraining."""

from functools import partial

import torch
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
from transformers.modeling_outputs import CausalLMOutput

from .configuration_recur import RecurConfig
from .modeling_recur import DepthHistory, RecurForCausalLM, attention_masks


@torch.compiler.disable
def fused_loss_sum(hidden, weight, targets):
    # Liger 0.6.2 reads the target count on the host and saves precomputed gradients.
    loss, _ = LigerFusedLinearCrossEntropyFunction.apply(
        hidden, weight, targets, None, None, -100, 0.0, 0.0, "sum"
    )
    return loss


class BlockAttnResLigerForCausalLM(RecurForCausalLM):
    config_class = RecurConfig

    def __init__(self, config):
        if config.variant != "block_attnres":
            raise ValueError("The Liger experiment requires variant='block_attnres'.")
        super().__init__(config)

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
        if not self.training or labels is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                labels=labels, inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, return_dict=return_dict,
            )
        if use_cache:
            raise ValueError("RecurForCausalLM requires use_cache=False")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        x = self.token_embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if not x.is_cuda:
            raise ValueError("Liger Block AttnRes training requires CUDA")
        batch, length, width = x.shape
        if not 1 <= length <= self.config.max_position_embeddings:
            raise ValueError(f"Sequence length must be between 1 and {self.config.max_position_embeddings}")
        if labels.shape != (batch, length):
            raise ValueError("labels must have shape (batch, sequence)")

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
            if visible is None or bool(visible.all()):
                flash_attention = partial(self._flash_self_attention, attention_mask=None)
            elif bool(visible.any(dim=-1).all()):
                flash_attention = partial(self._flash_self_attention, attention_mask=visible)
        mask = None
        if flash_attention is None:
            mask, _ = attention_masks(length, x.device, visible, self.config.num_attention_heads)

        history = DepthHistory(x, self.config.block_size)
        for group, (_, repeats) in zip(self.layers, self.config.recur_pattern):
            for repeat_index in range(repeats):
                for layer in group:
                    layer(history, mask, repeat_index, flash_attention)
        hidden = self.final_norm(self.output_attn_res(history.sources()))

        targets = labels[:, 1:].to(hidden.device)
        if visible is not None:
            targets = targets.masked_fill(~(visible[:, :-1] & visible[:, 1:]), -100)
        if length == 1:
            # Keep zero gradients connected to both the model body and the output head.
            loss = hidden[:, :0].sum().float() + self.lm_head.weight[:0].sum().float()
        else:
            total = fused_loss_sum(hidden[:, :-1, :].reshape(-1, width), self.lm_head.weight, targets.reshape(-1))
            loss = total / targets.ne(-100).sum().clamp_min(1)

        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            return (loss,)
        return CausalLMOutput(loss=loss)
