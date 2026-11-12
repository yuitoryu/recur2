from transformers import PretrainedConfig


VARIANTS = {
    "RecurrentTransformer": "plain",
    "RecurrentTransformerW": "weighted",
    "RecurrentTransformerAttnRes": "full_attnres",
    "RecurrentTransformerBlockAttnRes": "block_attnres",
}


class RecurConfig(PretrainedConfig):
    model_type = "recur"

    def __init__(
        self,
        variant="plain",
        hidden_size=768,
        num_attention_heads=16,
        intermediate_size=2048,
        max_position_embeddings=1024,
        vocab_size=50257,
        recur_pattern=None,
        activation="relu",
        dropout=0.1,
        block_size=None,
        tie_word_embeddings=False,
        use_cache=False,
        **kwargs,
    ):
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", num_attention_heads),
            ("intermediate_size", intermediate_size),
            ("max_position_embeddings", max_position_embeddings),
            ("vocab_size", vocab_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if variant not in VARIANTS.values():
            raise ValueError(f"Unknown model variant: {variant!r}")
        if recur_pattern is None:
            recur_pattern = ((16, 1),)
        if not isinstance(recur_pattern, (list, tuple)) or not recur_pattern:
            raise ValueError("recur_pattern must contain at least one [layers, repeats] group")
        for group in recur_pattern:
            if (
                not isinstance(group, (list, tuple))
                or len(group) != 2
                or any(type(value) is not int or value < 1 for value in group)
            ):
                raise ValueError("Each recur_pattern group must contain two positive integers")
        if variant == "weighted" and hidden_size < 12 and any(repeats > 1 for _, repeats in recur_pattern):
            raise ValueError("A weighted recurrent group requires hidden_size >= 12")
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if variant == "block_attnres" and block_size is None:
            block_size = 4
        if block_size is not None and (type(block_size) is not int or block_size < 1):
            raise ValueError("block_size must be a positive integer")

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.variant = variant
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.recur_pattern = [list(group) for group in recur_pattern]
        self.activation = activation
        self.dropout = dropout
        self.block_size = block_size
        self.use_cache = use_cache

    @classmethod
    def from_legacy(cls, data, **kwargs):
        variant = VARIANTS.get(data.get("use_model"))
        if variant is None:
            raise ValueError(f"Unknown use_model: {data.get('use_model')!r}")
        return cls(
            variant=variant,
            hidden_size=data["d_model"],
            num_attention_heads=data["nhead"],
            intermediate_size=data["dim_feedforward"],
            max_position_embeddings=data["max_seq_len"],
            vocab_size=data["vocab_size"],
            recur_pattern=data["recur_pattern"],
            activation=data.get("activation", "relu"),
            dropout=data.get("dropout", 0.1),
            block_size=data.get("block_size"),
            **kwargs,
        )

    def validate_cutoff(self, cutoff_len):
        if type(cutoff_len) is not int or not 1 <= cutoff_len <= self.max_position_embeddings:
            raise ValueError(f"cutoff_len must be between 1 and {self.max_position_embeddings}")
