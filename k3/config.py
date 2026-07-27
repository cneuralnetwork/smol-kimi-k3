from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class K3Config:
    vocab_size: int = 8192
    hidden_size: int = 320
    num_layers: int = 13
    num_heads: int = 5
    mla_latent_size: int = 128
    decay_rank: int = 32
    short_conv_kernel: int = 4
    attnres_block_size: int = 4
    latent_moe_size: int = 160
    routed_expert_hidden_size: int = 160
    shared_expert_hidden_size: int = 192
    num_routed_experts: int = 32
    num_shared_experts: int = 2
    experts_per_token: int = 4
    situ_beta1: float = 4.0
    situ_beta2: float = 25.0
    kda_gmin: float = -5.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers < 5 or (self.num_layers - 1) % 4:
            raise ValueError(
                "num_layers must be 4n+1 to preserve 3 KDA + 1 MLA blocks "
                "and the report's final extra MLA layer"
            )
        if not 0 < self.experts_per_token < self.num_routed_experts:
            raise ValueError("experts_per_token must be between 1 and num_routed_experts-1")
        if self.mla_latent_size <= 0 or self.latent_moe_size <= 0:
            raise ValueError("latent dimensions must be positive")

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_heads

    def attention_type(self, layer_index: int) -> str:
        if layer_index == self.num_layers - 1:
            return "mla"
        return "mla" if layer_index % 4 == 3 else "kda"

    @classmethod
    def from_json(cls, path: str | Path) -> "K3Config":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict:
        return asdict(self)
