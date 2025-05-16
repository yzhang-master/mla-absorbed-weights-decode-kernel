from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLAConfig:
    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5
