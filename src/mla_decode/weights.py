from __future__ import annotations

from dataclasses import dataclass

import torch

from mla_decode.config import MLAConfig


@dataclass
class MLAWeights:
    w_dq: torch.Tensor
    w_uq: torch.Tensor
    w_dkv: torch.Tensor
    w_uk: torch.Tensor
    w_uv: torch.Tensor
    w_o: torch.Tensor

    @classmethod
    def random(
        cls,
        config: MLAConfig,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> MLAWeights:
        def matrix(*shape: int) -> torch.Tensor:
            values = torch.randn(
                *shape, generator=generator, device=device, dtype=torch.float32
            )
            return values * shape[0] ** -0.5

        c = config
        return cls(
            w_dq=matrix(c.hidden_size, c.q_lora_rank),
            w_uq=matrix(c.q_lora_rank, c.num_heads, c.qk_head_dim),
            w_dkv=matrix(c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim),
            w_uk=matrix(c.kv_lora_rank, c.num_heads, c.qk_nope_head_dim),
            w_uv=matrix(c.kv_lora_rank, c.num_heads, c.v_head_dim),
            w_o=matrix(c.num_heads * c.v_head_dim, c.hidden_size),
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.w_dq, self.w_uq, self.w_dkv, self.w_uk, self.w_uv, self.w_o

    def to(self, device: torch.device | str) -> MLAWeights:
        return MLAWeights(*(tensor.to(device=device) for tensor in self.tensors()))
