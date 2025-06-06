from __future__ import annotations

from dataclasses import dataclass

import torch

from mla_decode.config import MLAConfig


@dataclass(frozen=True)
class KVCacheAccounting:
    latent_bytes_per_token: int
    materialized_mha_bytes_per_token: int

    @property
    def bytes_saved_per_token(self) -> int:
        return self.materialized_mha_bytes_per_token - self.latent_bytes_per_token

    @property
    def compression_ratio(self) -> float:
        return self.materialized_mha_bytes_per_token / self.latent_bytes_per_token


def kv_cache_accounting(
    config: MLAConfig, dtype: torch.dtype = torch.float32
) -> KVCacheAccounting:
    """Compare one sequence token's shared MLA cache with per-head MHA K and V."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    latent_elements = config.kv_lora_rank + config.qk_rope_head_dim
    mha_elements = config.num_heads * (config.qk_head_dim + config.v_head_dim)
    return KVCacheAccounting(
        latent_bytes_per_token=latent_elements * element_size,
        materialized_mha_bytes_per_token=mha_elements * element_size,
    )
