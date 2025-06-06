from __future__ import annotations

import torch

from mla_decode.cache import LatentKVCache
from mla_decode.config import MLAConfig
from mla_decode.kernel import mla_absorbed_decode
from mla_decode.reference import (
    absorb_query,
    explicit_mla_heads,
    project_output,
)
from mla_decode.weights import MLAWeights


class MLADecoder:
    """Single-token multi-head latent attention with absorbed K and V projections."""

    def __init__(self, config: MLAConfig, weights: MLAWeights | None = None) -> None:
        self.config = config
        self.weights = MLAWeights.random(config) if weights is None else weights
        c = config
        w = self.weights
        expected = (
            (c.hidden_size, c.q_lora_rank),
            (c.q_lora_rank, c.num_heads, c.qk_head_dim),
            (c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim),
            (c.kv_lora_rank, c.num_heads, c.qk_nope_head_dim),
            (c.kv_lora_rank, c.num_heads, c.v_head_dim),
            (c.num_heads * c.v_head_dim, c.hidden_size),
        )
        if tuple(tensor.shape for tensor in w.tensors()) != expected:
            raise ValueError("MLA weight shapes do not match the configuration")
        if any(tensor.dtype != torch.float32 for tensor in w.tensors()):
            raise ValueError("MLADecoder currently requires fp32 weights")
        devices = {tensor.device for tensor in w.tensors()}
        if len(devices) != 1:
            raise ValueError("all MLA weights must be on one device")
        self.device = w.w_dq.device

    def new_cache(self, batch_size: int, max_seq_len: int) -> LatentKVCache:
        return LatentKVCache(
            batch_size,
            max_seq_len,
            self.config.kv_lora_rank,
            self.config.qk_rope_head_dim,
            device=self.device,
        )

    def project(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        if hidden.shape != (hidden.shape[0], c.hidden_size) or hidden.ndim != 2:
            raise ValueError(f"hidden must have shape [batch, {c.hidden_size}]")
        if hidden.device != self.device or hidden.dtype != torch.float32:
            raise ValueError("hidden must be fp32 and on the weight device")
        q_latent = hidden @ self.weights.w_dq
        q = torch.einsum("br,rhd->bhd", q_latent, self.weights.w_uq)
        q_nope, q_rope = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        kv = hidden @ self.weights.w_dkv
        latent, rope = kv.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        return q_nope, q_rope, latent, rope

    def attend(
        self, q_nope: torch.Tensor, q_rope: torch.Tensor, cache: LatentKVCache
    ) -> torch.Tensor:
        q_absorbed = absorb_query(q_nope, self.weights.w_uk)
        heads = mla_absorbed_decode(
            q_absorbed,
            q_rope,
            cache,
            self.weights.w_uv,
            self.config.softmax_scale,
        )
        return project_output(heads, self.weights.w_o)

    def decode(
        self, hidden: torch.Tensor, cache: LatentKVCache, *, append: bool = True
    ) -> torch.Tensor:
        q_nope, q_rope, latent, rope = self.project(hidden)
        if append:
            cache.append(latent, rope)
        return self.attend(q_nope, q_rope, cache)

    def reference_decode(
        self, hidden: torch.Tensor, cache: LatentKVCache, *, append: bool = True
    ) -> torch.Tensor:
        q_nope, q_rope, latent, rope = self.project(hidden)
        if append:
            cache.append(latent, rope)
        heads = explicit_mla_heads(
            q_nope,
            q_rope,
            cache,
            self.weights.w_uk,
            self.weights.w_uv,
            self.config.softmax_scale,
        )
        return project_output(heads, self.weights.w_o)
