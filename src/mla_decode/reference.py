from __future__ import annotations

import torch

from mla_decode.cache import LatentKVCache


def absorb_query(q_nope: torch.Tensor, w_uk: torch.Tensor) -> torch.Tensor:
    """Absorb K up-projection into each query: q_abs[b,h,c] = q_nope @ W_UK[h]."""
    return torch.einsum("bhd,chd->bhc", q_nope, w_uk)


def _probabilities(
    scores: torch.Tensor, lengths: torch.Tensor, scale: float
) -> torch.Tensor:
    scores = scores * scale
    positions = torch.arange(scores.shape[-1], device=scores.device)
    valid = positions[None, None, :] < lengths[:, None, None]
    scores = scores.masked_fill(~valid, -torch.inf)
    return torch.softmax(scores, dim=-1)


def explicit_mla_heads(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    cache: LatentKVCache,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Explicit fp32 MLA: up-project non-RoPE K and all V, then attend per head."""
    key_nope = torch.einsum("chd,bsc->bshd", w_uk, cache.latent)
    scores_nope = torch.einsum("bhd,bshd->bhs", q_nope, key_nope)
    scores_rope = torch.einsum("bhd,bsd->bhs", q_rope, cache.rope)
    probabilities = _probabilities(scores_nope + scores_rope, cache.lengths, scale)
    values = torch.einsum("chv,bsc->bshv", w_uv, cache.latent)
    return torch.einsum("bhs,bshv->bhv", probabilities, values).float()


def absorbed_mla_heads(
    q_absorbed: torch.Tensor,
    q_rope: torch.Tensor,
    cache: LatentKVCache,
    w_uv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Torch counterpart of the kernel, with no materialized per-head K or V."""
    scores_latent = torch.einsum("bhc,bsc->bhs", q_absorbed, cache.latent)
    scores_rope = torch.einsum("bhd,bsd->bhs", q_rope, cache.rope)
    probabilities = _probabilities(scores_latent + scores_rope, cache.lengths, scale)
    latent_context = torch.einsum("bhs,bsc->bhc", probabilities, cache.latent)
    return torch.einsum("bhc,chv->bhv", latent_context, w_uv).float()


def project_output(heads: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    batch = heads.shape[0]
    return heads.reshape(batch, -1) @ w_o
