from __future__ import annotations

import torch

from mla_decode._triton import tl, triton
from mla_decode.cache import LatentKVCache


@triton.jit
def _mla_absorbed_decode_kernel(
    q_absorbed_ptr,
    q_rope_ptr,
    latent_ptr,
    rope_ptr,
    lengths_ptr,
    w_uv_ptr,
    output_ptr,
    q_batch_stride,
    q_head_stride,
    q_rope_batch_stride,
    q_rope_head_stride,
    latent_batch_stride,
    latent_seq_stride,
    rope_batch_stride,
    rope_seq_stride,
    w_latent_stride,
    w_head_stride,
    output_batch_stride,
    output_head_stride,
    latent_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    value_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offsets_c = tl.arange(0, BLOCK_C)
    offsets_r = tl.arange(0, BLOCK_R)
    valid_c = offsets_c < latent_dim
    valid_r = offsets_r < rope_dim
    q_offset = batch * q_batch_stride + head * q_head_stride
    q_rope_offset = batch * q_rope_batch_stride + head * q_rope_head_stride
    q_absorbed = tl.load(
        q_absorbed_ptr + q_offset + offsets_c, mask=valid_c, other=0.0
    ).to(tl.float32)
    q_rope = tl.load(
        q_rope_ptr + q_rope_offset + offsets_r, mask=valid_r, other=0.0
    ).to(tl.float32)
    length = tl.load(lengths_ptr + batch)
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)
    latent_context = tl.zeros((BLOCK_C,), dtype=tl.float32)
    offsets_l = tl.arange(0, BLOCK_L)
    for block in range(tl.cdiv(length, BLOCK_L)):
        positions = block * BLOCK_L + offsets_l
        valid_l = positions < length
        latent_offsets = (
            batch * latent_batch_stride
            + positions[:, None] * latent_seq_stride
            + offsets_c[None, :]
        )
        latent = tl.load(
            latent_ptr + latent_offsets,
            mask=valid_l[:, None] & valid_c[None, :],
            other=0.0,
        ).to(tl.float32)
        rope_offsets = (
            batch * rope_batch_stride
            + positions[:, None] * rope_seq_stride
            + offsets_r[None, :]
        )
        rope = tl.load(
            rope_ptr + rope_offsets,
            mask=valid_l[:, None] & valid_r[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(latent * q_absorbed[None, :], axis=1)
        scores += tl.sum(rope * q_rope[None, :], axis=1)
        scores = tl.where(valid_l, scores * scale, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(scores, axis=0))
        correction = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)
        latent_context = latent_context * correction + tl.sum(
            probabilities[:, None] * latent, axis=0
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
        running_max = next_max
    latent_context = latent_context / running_sum
    offsets_v = tl.arange(0, BLOCK_V)
    valid_v = offsets_v < value_dim
    w_offsets = (
        offsets_c[:, None] * w_latent_stride
        + head * w_head_stride
        + offsets_v[None, :]
    )
    w_uv = tl.load(
        w_uv_ptr + w_offsets,
        mask=valid_c[:, None] & valid_v[None, :],
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(latent_context[:, None] * w_uv, axis=0)
    output_offset = batch * output_batch_stride + head * output_head_stride
    tl.store(output_ptr + output_offset + offsets_v, output, mask=valid_v)


def mla_absorbed_decode(
    q_absorbed: torch.Tensor,
    q_rope: torch.Tensor,
    cache: LatentKVCache,
    w_uv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Decode directly from latent KV and apply each head's V up-projection in Triton."""
    if q_absorbed.ndim != 3 or q_rope.ndim != 3:
        raise ValueError("queries must have shape [batch, heads, dimensions]")
    batch, heads, latent_dim = q_absorbed.shape
    if q_rope.shape[:2] != (batch, heads):
        raise ValueError("non-RoPE and RoPE queries must have matching batch and heads")
    if latent_dim != cache.latent_dim or q_rope.shape[-1] != cache.rope_dim:
        raise ValueError("query and cache dimensions do not match")
    if w_uv.ndim != 3 or w_uv.shape[:2] != (latent_dim, heads):
        raise ValueError("W_UV must have shape [latent_dim, heads, value_dim]")
    tensors = (q_absorbed, q_rope, cache.latent, cache.rope, w_uv)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("the native MLA decode kernel currently requires fp32 tensors")
    if any(tensor.device != cache.device for tensor in tensors):
        raise ValueError("queries, weights and cache must be on the same device")
    if torch.any(cache.lengths <= 0).item():
        raise ValueError("MLA decode requires nonempty caches")
    value_dim = w_uv.shape[-1]
    q_absorbed = q_absorbed.contiguous()
    q_rope = q_rope.contiguous()
    w_uv = w_uv.contiguous()
    output = torch.empty(
        batch, heads, value_dim, dtype=torch.float32, device=cache.device
    )
    grid = (batch, heads)
    _mla_absorbed_decode_kernel[grid](
        q_absorbed,
        q_rope,
        cache.latent,
        cache.rope,
        cache.lengths,
        w_uv,
        output,
        q_absorbed.stride(0),
        q_absorbed.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        cache.latent.stride(0),
        cache.latent.stride(1),
        cache.rope.stride(0),
        cache.rope.stride(1),
        w_uv.stride(0),
        w_uv.stride(1),
        output.stride(0),
        output.stride(1),
        latent_dim=latent_dim,
        rope_dim=cache.rope_dim,
        value_dim=value_dim,
        scale=scale,
        BLOCK_L=32,
        BLOCK_C=triton.next_power_of_2(latent_dim),
        BLOCK_R=triton.next_power_of_2(cache.rope_dim),
        BLOCK_V=triton.next_power_of_2(value_dim),
        num_warps=4,
    )
    return output
