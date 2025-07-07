from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate the final even dimension in halves at each request's token position."""
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    if positions.shape != (x.shape[0],):
        raise ValueError("positions must contain one token position per request")
    frequencies = torch.arange(x.shape[-1] // 2, device=x.device, dtype=torch.float32)
    inverse_frequency = 10000 ** (-frequencies / (x.shape[-1] // 2))
    angles = positions.float()[:, None] * inverse_frequency[None, :]
    broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1] // 2,)
    cosine = angles.cos().reshape(broadcast_shape)
    sine = angles.sin().reshape(broadcast_shape)
    first, second = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1)
