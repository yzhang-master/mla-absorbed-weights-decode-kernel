from __future__ import annotations

import torch


class LatentKVCache:
    """Contiguous ragged cache storing one shared latent and RoPE key per token."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        latent_dim: int,
        rope_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(batch_size, max_seq_len, latent_dim) <= 0 or rope_dim < 0:
            raise ValueError("batch, sequence and latent dimensions must be positive")
        if not dtype.is_floating_point:
            raise ValueError("latent cache dtype must be floating point")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self.latent = torch.zeros(
            batch_size, max_seq_len, latent_dim, device=self.device, dtype=dtype
        )
        self.rope = torch.zeros(
            batch_size, max_seq_len, rope_dim, device=self.device, dtype=dtype
        )
        self.lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

    @property
    def bytes_per_token(self) -> int:
        return (self.latent_dim + self.rope_dim) * self.latent.element_size()

    @property
    def logical_bytes(self) -> int:
        return int(self.lengths.sum().item()) * self.bytes_per_token

    @property
    def allocated_bytes(self) -> int:
        latent_bytes = self.latent.numel() * self.latent.element_size()
        rope_bytes = self.rope.numel() * self.rope.element_size()
        return latent_bytes + rope_bytes

    def append(self, latent: torch.Tensor, rope: torch.Tensor) -> None:
        expected_latent = (self.batch_size, self.latent_dim)
        expected_rope = (self.batch_size, self.rope_dim)
        if latent.shape != expected_latent or rope.shape != expected_rope:
            raise ValueError(f"expected latent {expected_latent} and rope {expected_rope}")
        if latent.device != self.device or rope.device != self.device:
            raise ValueError("cache updates must be on the cache device")
        if latent.dtype != self.dtype or rope.dtype != self.dtype:
            raise ValueError("cache updates must have the cache dtype")
        if torch.any(self.lengths >= self.max_seq_len).item():
            raise RuntimeError("cannot append past cache capacity")
        rows = torch.arange(self.batch_size, device=self.device)
        slots = self.lengths
        self.latent[rows, slots] = latent
        self.rope[rows, slots] = rope
        self.lengths.add_(1)

    def clone(self) -> LatentKVCache:
        copied = LatentKVCache(
            self.batch_size,
            self.max_seq_len,
            self.latent_dim,
            self.rope_dim,
            device=self.device,
            dtype=self.dtype,
        )
        copied.latent.copy_(self.latent)
        copied.rope.copy_(self.rope)
        copied.lengths.copy_(self.lengths)
        return copied
