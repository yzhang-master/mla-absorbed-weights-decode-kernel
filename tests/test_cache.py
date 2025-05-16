import pytest
import torch

from mla_decode.cache import LatentKVCache


def test_cache_appends_shared_latent_and_rope_tokens() -> None:
    cache = LatentKVCache(2, 4, 3, 2)
    latent = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rope = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    cache.append(latent, rope)
    cache.append(latent + 1, rope + 1)
    assert cache.lengths.tolist() == [2, 2]
    torch.testing.assert_close(cache.latent[:, :2], torch.stack([latent, latent + 1], dim=1))
    torch.testing.assert_close(cache.rope[:, :2], torch.stack([rope, rope + 1], dim=1))
    assert cache.bytes_per_token == 20
    assert cache.logical_bytes == 80
    assert cache.allocated_bytes == 160


def test_cache_capacity_is_checked_before_update() -> None:
    cache = LatentKVCache(1, 1, 2, 1)
    cache.append(torch.ones(1, 2), torch.ones(1, 1))
    before = cache.latent.clone()
    with pytest.raises(RuntimeError, match="capacity"):
        cache.append(torch.full((1, 2), 9.0), torch.ones(1, 1))
    torch.testing.assert_close(cache.latent, before)
    assert cache.lengths.item() == 1


def test_clone_has_independent_storage() -> None:
    cache = LatentKVCache(1, 2, 2, 1)
    cache.append(torch.ones(1, 2), torch.ones(1, 1))
    copied = cache.clone()
    copied.latent.add_(3)
    assert not torch.equal(copied.latent, cache.latent)
    torch.testing.assert_close(copied.lengths, cache.lengths)
