import pytest
import torch

from mla_decode import MLAConfig, MLADecoder, MLAWeights
from mla_decode.kernel import mla_absorbed_decode
from mla_decode.reference import (
    absorb_query,
    absorbed_mla_heads,
    explicit_mla_heads,
)


def make_decoder(seed: int = 17, **overrides: int) -> MLADecoder:
    values = dict(
        hidden_size=12,
        num_heads=3,
        q_lora_rank=5,
        kv_lora_rank=7,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=6,
    )
    values.update(overrides)
    config = MLAConfig(**values)
    weights = MLAWeights.random(config, generator=torch.Generator().manual_seed(seed))
    return MLADecoder(config, weights)


def fill_ragged_cache(
    model: MLADecoder, lengths: list[int], generator: torch.Generator
):
    cache = model.new_cache(len(lengths), max(lengths) + 3)
    for batch, length in enumerate(lengths):
        latent = torch.randn(
            length, model.config.kv_lora_rank, generator=generator
        )
        rope = torch.randn(length, model.config.qk_rope_head_dim, generator=generator)
        cache.latent[batch, :length] = latent
        cache.rope[batch, :length] = rope
        cache.lengths[batch] = length
    return cache


@pytest.mark.parametrize(
    ("lengths", "config"),
    [
        ([1, 5, 9], {}),
        ([17, 33], dict(num_heads=2, kv_lora_rank=13, qk_rope_head_dim=4)),
        ([2, 7], dict(num_heads=4, kv_lora_rank=4, qk_nope_head_dim=6, v_head_dim=5)),
    ],
)
def test_native_kernel_matches_explicit_and_absorbed_fp32_references(
    lengths: list[int], config: dict[str, int]
) -> None:
    model = make_decoder(**config)
    generator = torch.Generator().manual_seed(31 + len(lengths))
    cache = fill_ragged_cache(model, lengths, generator)
    q_nope = torch.randn(
        len(lengths),
        model.config.num_heads,
        model.config.qk_nope_head_dim,
        generator=generator,
    )
    q_rope = torch.randn(
        len(lengths),
        model.config.num_heads,
        model.config.qk_rope_head_dim,
        generator=generator,
    )
    q_absorbed = absorb_query(q_nope, model.weights.w_uk)
    actual = mla_absorbed_decode(
        q_absorbed, q_rope, cache, model.weights.w_uv, model.config.softmax_scale
    )
    explicit = explicit_mla_heads(
        q_nope,
        q_rope,
        cache,
        model.weights.w_uk,
        model.weights.w_uv,
        model.config.softmax_scale,
    )
    absorbed = absorbed_mla_heads(
        q_absorbed, q_rope, cache, model.weights.w_uv, model.config.softmax_scale
    )
    torch.testing.assert_close(absorbed, explicit, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual, explicit, atol=2e-5, rtol=2e-5)


def test_explicit_reference_matches_naive_per_head_mla_loop() -> None:
    model = make_decoder()
    cache = fill_ragged_cache(model, [3, 5], torch.Generator().manual_seed(51))
    q_nope = torch.randn(2, 3, 4)
    q_rope = torch.randn(2, 3, 2)
    expected = torch.empty(2, 3, 6)
    for batch, length in enumerate([3, 5]):
        for head in range(3):
            keys = cache.latent[batch, :length] @ model.weights.w_uk[:, head]
            scores = keys @ q_nope[batch, head]
            scores = scores + cache.rope[batch, :length] @ q_rope[batch, head]
            probabilities = torch.softmax(scores * model.config.softmax_scale, dim=0)
            values = cache.latent[batch, :length] @ model.weights.w_uv[:, head]
            expected[batch, head] = probabilities @ values
    actual = explicit_mla_heads(
        q_nope,
        q_rope,
        cache,
        model.weights.w_uk,
        model.weights.w_uv,
        model.config.softmax_scale,
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_full_decode_matches_reference_at_every_append() -> None:
    model = make_decoder()
    actual_cache = model.new_cache(2, 6)
    reference_cache = model.new_cache(2, 6)
    hidden = torch.randn(6, 2, 12, generator=torch.Generator().manual_seed(77))
    for token in hidden:
        actual = model.decode(token, actual_cache)
        expected = model.reference_decode(token, reference_cache)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual_cache.lengths.tolist() == [6, 6]


def test_unused_cache_tail_is_masked() -> None:
    model = make_decoder()
    generator = torch.Generator().manual_seed(81)
    cache = fill_ragged_cache(model, [4, 7], generator)
    q_nope = torch.randn(2, 3, 4, generator=generator)
    q_rope = torch.randn(2, 3, 2, generator=generator)
    q_absorbed = absorb_query(q_nope, model.weights.w_uk)
    before = mla_absorbed_decode(
        q_absorbed, q_rope, cache, model.weights.w_uv, model.config.softmax_scale
    )
    cache.latent[0, 4:] = 1e6
    cache.latent[1, 7:] = -1e6
    cache.rope[0, 4:] = -1e6
    cache.rope[1, 7:] = 1e6
    after = mla_absorbed_decode(
        q_absorbed, q_rope, cache, model.weights.w_uv, model.config.softmax_scale
    )
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)


def test_online_softmax_is_stable_for_large_scores() -> None:
    model = make_decoder()
    cache = fill_ragged_cache(model, [35, 18], torch.Generator().manual_seed(91))
    cache.latent.mul_(20)
    cache.rope.mul_(20)
    q_nope = torch.randn(2, 3, 4) * 20
    q_rope = torch.randn(2, 3, 2) * 20
    q_absorbed = absorb_query(q_nope, model.weights.w_uk)
    actual = mla_absorbed_decode(
        q_absorbed, q_rope, cache, model.weights.w_uv, model.config.softmax_scale
    )
    expected = explicit_mla_heads(
        q_nope,
        q_rope,
        cache,
        model.weights.w_uk,
        model.weights.w_uv,
        model.config.softmax_scale,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_absorbed_weights_change_the_native_output() -> None:
    model = make_decoder()
    cache = fill_ragged_cache(model, [5, 8], torch.Generator().manual_seed(101))
    q_nope = torch.randn(2, 3, 4)
    q_rope = torch.randn(2, 3, 2)
    q_absorbed = absorb_query(q_nope, model.weights.w_uk)
    baseline = mla_absorbed_decode(
        q_absorbed, q_rope, cache, model.weights.w_uv, model.config.softmax_scale
    )
    changed_values = model.weights.w_uv.clone()
    changed_values[:, 1] *= -2
    changed = mla_absorbed_decode(
        q_absorbed, q_rope, cache, changed_values, model.config.softmax_scale
    )
    torch.testing.assert_close(changed[:, 0], baseline[:, 0])
    torch.testing.assert_close(changed[:, 2], baseline[:, 2])
    assert not torch.allclose(changed[:, 1], baseline[:, 1])


def test_decode_rejects_empty_cache() -> None:
    model = make_decoder()
    cache = model.new_cache(1, 2)
    with pytest.raises(ValueError, match="nonempty"):
        model.decode(torch.randn(1, 12), cache, append=False)
