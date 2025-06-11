import torch

from mla_decode import MLAConfig, kv_cache_accounting


def test_kv_bytes_are_shared_per_sequence_not_charged_per_head() -> None:
    config = MLAConfig(
        hidden_size=32,
        num_heads=3,
        q_lora_rank=8,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )
    result = kv_cache_accounting(config, torch.float32)
    assert result.latent_bytes_per_token == (16 + 4) * 4
    assert result.materialized_mha_bytes_per_token == 3 * (8 + 4 + 8) * 4
    assert result.bytes_saved_per_token == 160
    assert result.compression_ratio == 3


def test_accounting_respects_element_size() -> None:
    config = MLAConfig(16, 8, 4, 12, 6, 2, 6)
    fp32 = kv_cache_accounting(config, torch.float32)
    fp16 = kv_cache_accounting(config, torch.float16)
    assert fp16.latent_bytes_per_token * 2 == fp32.latent_bytes_per_token
    assert fp16.materialized_mha_bytes_per_token * 2 == fp32.materialized_mha_bytes_per_token
