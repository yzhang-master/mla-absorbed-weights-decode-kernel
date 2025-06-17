from __future__ import annotations

import platform
from dataclasses import asdict
from datetime import datetime
from typing import Any

import torch

from mla_decode.accounting import kv_cache_accounting
from mla_decode.config import MLAConfig
from mla_decode.model import MLADecoder
from mla_decode.weights import MLAWeights

GPU_CONFIG = MLAConfig(
    hidden_size=1024,
    num_heads=16,
    q_lora_rank=128,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=16,
    v_head_dim=64,
)


def cpu_smoke() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(141)
    config = MLAConfig(12, 3, 5, 7, 4, 2, 6)
    model = MLADecoder(config, MLAWeights.random(config, generator=generator))
    actual_cache = model.new_cache(2, 4)
    reference_cache = model.new_cache(2, 4)
    hidden = torch.randn(4, 2, 12, generator=generator)
    max_error = 0.0
    for token in hidden:
        actual = model.decode(token, actual_cache)
        expected = model.reference_decode(token, reference_cache)
        max_error = max(max_error, (actual - expected).abs().max().item())
    return {
        "status": "correctness_only",
        "device": "cpu",
        "triton_execution": "interpreter",
        "tokens_per_request": 4,
        "batch_size": 2,
        "max_absolute_error": max_error,
        "kv_cache": asdict(kv_cache_accounting(config)),
        "performance_data": False,
    }


def _cuda_milliseconds(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _gpu_case(
    model: MLADecoder,
    batch_size: int,
    context_length: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    device = model.device
    generator = torch.Generator(device=device).manual_seed(151 + batch_size)
    cache = model.new_cache(batch_size, context_length)
    cache.latent.normal_(generator=generator)
    cache.rope.normal_(generator=generator)
    cache.lengths.fill_(context_length)
    hidden = torch.randn(
        batch_size, model.config.hidden_size, generator=generator, device=device
    )

    def absorbed() -> torch.Tensor:
        return model.decode(hidden, cache, append=False)

    def materialized() -> torch.Tensor:
        return model.reference_decode(hidden, cache, append=False)

    actual = absorbed()
    expected = materialized()
    max_error = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    absorbed_ms = _cuda_milliseconds(absorbed, warmup, iterations)
    materialized_ms = _cuda_milliseconds(materialized, warmup, iterations)
    accounting = kv_cache_accounting(model.config)
    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "max_absolute_error": max_error,
        "absorbed_milliseconds": absorbed_ms,
        "materialized_mha_milliseconds": materialized_ms,
        "speedup_vs_materialized_mha": materialized_ms / absorbed_ms,
        "latent_bytes_per_token": accounting.latent_bytes_per_token,
        "materialized_mha_bytes_per_token": accounting.materialized_mha_bytes_per_token,
        "latent_cache_logical_mib": cache.logical_bytes / 2**20,
    }


def benchmark(warmup: int = 10, iterations: int = 50) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    if not torch.cuda.is_available():
        result.update(
            status="pending",
            reason="CUDA GPU unavailable; CPU interpreter timings are not performance data",
            correctness_smoke=cpu_smoke(),
        )
        return result
    result.update(status="measured", gpu_name=torch.cuda.get_device_name())
    generator = torch.Generator(device="cuda").manual_seed(137)
    model = MLADecoder(
        GPU_CONFIG, MLAWeights.random(GPU_CONFIG, generator=generator, device="cuda")
    )
    result["config"] = asdict(GPU_CONFIG)
    result["cases"] = [
        _gpu_case(model, batch, context, warmup, iterations)
        for batch, context in [(1, 512), (4, 512), (4, 2048), (16, 2048)]
    ]
    return result
