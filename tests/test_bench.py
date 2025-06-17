from mla_decode.bench import benchmark, cpu_smoke


def test_cpu_smoke_is_correctness_not_performance() -> None:
    result = cpu_smoke()
    assert result["status"] == "correctness_only"
    assert result["triton_execution"] == "interpreter"
    assert result["performance_data"] is False
    assert result["max_absolute_error"] < 2e-5
    assert "milliseconds" not in result


def test_cpu_benchmark_reports_gpu_pending() -> None:
    result = benchmark(warmup=1, iterations=1)
    if result["environment"]["cuda_available"]:
        assert result["status"] == "measured"
    else:
        assert result["status"] == "pending"
        assert result["correctness_smoke"]["performance_data"] is False
        assert "cases" not in result
