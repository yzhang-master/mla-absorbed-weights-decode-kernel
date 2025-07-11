# mla-absorbed-weights-decode-kernel

> **Era:** 2025+ · Multi-head latent attention technique from 2024, reimplemented in 2026 with
> PyTorch 2.x and Triton 3.x.

A complete single-token MLA decode path with query/key low-rank projections, decoupled RoPE, a
shared latent KV cache, absorbed key and value projections, and a native Triton attention kernel.
CPU execution uses Triton's interpreter for correctness only.

## Problem

Ordinary MHA decode stores a key and value for every head at every token. MLA instead stores a
shared latent vector and a small shared rotary key. Reconstructing per-head K/V before attention
would discard both the storage and compute benefits, so decode must consume the latent cache
directly and apply value up-projection after attention.

## Method

For hidden state `h`, the model computes

```
q_c = h W_DQ                         [batch, q_lora_rank]
[q_nope, q_rope] = q_c W_UQ          [batch, heads, nope_dim + rope_dim]
[c_kv, k_rope] = h W_DKV             [batch, kv_lora_rank + rope_dim]
```

`q_rope` and `k_rope` are rotated at each request's current position. Only `c_kv` and the rotated
shared `k_rope` enter the cache; there are no per-head K/V cache tensors. Ragged batches use a
contiguous `[batch, slot, dimension]` allocation plus a logical length per request.

Key absorption folds the non-RoPE key up-projection into the query:

```
q_abs[b,h,c] = Σ_d q_nope[b,h,d] W_UK[c,h,d]
score[b,h,s] = (Σ_c q_abs[b,h,c] c_kv[b,s,c]
                + Σ_r q_rope[b,h,r] k_rope[b,s,r]) / sqrt(nope_dim + rope_dim)
```

One native Triton program handles one `(request, head)`. It streams 32 cache positions at a time
with online softmax and accumulates a **latent** context. It then applies that head's value
up-projection inside the same kernel:

```
latent_context[b,h,c] = Σ_s softmax(score)[b,h,s] c_kv[b,s,c]
head_out[b,h,v]       = Σ_c latent_context[b,h,c] W_UV[c,h,v]
```

Only the final output projection `W_O` is a separate PyTorch matrix multiplication. The kernel
never allocates or reads materialized ordinary-MHA K/V tensors. Those appear only in the explicit
fp32 reference and the comparison baseline.

### KV bytes per token

For one sequence token, excluding lengths and allocator overhead:

```
MLA = (kv_lora_rank + rope_dim) * dtype_bytes
MHA = heads * (nope_dim + rope_dim + value_dim) * dtype_bytes
```

The latent cost is shared across heads, not multiplied by head count. `LatentKVCache` separately
reports logical bytes from actual lengths and allocated bytes from fixed capacity.

## Results

Local environment: Python 3.12.3, PyTorch 2.15.0+cpu, Triton 3.9.0, no CUDA GPU. The real CPU
interpreter smoke decodes four tokens for each of two requests and compares every step with the
explicit fp32 MLA reference:

| Check | Result |
|---|---:|
| maximum absolute output error | **4.76837158203125e-07** |
| smoke latent KV | **36 bytes/token/request** |
| equivalent materialized MHA KV | **144 bytes/token/request** |
| execution | Triton CPU interpreter |
| performance data | **no** |

The 18-test suite also checks an independent naive per-head MLA loop, absorbed versus explicit
algebra, ragged and multi-block sequences, non-power-of-two dimensions, poisoned unused cache
slots, large-score softmax stability, rotary positions, weight sensitivity, and byte accounting.

For the GPU benchmark configuration (`kv_lora_rank=64`, `rope_dim=16`, 16 heads, 64-dimensional
non-RoPE keys and values, fp32), latent storage is **320 bytes/token**, versus **9,216
bytes/token** for materialized MHA K/V: a **28.8×** compression ratio before metadata.

GPU decode latency and speedup are **pending** on this CPU-only machine. The benchmark checks
correctness before taking CUDA-event timings.

| GPU | Batch | Context | Absorbed ms | Materialized MHA ms | Speedup |
|---|---:|---:|---:|---:|---:|
| _pending_ | 1, 4, 16 | 512, 2048 | _pending_ | _pending_ | _pending_ |

The machine-generated status and smoke values are in `results/gpu_latency.json`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # use a CUDA build on GPU
pip install -e ".[dev]"
ruff check .
pytest
python scripts/bench_decode.py --smoke   # CPU interpreter correctness only
python scripts/bench_decode.py           # pending JSON on CPU; CUDA-event timings on GPU
```

Without installing, prefix Python commands with `PYTHONPATH=src`. On a CUDA host, run:

```bash
python scripts/bench_decode.py --warmup 10 --iterations 50
```

This writes absorbed latency, materialized-MHA latency, speedup, correctness error, and cache
sizes to `results/gpu_latency.json`.

## Limitations

- Decode only: one query token per request; no latent prefill or paged-cache kernel.
- FP32 tensors only. BF16/FP16 cache accounting is available, but those kernel dtypes are not.
- Fixed-capacity contiguous ragged storage; no eviction, prefix sharing, or tensor parallelism.
- Query absorption and final `W_O` projection use PyTorch; attention and `W_UV` use native Triton.
- The simple one-program-per-head schedule has no split-context reduction or autotuning. GPU
  latency and speedup are unmeasured, and CPU interpreter timings are deliberately not reported.

## References

- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
  Model.* 2024. arXiv:2405.04434 (multi-head latent attention and weight absorption)
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. arXiv:2412.19437
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021.
  arXiv:2104.09864
- T. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*
  2022. arXiv:2205.14135 (online-softmax background)

## License

MIT
