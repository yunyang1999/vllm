# DGX Spark (GB10 / sm_121) DSv4-Flash sparse-MLA: beating flashinfer, and the e2e reality

Goal: optimize the user's Triton sparse-MLA kernels for GB10, land on vLLM, and beat the
flashinfer kernel vLLM's DSv4 runs — measured at kernel + e2e level. All numbers below are
from real runs on GB10 (48 SMs, LPDDR5X ~273 GB/s, 99 KB smem, no fp8-mma / TMA / warp-spec
in Triton), DSv4-Flash-0731 shape H=64, latent 512 (448 fp8 + 64 rope), topk=512, 584 B cache.
Cross-validated ≥3 independent runs; correctness cos=1.0000 vs fp32 ref.

## Kernel-level vs flashinfer — WE BEAT IT on prefill and on graphed T=1 decode

**Prefill** (user's bf16 `sparse_mla_prefill`, GB10 config `head_tile=32, BLOCK_N=32, w4, ns3`):
| T | flashinfer | user Triton | speedup |
|---:|---:|---:|---:|
| 512  | 1374 µs | **1013 µs** | **1.36×** |
| 1024 | 2679 µs | **1888 µs** | **1.42×** |
| 2048 | 5270 µs | **3763 µs** | **1.40×** |
| 4096 | 10484 µs | **7309 µs** | **1.43×** |

Why: flashinfer has no dedicated sparse-*prefill* kernel; it runs its split-K *decode* kernel at
batch=T, where the split-K is dead weight (T×H already fills 48 SMs). The user's purpose-built
prefill kernel (one program per token×head-tile, register online-softmax, no HBM partials) wins.

**Decode, in the CUDA-graph regime vLLM actually uses** (`sm121_decode_v2.py`: head-tiled split-K,
dequant→bf16 bf16-dot, parallelized merge, splits {T=1:12, T≤32:3, T=64:1}):
| T | flashinfer (graph) | user Triton (graph) | ratio |
|---:|---:|---:|---:|
| 1  | 15.6 µs | **13.5 µs** | **0.87× ✅ (13% faster)** |
| 8  | 38.1 µs | 42.2 µs | 1.11× (FI wins) |
| 16 | 70.9 µs | 77.0 µs | 1.09× |
| 32 | 130 µs | 155 µs | 1.19× |
| 64 | 260 µs | 313 µs | 1.19× |

At T=1 flashinfer under-fills (32 blocks → 16 of 48 SMs idle); our 12-way split fills the GPU and
the graph makes the extra merge launch free → we win. At T≥8 flashinfer fills the GPU and its
sm_121 primitives Triton can't emit (TMA/`cp.async.bulk` gather-overlap, block-scaled fp8-MMA,
2× occupancy) decide — an irreducible *form* ceiling, NCU-proven (both kernels latency-bound,
never compute/BW-bound).

## End-to-end reality — the attention kernel is ~2% of a decode step (MoE-bound)

Deployed vLLM v0.25.1 (flashinfer), unique/uncacheable prompts, MTP dspark (8 tok/step), graph batch [1,2,4]:
| prompt | TTFT | ITL (ms/token) |
|---:|---:|---:|
| 128  | 256 ms | 87 ms |
| 2048 | 342 ms | 91 ms |
| 4096 | 354 ms | 92 ms |

**Decode ITL ≈ 90 ms/token.** The sparse-MLA attention is 43 layers × ~40 µs ≈ **1.7 ms = ~2% of a
step**; the other ~98% is the 256-expert MoE + Lightning indexer + norms. TTFT grows only ~25 µs/token
with length, so prefill is not attention-dominated either. **Consequence: beating the attention kernel
(prefill 1.4×, decode-T1 13%) moves e2e latency by ≲2% — within noise.** The real e2e optimization
lever on GB10 is the **MoE (256 experts) and the Lightning indexer**, not the attention kernel.

## What's landed here
- `vllm/v1/attention/ops/triton_sparse_mla_dsv4_decode_sm121.py` — the GB10 decode kernel (beats FI graphed T=1).
- `vllm/v1/attention/ops/triton_sparse_mla_dsv4.py` — the earlier split-KV decode (superseded; kept for the diagnosis).
- GB10 prefill pin (for the sglang kernel `_PINNED`): `_PINNED_HEAD_TILE[(12,1)]={32:32,64:32}`, `_PINNED[(12,1)]=(32,4,3)`.
- Full evidence: `DGX_SPARK_DSV4_decode_ceiling.md`, `DGX_SPARK_DSV4_prefill.md`, `DGX_SPARK_DSV4_SPARSE_MLA.md`;
  harnesses in `dsv4_max_probe/kernel_test/{ab_vllm,prefill_ab}/`.

## Recommendation
- **Kernel goal met**: the user's kernels beat flashinfer on GB10 prefill (1.4×) and graphed T=1 decode (1.13×).
- **Prefill win is worth landing** (pin the GB10 config); the decode kernel is a fast fallback but flashinfer
  stays best for the batched MTP decode shape (T=8–32).
- **For real e2e speedup on GB10, optimize the MoE/indexer** — that's 98% of the decode step. The attention
  kernel is already near its sm_121 ceiling and is not the bottleneck.
