# DGX Spark (GB10 / sm_121) sparse-MLA decode optimization for DeepSeek-V4-Flash

Branch: `dgx-spark-dsv4`. Goal: optimize the DSV4 sparse-MLA attention kernel for the
DGX Spark's specific hardware, for **both prefill and decode**, with NCU evidence.

## Hardware that defines the target (GB10, measured)

| property | value | consequence |
|---|---|---|
| Compute capability | **sm_121** (consumer Blackwell) | no tcgen05 / TMEM / CLC / 2-SM MMA |
| SMs | **48** | need ≥48 concurrent CTAs to fill |
| Memory | **unified LPDDR5X, ~273 GB/s** | ~30× less BW than B200 HBM → attention is bandwidth-bound |
| Shared mem / CTA | **~99 KB** (101376 B) | vs 228 KB datacenter → limits tile/stage |
| fp8 `tl.dot` | contraction **K ≥ 32** | PV dot contracts over BLOCK_N ⇒ each KV split ≥ 32 keys |

DSV4-Flash-0731 real shape: H=64, MQA (kv heads=1), kv_lora_rank(d_v)=448, qk_rope=64,
**index_topk=512**, packed KV = 584 B/token (448 fp8 nope + 128 bf16 rope + 8 scale).

## Phase 1 — diagnosis (the original per-query kernel)

The upstream SGLang triton sparse-MLA kernel launches `grid=(seq,)` — **one CTA per query**.
It is a *prefill* kernel (its docstring says so). Bench + NCU on GB10:

| shape | grid | latency | mem BW | achieved occ | verdict |
|---|---|---|---|---|---|
| decode s1   | **1 CTA**   | 155 µs | **0.8 %** (1.46% NCU) | 8.3 % | 1 CTA on 48 SMs — GPU 98% idle |
| decode s8   | 8 CTA       | 175 µs | 5.9 %  | 8.3 % | still starved |
| prefill s2048 | 2048 CTA  | 12 ms  | **59.5 %** | 8.3 % | form is right, but occ capped |

Root causes (NCU): (1) `grid=(seq,)` ⇒ 1 CTA at decode; (2) `acc[H=64,d_v=512]` fp32 accumulator
⇒ **255 registers/thread + 92 KB smem** ⇒ only **1 block/SM** ⇒ occupancy pinned at 8.3% everywhere;
(3) never compute-bound (≤22% SM) — purely bandwidth-bound + latency-exposed.

## Phase 3 — the fix: split-KV + split-head flash-decoding

- **split-KV**: grid `(seq, G, S)` partitions topk into `S ≤ 16` chunks (BLOCK_N ≥ 32), each CTA does a
  partial flash-attention; a second **LSE-combine** pass merges the S partials. Fills the SMs at seq=1.
  Adaptive `S ≈ clamp(48 / (seq·G), 1, topk/32)` → **S collapses to 1 at prefill**, where the single
  pass is bitwise-identical to the original (only head partitioning differs).
- **split-head**: `BLOCK_H=16` heads/CTA ⇒ `acc[16,512]` not `[64,512]` ⇒ ~4× fewer registers & smem
  ⇒ more blocks/SM ⇒ **raises the 8.3% occupancy ceiling for BOTH regimes**. (smem 92 KB → 37 KB.)

Mirrors FlashMLA's `num_sm_parts = num_sms / s_q`; validated against KernelWiki precedents
(FlashMLA sparse decode, FlashInfer trtllm sm120, SGLang tilelang sparse-decode partial+combine).

## Phase 4 — results (same-session paired A/B, real shapes, fp8)

| shape | original | optimized | speedup | correctness |
|---|---|---|---|---|
| decode s1        | 155.6 µs | **28.4 µs** | **5.5×** | rel-err 0.031 vs fp32 ref |
| decode s8 (MTP)  | 175.6 µs | **49.1 µs** | **3.6×** | rel-err 0.027 |
| prefill s512     | 3131 µs  | **2114 µs** | **1.48×** | **bitwise-identical** to original |
| prefill s2048    | 12489 µs | **8415 µs** | **1.48×** | **bitwise-identical** |

(32.5 µs at the default config; a config sweep gives 28.4 µs at `S=12, block_hc=4` — the landed
default. Tuning is flat across configs, confirming the residual cost is structural, not a knob.)

Both regimes improved; prefill output is provably exact.

### NCU on the optimized decode (the remaining headroom the user asked about)
| kernel | grid | duration | mem BW | regs | smem | occ |
|---|---|---|---|---|---|---|
| `_partial_kernel` | **48** | ~25 µs | 10.4% | 255 | 37 KB | 8.3% |
| `_combine_kernel` | **4**  | **~21 µs** | 3.7% | 187 | 64 B | 8.3% |

Two structural levers remain (neither is a config knob):
1. **The combine is ~46% of decode latency** yet does trivial work — it is launch-overhead + under-
   occupancy (4 CTAs). Fusing partial+combine into one kernel (atomic/flag reduction, or a persistent
   kernel) would remove the second launch (~10 µs) and the pacc round-trip.
2. **The partial is still 255 regs/thread** (fp32 `acc[16,512]`), pinning occupancy at 8.3%; a
   register-lighter accumulation (e.g. d_v sub-tiling) could hide LPDDR latency better.

Baseline yardstick for future e2e: vLLM's production `flashinfer trtllm sparse-MLA sm120`, the only
sparse decode kernel dispatched on sm_121 (`vllm/platforms/cuda.py` MLA priority for cc major==12).

## Files
- `vllm/v1/attention/ops/triton_sparse_mla_dsv4.py` — the optimized kernel (partial + combine).
- Kernel test harnesses + NCU reports: `dsv4_max_probe/kernel_test/` (diagnosis, A/B, sweep, PHASE1_DIAGNOSIS.md).

*Note: this kernel operates on the un-packed fp8 layout (matching the SGLang contract). Wiring it to
vLLM's 584 B packed `fp8_ds_mla` cache (per-token fp8 scale dequant + bf16 rope) + the sm120 sparse
backend is the e2e integration step.*
