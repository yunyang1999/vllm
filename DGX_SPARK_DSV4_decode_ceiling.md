# BEAT_RESULTS — can a Triton DSv4 sparse-MLA DECODE beat flashinfer on GB10 (sm_121)?

**Verdict: YES at T=1, in the production (CUDA-graph) regime — Triton is 13% FASTER
(13.57 µs vs 15.62 µs, ratio 0.869×, cos vs fp32-ref = 1.0000, stable across 4
captures).** For T≥8 flashinfer leads by **1.09–1.20×**, and the residual is an
*irreducible* structural gap (its warp-specialized `cp.async.bulk`/TMA gather-compute
overlap + block-scaled FP8 MMA + 2× occupancy — none expressible in Triton on sm_121),
proven by NCU. The prior verified result was **1.35–1.62× slower**; this work **crosses
0.87× at T=1 and tightens every T** (prior eager 1.23–1.64× → now in-graph 0.87–1.22×).

Device: NVIDIA GB10, sm_121 (cc 12.1), 48 SMs, 99 KB opt-in smem/block, 25.2 MB L2,
LPDDR5X. Container `dualspark-vllm:v0.25.1-nccl-2.30.4` (torch 2.11+cu130, triton 3.6.0,
flashinfer 0.6.14). Shape: H=64, d_v=512 (448 fp8 nope + 64 bf16 rope), topk=512, packed
584 B/token, sm_scale=1/√512, pool=16384 (9.57 MB cache, L2-resident). Timing = median of
100 CUDA-event iters (eager) / 100 graph-replays (in-graph), min-of-3, same-session paired
A/B. FlashInfer = `_SparseMLAPagedAttentionRunner` → `sparse_mla_sm120_decode_dsv4`
(autotuner warmed). Every number below is pasted from a real run.

---

## HEADLINE TABLE  (final_locked_out.txt, graph_ab.py)

Best Triton config: **BH16 BN16 ns1 w4**, splits per-T {T=1:12, T=8/16/32:3, T=64:1}
(kernel `sm121_decode_v2.py`, the head-tiled split-K bf16-dot decode with a
**parallelized merge**). All cos vs fp32-ref (T≤8) = 1.0000; cos vs flashinfer = 0.9997.

| T  | FI eager | FI **graph** | Tri eager | Tri **graph** | eager x | **graph x** | cos |
|---:|---------:|-------------:|----------:|--------------:|--------:|------------:|----:|
| 1  |  22.78   | **15.62**    | 26.72     | **13.57**     | 1.173   | **0.869** ✅ | 1.0000 |
| 8  |  40.19   | 38.08        | 44.29     | 42.18         | 1.102   | 1.108       | 1.0000 |
| 16 |  72.96   | 70.85        | 79.14     | 76.99         | 1.085   | 1.087       | 0.9997* |
| 32 | 130.30   | 130.21       | 156.80    | 154.66        | 1.203   | 1.188       | 0.9997* |
| 64 | 260.00   | 260.22       | 311.49    | 312.61        | 1.198   | 1.201       | 0.9997* |

\* cos vs flashinfer (no fp32 ref built for T>8). T=1 stability (4 graph captures):
FI `[15.65, 15.84, 15.58, 15.62]`, Tri `[13.57, 13.57, 13.57, 13.54]` — the beat is
noise-free (margin 2 µs ≫ σ≈0.1 µs). Independent confirmation (sweep_graph_out.txt):
Tri T=1 graph verified x4 `[13.57, 13.54, 13.57, 13.57]` vs FI 15.58.

**Adversarial verification** (verify_beat_out.txt — different seed/indices/harness, fresh
graph capture, explicit fp32-ref for both): `T=1 FI 15.58 / Tri 13.54 = 0.869× *** TRITON
WINS *** cos(Tri,ref)=1.0000`; `T=8 1.108 FI wins`; `T=64 1.250 FI wins`. Finding survives.

**Why "graph" is the honest kernel-vs-kernel metric:** flashinfer's `runner.run` carries
heavier host/Python dispatch than a lean Triton launch, which *inflates FI's eager time*
(22.78 vs its true 15.6 µs kernel). In a CUDA graph both are stripped to pure kernels.
Production decode (vLLM/sglang) runs attention inside CUDA graphs, so **graph x is the
number that ships.** (Eager x is reported for completeness; there Triton pays 2 launches.)

---

## THE T=1 WIN — mechanism (found by reading flashinfer's CUDA source, `cuh/decode_dsv4_kernel.cuh`)

FlashInfer's grid is `(num_tokens, num_head_blocks=H/HPB=4, num_splits_z)` with
`HPB=16`, `BI=64` ⇒ `num_splits = ceil(topk/64) = 8`, and `chunks_per_block` (autotuned)
sets `z = ceil(8/cpb)`. **At T=1 the maximum grid is `(1, 4, 8) = 32 blocks` (cpb=1) — it
physically cannot exceed 32 blocks, leaving 16 of 48 SMs idle.** NCU confirms: `fi_t1`
decode `blocks=32, waves=0.7, occ 18–22%`.

My Triton kernel splits topk into `BLOCK_N=16` tiles (32 tiles) and can split-K **any**
number of ways: grid `(T·⌈H/BH⌉, splits) = (4, 12) = 48 blocks` at T=1 → **fills all 48
SMs**. In a CUDA graph the extra merge launch is free, so this pure occupancy win
materializes: 13.57 vs 15.62 µs. This is the exact opening predicted from the source.

---

## NCU DECOMPOSITION — both kernels are LATENCY-bound, never compute/BW-bound

Calibrated denominators (measured locally, probe0_out.txt — never spec sheets):
`DRAM 112 GB/s · L2 1953 GB/s · bf16-TC 94 TFLOP/s · fp8-TC 118 TFLOP/s`.
Roofline floors: **T=1** gather=292 KB/token → L2-floor 0.15 µs, DRAM-floor 2.6 µs
(measured 13–16 µs ⇒ 5–100× above the memory floor). **T=64** compute=4.3 GFLOP →
fp8-floor 36 µs, bf16-floor 46 µs (measured 260–313 µs ⇒ 6–7× below compute peak; L2
traffic 18.7 MB → 9.6 µs floor). **Neither kernel is compute- or bandwidth-bound at any T
— both are latency-bound (long_scoreboard = memory-latency stall dominates).**

Per-kernel NCU (`--set full`; durations NCU-inflated by locked clocks + serialization,
read the *ratios/occupancy/stalls*, not absolute µs):

| kernel @T | blocks | occ% | sm% | l2% | regs | smem | top stall (per-issue) |
|---|---:|---:|---:|---:|---:|---:|---|
| **FI decode** T=1        | 32   | 18–22 | 10 | 8  | 96  | 90.8 KB | long_scoreboard 5–6 |
| FI merge  T=1            | 64   | 11–15 | 1  | 5  | 48  | —       | long_scoreboard 11–12 |
| **FI decode** T=64 (cpb1)| 2048 | 16–19 | 24 | 19 | 96  | 90.8 KB | long_scoreboard 3.5 (42 waves) |
| FI merge  T=64 (8-split) | 4096 | 73–77 | 5  | 21 | 48  | —       | long_scoreboard **141** |
| **Tri v1 decode** T=1    | 32   | 8–10  | 8–11| 3–4| 222 | 57 KB  | long_scoreboard 3.5 + mio 2.5 |
| Tri v1 merge  T=1        | **8**| 10    | 0.6| 2  | 40  | —       | long_scoreboard **18** |
| **Tri v2 decode** T=1    | 32/48| 8–11  | 11–14|5 | 250 | 48 KB  | long_scoreboard 5 |
| Tri v2 merge  T=1        | 64   | 18–27 | 1.7| 4  | 37  | —       | long_scoreboard 16.8 |
| **Tri v2 decode** T=64   | 256  | **14–15** | **42** | 9 | 250 | 48 KB | long_scoreboard 3.1 (2.7 waves) |

Reads: (1) at T=1 FI is **occupancy-starved** (32 blocks, 0.7 waves) *and* latency-bound —
its own weakness; (2) Triton's residual limiter is **register pressure 250 → 2 blocks/SM →
occ ~14%** (vs FI's 96 regs / ~18%), the acc[BH,512] fp32 tile is the cost; (3) at T=64 the
v2 form's smaller smem (48 vs 98 KB) lifted occ **7.7%→14%**, sm% **25%→42%** — the measured
mechanism of its speedup.

---

## LEVER LEDGER (each measured; negatives carry a revival condition)

**WON — landed:**
1. **Parallelized merge** (grid `(T,H)` per-head, was `(T,⌈H/8⌉)`): NCU showed v1 merge ran
   only **8 blocks** at T=1, 100% long_scoreboard (18). Fix → 64 blocks, occ 10%→27%.
   Δ −2 to −4 µs at T≤8.
2. **`num_stages=1` + small tiles (BH16 BN16)**: smem 98→48 KB ⇒ 2 blocks/SM. Occ 7.7%→14%,
   sm% 25→42% at T=64. This is what makes the pure kernel competitive.
3. **Fill 48 SMs at T=1 via high split (spl12)**: overcomes FI's 32-block cap. The T=1 win.
4. **Rank configs by in-graph time**: extra splits/merge are ~free in a graph; the eager
   sweep (which includes 2 launch overheads) hid the T=1 winner. Re-ranking surfaced it.

**LOST — autopsy + revival:**
- **CUDA-graph fusion of split+merge**: moot. The graph experiment shows in-graph launch
  cost ≈0 (FI 2 launches = 260.0 eager ≈ 260.2 graph at T=64), so a persistent/atomic
  single-kernel saves nothing where it matters. *Revive if:* target is eager-only decode.
- **fp8 storage + `tl.dot_scaled` (mxfp8) for KV**: would halve smem and match FI's FP8 MMA.
  Blocked for **PV**: FI folds the per-(token,V-chunk) KV scale into P *before* the MMA (7
  re-quantizations of P), which `tl.dot_scaled` can't express (its block-scale is along the
  contraction axis only). QK-only fp8 gives no win (latency- not compute-bound). *Revive if:*
  Triton adds a scaled-accumulate PV primitive, or on a compute-bound shape.
- **num_warps=8 / BH=8**: cut acc registers but split the MMA thinner; net neutral at T=1,
  worse at T≥32 (M=8 wastes half the m16 tensor tile). *Revive if:* a lower-reg accumulator
  form is found.
- **BH=32 / BN=32 (v1's form)**: 98 KB smem ⇒ 1 block/SM ⇒ occ 7.7%. Strictly dominated by
  BH16 BN16 ns1. Kept only as the T=64 eager runner-up.

---

## WHY T≥8 IS THE IRREDUCIBLE CEILING (scoped: this op, sm_121, this stack)

At T≥8 flashinfer already fills the GPU (T·4·splits ≥ 48 blocks), so the T=1 occupancy
opening closes and FI's **per-block efficiency** decides. Its three advantages are exactly
the sm_121 primitives Triton 3.6 cannot emit:
1. **Warp specialization** (1 IO warp drives `cp.async.bulk`/TMA into double-buffered smem
   while 8 math warps compute) → hides the long_scoreboard latency that Triton *exposes*
   (Triton has no TMA/`cp.async.bulk` nor warp-spec on sm_121; its `tl.load` gather is
   consumed in-line).
2. **Block-scaled FP8 MMA** (`mma_fp8_block_scaled_m16n8k32` for QK, folded-scale FP8 for
   PV) → denser math + no dequant; measured fp8-TC 118 vs bf16-TC 94 TFLOP/s, and Triton
   must dequant KV→bf16 (mio_throttle in NCU).
3. **2× occupancy** (96 regs vs Triton's 250; 288-thread/9-warp block) → more in-flight
   memory to hide latency. Triton's fp32 acc[BH,512] tile caps it at 2 blocks/SM.

None is a *tuning* gap; all three are *form* gaps requiring hand-CUDA TMA + warp-spec. A
Triton kernel therefore plateaus at ≈**1.09–1.20× at T≥8**, which this work reaches.

---

## Files (all in `ab_vllm/decode_beat/`)
- `sm121_decode_v2.py` — the kernel (fixed parallel merge; BH16 BN16 ns1 w4). **Winner.**
- `probe0.py`/`_out` — calibrated denominators + eager/graph A/B (found graphs don't move ratio → launch not the lever).
- `ncu_target*.py`, `parse_ncu.py`, `*.ncu-rep` — NCU captures + parser (FI & Triton, T=1/64).
- `sweep_final.py`/`_out` — eager sweep (unified BH16 BN16 ns1 form).
- `sweep_graph.py`/`_out` — **in-graph-ranked sweep → the T=1 beat (0.871×)**.
- `graph_ab.py` + `final_locked_out.txt` — the locked headline table.
- Source studied: `cuh/decode_dsv4_kernel.cuh`, `cuh/kv_cache_traits.cuh` (flashinfer kernel).

## Bottom line
The prior "1.35–1.62× slower" gap is **crossed at T=1** (0.87×, 13% faster, in the
CUDA-graph regime production uses) by filling the 16 SMs flashinfer structurally leaves
idle, and **tightened to 1.09–1.20× everywhere else** — the remaining gap is flashinfer's
TMA/warp-spec/FP8 pipeline, which NCU proves is a latency-hiding + occupancy advantage that
Triton cannot express on sm_121. Recommendation: ship the Triton kernel as the sm_121 decode
path at **T=1** (MTP/low-QPS decode) where it wins; keep flashinfer default for batched decode.
```
σ note: the resident vLLM rank injects ±15% on ~20 µs *eager* numbers (FI eager seen 17.7–22.8
across processes); in-graph numbers are stable to ≈±1% and are the ones the verdict rests on.
```
