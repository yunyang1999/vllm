# DSv4-Flash sparse-MLA **PREFILL** on GB10 (sm_121): user's Triton kernel vs FlashInfer

**Verdict (plainly): YES — the user's Triton bf16 sparse-MLA prefill kernel BEATS FlashInfer
on GB10 by 1.36–1.43× at T=512/1024/2048/4096, correct to cos=1.0000.** This is the *opposite*
of the decode result (where FlashInfer won 1.2–1.6×). It holds even after the desktop-tuned
config is left untouched (the shipping default already wins 1.20–1.24×); GB10 re-tuning adds
another 1.10–1.19× on top. The winning GB10 config is **head_tile=32, (BLOCK_N,warps,stages)=(32,4,3)**
and it is verified turnkey as a `_PINNED_*` edit.

Device: **NVIDIA GB10, sm_121 (cc 12.1), 48 SMs, ~99 KB opt-in smem/block, 25 MB L2, LPDDR5X 273 GB/s.**
Container `dualspark-vllm:v0.25.1-nccl-2.30.4` (torch 2.11+cu130, triton 3.6.0, flashinfer 0.6.14).
Shape: H=64, d_qk=d_v=512 (448 fp8-e4m3 nope + 64 bf16 rope), topk=512, packed 584 B/token,
sm_scale=1/sqrt(512), pool=16384 tokens (9.6 MB, L2-resident). Same packed fp8 cache bytes / same
bf16 dequant of it / same indices / same scale for every arm. Timing = median CUDA-event iters
after warmup (3 reps; run-to-run spread <1% at these T). Reuses the validated `ab_bench.py`
DSv4 fp8 packing + fp32 reference. Every number below is pasted from a real container run.

---

## HEADLINE TABLE (locked, `final_ab.py`)

| T | FlashInfer prefill (µs) | user **bf16** best (µs) | config | user **fp8** (µs) | **bf16 speedup** | cos(bf16 vs fp32) |
|---:|---:|---:|:--|---:|---:|---:|
| 512  | 1374.14 | **1012.99** | ht32 BN32 w4 ns3 | 35978.4  | **1.36×** | 1.0000 |
| 1024 | 2678.98 | **1888.48** | ht32 BN32 w4 ns3 | 71941.8  | **1.42×** | 1.0000 |
| 2048 | 5269.57 | **3763.33** | ht32 BN32 w4 ns3 | 143963.4 | **1.40×** | 1.0000 |
| 4096 | 10483.90| **7308.51** | ht32 BN32 w4 ns3 | 288272.2 | **1.43×** | 1.0000 |

FlashInfer cos vs fp32 ref = 0.9997 (its inline-fp8 path); the user's bf16 kernel is **cos=1.0000**,
i.e. *closer* to the fp32 oracle than FlashInfer. All arms numerically correct.

The user's **fp8** paged path (`sparse_mla_prefill_paged_fp8`) is ~26–27× slower and is **not** a
contender on GB10 — see "Why fp8 is crippled" below. The bf16 path (`sparse_mla_prefill`) is the
GB10 prefill winner.

### Fairness: the bf16 arm's one-time pool dequant
FlashInfer and the user-fp8 arm read the packed 584 B fp8 cache directly. The user-bf16 arm reads a
pre-dequantized bf16 pool, which in production is produced once per layer by a dequant kernel.
Charged conservatively (naive full fp32→bf16 cast of 16384 rows) that prep is **~174 µs** — an upper
bound (a fused fp8-unpack reads 9.6 MB not 32 MB, ~½ that). Even with the full 174 µs added, the bf16
arm still wins at every T: `(bf16_best+prep)/FlashInfer` = **0.86 / 0.77 / 0.75 / 0.71** (T=512→4096),
i.e. 1.16–1.41×. At T≥1024 the prep is <9% and shrinks with T; it is also shared across the whole
prefill, not per-token.

---

## The winning GB10 config, and the re-tuning delta

The kernel ships with desktop-Blackwell pins (RTX 5080/5090/PRO 6000). On GB10 for H=64 those resolve
(measured) to **head_tile=16, tile config (BLOCK_N=64,warps=4,stages=2)** — the head-tiled path
re-derives `_config(dev, tile_h=16)` = `_PINNED[(12,1)]=(64,4,2)`, *not* the `_PINNED_WIDE_H` (16,8,2).
That default already beats FlashInfer. Re-tuning the (head_tile, tile) pair for GB10's 99 KB / 48-SM
budget moves the optimum to **head_tile=32, (32,4,3)**:

| T | FlashInfer | shipping default (ht16, 64/4/2) | GB10-retuned (ht32, 32/4/3) | default vs FI | retuned vs FI | retune gain |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 1374.14 | 1111.01 | 1012.99 | 1.24× | 1.36× | 1.10× |
| 1024 | 2678.98 | 2201.44 | 1888.48 | 1.22× | 1.42× | 1.17× |
| 2048 | 5269.57 | 4365.54 | 3763.33 | 1.20× | 1.40× | 1.16× |
| 4096 | 10483.90| 8684.67 | 7308.51 | 1.20× | 1.43× | 1.19× |

**Config landscape @ T=2048** (`final_ab.py`, real feasibility only, no silent smem step-down):
```
  ht=32 BN= 32 w=4 ns=3:   3755.26 us   <- winner
  ht=32 BN= 32 w=8 ns=2:   4168.99 us
  ht=32 BN= 32 w=4 ns=1:   4252.22 us
  ht=16 BN= 64 w=4 ns=2:   4367.42 us   (== shipping default; head-tile-16 unlocks BN64 but it loses)
  ht=16 BN= 64 w=8 ns=3:   4513.02 us
  ht=16 BN= 32 w=4 ns=3:   5552.13 us
  ht=64 BN= 16 w=8 ns=2:   5638.18 us   (monolithic H=64 -> BN capped at 16 by the 99KB wall)
  ht=16 BN= 16 w=8 ns=2:   6869.38 us
```
The lever is the **(head_tile, BLOCK_N) pair**: head_tile=32 shrinks the `[BLOCK_H,d_v]` fp32
accumulator (halving register pressure vs the H=64 monolith → 1 block/SM) while still leaving smem
room for BLOCK_N=32. Going further to head_tile=16 unlocks BLOCK_N=64 but the extra re-gather
(4 head-tiles instead of 2) and wider K tile don't pay — ht32/BN32 wins by ~14% over ht16/BN64.

### Pin recommendation (verified turnkey — `verify_pin.py`)
For `(12,1)` at H=64, set:
```python
_PINNED_HEAD_TILE[(12, 1)] = {32: 32, 64: 32}   # H=64 -> 2 head tiles of 32 (was ht=16)
_PINNED[(12, 1)]           = (32, 4, 3)          # tile_h=32 re-derives _config(dev,32); was (64,4,2)
```
With those two edits and **no call-site override** (smem fallback left ENABLED), the plain
`sparse_mla_prefill(...)` reproduces the swept best:
```
    T | no-override us | cos
  512 |         982.27 | 1.0000
 1024 |        1922.30 | 1.0000
 2048 |        3659.68 | 1.0000
 4096 |        7225.38 | 1.0000
```
(982/1922/3660/7225 µs = 1.40 / 1.39 / 1.44 / 1.45× over FlashInfer — matches/slightly beats the
explicit-override run.) **Caveat:** editing `_PINNED[(12,1)]` also changes the tile config for H≤32
monolithic callers and the split-K path on this arch. DSv4 at TP8 uses H=8 (which takes
`_PINNED_NARROW_H`, unaffected) and at TP1 H=64 (head-tiled, the case tuned here); the H=16/32
monolithic cases should be re-swept before landing, or wire the win as an explicit
`sparse_mla_prefill(..., head_tile=32, config=(32,4,3))` from the sm121 dispatch instead.

---

## Why FlashInfer LOSES at prefill (but wins at decode)

The only FlashInfer sparse-MLA path on sm120/sm121 is the **DSv4-native sparse *decode* kernel**
(`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` → `_SparseMLAPagedAttentionRunner`, the same
kernel used as the decode baseline). There is **no dedicated FlashInfer sparse *prefill* kernel**:
`flashinfer.mla.sparse_mla_prefill` does not exist, and vLLM's `FlashInferMLASparseSM120Impl.forward_mqa`
routes *every* token of a prefill through this decode kernel at `batch = num_actual_toks, q_len=1`.
So "FlashInfer prefill" = the sparse decode kernel run at batch=T. That kernel is a **split-K over the
topk dimension** with an HBM partial write + a separate merge pass — the right structure for decode
(T=1–64, little parallelism, split to fill 48 SMs) and dead weight for prefill:

* At prefill T, there are T×H = 32k–262k independent program-rows — the 48 SMs are saturated many
  times over, so split-K buys **no** occupancy and just adds partial/merge traffic. **Measured:**
  FlashInfer num_splits ∈ {1,2,4} gave identical time at T≤2048 (e.g. 1378.46 vs 1379.36 vs 1378.40 µs
  at T=512), and num_splits=2 was **1.9× WORSE** at T=4096 (19701 vs 10500 µs). It cannot fully shed
  the decode structure (it still carries `mid_out`/`mid_lse` and the merge) even at splits=1.

The user's kernel is a **purpose-built prefill kernel**: one Triton program per (token, head-tile),
online softmax held in registers, `splits=1`, **no HBM partials, no merge**. At prefill T the abundant
parallelism hides gather/mma latency via occupancy (the exact thing that hurt it at decode T=1), and
its leaner memory traffic (no partial write-back) is what the 1.4× is made of. Same op, opposite wall:
**decode is latency-exposed → FlashInfer's TMA/warp-spec async pipeline wins; prefill is
throughput/occupancy-saturated → the leaner Triton kernel wins.**

## Why the user's fp8 paged path is crippled on GB10 (~26–27× slower)

`sparse_mla_prefill_paged_fp8` launches **monolithic over heads** (`grid=(T,)`, BLOCK_H=64, no
head_tile knob) and holds an extra fp32 dequant staging tile. A [64,512] bf16 q-tile is 64 KB, so with
the fp32 staging tile the 99 KB smem admits **only BLOCK_N=16, num_stages=1** (feasibility map,
`smoke2.py`: BN=16 is the *only* fp8 config that fits; BN≥32 all OOR). Locked at BN=16 it does 32
scattered-gather iterations/token and cannot escape the wall the way the bf16 path does by tiling the
head axis. On sm_121 `_has_fp8_mma` is False so it upcasts to bf16 anyway (no fp8-mma benefit), and the
per-row byte-gather/dequant cost dominates. Net: 35978 / 71942 / 143963 / 288272 µs — a non-starter.
The fp8 path is a memory-lean *decode* / correctness form, not a GB10 prefill contender.

## The GB10 smem wall (true feasibility, fallback disabled — `smoke2.py`)
`sparse_mla_prefill` silently steps BLOCK_N down on OutOfResources; with that disabled the real map is:
```
bf16:  ht=64  BN16 OK  | BN32 OOR | BN64 OOR | BN128 OOR   (monolithic q[64,512]=64KB -> BN16 only)
       ht=32  BN16 OK  | BN32 OK  | BN64 OOR | BN128 OOR   (q[32,512]=32KB -> up to BN32)
       ht=16  BN16 OK  | BN32 OK  | BN64 OK(ns>=2) | BN128 OOR   (q[16,512]=16KB -> up to BN64)
fp8:   BN16(ns1) OK    | BN32 OOR | BN64 OOR | BN128 OOR
```
Head-tiling is the escape: smaller head_tile → smaller q-tile → room for a larger BLOCK_N. BLOCK_N=128
is smem-infeasible at any head_tile (kv + its transpose alone exceed 99 KB). The optimum sits at the
knee: head_tile=32 / BN=32.

## Limiter of the winning kernel (qualitative; NCU not required to settle the verdict)
The user's kernel is throughput-bound on GB10 prefill. Two structural caps remain, both inherent to
the Triton form (not tunable further here): (1) the `[BLOCK_H,d_v]` fp32 **register** accumulator
(128 regs/thread even at BLOCK_H=32 → limited blocks/SM), and (2) the **99 KB smem** cap on BLOCK_N
(≤32 at head_tile=32; ≤64 only by shrinking head_tile, which then over-re-gathers). Closing them would
need TMA async-gather + warp specialization + a smem (not register) accumulator — surfaces Triton
cannot emit on sm_121. But unlike decode, that residual does **not** cost the race here: the kernel is
already 1.4× ahead because FlashInfer's only available sparse kernel is the wrong *form* (decode
split-K) for the prefill regime.

---

## Method notes / honesty
* **Baseline identity.** DSv4-Flash is 512-wide latent (448+64), so vLLM's generic sm120 sparse wrapper
  (`trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k>0)`, which validates q head_dim==576, the
  v3.2/GLM 656 B path) does not accept it; the DSv4-native `trtllm_batch_decode_sparse_mla_dsv4` /
  `_SparseMLAPagedAttentionRunner` (584 B, q=512) is the correct FlashInfer kernel and is what is
  benchmarked — the same primary baseline the decode study used, extended to prefill T.
* **Dense ragged is a different computation.** `flashinfer.prefill.trtllm_ragged_attention_deepseek`
  and `BatchPrefillWithRaggedKVCacheWrapper` exist but attend the *full causal prefix* (dense), not the
  topk=512 sparse set DSv4 selects — for T=4096 that is ~8× more key work and a different (denser)
  result, so it is not a same-inputs baseline and is not what DSv4 dispatches.
* Every timed config is its real `(BLOCK_N, num_stages)` (the `_smem_fallbacks` silent step-down was
  patched to identity for the sweep/feasibility runs), so no reported config secretly ran a smaller tile.
* FlashInfer given its best num_splits per T (swept 1/2/4); the AutoTuner warmed before timing.
* Correctness cos measured on the fp32 reference over the dequantized cache (`ab_bench.fp32_reference`).

## Files (all in `kernel_test/prefill_ab/`)
- `probe_prefill.py` / stdout — enumerates FlashInfer sparse entrypoints on sm_121; confirms no
  dedicated sparse-prefill kernel; `_SparseMLAPagedAttentionRunner` runs at prefill T (cos 0.9997).
- `smoke.py`, `smoke2.py` (+ `_out.txt`) — correctness + **true** GB10 smem-feasibility map.
- `sweep_time.py` / `sweep_out.txt` — full (head_tile × BN × warps × stages) timing sweep, all 4 T.
- `final_ab.py` / `final_out.txt` — locked 3-rep A/B: FlashInfer vs default-pins vs retuned vs fp8 + landscape.
- `verify_pin.py` / `verify_pin_out.txt` — proves the `_PINNED_*` edit makes the no-override call
  reproduce the win (fallback ENABLED).

## Bottom line
On GB10 the user's Triton **bf16** sparse-MLA prefill kernel is **1.36–1.43× faster than FlashInfer**
across T=512–4096, correct to cos=1.0000 — a real win worth landing as the sm_121 prefill path. Pin
**head_tile=32, config=(32,4,3)** for `(12,1)` (verified turnkey). The **fp8** paged path is
smem-crippled (BN=16-locked, ~26×) and should not be used for prefill on GB10. FlashInfer loses here
not because it is badly tuned but because it has no sparse-*prefill* kernel on sm120/121 — its sparse
*decode* split-K kernel is the wrong form once the T×head parallelism saturates the 48 SMs.
```
(σ note: three independent in-session runs agree within ~2%; e.g. user bf16 T=2048 = 3737 / 3763 /
3660 µs across sweep/final/pinned, FlashInfer = 5288 / 5270 µs — the 1.4× gap is stable, not noise.)
```
