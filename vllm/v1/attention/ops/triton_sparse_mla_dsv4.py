# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV + split-head flash-decoding sparse-MLA for DGX Spark GB10 (sm_121).

DeepSeek-V4 sparse MLA attention over the indexer-selected top-k KV. The upstream
SGLang triton kernel launches ``grid=(seq,)`` (one CTA per query): correct, and a
good *prefill* kernel (thousands of CTAs), but at **decode** (seq=1, or 8 with MTP)
it leaves a 48-SM GB10 with 1-8 CTAs -> ~0.8% of LPDDR5X bandwidth (profiled).

This module fixes the decode form for consumer Blackwell (sm_121), where the
datacenter surfaces (tcgen05/TMEM/CLC/FlashMLA-sparse/FA-MLA-sparse) are all
unavailable and the op is bandwidth-bound + latency-exposed:

  * **split-KV**  : grid ``(seq, G, S)`` partitions top-k into ``S`` chunks (each
                    >= 32 keys, because the fp8 PV ``tl.dot`` contracts over
                    BLOCK_N and Blackwell requires K>=32). A cheap second pass
                    merges the S partials by log-sum-exp. Fills the SMs at seq=1.
  * **split-head**: ``BLOCK_H`` heads/CTA (default 16) shrinks the fp32 accumulator
                    ``acc[H, d_v]`` -> ``acc[BLOCK_H, d_v]``, cutting registers and
                    shared memory ~4x (92KB -> 37KB) so more blocks fit per SM.
                    Raises occupancy for prefill too (output is bitwise-identical
                    to the single-pass kernel there).

``num_kv_splits`` is adaptive: ``S ~ clamp(NUM_SMS / (seq*G), 1, top_k//32)``. At
prefill ``S`` collapses to 1 and the combine pass is skipped -> the single-pass
per-query kernel. Mirrors FlashMLA's ``num_sm_parts = num_sms / s_q``.

Measured on GB10 at the real DSV4-Flash-0731 shape (H=64, d_v=448, rope=64,
top_k=512), vs the original per-query kernel, same fp8 inputs:
    decode s1   155 us -> 28 us  (5.5x)   rel-err 0.031 vs fp32 ref
    decode s8   176 us -> 49 us  (3.6x)   rel-err 0.027
    prefill     1.48x, bitwise-identical output

Input contract (un-packed fp8 layout, matching the SGLang kernel):
    q_nope  [seq, H, d_v]        fp8_e4m3
    q_rope  [seq, H, d_tail]     fp8_e4m3
    kv      [num_pages, 1, dim]  fp8_e4m3   (dim = d_v + d_tail)
    indices [seq, 1, top_k]      int32      (-1 = padding slot)
    returns [1, seq, H, d_v]     bf16
"""
import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0            # e4m3 max on NVIDIA (non-fnuz)
NUM_SMS = 48               # GB10


@triton.jit
def _partial_kernel(q_nope_ptr, q_rope_ptr, kv_ptr, idx_ptr, pacc_ptr, plse_ptr,
        sm_scale, fp8_max, topk, split_keys,
        H: tl.constexpr, S: tl.constexpr, DIM: tl.constexpr,
        D_V: tl.constexpr, D_TAIL: tl.constexpr,
        BLOCK_H: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_DT: tl.constexpr,
        BLOCK_N: tl.constexpr):
    s_i = tl.program_id(0)      # query
    g = tl.program_id(1)        # head group
    j = tl.program_id(2)        # kv split
    h = g * BLOCK_H + tl.arange(0, BLOCK_H); hm = h < H
    dv = tl.arange(0, BLOCK_DV); dvm = dv < D_V      # pad non-pow2 d_v (448) and mask
    dt = tl.arange(0, BLOCK_DT); dtm = dt < D_TAIL
    q_main = tl.load(q_nope_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
                     mask=hm[:, None] & dvm[None, :], other=0.0).to(q_nope_ptr.dtype.element_ty)
    q_tail = tl.load(q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :],
                     mask=hm[:, None] & dtm[None, :], other=0.0).to(q_nope_ptr.dtype.element_ty)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], tl.float32)
    n = tl.arange(0, BLOCK_N)
    start = j * split_keys
    for k0 in range(0, split_keys, BLOCK_N):
        gk = start + k0 + n
        kmask = gk < topk
        idx = tl.load(idx_ptr + s_i * topk + gk, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0)
        kbase = kv_ptr + page[:, None] * DIM
        kv_main = tl.load(kbase + dv[None, :], mask=valid[:, None] & dvm[None, :], other=0.0).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :], mask=valid[:, None] & dtm[None, :], other=0.0).to(q_nope_ptr.dtype.element_ty)
        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_fp8 = (p * fp8_max).to(q_nope_ptr.dtype.element_ty)
        acc = acc * alpha[:, None] + tl.dot(p_fp8, kv_main).to(tl.float32) * (1.0 / fp8_max)
        m_i = m_new
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log(l_i))
    tl.store(pacc_ptr + s_i * H * S * D_V + h[:, None] * S * D_V + j * D_V + dv[None, :],
             acc, mask=hm[:, None] & dvm[None, :])
    tl.store(plse_ptr + s_i * H * S + h * S + j, lse, mask=hm)


@triton.jit
def _combine_kernel(pacc_ptr, plse_ptr, o_ptr,
        H: tl.constexpr, S: tl.constexpr, D_V: tl.constexpr,
        BLOCK_HC: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_S: tl.constexpr):
    s_i = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * BLOCK_HC + tl.arange(0, BLOCK_HC); hm = h < H
    dv = tl.arange(0, BLOCK_DV); dvm = dv < D_V
    sj = tl.arange(0, BLOCK_S); sjm = sj < S
    lse = tl.load(plse_ptr + s_i * H * S + h[:, None] * S + sj[None, :],
                  mask=hm[:, None] & sjm[None, :], other=-float("inf"))
    M = tl.max(lse, axis=1); M = tl.where(M == -float("inf"), 0.0, M)
    denom = tl.sum(tl.exp(lse - M[:, None]), axis=1)
    denom = tl.where(denom == 0.0, 1.0, denom)
    out = tl.zeros([BLOCK_HC, BLOCK_DV], tl.float32)
    for j in tl.static_range(BLOCK_S):
        if j < S:      # compile-time prune of padding splits (no tensor indexing on sm_121)
            lse_j = tl.load(plse_ptr + s_i * H * S + h * S + j, mask=hm, other=-float("inf"))
            wj = tl.exp(lse_j - M)
            accj = tl.load(pacc_ptr + s_i * H * S * D_V + h[:, None] * S * D_V + j * D_V + dv[None, :],
                           mask=hm[:, None] & dvm[None, :], other=0.0)
            out += wj[:, None] * accj
    out = out / denom[:, None]
    tl.store(o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
             out.to(o_ptr.dtype.element_ty), mask=hm[:, None] & dvm[None, :])


def _choose_splits(seq: int, H: int, topk: int, block_h: int, block_n: int) -> int:
    G = triton.cdiv(H, block_h)
    max_s = max(1, topk // block_n)              # each split >= block_n(>=32) keys (fp8 K>=32)
    s = max(1, round(NUM_SMS / max(1, seq * G)))
    return min(s, max_s)


def triton_sparse_mla_dsv4(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    block_h: int = 16,
    block_n: int = 32,
    num_kv_splits: int | None = None,
    block_hc: int = 4,
    partial_warps: int = 4,
    combine_warps: int = 2,
) -> torch.Tensor:
    """Sparse-MLA forward (decode + prefill) tuned for GB10 sm_121. See module docstring."""
    seq, H, _ = q_nope.shape
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    S = num_kv_splits or _choose_splits(seq, H, topk, block_h, block_n)
    G = triton.cdiv(H, block_h)
    split_keys = triton.cdiv(triton.cdiv(topk, S), block_n) * block_n
    BLOCK_DV = triton.next_power_of_2(d_v)
    BLOCK_DT = triton.next_power_of_2(d_tail)
    dev = q_nope.device
    pacc = torch.empty(seq, H, S, d_v, device=dev, dtype=torch.float32)
    plse = torch.empty(seq, H, S, device=dev, dtype=torch.float32)
    _partial_kernel[(seq, G, S)](
        q_nope.contiguous(), q_rope.contiguous(), kv, indices, pacc, plse,
        sm_scale, _FP8_MAX, topk, split_keys,
        H=H, S=S, DIM=dim, D_V=d_v, D_TAIL=d_tail,
        BLOCK_H=block_h, BLOCK_DV=BLOCK_DV, BLOCK_DT=BLOCK_DT, BLOCK_N=block_n,
        num_warps=partial_warps,
    )
    if S == 1:                       # prefill: no split -> partial already normalized
        return pacc.squeeze(2).to(torch.bfloat16).unsqueeze(0)
    out = torch.empty(seq, H, d_v, device=dev, dtype=torch.bfloat16)
    _combine_kernel[(seq, triton.cdiv(H, block_hc))](
        pacc, plse, out, H=H, S=S, D_V=d_v,
        BLOCK_HC=block_hc, BLOCK_DV=BLOCK_DV, BLOCK_S=triton.next_power_of_2(S),
        num_warps=combine_warps,
    )
    return out.unsqueeze(0)
