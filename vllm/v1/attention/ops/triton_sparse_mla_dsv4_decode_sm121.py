"""v2 decode: same head-tiled split-K bf16-dot core as sm121_decode, but with a
PROPERLY PARALLELIZED merge (grid (T,H), one block per head, full-D_V row per
block, vec loads) — the NCU on v1 showed the T=1 merge ran only 8 blocks and was
100% long-scoreboard bound. Also exposes num_warps for the decode so the sweep
can trade register pressure (222 regs @ w4 → occ 9%) for more warps.
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn


@triton.jit
def _decode_kernel(
    q_ptr, fp8_ptr, bf16_ptr, u8_ptr, idx_ptr,
    mid_o_ptr, mid_m_ptr, mid_l_ptr, o_ptr,
    sm_scale, topk, splits, HB,
    H: tl.constexpr, BLOCK_H: tl.constexpr, D: tl.constexpr, D_NOPE: tl.constexpr,
    SCALE_TILE: tl.constexpr, PAGE_SIZE: tl.constexpr, BYTES_PER_PAGE: tl.constexpr,
    ROW_BYTES: tl.constexpr, SCALE_BYTES_PER_TOKEN: tl.constexpr, S_OFFSET_BYTES: tl.constexpr,
    N_LANES: tl.constexpr, NOPE_LANES: tl.constexpr,
    BLOCK_N: tl.constexpr, NORMALIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    t = pid0 // HB
    hblk = pid0 % HB
    s = tl.program_id(1)

    h = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    d = tl.arange(0, D)
    is_nope = d < D_NOPE
    rope_col = tl.where(is_nope, 0, d - D_NOPE)
    s_lane = tl.arange(0, N_LANES)
    n = tl.arange(0, BLOCK_N)

    q = tl.load(q_ptr + t * H * D + h[:, None] * D + d[None, :],
                mask=hmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)

    n_tiles = tl.cdiv(topk, BLOCK_N)
    cpt = tl.cdiv(n_tiles, splits)
    k_lo = s * cpt * BLOCK_N
    k_hi = tl.minimum((s + 1) * cpt * BLOCK_N, topk)

    for k0 in tl.range(k_lo, k_hi, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < k_hi, other=-1)
        valid = idx >= 0
        loc = tl.where(valid, idx, 0).to(tl.int64)
        page = loc // PAGE_SIZE
        in_page = loc % PAGE_SIZE
        data_base = page * BYTES_PER_PAGE + in_page * ROW_BYTES
        scale_base = page * BYTES_PER_PAGE + S_OFFSET_BYTES + in_page * SCALE_BYTES_PER_TOKEN

        fp8v = tl.load(fp8_ptr + data_base[:, None] + d[None, :],
                       mask=valid[:, None] & is_nope[None, :], other=0.0).to(tl.float32)
        sc = tl.load(u8_ptr + scale_base[:, None] + s_lane[None, :],
                     mask=valid[:, None] & (s_lane < NOPE_LANES)[None, :], other=127).to(tl.float32)
        scale = tl.reshape(
            tl.broadcast_to(tl.exp2(sc - 127.0)[:, :, None], (BLOCK_N, N_LANES, SCALE_TILE)),
            (BLOCK_N, N_LANES * SCALE_TILE))
        nope = fp8v * scale
        rope = tl.load(bf16_ptr + (data_base[:, None] + D_NOPE) // 2 + rope_col[None, :],
                       mask=valid[:, None] & (is_nope[None, :] == 0), other=0.0)
        kv = tl.where(is_nope[None, :], nope.to(tl.bfloat16), rope)

        qk = tl.dot(q, tl.trans(kv)) * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
        m_i = m_new

    if NORMALIZE:
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        tl.store(o_ptr + t * H * D + h[:, None] * D + d[None, :],
                 (acc / l_safe[:, None]).to(o_ptr.dtype.element_ty), mask=hmask[:, None])
    else:
        mbase = ((t * H + h).to(tl.int64) * splits) + s
        tl.store(mid_m_ptr + mbase, m_i, mask=hmask)
        tl.store(mid_l_ptr + mbase, l_i, mask=hmask)
        tl.store(mid_o_ptr + mbase[:, None] * D + d[None, :],
                 acc.to(mid_o_ptr.dtype.element_ty), mask=hmask[:, None])


@triton.jit
def _merge_kernel2(
    mid_o_ptr, mid_m_ptr, mid_l_ptr, o_ptr, splits,
    H: tl.constexpr, D_V: tl.constexpr,
):
    """One block per (t,h). Full D_V row per block, split across warps' lanes.
    Grid (T, H) => T=1 gives H=64 blocks (v1 gave only 8)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D_V)
    bs = (t * H + h).to(tl.int64) * splits
    gm = -float("inf")
    for s in tl.range(0, splits):
        gm = tl.maximum(gm, tl.load(mid_m_ptr + bs + s))
    gm = tl.where(gm == -float("inf"), 0.0, gm)
    acc = tl.zeros([D_V], tl.float32)
    den = 0.0
    for s in tl.range(0, splits):
        m = tl.load(mid_m_ptr + bs + s)
        w = tl.where(m == -float("inf"), 0.0, tl.exp(m - gm))
        den += w * tl.load(mid_l_ptr + bs + s)
        acc += w * tl.load(mid_o_ptr + (bs + s) * D_V + d).to(tl.float32)
    den = tl.where(den == 0.0, 1.0, den)
    tl.store(o_ptr + (t * H + h).to(tl.int64) * D_V + d,
             (acc / den).to(o_ptr.dtype.element_ty))


_WS = {}


def _ws(device, T, H, splits, D, dtype=torch.bfloat16):
    n_o = T * H * splits * D
    n_s = T * H * splits
    key = (device, dtype)
    w = _WS.get(key)
    if w is None or w[0].numel() < n_o or w[1].numel() < n_s:
        w = (torch.empty(n_o, dtype=dtype, device=device),
             torch.empty(n_s, dtype=torch.float32, device=device),
             torch.empty(n_s, dtype=torch.float32, device=device))
        _WS[key] = w
    return w


def layout(cache, page_size, d_nope=448, d_rope=64, scale_tile=64):
    u8 = cache.view(torch.uint8)
    assert u8.is_contiguous()
    row_bytes = d_nope + d_rope * 2
    bpp = u8.shape[-1]
    return dict(
        fp8=u8.view(FP8).reshape(-1),
        bf16=u8.view(torch.bfloat16).reshape(-1),
        u8=u8.reshape(-1),
        bpp=bpp, row_bytes=row_bytes,
        sbpt=triton.next_power_of_2(d_nope // scale_tile),
        s_off=page_size * row_bytes,
        n_lanes=(d_nope + d_rope) // scale_tile,
        nope_lanes=d_nope // scale_tile,
    )


def sm121_sparse_decode(q, cache, idx, sm_scale, page_size, *, splits, block_n,
                        block_h=32, num_warps=4, num_stages=2, merge_warps=4,
                        d_nope=448, d_rope=64, scale_tile=64, out=None, lay=None):
    T, H, D = q.shape
    topk = idx.shape[-1]
    if lay is None:
        lay = layout(cache, page_size, d_nope, d_rope, scale_tile)
    if out is None:
        out = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    idx = idx.contiguous()
    HB = triton.cdiv(H, block_h)
    common = dict(
        H=H, BLOCK_H=block_h, D=D, D_NOPE=d_nope, SCALE_TILE=scale_tile,
        PAGE_SIZE=page_size, BYTES_PER_PAGE=lay["bpp"], ROW_BYTES=lay["row_bytes"],
        SCALE_BYTES_PER_TOKEN=lay["sbpt"], S_OFFSET_BYTES=lay["s_off"],
        N_LANES=lay["n_lanes"], NOPE_LANES=lay["nope_lanes"],
        BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages)

    if splits == 1:
        _decode_kernel[(T * HB, 1)](
            q, lay["fp8"], lay["bf16"], lay["u8"], idx, out, out, out, out,
            sm_scale, topk, 1, HB, NORMALIZE=True, **common)
        return out

    mid_o, mid_m, mid_l = _ws(q.device, T, H, splits, D, torch.bfloat16)
    _decode_kernel[(T * HB, splits)](
        q, lay["fp8"], lay["bf16"], lay["u8"], idx, mid_o, mid_m, mid_l, out,
        sm_scale, topk, splits, HB, NORMALIZE=False, **common)
    _merge_kernel2[(T, H)](
        mid_o, mid_m, mid_l, out, splits,
        H=H, D_V=D, num_warps=merge_warps, num_stages=1)
    return out
