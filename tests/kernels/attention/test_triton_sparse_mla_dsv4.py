# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness for the GB10 split-KV sparse-MLA decode kernel (DeepSeek-V4).

Covers decode (seq=1, seq=8 MTP) and prefill (seq=512) at the real DSV4-Flash-0731
shape H=64, d_v=448, rope=64, top_k=512. fp8 inputs vs an fp32 torch reference.
"""
import pytest
import torch

triton = pytest.importorskip("triton")
from vllm.v1.attention.ops.triton_sparse_mla_dsv4 import triton_sparse_mla_dsv4  # noqa: E402

FP8 = torch.float8_e4m3fn


def _ref(q_nope, q_rope, kv, indices, sm_scale, d_v):
    seq, H, _ = q_nope.shape
    d_tail = q_rope.shape[-1]
    qn, qr, kvf = q_nope.float(), q_rope.float(), kv.float().squeeze(1)
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.float32)
    for s in range(seq):
        idx = indices[s, 0]
        valid = idx >= 0
        gk = kvf[idx.clamp(min=0)]
        k_main, k_tail = gk[:, :d_v], gk[:, d_v:d_v + d_tail]
        qk = (qn[s] @ k_main.T + qr[s] @ k_tail.T) * sm_scale
        qk = qk.masked_fill(~valid[None, :], float("-inf"))
        out[s] = torch.softmax(qk, dim=-1) @ k_main
    return out.unsqueeze(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("seq", [1, 8, 512])
def test_sparse_mla_dsv4_matches_reference(seq):
    torch.manual_seed(0)
    H, D_V, D_TAIL, TOPK, PAGES = 64, 448, 64, 512, 8192
    dim = D_V + D_TAIL
    sm_scale = 1.0 / dim ** 0.5
    dev = "cuda"
    qn = (torch.randn(seq, H, D_V, device=dev) * 0.3).to(FP8)
    qr = (torch.randn(seq, H, D_TAIL, device=dev) * 0.3).to(FP8)
    kv = (torch.randn(PAGES, 1, dim, device=dev) * 0.3).to(FP8)
    idx = torch.stack([torch.randperm(PAGES, device=dev)[:TOPK] for _ in range(seq)])
    indices = idx.to(torch.int32).unsqueeze(1)

    out = triton_sparse_mla_dsv4(qn, qr, kv, indices, sm_scale, d_v=D_V)
    ref = _ref(qn, qr, kv, indices, sm_scale, D_V)
    assert out.shape == (1, seq, H, D_V)
    rel = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 0.05, f"rel-err {rel} too high at seq={seq}"
