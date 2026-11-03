# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated knobs for the decode expert-balance study. All off by default."""

import torch

import vllm.envs as envs


def maybe_force_uniform_routing(
    topk_ids: torch.Tensor, num_logical_experts: int, ep_size: int = 1, ep_rank: int = 0
) -> torch.Tensor:
    """Replace the routed top-k with a perfectly balanced assignment.

    Every rank numbers its own tokens from zero, so a plain ``(t*k + j) % E``
    sends every rank's tokens to the same low experts and *concentrates* load
    (measured 4.2x on 16 ranks). Instead each source rank deals its
    assignments round-robin over the destination ranks, starting at its own
    rank so the leftover of a partial cycle lands on different destinations for
    different sources, and cycles through that rank's local experts: per destination rank the load is equal to within one
    assignment per source rank, and per expert within a rank likewise. Assumes
    the identity layout (rank r holds logical experts [r*E/ep, (r+1)*E/ep)),
    i.e. run it without redundancy. The model's outputs are garbage, so this is
    only a timing floor. Gated by ``EVENMOE_FORCE_UNIFORM_ROUTING=1``.
    """
    if not envs.EVENMOE_FORCE_UNIFORM_ROUTING:
        return topk_ids
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    idx = (
        torch.arange(num_tokens, device=device, dtype=torch.int64)[:, None] * top_k
        + torch.arange(top_k, device=device, dtype=torch.int64)[None, :]
    )
    if ep_size > 1 and num_logical_experts % ep_size == 0:
        per_rank = num_logical_experts // ep_size
        dest_rank = (idx + ep_rank) % ep_size
        local = (idx // ep_size + ep_rank) % per_rank
        ids = dest_rank * per_rank + local
    else:
        ids = idx % num_logical_experts
    return ids.to(topk_ids.dtype)
