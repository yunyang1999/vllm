# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated knobs for the decode expert-balance study. All off by default."""

import torch

import vllm.envs as envs


def maybe_force_uniform_routing(
    topk_ids: torch.Tensor, num_logical_experts: int
) -> torch.Tensor:
    """Replace the routed top-k with a perfectly balanced round-robin assignment.

    Token ``t`` takes logical experts ``(t*k + j) % E`` for ``j < k``: distinct
    experts per token and, over any run of tokens, counts that differ by at most
    one per expert. The model's outputs are garbage, so this is only a timing
    floor: the step time a layer would reach if the expert load were perfectly
    balanced, measured with the same kernels, batch and shapes. Gated by
    ``EVENMOE_FORCE_UNIFORM_ROUTING=1``.
    """
    if not envs.EVENMOE_FORCE_UNIFORM_ROUTING:
        return topk_ids
    num_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    base = torch.arange(num_tokens, device=device, dtype=torch.int64) * top_k
    offsets = torch.arange(top_k, device=device, dtype=torch.int64)
    ids = (base[:, None] + offsets[None, :]) % num_logical_experts
    return ids.to(topk_ids.dtype)
