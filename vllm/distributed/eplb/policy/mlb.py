# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned glue for the framework-decoupled MoE Load Balancer (MLB).

MLB (``moe-load-balancer``) implements expert-placement and token-routing
policies once, framework-neutrally, and each serving framework keeps a thin
adapter.  This module is that adapter for vLLM's L1 placement slot: it
translates :class:`AbstractEplbPolicy` arguments into an MLB
``PlacementRequest``, runs the selected MLB policy, and translates the result
back into the ``physical_to_logical_map`` vLLM expects.

Enable with ``--eplb-config '{"policy": "mlb", "use_async": false}'``.
The MLB placement algorithm is selected with ``VLLM_MLB_L1_ALGORITHM``
(default ``auto``); see ``moe_load_balancer`` for the supported names.

Note that ``use_async`` must be disabled: vLLM's async rearrangement path is
validated only against the built-in policy (see ``EPLBConfig``), and MLB's
planner runs synchronously on the caller's thread.
"""

from __future__ import annotations

import os

import torch

from vllm.distributed.eplb.policy.abstract import AbstractEplbPolicy
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_ALGORITHM = "auto"
ALGORITHM_ENV = "VLLM_MLB_L1_ALGORITHM"


def _algorithm() -> str:
    return os.environ.get(ALGORITHM_ENV, DEFAULT_ALGORITHM)


class MlbEplbPolicy(AbstractEplbPolicy):
    """Delegate expert placement to the MoE Load Balancer core."""

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        try:
            from moe_load_balancer import MoELoadBalancer
            from moe_load_balancer.adapters.vllm import (
                to_placement_request,
                to_vllm_physical_to_logical,
            )
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "EPLB policy 'mlb' requires the moe-load-balancer package. "
                "Install it into this environment with "
                "`pip install -e /path/to/moe_load_balancer --no-deps`."
            ) from exc

        algorithm = _algorithm()

        # vLLM passes num_groups=0 for models without expert groups; MLB
        # expresses "no grouping" as a single group.
        request = to_placement_request(
            weight.float().cpu(),
            num_replicas=num_replicas,
            num_ranks=num_ranks,
            num_groups=num_groups or 1,
            num_nodes=num_nodes,
            algorithm=algorithm,
            old_physical_to_logical_map=(
                None
                if old_global_expert_indices is None
                else old_global_expert_indices.cpu()
            ),
        )

        plan = MoELoadBalancer().plan_placement(request)
        phy2log = to_vllm_physical_to_logical(plan).cpu().to(torch.int64)

        # MLB's L1 policies do not consume a previous placement, so the
        # slot-preservation pass that the built-in policy performs inside
        # rebalance_experts is applied here instead.  It only permutes slots
        # within a rank, so the placement MLB decided (which logical experts
        # live on which rank, and how many replicas each gets) is unchanged --
        # only their slot positions are, which is what avoids weight copies.
        if old_global_expert_indices is not None:
            phy2log = torch.from_numpy(
                DefaultEplbPolicy.preserve_intragpu_slots(
                    phy2log.numpy(),
                    num_ranks,
                    old_global_expert_indices.cpu().numpy(),
                )
            )

        logger.info_once(
            "EPLB placement delegated to MLB (algorithm=%s, resolved=%s)",
            algorithm,
            plan.metadata.get("resolved_policy", "unknown"),
        )
        return phy2log
