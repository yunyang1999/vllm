# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned glue for MLB's L2 token-routing layer.

vLLM's built-in replica choice is a Knuth hash of the token index, fused into
the same Triton kernel that records expert load
(``fused_moe/router/base_router.py``).  When an MLB routing algorithm is
selected, that kernel is bypassed: MLB decides the physical replica and this
module records the load separately.

Enable with ``VLLM_MLB_L2_ALGORITHM``, e.g. ``lplb``, ``dynamic``, ``static``.
Unset (the default) leaves vLLM's fused kernel in charge, so this module costs
nothing when it is not used.

Two things are deliberately *not* supported and fail loudly rather than
silently degrading:

* **Waterfill / shared-expert routing.**  On vLLM's CUDA path the shared expert
  is a per-rank replicated MLP that never enters EP dispatch, so there is no
  destination-rank decision to write back.
* **DBO (dual-batch overlap).**  MLB's ``RoutingRequest`` carries a single
  token count and its policies keep one solver state per layer, so two
  concurrently-executing micro-batches would clobber each other.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbLayerState

logger = init_logger(__name__)

ALGORITHM_ENV = "VLLM_MLB_L2_ALGORITHM"

_runtime: MlbRoutingRuntime | None = None


def mlb_l2_algorithm() -> str:
    """Configured MLB routing expression, or "" when MLB routing is off."""
    return os.environ.get(ALGORITHM_ENV, "").strip()


class VllmRoutingCollectives:
    """Expose vLLM's EP process group through MLB's generic transport API.

    Validated to be CUDA-graph capturable: vLLM routes EP collectives through
    pynccl, and capture happens inside vLLM's own ``graph_capture()`` context.
    """

    @property
    def rank(self) -> int:
        from vllm.distributed.parallel_state import get_ep_group

        return get_ep_group().rank_in_group

    @property
    def world_size(self) -> int:
        from vllm.distributed.parallel_state import get_ep_group

        return get_ep_group().world_size

    def all_reduce_sum(self, payload: torch.Tensor) -> torch.Tensor:
        from vllm.distributed.parallel_state import get_ep_group

        return get_ep_group().all_reduce(payload)


def _record_physical_load(
    physical_topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None,
) -> None:
    """Record per-physical-expert load for topk ids MLB already mapped.

    Reproduces the second half of ``_eplb_map_and_record_i32_kernel``: physical
    counting, the record gate, and the padding mask.  Everything stays on the
    device and is expressed as tensor arithmetic, so it is CUDA-graph safe --
    in particular ``record_enabled`` is folded in as a mask instead of being
    branched on, which would force a host sync.
    """
    if expert_load_view is None:
        return
    num_active_experts = physical_topk_ids.shape[-1]
    flat = physical_topk_ids.reshape(-1)
    valid = flat >= 0
    if num_unpadded_tokens is not None:
        positions = torch.arange(flat.numel(), device=flat.device)
        valid = valid & (
            (positions // num_active_experts) < num_unpadded_tokens.reshape(())
        )
    valid = valid & (record_enabled.reshape(()) != 0)
    expert_load_view.scatter_add_(
        0,
        flat.clamp(min=0).long(),
        valid.to(expert_load_view.dtype),
    )


class MlbRoutingRuntime:
    """Holds the MoELoadBalancer instance and the placement geometry."""

    def __init__(
        self,
        algorithm: str,
        *,
        ep_size: int,
        ep_rank: int,
        num_logical_experts: int,
        num_physical_experts: int,
        physical_to_logical_map: torch.Tensor,
    ) -> None:
        from moe_load_balancer import MoELoadBalancer

        self.algorithm = algorithm
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts
        # Model-level [layers, num_physical] map.  Kept here rather than on the
        # layer state so that MixtureOfExperts.set_eplb_state -- a public model
        # interface -- does not have to change.
        self.physical_to_logical_map = physical_to_logical_map
        self._default_replicas: torch.Tensor | None = None
        self._mlb = MoELoadBalancer.from_algorithm(
            algorithm,
            ep_size=ep_size,
            source_rank=ep_rank,
            experts_per_rank=num_physical_experts // ep_size,
            collectives=VllmRoutingCollectives(),
        )
        logger.info(
            "MLB L2 routing enabled (algorithm=%s, ep_size=%d, "
            "logical=%d, physical=%d)",
            algorithm,
            ep_size,
            num_logical_experts,
            num_physical_experts,
        )

    def set_physical_to_logical_map(self, mapping: torch.Tensor) -> None:
        self.physical_to_logical_map = mapping

    def _snapshot(self, layer_state: EplbLayerState, layer_id: int) -> Any:
        from moe_load_balancer.adapters.vllm import to_placement_snapshot

        defaults = self._default_replicas
        return to_placement_snapshot(
            layer_state,
            layer_id,
            physical_to_logical_map=self.physical_to_logical_map[layer_id],
            num_logical_experts=self.num_logical_experts,
            num_physical_experts=self.num_physical_experts,
            ep_size=self.ep_size,
            default_physical_for_logical=(
                None if defaults is None else defaults[layer_id]
            ),
        )

    def _rebuild_default_replicas(self) -> None:
        """Synthesize the per-rank default replica table vLLM does not keep.

        Recomputed only when a placement is committed, never per forward.
        """
        from moe_load_balancer.adapters.vllm import nearest_replica_table

        self._default_replicas = torch.stack(
            [
                nearest_replica_table(
                    self._logical_to_physical_map[layer],
                    self._logical_replica_count[layer],
                    ep_rank=self.ep_rank,
                    num_local_physical_experts=(
                        self.num_physical_experts // self.ep_size
                    ),
                )
                for layer in range(self._logical_to_physical_map.shape[0])
            ]
        )

    def register_logical_maps(
        self, logical_to_physical_map: torch.Tensor, logical_replica_count: torch.Tensor
    ) -> None:
        """Keep the model-level maps so snapshots can be built without a layer.

        vLLM commits a rearrangement with in-place ``copy_`` into these
        tensors, so holding them here observes every future placement without
        re-registration.
        """
        self._logical_to_physical_map = logical_to_physical_map
        self._logical_replica_count = logical_replica_count
        self._rebuild_default_replicas()

    def on_placement_committed(self) -> None:
        """Rebuild placement-derived policy state after a committed rearrange.

        MLB does not track a placement version; it refreshes derived state only
        at this explicit lifecycle event.  Must be called after vLLM has moved
        weights and updated its live metadata -- never from an uncommitted plan.
        """
        from moe_load_balancer.core.types import PlacementSnapshot

        # KNOWN ISSUE (measured 2026-08-11, DeepSeek-style MoE, TP2/EP2):
        # with the `lplb` policy this hook kills the worker at the first
        # rearrangement -- silently, with no Python traceback.  Steady-state
        # lplb routing is fine: with rearrangement disabled the same build
        # serves correctly, and with this hook skipped rearrangement also runs
        # clean, so the fault is inside the per-layer rebuild below rather than
        # in lplb's routing or in vLLM's weight movement.
        #
        # Ruled out by experiment: racing the `non_blocking=True` map commit (a
        # full torch.cuda.synchronize() here does not help) and GPU memory
        # pressure (unchanged at gpu_memory_utilization=0.55).  The leading
        # remaining hypothesis is a candidate-table *width* change: vLLM pads
        # `logical_to_physical_map` with `_pad_out_tensor`, so max-replicas can
        # differ before and after a rearrangement, and LPLBL2Router.prepare_layer
        # documents shape-changing updates as the path that replaces -- rather
        # than updates in place -- its prepared state.
        #
        # Until that is fixed, `MLB_SKIP_COMMIT_HOOK=1` is the workaround: the
        # policy then keeps serving from its initial placement state.  That is
        # only correct while the placement it derived state from is still live,
        # so it is a triage aid, not a supported configuration.
        if os.environ.get("MLB_SKIP_COMMIT_HOOK") == "1":
            logger.warning("MLB placement-commit hook skipped (MLB_SKIP_COMMIT_HOOK=1)")
            return

        self._rebuild_default_replicas()
        num_layers = self._logical_to_physical_map.shape[0]
        for layer_id in range(num_layers):
            self._mlb.on_placement_committed(
                PlacementSnapshot(
                    layer_id=layer_id,
                    num_logical_experts=self.num_logical_experts,
                    num_physical_experts=self.num_physical_experts,
                    ep_size=self.ep_size,
                    num_local_physical_experts=(
                        self.num_physical_experts // self.ep_size
                    ),
                    physical_to_logical_map=self.physical_to_logical_map[layer_id],
                    logical_to_physical_candidates=self._logical_to_physical_map[
                        layer_id
                    ],
                    logical_to_physical_count=self._logical_replica_count[layer_id],
                )
            )

    def route(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return physical topk ids chosen by MLB, and record the load."""
        from moe_load_balancer.adapters.vllm import (
            to_routing_request,
            to_vllm_topk_ids,
        )

        layer_id = layer_state.moe_layer_idx
        if layer_id is None:
            raise RuntimeError(
                "MLB routing requires EplbLayerState.moe_layer_idx; the layer "
                "was registered by a path that does not set it."
            )

        decision = self._mlb.route_tokens(
            to_routing_request(
                layer_id=layer_id,
                logical_topk_ids=topk_ids,
                topk_weights=topk_weights,
                placement=self._snapshot(layer_state, layer_id),
                token_count=num_unpadded_tokens,
            )
        )
        physical = to_vllm_topk_ids(decision)
        _record_physical_load(
            physical,
            layer_state.expert_load_view,
            layer_state.should_record_tensor,
            num_unpadded_tokens,
        )
        return physical


def init_mlb_routing(
    *,
    ep_size: int,
    ep_rank: int,
    num_logical_experts: int,
    num_physical_experts: int,
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> MlbRoutingRuntime | None:
    """Create the routing runtime if ``VLLM_MLB_L2_ALGORITHM`` is set."""
    global _runtime
    algorithm = mlb_l2_algorithm()
    if not algorithm:
        return None
    _runtime = MlbRoutingRuntime(
        algorithm,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        physical_to_logical_map=physical_to_logical_map,
    )
    _runtime.register_logical_maps(logical_to_physical_map, logical_replica_count)
    return _runtime


def get_mlb_routing() -> MlbRoutingRuntime | None:
    return _runtime


def reset_mlb_routing() -> None:
    """Test hook."""
    global _runtime
    _runtime = None
