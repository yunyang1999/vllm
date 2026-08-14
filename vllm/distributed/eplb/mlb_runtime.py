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


def _current_stage() -> str | None:
    """Map vLLM's current batch onto MLB's routing stage.

    SGLang hands MLB a discrete ForwardMode (EXTEND / DECODE / IDLE / ...).
    vLLM has no equivalent: with continuous batching and chunked prefill a
    single batch routinely mixes prefill and decode tokens, so a clean
    "prefill" stage does not exist.  Only decode is provable -- every request
    contributing exactly one token -- and everything else is reported as mixed
    rather than guessed at, so a stage-gated policy never runs on a stage it
    did not ask for.
    """
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    descriptor = getattr(get_forward_context(), "batch_descriptor", None)
    if descriptor is None:
        return None
    if (
        descriptor.uniform
        and descriptor.num_reqs is not None
        and descriptor.num_reqs > 0
        and descriptor.num_tokens == descriptor.num_reqs
    ):
        return "decode"
    return "mixed"


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
        # MLB declares what the framework has to prepare for the selected
        # policy, so none of the work below is done unconditionally: `lplb`
        # needs placement state refreshed on commit, `static` needs a per-rank
        # dispatch table, and only a pipeline with a post-TopK policy wants the
        # routing boundary at all.
        caps = self._mlb.routing_capabilities
        self.requires_post_topk_routing = caps.requires_post_topk_routing
        self.requires_placement_state = caps.requires_placement_state
        self.requires_rank_dispatch_map = caps.requires_rank_dispatch_map
        logger.info(
            "MLB L2 routing enabled (algorithm=%s, ep_size=%d, logical=%d, "
            "physical=%d, caps: post_topk=%s placement_state=%s "
            "rank_dispatch_map=%s)",
            algorithm,
            ep_size,
            num_logical_experts,
            num_physical_experts,
            self.requires_post_topk_routing,
            self.requires_placement_state,
            self.requires_rank_dispatch_map,
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

        Only `static` replica routing reads it (MLB reports that through
        ``requires_rank_dispatch_map``), and it is recomputed only when a
        placement is committed -- never per forward.
        """
        if not self.requires_rank_dispatch_map:
            self._default_replicas = None
            return

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

    def announce_initial_placement(self) -> None:
        """Tell MLB about the placement the model just loaded with.

        The contract is "after initial weight loading **and** after each
        committed update"; SGLang does the first half in
        ``ModelRunner._prepare_moe_topk``.  Skipping it leaves a policy to build
        its per-layer state lazily inside the first forward, from whatever
        placement that forward happens to see.
        """
        if not self.requires_placement_state:
            return
        num_layers = self._logical_to_physical_map.shape[0]
        self._commit_layers(range(num_layers))
        logger.info("MLB: announced initial placement for %d layers", num_layers)

    def _commit_layers(self, layer_ids) -> None:
        """Hand MLB the committed placement of the given layers."""
        from moe_load_balancer.core.types import PlacementSnapshot

        for layer_id in layer_ids:
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
                    default_physical_for_logical=(
                        None
                        if self._default_replicas is None
                        else self._default_replicas[layer_id]
                    ),
                )
            )

    def on_placement_committed(
        self, changed_layer_ids: list[int] | None = None
    ) -> None:
        """Refresh policy state for layers whose placement just changed.

        Must be called after vLLM has moved weights and updated its live
        metadata -- never from an uncommitted plan.

        ``changed_layer_ids`` matters for cost, not just tidiness.  A policy
        whose per-layer state layout changes has to rebuild it, and for `lplb`
        that means a JIT build plus warmup -- about 14.5 s per layer measured
        here.  Refreshing all 40 layers stalled the worker for ~102 s, long
        enough for the peer's collective to time out and kill it.  SGLang
        avoids this by passing the framework's own ``update_layer_ids``
        (``ModelRunner._notify_mlb_placement_committed``); vLLM does not track
        them, so Glue diffs the maps before committing and passes the result.
        """
        if not self.requires_placement_state:
            return
        self._rebuild_default_replicas()
        if changed_layer_ids is None:
            changed_layer_ids = list(range(self._logical_to_physical_map.shape[0]))
        if not changed_layer_ids:
            return
        self._commit_layers(changed_layer_ids)
        logger.info(
            "MLB: refreshed placement state for %d/%d layers",
            len(changed_layer_ids),
            self._logical_to_physical_map.shape[0],
        )

    def route(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """Return physical topk ids chosen by MLB, and record the load.

        ``routed_scaling_factor`` is plumbed but left at its default: MLB only
        reads it when materializing a shared-expert decision, and vLLM's CUDA
        path has no shared-expert dispatch to materialize.  Threading the real
        value down from the MoE layer would add invasiveness for a value
        nothing on this path consumes.
        """
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
                stage=_current_stage(),
                token_count=num_unpadded_tokens,
                routed_scaling_factor=routed_scaling_factor,
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
