# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned glue for MLB's L2 token-routing layer.

vLLM's built-in replica choice is a Knuth hash of the token index, fused into
the same Triton kernel that records expert load
(``fused_moe/router/base_router.py``).  When an MLB routing algorithm is
selected the kernel is kept: MLB supplies the per-expert replica shares it
solved for, and the kernel samples from them instead of hashing.  Recording,
the record gate and the padding mask therefore stay where vLLM put them, with
nothing reimplemented alongside.

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
import time
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
        balancer: object | None = None,
        expert_weights: "Any | None" = None,
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
        # A logical expert can hold at most one slot plus every redundant one,
        # so this bounds the replica count without reading the map off device.
        self._max_replicas = num_physical_experts - num_logical_experts + 1
        self._default_replicas: torch.Tensor | None = None
        # Until a placement commits, assume there is a choice to make; the
        # first commit replaces this with what the placement actually offers.
        self._placement_offers_replica_choice: bool = True
        # The same table in the layout the fused kernel reads: one candidate
        # column per logical expert, and a replica count of 1 so the kernel's
        # `hash % count` lands on it. Built once per placement.
        self._fixed_map: torch.Tensor | None = None
        self._fixed_counts: torch.Tensor | None = None
        # [num_physical_experts] — slot i belongs to rank (i // experts_per_rank).
        # Passed to every PlacementSnapshot so LPLB can incorporate cross-GPU
        # transfer cost; SGLang always provides this, we build it from topology.
        from moe_load_balancer.adapters.vllm import build_physical_to_rank_map
        self._physical_to_rank_map = build_physical_to_rank_map(
            num_physical_experts,
            ep_size,
            device=physical_to_logical_map.device,
        )
        # Injected when an engine-scoped integration owns the balancer, so L1
        # and L2 are served by one instance rather than two that cannot see
        # each other.
        self._mlb = balancer if balancer is not None else MoELoadBalancer.from_algorithm(
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
        # A policy whose answer depends only on the committed placement does
        # not need the per-forward call at all: bake the answer into a map the
        # fused kernel already knows how to read, and the boundary disappears.
        # getattr, because an older MLB has no such field.
        self.dispatch_fixed_by_placement = getattr(
            caps, "dispatch_fixed_by_placement", False
        )
        # Declared by the pipeline rather than inferred from "has a post-TopK
        # policy": a replica policy can route from placement alone, and
        # gathering the EP-wide load for it costs a collective per step that is
        # then discarded.
        self.consumes_global_logical_count = getattr(
            caps, "consumes_global_logical_count", caps.requires_post_topk_routing
        )
        self.max_input_staleness_steps = getattr(caps, "max_input_staleness_steps", 0)
        self.supports_concurrent_microbatches = getattr(
            caps, "supports_concurrent_microbatches", False
        )
        self.graph_stability = getattr(caps, "graph_stability", "stable")
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

        # Graph-safe global count cache for LPLB.
        #
        # LPLB's LP solve needs global_logical_count: how many tokens each
        # logical expert received across all EP ranks.  The naive path inside
        # MLB does count_logical_experts(topk_ids) + EP all_reduce per layer
        # per forward, which is 40 NCCL collectives per step -- and NCCL
        # cannot be captured in a CUDA graph.
        #
        # The fix: split "compute local count" (CUDA kernel, graph-safe) from
        # "all_reduce" (outside the graph, once per step).
        #
        # _logical_count_local[layer, expert]: local counts accumulated per-layer
        #   inside the graph by count_logical_experts; reset each step.
        # _logical_count_global[layer, expert]: all-reduced result of the previous
        #   step, stable GPU address, read by LP solve inside the graph.
        #
        # The LP solve therefore uses counts stale by one step, which is an
        # excellent approximation for decode (distribution barely changes) and
        # acceptable for prefill (first step uses zeros → hash routing, then
        # converges).
        num_moe_layers = physical_to_logical_map.shape[0]
        dev = physical_to_logical_map.device
        # Gated on redundancy as well as on the policy.  With no redundant
        # experts every logical has exactly one replica, the LP is degenerate,
        # and LPLB answers with identity dispatch without ever reading the
        # count -- so keeping these buffers would buy a per-layer count kernel
        # plus one collective per step for a result that is thrown away.
        # Leaving them None also makes MLB decline to gather the count itself
        # (needs_global_logical_count), so nothing downstream pays for it.
        # MLB_KEEP_ZERO_REDUNDANCY_COUNTS=1 restores the ungated behaviour so the
        # cost of that discarded work can be measured against this, rather than
        # only argued for.
        keep_degenerate = os.environ.get("MLB_KEEP_ZERO_REDUNDANCY_COUNTS") == "1"
        if self.consumes_global_logical_count and (
            self._max_replicas > 1 or keep_degenerate
        ):
            self._logical_count_local: torch.Tensor | None = torch.zeros(
                num_moe_layers, num_logical_experts, dtype=torch.float32, device=dev
            )
            self._logical_count_global: torch.Tensor | None = torch.zeros(
                num_moe_layers, num_logical_experts, dtype=torch.float32, device=dev
            )
        else:
            self._logical_count_local = None
            self._logical_count_global = None
        # True once finalize_step_counts has run at least once (first step uses
        # zeros → hash routing fallback via selection is None).
        self._logical_count_ready = False
        # Reduced by the dummy path so the collective stays symmetric.
        self._count_scratch: torch.Tensor | None = None

        # MLB_FRESH_COUNTS=1 withholds the pre-computed count so MLB gathers it
        # itself, per layer per forward. That is what the SGLang adapter does and
        # what LPLB's published numbers were measured against; the count here is
        # one step stale so the collective can live outside the CUDA graph.
        # Staleness is a good approximation in decode, where the routing
        # distribution moves slowly, and a worse one in chunked prefill, where
        # consecutive chunks can differ sharply -- so a measured LPLB regression
        # cannot be attributed to the algorithm without checking this. Read once:
        # replica_shares runs per layer per forward.
        self._fresh_counts = os.environ.get("MLB_FRESH_COUNTS") == "1"

        # Diagnostic capture of the LP's inputs and output. Off unless a
        # directory is named; bounded so a long run cannot fill the disk.
        self._dump_lp_dir = os.environ.get("MLB_DUMP_LP") or None
        self._dump_lp_left = int(os.environ.get("MLB_DUMP_LP_N", "120"))
        # Skip the opening calls: the first steps legitimately carry zeros
        # while the one-step-stale pipeline fills, and sampling only those
        # would mistake a warm-up transient for steady state.
        self._dump_lp_skip = int(os.environ.get("MLB_DUMP_LP_SKIP", "0"))

        # MLB_TIME_L2=<n> times n route_tokens calls with a device sync on each
        # side. The sync makes the number meaningful and the measurement
        # invasive, so it is off unless asked for -- diagnostic only.
        # Set by replica_shares when a policy answers with ids; read by
        # resolve_routing in the same call.
        self._last_physical_ids: torch.Tensor | None = None
        # Set by plan_placement() after an UltraEP L1 solve, read by
        # _snapshot() on every forward until the next solve replaces them.
        # None for every other L1 policy, and before the first solve.
        #
        # The candidate table and quota must come from the same solve: the
        # quota indexes replicas by column against MLB's own candidate
        # ordering, which is not guaranteed to be the same table -- same
        # width, even -- as vLLM's own logical_to_physical_map, independently
        # rebuilt from physical_to_logical_map through compute_logical_maps.
        # So route with MLB's own table for this policy, not vLLM's.
        self._ultraep_rank_quota_prefix: torch.Tensor | None = None
        self._ultraep_logical_to_physical: torch.Tensor | None = None
        self._ultraep_replica_counts: torch.Tensor | None = None
        self._time_l2 = int(os.environ.get("MLB_TIME_L2", "0"))
        self._l2_us: list[float] = []

        # Real, traffic-driven placement refresh with real weight transfer,
        # additive to plan_placement()'s slow, wall-clock-cadence path above:
        # that path still owns this model's *initial* placement (the buffers
        # this refresh later writes per-layer slices into do not exist
        # before its first solve -- see the None-guard in
        # _ultraep_fast_refresh), this only adds much more frequent updates
        # on top of it. The weight transfer itself is UltraEP's own private
        # execution detail (moe_load_balancer.policies.fused.
        # ultraep_weight_transfer.UltraEPWeightTransfer) for making memory
        # match a placement its L1 solve already decided -- not a general
        # MLB protocol, since MLB has never defined one for that step; this
        # file only solves placement and feeds it topk_ids, never touches
        # ultra_ep's Manager directly. UltraEPL2Router (unchanged) stays the
        # dispatch mechanism, fed by the same deterministic solve this
        # refresh also hands the transfer backend, so the two agree without
        # an explicit data bridge.
        self._ultraep_transfer: Any = None
        self._ultraep_refresh_gate: Any = None
        self._ultraep_min_representative_tokens = int(
            os.environ.get("MLB_ULTRAEP_REFRESH_MIN_TOKENS", "8")
        )
        if algorithm == "ultraep" and expert_weights is not None:
            self._init_ultraep_fast_refresh(expert_weights)

    def _init_ultraep_fast_refresh(self, expert_weights: Any) -> None:
        """One-time setup: hand UltraEP's own weight-transfer backend this
        rank's own master expert weights so it has real memory to move data
        between.

        Eager, not lazy like the SGLang reference's ``UltraEPExpertTransfer``
        (which registers on its *first* transfer() call, because the object
        that owns it there is constructed before weights are loaded): vLLM
        hands this runtime ``model.expert_weights`` at the same call site
        that constructs it, so there is no "not loaded yet" window to defer
        past.

        Assumes exactly two per-layer weight tensors (fc1/w13-fused, then
        fc2/w2), each shaped ``[num_local_physical_experts, ...]`` -- true
        for the unquantized case this was verified against. A model whose
        ``expert_weights`` also carries separate quantization-scale tensors
        needs ``UltraEPWeightTransfer.register_weights`` to grow scale
        support; not attempted here.
        """
        try:
            import ultra_ep  # noqa: F401 -- import-only probe, see below.
        except ImportError:
            logger.warning_once(
                "ultraep fast refresh requires the ultra_ep package (not "
                "installed); staying on the slow EplbState.rearrange() "
                "cadence only."
            )
            return

        # Probed above, before get_ep_group(): a caller-side check, not
        # register_weights()'s own -- that one still does its own `from
        # ultra_ep import Manager` internally (the actual point of use, and
        # the right behavior for a caller that skips this pre-check). This
        # one exists only so a process_group=get_ep_group().device_group
        # argument -- evaluated eagerly, before register_weights's body ever
        # runs -- is never computed when ultra_ep is not installed. A test
        # environment with no real EP process group but no ultra_ep either
        # (test_ultraep_fast_refresh_degrades_gracefully_without_ultra_ep)
        # depends on this ordering: get_ep_group() would assert before
        # register_weights() got a chance to raise ImportError.
        from moe_load_balancer.policies.fused.ultraep_weight_transfer import (
            UltraEPWeightTransfer,
        )
        from vllm.distributed import get_ep_group

        num_local_master = self.num_logical_experts // self.ep_size
        num_local_physical = self.num_physical_experts // self.ep_size
        num_local_redundant = num_local_physical - num_local_master

        transfer = UltraEPWeightTransfer()
        transfer.register_weights(
            expert_weights,
            ep_size=self.ep_size,
            num_local_master_experts=num_local_master,
            num_local_redundant_experts=num_local_redundant,
            process_group=get_ep_group().device_group,
        )

        from moe_load_balancer import RefreshGate

        self._ultraep_transfer = transfer
        interval = int(os.environ.get("MLB_ULTRAEP_REFRESH_INTERVAL", "64"))
        self._ultraep_refresh_gate = RefreshGate(interval)
        logger.info(
            "UltraEP fast refresh enabled (interval=%d representative "
            "batches, min_tokens=%d, num_layers=%d, "
            "num_local_master=%d, num_local_redundant=%d)",
            interval,
            self._ultraep_min_representative_tokens,
            len(expert_weights),
            num_local_master,
            num_local_redundant,
        )

    def finalize_step_counts(self) -> None:
        """All-reduce per-layer local counts and update the stable LP input buffer.

        Call at the START of each forward pass (from EplbState.prepare_forward).
        By the time this runs, the previous step's count_logical_experts results
        are already in _logical_count_local (written per-layer inside the graph or
        in the eager forward).

        One EP collective for all 40 layers combined replaces the 40 per-layer
        collectives that MLB's _global_logical_count would otherwise issue.
        Running outside the graph means NCCL is never captured -- graphs only
        see the LP solve kernels reading from _logical_count_global.
        """
        if self._logical_count_local is None or self._logical_count_global is None:
            return
        # Never run inside a CUDA graph capture stream.  prepare_forward is
        # normally called outside graph context, but guard explicitly.
        if torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_ep_group
        ep_group = get_ep_group()
        ep_group.all_reduce(self._logical_count_local)
        self._logical_count_global.copy_(self._logical_count_local)
        self._logical_count_local.zero_()
        self._logical_count_ready = True

    def match_step_counts_collective(self) -> None:
        """Issue the count collective without touching the count pipeline.

        A rank running a dummy batch has to take part in the collective, or the
        EP group falls out of order and its peers wait in dispatch until DeepEP
        times out. It must not run `finalize_step_counts`, though: that copies
        the local buffer into the global one and then clears the local. Called a
        second time within a step -- which is exactly what a dummy forward
        beside a real one does -- it copies the freshly cleared buffer over the
        counts that were just published, and the LP then solves against an
        all-zero load for the rest of the run.

        So the dummy path reduces a scratch tensor of the same shape instead:
        same collective, same size, no effect on what the solve reads.
        """
        if self._logical_count_local is None:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_ep_group

        if self._count_scratch is None:
            self._count_scratch = torch.zeros_like(self._logical_count_local)
        get_ep_group().all_reduce(self._count_scratch)

    def set_physical_to_logical_map(self, mapping: torch.Tensor) -> None:
        self.physical_to_logical_map = mapping

    def _snapshot(
        self,
        layer_state: EplbLayerState,
        layer_id: int,
    ) -> Any:
        from moe_load_balancer.adapters.vllm import (
            collapse_candidates_to_local,
            to_placement_snapshot,
        )

        # Narrowing an expert's candidates to its local replica is a *routing
        # decision* ("never pay for a cross-GPU hop"), not a view of the
        # placement, so it belongs to the policy that asked for it rather than
        # to every policy.  SGLang draws the line the same way: the collapse
        # lives in `logical_to_rank_dispatch_physical_map`, which it only builds
        # when the pipeline reports `requires_rank_dispatch_map` -- while
        # `init_by_eplb`, the path taken on every rebalance, hands the policy
        # the full global list.
        #
        # Applying it unconditionally answers the question LPLB exists to ask:
        # with a local replica always winning, a rank holding a copy has nothing
        # left to solve, so the LP is skipped outright and the rest see only the
        # experts they do not host.
        # vLLM pads the candidate map to MAX_EXPERT_REDUNDANCY + 1 (1024)
        # columns whatever the configured redundancy, and MLB answers with a
        # table the same width as the candidates it was given. The kernel scans
        # that table with a compile-time loop, so handing over the padded width
        # would have made a share-table answer impossible to compile. Every
        # column past `_max_replicas` is padding on both sides, so trimming to
        # it changes no decision.
        candidates = layer_state.logical_to_physical_map[:, : self._max_replicas]
        counts = layer_state.logical_replica_count
        # `requires_rank_dispatch_map` is true only for `static`, and static no
        # longer calls this method at all -- its answer is resolved once per
        # placement instead (see `dispatch_fixed_by_placement`). Kept rather
        # than deleted: it is what a future rank-dispatch-map policy would need,
        # and removing it now would be removing untested surface, not dead code.
        if self.requires_rank_dispatch_map:
            candidates, counts = collapse_candidates_to_local(
                candidates,
                counts,
                ep_rank=self.ep_rank,
                num_local_physical_experts=(
                    self.num_physical_experts // self.ep_size
                ),
            )

        defaults = self._default_replicas
        quota = self._ultraep_rank_quota_prefix
        if quota is not None:
            # The quota's replica columns are only meaningful against the
            # candidate table MLB solved them from -- not vLLM's own
            # candidates, independently rebuilt from physical_to_logical_map
            # and not guaranteed to share its width, let alone its ordering.
            candidates = self._ultraep_logical_to_physical[layer_id]
            counts = self._ultraep_replica_counts[layer_id]
        return to_placement_snapshot(
            layer_state,
            layer_id,
            physical_to_logical_map=self.physical_to_logical_map[layer_id],
            num_logical_experts=self.num_logical_experts,
            num_physical_experts=self.num_physical_experts,
            ep_size=self.ep_size,
            candidates=candidates,
            counts=counts,
            default_physical_for_logical=(
                None if defaults is None else defaults[layer_id]
            ),
            physical_to_rank_map=self._physical_to_rank_map,
            rank_quota_prefix=None if quota is None else quota[layer_id],
        )

    def _rebuild_default_replicas(self) -> None:
        """Synthesize the per-rank default replica table vLLM does not keep.

        Only `static` replica routing reads it (MLB reports that through
        ``requires_rank_dispatch_map``), and it is recomputed only when a
        placement is committed -- never per forward.
        """
        if not self.requires_rank_dispatch_map:
            self._default_replicas = None
            self._fixed_map = None
            self._fixed_counts = None
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
                    # Passed rather than inferred: inference reads the largest
                    # physical id present, which undercounts whenever the top
                    # ranks hold no replica, and this rank can then fall outside
                    # the table it just built.
                    ep_size=self.ep_size,
                )
                for layer in range(self._logical_to_physical_map.shape[0])
            ]
        )

        if not self.dispatch_fixed_by_placement:
            self._fixed_map = None
            self._fixed_counts = None
            return

        # [layers, num_logical] -> [layers, num_logical, 1]. One column, so the
        # kernel's gather strides by 1 instead of by the candidate map's padded
        # width of MAX_EXPERT_REDUNDANCY + 1, and reads a table small enough to
        # stay cached.
        self._fixed_map = (
            self._default_replicas.to(self._logical_to_physical_map.dtype)
            .unsqueeze(-1)
            .contiguous()
        )
        # Every expert has exactly one candidate in that layout, so `hash %
        # count` is 0 for every token and the kernel returns column 0 -- which
        # is the replica this policy chose. Same answer as calling the policy
        # per forward, without the call.
        self._fixed_counts = torch.ones(
            self._default_replicas.shape[1],
            dtype=self._logical_replica_count.dtype,
            device=self._default_replicas.device,
        ).contiguous()

    def fixed_dispatch_maps(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The placement-resolved map and counts for a layer, if there is one.

        Returned in place of calling the routing pipeline: for a policy that
        decides from placement alone, the per-forward call produced this same
        answer at a measured 426 us per layer against a policy-free path that
        costs nothing.
        """
        if self._fixed_map is None or self._fixed_counts is None:
            return None
        return self._fixed_map[layer_id], self._fixed_counts

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
        # Does this placement give any expert a second copy? If not, every
        # replica policy -- static, dynamic, dynamic_random, lplb alike -- can
        # only return the identity, and consulting one is pure cost.
        #
        # Read once here rather than per forward: `.any()` on a CUDA tensor is
        # a device sync, which is cheap at a rearrangement and ruinous inside a
        # forward. Kept as a Python bool for that reason.
        #
        # This belongs to the placement, not to a policy. LPLB happens to
        # short-circuit itself (`num_red_log == 0` leaves its LP with nothing
        # to solve), static does not -- it consults its nearest-replica table
        # whatever the redundancy is. Leaving each policy to notice the
        # degenerate case independently is how that asymmetry arose, and it is
        # exactly what made a red0 row -- where by construction no policy can
        # differ -- report a 2.6% spread between them.
        self._placement_offers_replica_choice = bool(
            (logical_replica_count > 1).any().item()
        )
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
                    # Same trim as `_snapshot`, and it has to be the same: the
                    # policy sizes its per-layer state from whichever snapshot
                    # reaches it first, and this one does -- it runs for every
                    # layer at startup. Committing the padded width here left
                    # the solver emitting 1024-wide tables that the mapping
                    # kernel then could not compile.
                    logical_to_physical_candidates=self._logical_to_physical_map[
                        layer_id
                    ][:, : self._max_replicas],
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
        # The nearest-replica table is *this* side's derived state, not the
        # policy's, so it is refreshed on every commit. Gating it on
        # requires_placement_state left a policy that keeps no state of its
        # own -- static, which reads this table and nothing else -- routing
        # by the placement the run started with. After a rearrangement most
        # defaults are no longer in their expert's candidate list, the share
        # table falls back to column 0 for them, and the traffic those
        # replicas exist to spread lands on one of them.

        self._rebuild_default_replicas()

        if not self.requires_placement_state:
            return
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

    def resolve_routing(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Ask MLB to route.

        Returns ``(replica_shares, physical_ids)``. Every replica policy
        resolves its own ids now, so the first element is always ``None`` --
        kept in the return shape because the caller's fused kernel still has a
        share-table mode (``HAS_REPLICA_PROB``) it did not stop supporting,
        even though nothing on the MLB side asks for it any more.
        """
        shares = self.replica_shares(
            topk_ids, topk_weights, layer_state, num_unpadded_tokens
        )
        ids = None if shares is not None else self._last_physical_ids
        return shares, ids

    def _ultraep_fast_refresh(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> None:
        """Real, traffic-driven placement refresh + weight transfer, one layer.

        Not CUDA-graph-safe: issues real collectives (all_gather, a
        distributed weight transfer, a barrier) whenever it actually
        refreshes, so it must never run under capture -- same reason
        ``finalize_step_counts`` guards on the same check.

        Gated on ``_ultraep_rank_quota_prefix`` already being allocated:
        those whole-model buffers are only created by ``plan_placement()``'s
        slow-path solve (see ``mlb_runtime.py``'s ``plan_placement``), which
        still owns this model's *initial* placement. This refresh writes
        only a per-layer slice into buffers that solve already allocated; it
        does nothing before that first solve has run.

        Writes its result directly into the same whole-model buffers
        ``_snapshot()`` reads every forward (``physical_to_logical_map``,
        ``_ultraep_logical_to_physical``, ``_ultraep_replica_counts``,
        ``_ultraep_rank_quota_prefix``) rather than a separate pending store:
        ``replica_shares()`` calls ``_snapshot()`` again immediately after
        this returns, on its way to dispatch, so there is no window where a
        separate "pending" representation could go stale relative to what
        this just wrote.

        The placement solve above is this file's own (MLB's L1, through
        ``self._mlb.plan_placement``); physically moving weight data to
        match it is UltraEP's own private execution detail
        (``moe_load_balancer.policies.fused.ultraep_weight_transfer.
        UltraEPWeightTransfer``, constructed in
        ``_init_ultraep_fast_refresh``) -- this method hands it the same
        ``topk_ids`` the solve above was seeded from and nothing else, so
        the two agree without an explicit data bridge.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        if self._ultraep_rank_quota_prefix is None:
            return
        if _current_stage() == "decode":
            # UltraEP is explicitly a prefill-time algorithm (the paper's own
            # scoping; SGLang's reference should_refresh gates the same way,
            # rejecting only its provable "decode" and accepting everything
            # else, including its own "mixed"). A decode-only batch must not
            # count toward this layer's refresh interval at all -- it never
            # reaches is_due(), the same way SGLang's gate never touches its
            # own batch counter for a rejected stage.
            return

        # is_due() is pure Python counter bookkeeping -- no GPU access, never
        # syncs -- and is always cheap to call. Computing `representative`
        # below is not: num_unpadded_tokens.item() is a real CPU-GPU sync,
        # and paying it on every one of interval-1-out-of-interval calls
        # where should_refresh could not possibly say yes anyway is exactly
        # the class of per-forward cost this file goes to real lengths to
        # avoid elsewhere (see finalize_step_counts's whole reason for
        # existing). Check is_due first; only compute representative and
        # call should_refresh when it says a refresh could happen this call.
        if not self._ultraep_refresh_gate.is_due(layer_id):
            return

        representative = (
            num_unpadded_tokens is not None
            and int(num_unpadded_tokens.item())
            >= self._ultraep_min_representative_tokens
        )
        if not self._ultraep_refresh_gate.should_refresh(
            layer_id, representative=representative
        ):
            return

        from moe_load_balancer.adapters.vllm import to_placement_request
        from moe_load_balancer.kernels.expert_count import count_logical_experts
        from vllm.distributed import get_ep_group

        ep_group = get_ep_group().device_group
        local_count = count_logical_experts(
            topk_ids, self.num_logical_experts, dtype=torch.int32
        )
        per_rank_count = local_count.new_empty((self.ep_size, self.num_logical_experts))
        torch.distributed.all_gather_into_tensor(
            per_rank_count, local_count.contiguous(), group=ep_group
        )

        request = to_placement_request(
            per_rank_count[:, None, :],
            num_replicas=self.num_physical_experts,
            num_ranks=self.ep_size,
            num_groups=1,
            num_nodes=1,
            algorithm="ultraep",
            ep_rank=self.ep_rank,
        )
        plan = self._mlb.plan_placement(request)

        # The request carried one synthetic layer (dim 0 of per_rank_count),
        # so the plan's own layer index is always 0 regardless of layer_id --
        # layer_id only selects where in *our* whole-model buffers this one
        # layer's slice of the plan lands.
        self.physical_to_logical_map[layer_id].copy_(plan.physical_to_logical_map[0])
        self._ultraep_logical_to_physical[layer_id].copy_(
            plan.logical_to_all_physical_map[0]
        )
        self._ultraep_replica_counts[layer_id].copy_(plan.logical_to_physical_count[0])
        self._ultraep_rank_quota_prefix[layer_id].copy_(
            plan.metadata["rank_quota_prefix"][0]
        )

        self._ultraep_transfer.transfer(layer_id, topk_ids)

    def replica_shares(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | None:
        """Ask MLB to route this layer, recording resolved ids as a side effect.

        Always returns ``None``: every replica policy resolves its own ids
        (``self._last_physical_ids``) rather than answering with a share table
        for this method to hand back. The resolved ids are what
        ``resolve_routing`` reads after this call returns.

        ``routed_scaling_factor`` is plumbed but left at its default: MLB only
        reads it when materializing a shared-expert decision, and vLLM's CUDA
        path has no shared-expert dispatch to materialize.
        """

        from moe_load_balancer.adapters.vllm import to_routing_request
        from moe_load_balancer.kernels.expert_count import count_logical_experts

        self._last_physical_ids = None
        layer_id = layer_state.moe_layer_idx
        if layer_id is None:
            raise RuntimeError(
                "MLB routing requires EplbLayerState.moe_layer_idx; the layer "
                "was registered by a path that does not set it."
            )

        # Accumulate this layer's LOCAL count into the stable buffer. Skipped
        # during Dynamo tracing, where the counting kernel may not be handled;
        # under graph capture and eager it runs normally and is captured
        # alongside the copy_.
        #
        # This happens BEFORE the readiness check below, not after it. With the
        # write on the far side, the opening step returned early without
        # recording anything, the next step's reduction still saw zeros, and the
        # solve after that was the first to meet real data -- a three-step
        # warm-up where the design calls for two, leaving one step of every run
        # solving against an all-zero load.
        if self._logical_count_local is not None and not torch._dynamo.is_compiling():
            local = count_logical_experts(topk_ids, self.num_logical_experts)
            self._logical_count_local[layer_id].copy_(local)
        # A placement with no redundancy leaves a replica policy nothing to
        # decide: every logical expert has exactly one physical copy, so the
        # only answer any of them can give is the one the framework's own
        # kernel already produces. Return before the pipeline runs.
        #
        # After the count write above, not before it -- the counts feed the
        # next placement and LPLB's solve, and neither stops being wanted just
        # because this placement happens to be degenerate.
        if not self._placement_offers_replica_choice:
            return None

        # Only policies that read the load have to wait for it. Gating every
        # policy on this readiness flag disabled the ones that never asked for
        # a count: their buffers are not allocated, so the flag never flips and
        # they returned None for the lifetime of the run.
        if self.consumes_global_logical_count and not self._logical_count_ready:
            return None

        if self._ultraep_transfer is not None:
            self._ultraep_fast_refresh(layer_id, topk_ids, num_unpadded_tokens)

        # Pass the PREVIOUS step's global count to MLB.  The stable tensor
        # slice has a fixed GPU address, so it is safe to use inside a
        # captured CUDA graph: on replay the LP solve reads whatever value
        # finalize_step_counts deposited before the graph was launched.
        global_count = (
            None
            if self._fresh_counts
            else (
                self._logical_count_global[layer_id]
                if self._logical_count_global is not None
                else None
            )
        )

        if self._time_l2:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        decision = self._mlb.route_tokens(
            to_routing_request(
                layer_id=layer_id,
                logical_topk_ids=topk_ids,
                topk_weights=topk_weights,
                placement=self._snapshot(layer_state, layer_id),
                stage=_current_stage(),
                token_count=num_unpadded_tokens,
                routed_scaling_factor=routed_scaling_factor,
                global_logical_count=global_count,
            )
        )
        if self._time_l2:
            torch.cuda.synchronize()
            self._l2_us.append((time.perf_counter() - _t0) * 1e6)
            if len(self._l2_us) >= self._time_l2:
                import statistics

                logger.info(
                    "MLB L2 solve cost: n=%d  median=%.0f us  p90=%.0f us  "
                    "mean=%.0f us",
                    len(self._l2_us), statistics.median(self._l2_us),
                    sorted(self._l2_us)[int(0.9 * len(self._l2_us))],
                    statistics.mean(self._l2_us),
                )
                self._l2_us.clear()
                self._time_l2 = 0

        # MLB_DUMP_LP=<dir> captures what the solve was given and what it
        # returned, so the achievable headroom can be computed offline instead
        # of inferred from end-to-end throughput. Diagnostic only. Unconditional
        # on which policy is active -- global_count/candidates/counts come from
        # layer_state regardless, and lp_probability is None for anything but
        # LPLB, which is exactly what "no LP ran here" should look like.
        if self._dump_lp_skip > 0:
            self._dump_lp_skip -= 1
        elif self._dump_lp_dir is not None and self._dump_lp_left > 0:
            import os as _os

            self._dump_lp_left -= 1
            lp_probability = decision.metadata.get("lp_probability")
            torch.save(
                {
                    "layer_id": layer_id,
                    "ep_size": self.ep_size,
                    "num_local": self.num_physical_experts // self.ep_size,
                    "candidates": layer_state.logical_to_physical_map[
                        :, : self._max_replicas
                    ].cpu(),
                    "counts": layer_state.logical_replica_count.cpu(),
                    "global_count": (
                        None if global_count is None else global_count.float().cpu()
                    ),
                    "probability": (
                        None
                        if lp_probability is None
                        else lp_probability.float().cpu()
                    ),
                },
                _os.path.join(
                    self._dump_lp_dir,
                    f"lp_r{self.ep_rank}_l{layer_id}_{self._dump_lp_left}.pt",
                ),
            )

        # Every replica policy resolves its own ids now; none answers with a
        # share table for this runtime to apply. Record the ids for
        # resolve_routing to pick up.
        ids = decision.routed_physical_topk_ids
        if ids is None or ids is topk_ids:
            return None
        self._last_physical_ids = ids
        return None



class VllmMlbIntegration:
    """One balancer per engine, serving both placement and routing.

    SGLang gives an engine a single MoELoadBalancer and lets L1 and L2 share
    it. vLLM did not: the L1 policy built a throwaway instance on every
    rebalance and L2 held a second one in a module global. Two instances cannot
    see each other's state, which rules out any policy whose placement decision
    depends on what routing observed -- the premise of the predictive layer --
    and a module global also rules out more than one engine in a process.

    The lookup below stays module-level because vLLM's placement policy is a
    classmethod with nowhere to hang an instance. Ownership is not: the
    integration is created and released by EplbState, so its lifetime is the
    engine's.
    """

    def __init__(self, algorithm: str = "") -> None:
        from moe_load_balancer import MoELoadBalancer

        self.algorithm = algorithm
        self._balancer_kwargs: dict | None = None
        self._balancer = None if algorithm else MoELoadBalancer()
        self.routing: MlbRoutingRuntime | None = None

    def balancer(self):
        """The single MoELoadBalancer this engine uses."""
        if self._balancer is None:
            from moe_load_balancer import MoELoadBalancer

            if self._balancer_kwargs is None:
                # Placement can be asked for before the routing geometry is
                # known; a plain planner answers L1 and is replaced in place
                # once routing supplies the topology.
                self._balancer = MoELoadBalancer()
            else:
                self._balancer = MoELoadBalancer.from_algorithm(
                    self.algorithm, **self._balancer_kwargs
                )
        return self._balancer

    def bind_routing(self, **kwargs) -> MlbRoutingRuntime:
        """Create the routing runtime against this engine's balancer."""
        from moe_load_balancer import MoELoadBalancer

        self._balancer_kwargs = {
            "ep_size": kwargs["ep_size"],
            "source_rank": kwargs["ep_rank"],
            "experts_per_rank": kwargs["num_physical_experts"] // kwargs["ep_size"],
            "collectives": VllmRoutingCollectives(),
        }
        self._balancer = MoELoadBalancer.from_algorithm(
            self.algorithm, **self._balancer_kwargs
        )
        self.routing = MlbRoutingRuntime(
            self.algorithm, balancer=self._balancer, **kwargs
        )
        return self.routing


_integration: VllmMlbIntegration | None = None


def get_mlb_integration() -> VllmMlbIntegration:
    """The engine's integration, created on first use."""
    global _integration
    if _integration is None:
        _integration = VllmMlbIntegration(mlb_l2_algorithm())
    return _integration


def set_mlb_integration(integration: VllmMlbIntegration | None) -> None:
    global _integration, _runtime
    _integration = integration
    _runtime = None if integration is None else integration.routing


def l2_pipeline_capabilities(algorithm: str):
    """Capabilities the named L2 pipeline declares, or None if unknown.

    Stateless, so it answers before any engine exists -- configuration
    validation needs it. Living here rather than at each call site keeps the
    balancer a detail of one module: everything else in vLLM asks this file.
    """
    if not algorithm:
        return None
    try:
        from moe_load_balancer.core.routing_pipeline import RoutingPipeline

        return RoutingPipeline.from_value(algorithm).capabilities
    except Exception:
        return None


def l2_inapplicable_reason(algorithm: str, num_redundant_experts: int) -> str | None:
    """Why the named policy cannot act on this deployment, or None.

    The reason is the policy's own words. vLLM replicates the shared expert
    per rank and never dispatches it, which is what makes a shared-expert
    policy inapplicable here whatever the redundancy -- a distinction the
    policy draws, not one this side can assume.
    """
    if not algorithm:
        return None
    try:
        from moe_load_balancer import ExpertDeploymentConfig
        from moe_load_balancer.core.routing_pipeline import RoutingPipeline

        pipeline = RoutingPipeline.from_value(algorithm)
    except Exception:
        return None
    return pipeline.is_applicable(
        ExpertDeploymentConfig(
            num_redundant_experts=num_redundant_experts,
            routes_shared_expert=False,
        )
    )


def plan_placement(request):
    """Run L1 through the engine's balancer and hand back vLLM's one map."""
    from moe_load_balancer.adapters.vllm import to_vllm_physical_to_logical

    integration = get_mlb_integration()
    plan = integration.balancer().plan_placement(request)
    # UltraEP is the only L1 policy that publishes this; every other plan's
    # metadata simply lacks the key, so this stays a no-op for them. Routing
    # geometry can lag placement (see balancer()'s docstring), so there may be
    # no runtime to hand the quota to yet -- it reads whatever the next solve
    # after bind_routing() leaves here.
    #
    # The candidate table and replica counts come along too, not just the
    # quota: they must be read from this same plan, not rebuilt from
    # phy2log through vLLM's own compute_logical_maps, or the quota's column
    # ordering and width silently stop matching the table L2 routes against.
    #
    # Stashed here rather than after the caller commits the weight move: no
    # forward can observe this quota paired with the placement it belongs to
    # before that commit happens, because MlbEplbPolicy requires synchronous
    # (use_async=False) rearrangement -- nothing else runs on this thread
    # between this return and register_logical_maps(). An async L1 path would
    # need this to move to the commit step instead.
    quota = plan.metadata.get("rank_quota_prefix")
    if quota is not None and integration.routing is not None:
        integration.routing._ultraep_rank_quota_prefix = quota
        integration.routing._ultraep_logical_to_physical = plan.logical_to_all_physical_map
        integration.routing._ultraep_replica_counts = plan.logical_to_physical_count
    return to_vllm_physical_to_logical(plan), plan


def placement_request(*args, **kwargs):
    """Translate vLLM's rebalance arguments into the neutral request."""
    from moe_load_balancer.adapters.vllm import to_placement_request

    return to_placement_request(*args, **kwargs)

def _reject_graphs_with_rearranging_placement_state(rearranges: bool) -> None:
    """Refuse the one combination that can fault the GPU.

    A policy that keeps per-layer state rebuilds it when a committed placement
    changes that state's layout, and LPLB *replaces* the tensors in that case
    rather than updating them in place -- its own ``prepare_layer`` docstring
    notes that keeping a previously captured graph valid across such a change
    "requires a future fixed-shape LPLB state representation".

    A CUDA graph captures pointers. Once the solver's buffers move, replaying
    the graph reads freed memory: observed as ``Xid 43`` on two devices and a
    dead worker. It is intermittent -- it needs a rearrangement that actually
    changes the layout, so a short benchmark can pass and a long serving run
    can fault. That is worth failing loudly for.

    Both escapes keep MLB routing available: pin the placement (leave
    rearrangement off, which is how the reported LPLB numbers were measured),
    or run eager.
    """
    if not rearranges:
        return
    from vllm.config import get_current_vllm_config

    try:
        compilation = get_current_vllm_config().compilation_config
    except Exception:  # noqa: BLE001 - no config context: leave the decision alone
        return
    if getattr(compilation, "cudagraph_mode", None) is None:
        return
    if compilation.cudagraph_mode.name == "NONE":
        return
    raise ValueError(
        "VLLM_MLB_L2_ALGORITHM keeps per-layer solver state, and EPLB "
        "rearrangement can change that state's layout. LPLB replaces its "
        "tensors on such a change, which a captured CUDA graph cannot follow "
        "-- replay then reads freed memory (Xid 43). Either disable "
        "rearrangement (eplb_config step_interval high, or policy that does "
        "not re-plan), or set enforce_eager=True. Fixing this properly needs "
        "a fixed-shape solver state in moe_load_balancer."
    )


def init_mlb_routing(
    *,
    algorithm: str,
    ep_size: int,
    ep_rank: int,
    num_logical_experts: int,
    num_physical_experts: int,
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    rearranges: bool = False,
    expert_weights: Any | None = None,
) -> MlbRoutingRuntime | None:
    """Create the routing runtime for a configured L2 algorithm.

    ``algorithm`` comes from ``EPLBConfig.l2_algorithm``, which has already
    resolved the environment default and cleared itself for placements no L2
    policy can act on. Passing it in rather than re-reading the environment is
    what makes that decision binding.

    ``expert_weights`` is the model's own ``expert_weights`` (one entry per
    MoE layer, present at this same call site) -- only ``ultraep`` reads it,
    to register real weight pointers with its transfer runtime. Every other
    algorithm ignores it, so it is safe to leave unset.
    """
    global _runtime
    if not algorithm:
        return None
    integration = get_mlb_integration()
    integration.algorithm = algorithm
    _runtime = integration.bind_routing(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        physical_to_logical_map=physical_to_logical_map,
        expert_weights=expert_weights,
    )
    _runtime.register_logical_maps(logical_to_physical_map, logical_replica_count)
    # Keyed on the declared stability of the policy's state, not on whether it
    # keeps state at all. A policy whose placement-derived tensors keep their
    # addresses across a rearrangement is safe to capture; refusing it because
    # some other policy is not would bar a combination that never faults.
    if _runtime.graph_stability == "realloc_on_placement_change":
        _reject_graphs_with_rearranging_placement_state(rearranges)
    return _runtime


def get_mlb_routing() -> MlbRoutingRuntime | None:
    return _runtime


def reset_mlb_routing() -> None:
    """Test hook. Releases the engine's integration along with the runtime."""
    global _runtime, _integration
    _runtime = None
    _integration = None
