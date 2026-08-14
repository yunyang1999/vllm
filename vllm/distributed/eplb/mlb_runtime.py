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
        # [num_physical_experts] — slot i belongs to rank (i // experts_per_rank).
        # Passed to every PlacementSnapshot so LPLB can incorporate cross-GPU
        # transfer cost; SGLang always provides this, we build it from topology.
        from moe_load_balancer.adapters.vllm import build_physical_to_rank_map
        self._physical_to_rank_map = build_physical_to_rank_map(
            num_physical_experts,
            ep_size,
            device=physical_to_logical_map.device,
        )
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
        # _lplb_local_count[layer, expert]: local counts accumulated per-layer
        #   inside the graph by count_logical_experts; reset each step.
        # _lplb_global_count[layer, expert]: all-reduced result of the previous
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
        if self.requires_post_topk_routing and (
            self._max_replicas > 1 or keep_degenerate
        ):
            self._lplb_local_count: torch.Tensor | None = torch.zeros(
                num_moe_layers, num_logical_experts, dtype=torch.float32, device=dev
            )
            self._lplb_global_count: torch.Tensor | None = torch.zeros(
                num_moe_layers, num_logical_experts, dtype=torch.float32, device=dev
            )
        else:
            self._lplb_local_count = None
            self._lplb_global_count = None
        # True once finalize_step_counts has run at least once (first step uses
        # zeros → hash routing fallback via selection is None).
        self._lplb_count_initialized = False


    def finalize_step_counts(self) -> None:
        """All-reduce per-layer local counts and update the stable LP input buffer.

        Call at the START of each forward pass (from EplbState.prepare_forward).
        By the time this runs, the previous step's count_logical_experts results
        are already in _lplb_local_count (written per-layer inside the graph or
        in the eager forward).

        One EP collective for all 40 layers combined replaces the 40 per-layer
        collectives that MLB's _global_logical_count would otherwise issue.
        Running outside the graph means NCCL is never captured -- graphs only
        see the LP solve kernels reading from _lplb_global_count.
        """
        if self._lplb_local_count is None or self._lplb_global_count is None:
            return
        # Never run inside a CUDA graph capture stream.  prepare_forward is
        # normally called outside graph context, but guard explicitly.
        if torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_ep_group
        ep_group = get_ep_group()
        ep_group.all_reduce(self._lplb_local_count)
        self._lplb_global_count.copy_(self._lplb_local_count)
        self._lplb_local_count.zero_()
        self._lplb_count_initialized = True

    def set_physical_to_logical_map(self, mapping: torch.Tensor) -> None:
        self.physical_to_logical_map = mapping

    def _snapshot(self, layer_state: EplbLayerState, layer_id: int) -> Any:
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
        # experts they do not host.  It also keeps the shares MLB returns
        # indexable by vLLM's own `logical_to_physical_map` -- `replica_shares`
        # passes on the probabilities alone, and their columns line up with that
        # map only because the candidates handed to MLB are that map.
        # vLLM pads the candidate map to MAX_EXPERT_REDUNDANCY + 1 (1024)
        # columns whatever the configured redundancy, and MLB answers with a
        # table the same width as the candidates it was given. The kernel scans
        # that table with a compile-time loop, so handing over the padded width
        # is what made the share-table branch impossible to compile. Every
        # column past `_max_replicas` is padding on both sides, so trimming to
        # it changes no decision.
        candidates = layer_state.logical_to_physical_map[:, : self._max_replicas]
        counts = layer_state.logical_replica_count
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

    def replica_shares(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | None:
        """Ask MLB how this layer's traffic should be split across replicas.

        Returns a ``[num_logical, max_replicas]`` share table laid out like
        vLLM's own ``logical_to_physical_map``, or None when the policy did not
        produce one -- in which case the caller keeps its built-in choice.

        The point of asking for the table rather than for per-token ids is that
        vLLM fuses replica selection and expert-load recording into one kernel.
        Taking over the ids means taking over the recording too, and
        reimplementing the record gate, the padding mask and the physical
        counting alongside a kernel that already does all three is duplication
        that silently drifts. Handing back a table keeps every one of those in
        vLLM's kernel, and MLB keeps the part that is genuinely its own: the
        solve.

        ``routed_scaling_factor`` is plumbed but left at its default: MLB only
        reads it when materializing a shared-expert decision, and vLLM's CUDA
        path has no shared-expert dispatch to materialize.
        """
        # First step: global count buffer is zero-initialised (no real data yet).
        # Return None → hash routing until finalize_step_counts has run once.
        if not self._lplb_count_initialized:
            return None

        from moe_load_balancer.adapters.vllm import to_routing_request
        from moe_load_balancer.kernels.expert_count import count_logical_experts

        layer_id = layer_state.moe_layer_idx
        if layer_id is None:
            raise RuntimeError(
                "MLB routing requires EplbLayerState.moe_layer_idx; the layer "
                "was registered by a path that does not set it."
            )

        # Accumulate LOCAL count for this layer into the stable buffer.
        # Skip during Dynamo tracing: count_logical_experts_cuda is a custom
        # CUDA kernel that Dynamo may not handle.  During actual execution
        # (graph capture or eager) it runs normally and is captured into the
        # CUDA graph along with the copy_.  On replay, the captured kernels
        # re-execute with fresh topk_ids and update _lplb_local_count.
        if self._lplb_local_count is not None and not torch._dynamo.is_compiling():
            local = count_logical_experts(topk_ids, self.num_logical_experts)
            self._lplb_local_count[layer_id].copy_(local)

        # Pass the PREVIOUS step's global count to MLB.  The stable tensor
        # slice has a fixed GPU address, so it is safe to use inside a
        # captured CUDA graph: on replay the LP solve reads whatever value
        # finalize_step_counts deposited before the graph was launched.
        global_count = (
            self._lplb_global_count[layer_id]
            if self._lplb_global_count is not None
            else None
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
                defer_dispatch=True,
                global_logical_count=global_count,
            )
        )
        selection = decision.metadata.get("replica_selection")
        if selection is None:
            return None
        return selection.probability


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
) -> MlbRoutingRuntime | None:
    """Create the routing runtime for a configured L2 algorithm.

    ``algorithm`` comes from ``EPLBConfig.l2_algorithm``, which has already
    resolved the environment default and cleared itself for placements no L2
    policy can act on. Passing it in rather than re-reading the environment is
    what makes that decision binding.
    """
    global _runtime
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
    if _runtime.requires_placement_state:
        _reject_graphs_with_rearranging_placement_state(rearranges)
    return _runtime


def get_mlb_routing() -> MlbRoutingRuntime | None:
    return _runtime


def reset_mlb_routing() -> None:
    """Test hook."""
    global _runtime
    _runtime = None
