# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-parallel load-balancer connectors.

EPLB rearranges experts across EP ranks on a fixed cadence with a built-in
placement policy and picks a logical expert's replica per token with a hash
fused into the map-and-record kernel. A *connector* plugs an external load
balancer into that machinery at three points, without the balancer's code
living in vLLM:

* **placement** (L1): the :class:`AbstractEplbPolicy` an ``EplbState`` runs at
  a rearrangement (:meth:`EplbConnectorBase.placement_policy`);
* **routing** (L2): which replica of a logical expert serves each token and
  which rank serves the shared expert, decided per forward and fed to the
  existing fused kernel (:class:`EplbRoutingRuntimeBase`);
* **lifecycle**: one object per engine core, bound by ``EplbState`` to its
  maps, expert weights, staging buffer and communicator at construction and
  consulted by the router, the state and the model runner afterwards.

Nothing here is active unless ``--eplb-config`` names a connector
(``connector``, optionally ``connector_module_path`` for one outside the
vLLM tree; ``policy: "mlb"`` is an alias for the built-in ``mlb`` connector).
The classmethods answer from configuration alone, before any engine exists,
so config validation and layer construction can ask them; the instance
methods need the bound engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config.parallel import EPLBConfig
    from vllm.distributed.eplb.eplb_state import EplbLayerState
    from vllm.distributed.eplb.policy import AbstractEplbPolicy


@dataclass(frozen=True)
class EplbRoutingCapabilities:
    """What a connector's routing needs from the framework.

    Declared, not assumed: the framework prepares only what the selected
    policy asks for (a per-forward routing boundary, placement-derived state
    rebuilt on commit, a dispatched shared expert), and refuses combinations
    the policy says it cannot survive (concurrent micro-batches, captured
    graphs across a placement change).
    """

    requires_post_topk_routing: bool = False
    """The policy decides per token after TopK; the router must call it."""
    requires_placement_state: bool = False
    """The policy keeps per-layer state derived from the committed placement."""
    requires_rank_dispatch_map: bool = False
    """The policy wants a per-rank dispatch table materialised on commit."""
    routes_shared_expert: bool = False
    """The policy picks the shared expert's rank, so the shared expert must
    be dispatched through EP rather than replicated per rank."""
    supports_concurrent_microbatches: bool = True
    """Safe under dual-batch overlap (two micro-batches in flight per layer)."""
    graph_stability: str = "stable"
    """``"stable"``, or ``"realloc_on_placement_change"`` when a committed
    placement can move the policy's buffers under a captured CUDA graph."""


class EplbRoutingRuntimeBase(ABC):
    """Per-engine routing runtime a connector binds at ``EplbState`` build.

    The attributes mirror :class:`EplbRoutingCapabilities` for the bound
    policy; the framework reads them as plain attributes on the hot path.
    Every hook other than :meth:`resolve_routing` defaults to a no-op, so a
    connector implements only what its policy uses.
    """

    requires_post_topk_routing: bool = False
    requires_placement_state: bool = False
    requires_rank_dispatch_map: bool = False
    routes_shared_expert: bool = False
    dispatch_fixed_by_placement: bool = False
    """The replica choice is fixed by the committed placement, so the router
    substitutes :meth:`fixed_dispatch_maps` for the per-forward call."""
    supports_concurrent_microbatches: bool = True
    graph_stability: str = "stable"

    def fixed_dispatch_maps(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Candidate map and replica counts for a placement-fixed policy."""
        return None

    @abstractmethod
    def resolve_routing(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Per-expert replica shares and/or resolved physical ids for a batch.

        Either tensor may be ``None``; the fused kernel then keeps its own
        hash for that part. Returned shapes: shares ``[num_logical, slots]``,
        ids the shape of ``topk_ids``.
        """

    def resolve_shared_expert_rank(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: EplbLayerState,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Rank that serves each token's shared expert, or ``None`` for local."""
        return None

    def finalize_step_counts(self) -> None:  # noqa: B027 - optional hook
        """Called once per engine step before any layer routes."""

    def match_step_counts_collective(self) -> None:  # noqa: B027 - optional hook
        """Issue the same collectives :meth:`finalize_step_counts` would, on
        a dummy forward, so DP ranks without requests stay in lockstep."""

    def on_placement_committed(  # noqa: B027 - optional hook
        self, changed_layer_ids: list[int] | None = None
    ) -> None:
        """Weights moved and the live maps changed for these layers (all
        layers when ``None``)."""

    def announce_initial_placement(self) -> None:  # noqa: B027 - optional hook
        """The state's initial maps are final; log or seed whatever is needed."""


class EplbConnectorBase(ABC):
    """One load-balancer connector per engine core.

    Construct from :class:`EPLBConfig`; the classmethods answer without an
    instance so that configuration validation and layer construction can
    consult them before any engine exists.
    """

    def __init__(self, eplb_config: EPLBConfig) -> None:
        self.eplb_config = eplb_config
        self.routing: EplbRoutingRuntimeBase | None = None

    @classmethod
    def placement_policy(
        cls, eplb_config: EPLBConfig
    ) -> type[AbstractEplbPolicy] | None:
        """Placement policy class the connector supplies, or ``None`` to keep
        the built-in one selected by ``eplb_config.policy``."""
        return None

    @classmethod
    def routing_capabilities(
        cls, eplb_config: EPLBConfig
    ) -> EplbRoutingCapabilities | None:
        """Capabilities of the routing the config selects, or ``None`` when
        the connector does no per-token routing (or cannot tell yet)."""
        return None

    @classmethod
    def inapplicable_reason(
        cls, eplb_config: EPLBConfig, num_redundant_experts: int
    ) -> str | None:
        """Why the configured routing cannot act on this deployment, or
        ``None`` if it can. The framework disables routing and logs the
        reason; placement is unaffected."""
        return None

    @abstractmethod
    def bind_routing(
        self,
        *,
        ep_size: int,
        ep_rank: int,
        num_logical_experts: int,
        num_physical_experts: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_weights: Any | None = None,
        expert_buffer: Any | None = None,
        communicator: Any | None = None,
        rearranges: bool = False,
    ) -> EplbRoutingRuntimeBase | None:
        """Create the routing runtime against the engine's live EPLB state.

        The maps are the state's own tensors (updated in place on commit);
        ``expert_weights``, ``expert_buffer`` and ``communicator`` are the
        model's expert tensors and the state's staging buffer and P2P backend,
        for a policy that moves weight between rearrangements. Returns ``None``
        when the configuration asks for no per-token routing.
        """

    def shutdown(self) -> None:
        """Release engine-scoped resources."""
        self.routing = None
