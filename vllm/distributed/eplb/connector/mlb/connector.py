# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE Load Balancer (``moe_load_balancer``) as an EPLB connector.

One balancer per engine, serving both placement and routing. SGLang gives an
engine a single MoELoadBalancer and lets L1 and L2 share it; a throwaway
instance per rebalance plus a second one for routing could not see each
other's state, which rules out any policy whose placement decision depends on
what routing observed. The connector is that single instance's owner: created
by ``EplbState`` through :class:`EplbConnectorFactory`, released with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.distributed.eplb.connector.base import (
    EplbConnectorBase,
    EplbRoutingCapabilities,
    EplbRoutingRuntimeBase,
)
from vllm.distributed.eplb.connector.mlb.runtime import (
    MlbRoutingRuntime,
    VllmRoutingCollectives,
    _reject_graphs_with_rearranging_placement_state,
    l2_inapplicable_reason,
    l2_pipeline_capabilities,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import EPLBConfig
    from vllm.distributed.eplb.policy import AbstractEplbPolicy

logger = init_logger(__name__)


class MoeLoadBalancerConnector(EplbConnectorBase):
    """Placement through MLB's L1 policies, routing through its L2 pipeline."""

    def __init__(self, eplb_config: EPLBConfig) -> None:
        super().__init__(eplb_config)
        from moe_load_balancer import MoELoadBalancer

        # The L2 expression; "" means placement only. EPLBConfig has already
        # resolved the environment default and cleared it for deployments the
        # policy cannot act on, so this value is binding.
        self.algorithm = eplb_config.l2_algorithm
        self._balancer_kwargs: dict | None = None
        self._balancer = None if self.algorithm else MoELoadBalancer()

    # -- answers from configuration alone -------------------------------------

    @classmethod
    def placement_policy(
        cls, eplb_config: EPLBConfig
    ) -> type[AbstractEplbPolicy] | None:
        from vllm.distributed.eplb.connector.mlb.policy import MlbEplbPolicy

        return MlbEplbPolicy

    @classmethod
    def routing_capabilities(
        cls, eplb_config: EPLBConfig
    ) -> EplbRoutingCapabilities | None:
        caps = l2_pipeline_capabilities(eplb_config.l2_algorithm)
        if caps is None:
            return None
        return EplbRoutingCapabilities(
            requires_post_topk_routing=bool(
                getattr(caps, "requires_post_topk_routing", False)
            ),
            requires_placement_state=bool(
                getattr(caps, "requires_placement_state", False)
            ),
            requires_rank_dispatch_map=bool(
                getattr(caps, "requires_rank_dispatch_map", False)
            ),
            routes_shared_expert=bool(getattr(caps, "routes_shared_expert", False)),
            supports_concurrent_microbatches=bool(
                getattr(caps, "supports_concurrent_microbatches", True)
            ),
            graph_stability=str(getattr(caps, "graph_stability", "stable")),
        )

    @classmethod
    def inapplicable_reason(
        cls, eplb_config: EPLBConfig, num_redundant_experts: int
    ) -> str | None:
        # Whether the shared expert is dispatched: the explicit switch, or the
        # pipeline itself asking for it (waterfill picks a rank for it, and
        # there is no rank to pick unless it goes through dispatch). Answered
        # from configuration because this runs during config validation,
        # before any model exists.
        caps = cls.routing_capabilities(eplb_config)
        routes_shared_expert = bool(envs.VLLM_FUSE_SHARED_EXPERTS) or bool(
            caps is not None and caps.routes_shared_expert
        )
        return l2_inapplicable_reason(
            eplb_config.l2_algorithm, num_redundant_experts, routes_shared_expert
        )

    # -- the engine's balancer ------------------------------------------------

    def balancer(self):
        """The single MoELoadBalancer this engine uses."""
        if self._balancer is None:
            from moe_load_balancer import MoELoadBalancer

            if self._balancer_kwargs is None:
                self._balancer = MoELoadBalancer()
            else:
                self._balancer = MoELoadBalancer.from_algorithm(
                    self.algorithm, **self._balancer_kwargs
                )
        return self._balancer

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
        """Create the routing runtime against this engine's balancer.

        ``expert_weights`` is the model's own ``expert_weights`` (one entry per
        MoE layer); ``expert_buffer`` and ``communicator`` are ``EplbState``'s
        staging buffer and P2P backend for moving expert weight between ranks.
        Only ``ultraep`` reads them -- the one policy that re-plans placement
        often enough to need weight moved mid-run -- and it moves weight with
        this framework machinery rather than any of its own.
        """
        if not self.algorithm:
            return None
        from moe_load_balancer import MoELoadBalancer

        self._balancer_kwargs = {
            "ep_size": ep_size,
            "source_rank": ep_rank,
            "experts_per_rank": num_physical_experts // ep_size,
            "collectives": VllmRoutingCollectives(),
        }
        self._balancer = MoELoadBalancer.from_algorithm(
            self.algorithm, **self._balancer_kwargs
        )
        runtime = MlbRoutingRuntime(
            self.algorithm,
            ep_size=ep_size,
            ep_rank=ep_rank,
            num_logical_experts=num_logical_experts,
            num_physical_experts=num_physical_experts,
            physical_to_logical_map=physical_to_logical_map,
            balancer=self._balancer,
            expert_weights=expert_weights,
            expert_buffer=expert_buffer,
            communicator=communicator,
        )
        runtime.register_logical_maps(logical_to_physical_map, logical_replica_count)
        if runtime.graph_stability == "realloc_on_placement_change":
            _reject_graphs_with_rearranging_placement_state(rearranges)
        self.routing = runtime
        return runtime

    def plan_placement(self, request):
        """Run L1 through the engine's balancer and hand back vLLM's one map."""
        from moe_load_balancer.adapters.vllm import to_vllm_physical_to_logical

        plan = self.balancer().plan_placement(request)
        # UltraEP is the only L1 policy that publishes this; every other plan's
        # metadata simply lacks the key, so this stays a no-op for them. Routing
        # geometry can lag placement, so there may be no runtime to hand the
        # quota to yet -- it reads whatever the next solve after bind_routing()
        # leaves here.
        #
        # The candidate table and replica counts come along too, not just the
        # quota: they must be read from this same plan, not rebuilt from
        # phy2log through vLLM's own compute_logical_maps, or the quota's column
        # ordering and width silently stop matching the table L2 routes against.
        quota = plan.metadata.get("rank_quota_prefix")
        routing: Any = self.routing
        if quota is not None and routing is not None:
            routing._ultraep_rank_quota_prefix = quota
            routing._ultraep_logical_to_physical = plan.logical_to_all_physical_map
            routing._ultraep_replica_counts = plan.logical_to_physical_count
            routing._ultraep_committed_layers = set(range(quota.shape[0]))
        return to_vllm_physical_to_logical(plan), plan

    def shutdown(self) -> None:
        super().shutdown()
        self._balancer = None
        self._balancer_kwargs = None


def current_mlb_connector() -> MoeLoadBalancerConnector:
    """The engine's MLB connector, created from the current config on demand.

    The placement policy is a classmethod that vLLM calls at a rearrangement;
    by then ``EplbState`` has created the connector. Creating it here covers a
    policy run outside an engine (tests, offline scoring), the way the old
    module-level integration was created on first use.
    """
    from vllm.distributed.eplb.connector.state import (
        ensure_eplb_connector_initialized,
        get_eplb_connector,
    )

    connector = get_eplb_connector()
    if connector is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            eplb_config = vllm_config.parallel_config.eplb_config
        else:
            # No engine and no config context (a policy exercised directly):
            # the defaults, whose l2_algorithm is the environment's, exactly
            # what the module-level integration used to be built from.
            from vllm.config.parallel import EPLBConfig

            eplb_config = EPLBConfig()
        connector = ensure_eplb_connector_initialized(
            eplb_config, MoeLoadBalancerConnector
        )
    if not isinstance(connector, MoeLoadBalancerConnector):
        raise TypeError(
            "The engine's EPLB connector is not the MoE Load Balancer: "
            f"{type(connector).__name__}"
        )
    return connector
