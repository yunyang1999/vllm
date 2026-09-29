# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EPLB connector seam itself: registry, module-path loading, engine state
and the ``policy: "mlb"`` alias. Needs no load balancer installed."""

import pytest
import torch

from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.connector import (
    EplbConnectorBase,
    EplbConnectorFactory,
    EplbRoutingCapabilities,
    EplbRoutingRuntimeBase,
    ensure_eplb_connector_initialized,
    get_eplb_connector,
    get_eplb_routing,
    reset_eplb_connector,
)
from vllm.distributed.eplb.policy import EPLB_POLICIES, DefaultEplbPolicy


class _Runtime(EplbRoutingRuntimeBase):
    requires_post_topk_routing = True

    def resolve_routing(self, topk_ids, topk_weights, layer_state, num_unpadded):
        return None, topk_ids


class DummyConnector(EplbConnectorBase):
    bound_with: dict | None = None

    @classmethod
    def placement_policy(cls, eplb_config):
        return DefaultEplbPolicy

    @classmethod
    def routing_capabilities(cls, eplb_config):
        return EplbRoutingCapabilities(requires_post_topk_routing=True)

    @classmethod
    def inapplicable_reason(cls, eplb_config, num_redundant_experts):
        return None if num_redundant_experts else "needs redundant experts"

    def bind_routing(self, **kwargs):
        DummyConnector.bound_with = kwargs
        self.routing = _Runtime()
        return self.routing


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_eplb_connector()
    monkeypatch.setitem(EplbConnectorFactory._registry, "dummy", lambda: DummyConnector)
    yield
    reset_eplb_connector()


def test_no_connector_means_no_routing():
    cfg = EPLBConfig()
    assert cfg.connector is None
    assert ensure_eplb_connector_initialized(cfg) is None
    assert get_eplb_connector() is None
    assert get_eplb_routing() is None


def test_registered_connector_is_created_once_and_binds_routing():
    cfg = EPLBConfig(connector="dummy", use_async=False)
    first = ensure_eplb_connector_initialized(cfg)
    assert isinstance(first, DummyConnector)
    assert ensure_eplb_connector_initialized(cfg) is first
    assert get_eplb_routing() is None
    runtime = first.bind_routing(
        ep_size=2,
        ep_rank=0,
        num_logical_experts=4,
        num_physical_experts=6,
        physical_to_logical_map=torch.zeros(1, 6, dtype=torch.int64),
        logical_to_physical_map=torch.zeros(1, 4, 2, dtype=torch.int64),
        logical_replica_count=torch.ones(1, 4, dtype=torch.int64),
    )
    assert get_eplb_routing() is runtime
    assert DummyConnector.bound_with is not None
    assert DummyConnector.bound_with["ep_size"] == 2
    reset_eplb_connector()
    assert get_eplb_routing() is None


def test_module_path_loads_a_class_by_name():
    cfg = EPLBConfig(
        connector="DummyConnector",
        connector_module_path=__name__,
        use_async=False,
    )
    assert EplbConnectorFactory.get_connector_class(cfg) is DummyConnector


def test_unknown_names_fail_loudly():
    with pytest.raises(ValueError, match="Unknown EPLB connector"):
        EplbConnectorFactory.get_connector_class(
            EPLBConfig(connector="no-such-connector", use_async=False)
        )
    with pytest.raises(ValueError, match="not found"):
        EplbConnectorFactory.get_connector_class(
            EPLBConfig(
                connector="Missing", connector_module_path=__name__, use_async=False
            )
        )


def test_mlb_policy_is_an_alias_for_the_mlb_connector():
    cfg = EPLBConfig(policy="mlb", use_async=False)
    assert cfg.connector == "mlb"
    assert "mlb" not in EPLB_POLICIES
    # An explicit connector is not overridden by the alias.
    cfg = EPLBConfig(policy="mlb", connector="dummy", use_async=False)
    assert cfg.connector == "dummy"


def test_connector_requires_synchronous_placement():
    with pytest.raises(ValueError, match="use_async"):
        EPLBConfig(connector="dummy", use_async=True)
