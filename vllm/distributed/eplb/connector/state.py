# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine core's EPLB connector, if one is configured.

Module-level because the placement policy vLLM runs is a classmethod with
nowhere to hang an instance, and the router reaches the routing runtime from
inside a kernel wrapper. Ownership is not module-level: ``EplbState`` creates
the connector and ``reset_eplb_connector`` releases it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.eplb.connector.base import (
    EplbConnectorBase,
    EplbRoutingRuntimeBase,
)

if TYPE_CHECKING:
    from vllm.config.parallel import EPLBConfig

_CONNECTOR: EplbConnectorBase | None = None


def ensure_eplb_connector_initialized(
    eplb_config: EPLBConfig,
    connector_cls: type[EplbConnectorBase] | None = None,
) -> EplbConnectorBase | None:
    """Create the engine's connector on first use.

    Resolved through the factory from ``eplb_config.connector``; ``None`` when
    no connector is configured. ``connector_cls`` builds that class instead,
    for a connector's own entry points that run outside an engine (a placement
    policy called directly, as tests do).
    """
    global _CONNECTOR
    if _CONNECTOR is not None:
        return _CONNECTOR
    if connector_cls is not None:
        _CONNECTOR = connector_cls(eplb_config)
    elif eplb_config.connector:
        from vllm.distributed.eplb.connector.factory import EplbConnectorFactory

        _CONNECTOR = EplbConnectorFactory.create_connector(eplb_config)
    return _CONNECTOR


def get_eplb_connector() -> EplbConnectorBase | None:
    return _CONNECTOR


def has_eplb_connector() -> bool:
    return _CONNECTOR is not None


def get_eplb_routing() -> EplbRoutingRuntimeBase | None:
    """The bound routing runtime, or ``None`` when no connector routes."""
    return None if _CONNECTOR is None else _CONNECTOR.routing


def reset_eplb_connector() -> None:
    """Release the engine's connector (tests, and engine shutdown)."""
    global _CONNECTOR
    if _CONNECTOR is not None:
        _CONNECTOR.shutdown()
    _CONNECTOR = None
