# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Look up an EPLB connector class by name or module path."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.eplb.connector.base import EplbConnectorBase
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.parallel import EPLBConfig

logger = init_logger(__name__)


class EplbConnectorFactory:
    _registry: dict[str, Callable[[], type[EplbConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazily imported module and class."""
        if name in cls._registry:
            raise ValueError(f"EPLB connector '{name}' is already registered.")

        def loader() -> type[EplbConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_connector_class(cls, eplb_config: EPLBConfig) -> type[EplbConnectorBase]:
        """The class ``eplb_config.connector`` names.

        With ``connector_module_path`` set, ``connector`` is the class name
        inside that module (a connector maintained outside the vLLM tree);
        otherwise it is a registered name.
        """
        name = eplb_config.connector
        if not name:
            raise ValueError("eplb_config.connector is not set.")
        module_path = eplb_config.connector_module_path
        if module_path is not None and not module_path:
            raise ValueError("eplb_config.connector_module_path cannot be empty.")
        if module_path:
            module = importlib.import_module(module_path)
            try:
                connector_cls = getattr(module, name)
            except AttributeError as exc:
                raise ValueError(
                    f"EPLB connector class {name!r} not found in {module_path}"
                ) from exc
        elif name in cls._registry:
            connector_cls = cls._registry[name]()
        else:
            raise ValueError(
                f"Unknown EPLB connector {name!r}; registered: "
                f"{sorted(cls._registry)}. Set connector_module_path for a "
                "connector outside the vLLM tree."
            )
        if not (
            isinstance(connector_cls, type)
            and issubclass(connector_cls, EplbConnectorBase)
        ):
            raise TypeError(
                f"EPLB connector {name!r} must subclass EplbConnectorBase, "
                f"got {connector_cls!r}"
            )
        return connector_cls

    @classmethod
    def create_connector(cls, eplb_config: EPLBConfig) -> EplbConnectorBase:
        connector_cls = cls.get_connector_class(eplb_config)
        logger.info("Creating EPLB connector %s", connector_cls.__name__)
        return connector_cls(eplb_config)


# Built-in connectors.
EplbConnectorFactory.register_connector(
    "mlb", "vllm.distributed.eplb.connector.mlb", "MoeLoadBalancerConnector"
)
