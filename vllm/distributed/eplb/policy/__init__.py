# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import get_args

from vllm.config.parallel import EPLBPolicyOption

from .abstract import AbstractEplbPolicy
from .default import DefaultEplbPolicy

EPLB_POLICIES: dict[str, type[AbstractEplbPolicy]] = {"default": DefaultEplbPolicy}

# Every registered policy is a valid EPLBPolicyOption; the remaining option
# values ("mlb") name policies a connector supplies (see eplb.connector).
assert set(EPLB_POLICIES.keys()) <= set(get_args(EPLBPolicyOption))

__all__ = [
    "AbstractEplbPolicy",
    "DefaultEplbPolicy",
    "EPLB_POLICIES",
]
