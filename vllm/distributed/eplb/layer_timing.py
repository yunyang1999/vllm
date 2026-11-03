# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-MoE-layer CUDA event timing for the decode-balance study.

Enabled by ``VLLM_EPLB_LAYER_TIMING_EVERY`` (N > 0). Every MoE layer brackets
its routed-expert section (replica mapping, dispatch, grouped GEMM, combine)
with a pair of CUDA events on every forward. Every N-th logged EPLB step the
EPLB state synchronizes on the last layer's end event, reads each layer's
elapsed milliseconds, all-gathers them over the EP group and hands the
``[ep_size, num_layers]`` table to the per-step load dumper, which stores it
as ``layer_ms`` next to the expert loads. The events are created with
``external=True`` so recording them inside a captured CUDA graph is allowed
and every replay updates them (the same trick ``vllm.benchmarks.forward``
uses).

Any local failure disables timing on that rank; ``collect`` then agrees with
the other ranks through a tiny all-reduce, so no rank is left waiting in a
collective. Fused cooperative kernels (MegaMoE) finish together on all ranks,
so for them the per-rank numbers are the layer's step time, not that rank's
own compute time; the generic modular path times dispatch, local experts and
(unless finalize is deferred) combine.
"""

import torch
from torch.distributed import ProcessGroup, ReduceOp, all_gather_into_tensor, all_reduce

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


class LayerTimer:
    __slots__ = ("start", "end", "recorded")

    def __init__(self) -> None:
        kwargs = {"enable_timing": True, "external": True}
        try:
            self.start = torch.cuda.Event(**kwargs)
            self.end = torch.cuda.Event(**kwargs)
        except TypeError:  # torch without the external flag
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        self.recorded = False


_TIMERS: list[LayerTimer] = []
_BY_LAYER: dict[int, LayerTimer] = {}
_DISABLED = False


def every() -> int:
    """Sampling period in logged EPLB steps; 0 means timing is off."""
    return max(0, envs.VLLM_EPLB_LAYER_TIMING_EVERY)


def _disable(reason: str) -> None:
    global _DISABLED
    if not _DISABLED:
        logger.warning("MoE layer timing disabled on this rank: %s", reason)
    _DISABLED = True


def record_start(layer: object) -> LayerTimer | None:
    """Record the start event of ``layer``'s routed section (None when off)."""
    if every() == 0 or _DISABLED:
        return None
    timer = _BY_LAYER.get(id(layer))
    if timer is None:
        # Layers register in first-forward order, which is model order (a
        # draft model's MoE layers, if any, follow the target model's).
        timer = LayerTimer()
        _BY_LAYER[id(layer)] = timer
        _TIMERS.append(timer)
    try:
        timer.start.record()
    except Exception as exc:  # noqa: BLE001 -- timing must never break serving
        _disable(f"start.record failed: {exc!r}")
        return None
    return timer


def record_end(timer: LayerTimer | None) -> None:
    if timer is None:
        return
    try:
        timer.end.record()
        timer.recorded = True
    except Exception as exc:  # noqa: BLE001
        _disable(f"end.record failed: {exc!r}")


def collect(device: torch.device, ep_group: ProcessGroup) -> torch.Tensor | None:
    """Per-rank, per-layer milliseconds of the last forward, ``[ep, L]`` on CPU.

    Every EP rank must call this on the same step. Returns None on every rank
    when any rank has no usable timing (disabled, or a different layer count).
    """
    vals: list[float] = []
    ok = 1
    if _DISABLED or not _TIMERS or not all(t.recorded for t in _TIMERS):
        ok = 0
    else:
        try:
            _TIMERS[-1].end.synchronize()
            vals = [t.start.elapsed_time(t.end) for t in _TIMERS]
        except Exception as exc:  # noqa: BLE001
            _disable(f"elapsed_time failed: {exc!r}")
            ok = 0
    n = len(vals)
    flag = torch.tensor([ok, n, -n], dtype=torch.int64, device=device)
    all_reduce(flag, op=ReduceOp.MIN, group=ep_group)
    ok_all, n_min, n_max = int(flag[0]), int(flag[1]), -int(flag[2])
    if ok_all == 0 or n_min != n_max or n_min == 0:
        return None
    local = torch.tensor(vals, dtype=torch.float32, device=device)
    out = local.new_empty((ep_group.size() * n,))
    all_gather_into_tensor(out, local, group=ep_group)
    return out.view(ep_group.size(), n).cpu()
