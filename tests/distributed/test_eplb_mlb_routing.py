# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLB-backed L2 token routing.

Run on a single device without a process group: ``static`` and ``dynamic``
replica policies make a local decision, so no EP collective is issued.  The
``lplb`` policy does issue one and is covered by the multi-rank CUDA-graph
probe instead.

The important property under test is the accuracy-preserving invariant: MLB
may only change *which physical replica* serves a token, never which logical
expert the router chose.
"""

import pytest
import torch

pytest.importorskip("moe_load_balancer")

from vllm.distributed.eplb.eplb_state import (  # noqa: E402
    EplbLayerState,
    compute_logical_maps,
)
from vllm.distributed.eplb.mlb_runtime import (  # noqa: E402
    MlbRoutingRuntime,
    reset_mlb_routing,
)

NUM_LAYERS = 4
NUM_LOGICAL = 32
NUM_PHYSICAL = 40  # 8 redundant replicas
EP_SIZE = 4
NUM_TOKENS = 64
TOPK = 4


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_mlb_routing()


def _placement() -> torch.Tensor:
    """A placement where the first 8 logical experts get a second replica."""
    rows = []
    for _ in range(NUM_LAYERS):
        row = list(range(NUM_LOGICAL)) + list(range(NUM_PHYSICAL - NUM_LOGICAL))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _runtime(algorithm: str) -> MlbRoutingRuntime:
    phy2log = _placement()
    log2phy, replica_count = compute_logical_maps(phy2log, NUM_LOGICAL)
    rt = MlbRoutingRuntime(
        algorithm,
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=NUM_PHYSICAL,
        physical_to_logical_map=phy2log,
    )
    rt.register_logical_maps(log2phy, replica_count)
    return rt


def _layer_state(rt: MlbRoutingRuntime, layer_id: int = 0) -> EplbLayerState:
    state = EplbLayerState()
    state.set_layer_state(
        layer_id,
        torch.zeros(NUM_LAYERS, NUM_PHYSICAL, dtype=torch.int64),
        rt._logical_to_physical_map,
        rt._logical_replica_count,
    )
    state.should_record_tensor = torch.tensor(1, dtype=torch.int32)
    return state


def test_lplb_asks_for_the_table_rather_than_the_ids():
    """MLB is asked how to split traffic, not for per-token ids -- that is what
    keeps expert-load recording inside vLLM's kernel.

    `lplb` needs an EP all-reduce to solve, so it cannot run in a single
    process; assert on the request this path builds instead. The table is
    consumed by the kernel, which has its own tests.
    """
    from moe_load_balancer.adapters.vllm import to_routing_request

    request = to_routing_request(
        layer_id=0,
        logical_topk_ids=torch.zeros(2, 2, dtype=torch.int64),
        topk_weights=torch.zeros(2, 2),
        defer_dispatch=True,
    )
    assert request.defer_dispatch is True


def test_replica_shares_is_none_when_policy_cannot_produce_one():
    """`dynamic` makes a per-token choice with no table behind it; the caller
    must fall back rather than treat a missing table as 'all zeros'."""
    rt = _runtime("dynamic")
    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    assert rt.replica_shares(logical, torch.rand(NUM_TOKENS, TOPK), state, None) is None


def test_synthesized_default_prefers_local_replicas():
    """vLLM keeps no per-rank dispatch table, so the adapter builds one.
    It must prefer a replica this rank owns -- that is the locality SGLang
    gets for free by collapsing its candidate list."""
    from moe_load_balancer.adapters.vllm import nearest_replica_table

    phy2log = _placement()
    log2phy, counts = compute_logical_maps(phy2log, NUM_LOGICAL)
    slots = NUM_PHYSICAL // EP_SIZE

    for ep_rank in range(EP_SIZE):
        table = nearest_replica_table(
            log2phy[0], counts[0], ep_rank=ep_rank, num_local_physical_experts=slots
        )
        assert table.shape == (NUM_LOGICAL,)
        # Chosen slot must really hold that logical expert.
        assert torch.equal(phy2log[0][table], torch.arange(NUM_LOGICAL))
        low, high = ep_rank * slots, (ep_rank + 1) * slots
        for logical in range(NUM_LOGICAL):
            valid = log2phy[0, logical, : counts[0, logical]]
            local = valid[(valid >= low) & (valid < high)]
            if local.numel() > 0:
                assert int(table[logical]) in local.tolist(), (
                    f"expert {logical} has a replica on rank {ep_rank} "
                    "but the default table points elsewhere"
                )


def test_waterfill_is_rejected_rather_than_silently_dropped():
    """vLLM's CUDA path has no shared-expert dispatch, so a shared-expert
    decision has nowhere to be written back."""
    from moe_load_balancer.adapters.vllm import to_vllm_topk_ids
    from moe_load_balancer.core.types import RoutingDecision

    decision = RoutingDecision(
        routed_physical_topk_ids=torch.zeros(2, 2, dtype=torch.int64),
        topk_weights=torch.zeros(2, 2),
        shared_expert_rank=torch.zeros(2, dtype=torch.int64),
    )
    with pytest.raises(NotImplementedError, match="Shared-expert routing"):
        to_vllm_topk_ids(decision)


def test_placement_commit_refreshes_policy_state():
    """A committed rearrangement must be observable without re-registration:
    vLLM updates its maps in place, and the runtime holds those tensors."""
    rt = _runtime("dynamic")
    rt.on_placement_committed()  # must not raise
    # Simulate vLLM committing a new placement in place.
    rt.physical_to_logical_map[0, NUM_LOGICAL:] = torch.arange(
        NUM_PHYSICAL - NUM_LOGICAL, dtype=torch.int64
    ).flip(0)
    new_log2phy, new_counts = compute_logical_maps(
        rt.physical_to_logical_map, NUM_LOGICAL
    )
    rt._logical_to_physical_map.copy_(new_log2phy)
    rt._logical_replica_count.copy_(new_counts)
    rt.on_placement_committed()

    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    # The snapshot handed to MLB must reflect the committed maps, not the ones
    # captured at registration.
    snapshot = rt._snapshot(state, 0)
    assert torch.equal(
        snapshot.logical_to_physical_candidates, rt._logical_to_physical_map[0]
    )
    assert rt.replica_shares(logical, torch.rand(NUM_TOKENS, TOPK), state, None) is None


def test_capabilities_gate_the_work_the_framework_does():
    """MLB declares what the framework must prepare; Glue must honour it
    instead of doing every lifecycle step for every policy."""
    dyn = _runtime("dynamic")
    # dynamic keeps no placement-derived state and needs no dispatch table.
    assert dyn.requires_post_topk_routing
    assert not dyn.requires_placement_state
    assert not dyn.requires_rank_dispatch_map
    assert dyn._default_replicas is None, (
        "the rank dispatch table was built for a policy that never reads it"
    )

    st = _runtime("static")
    assert st.requires_rank_dispatch_map
    assert st._default_replicas is not None


def test_commit_is_skipped_for_policies_without_placement_state(monkeypatch):
    rt = _runtime("dynamic")
    calls = []
    monkeypatch.setattr(rt, "_commit_layers", lambda ids: calls.append(list(ids)))
    rt.on_placement_committed([0, 1])
    rt.announce_initial_placement()
    assert calls == [], "dynamic does not keep placement state; nothing to refresh"


def test_only_changed_layers_are_refreshed(monkeypatch):
    """Refreshing an untouched layer is pure stall: a shape-changing rebuild
    costs a JIT build plus warmup per layer."""
    rt = _runtime("static")  # any policy; we intercept the MLB call
    seen = []
    monkeypatch.setattr(rt, "requires_placement_state", True)
    monkeypatch.setattr(rt, "_commit_layers", lambda ids: seen.append(list(ids)))

    rt.on_placement_committed([1, 3])
    assert seen == [[1, 3]]

    seen.clear()
    rt.on_placement_committed([])
    assert seen == [], "an empty change set must not touch any layer"

    seen.clear()
    rt.on_placement_committed(None)  # unknown -> conservative full refresh
    assert seen == [list(range(NUM_LAYERS))]


def test_initial_placement_is_announced():
    """The contract is 'after initial weight loading AND after each commit';
    skipping the first half leaves the policy to build state lazily inside the
    first forward, from whatever placement that forward happens to see."""
    rt = _runtime("static")
    seen = []
    rt.requires_placement_state = True
    rt._commit_layers = lambda ids: seen.append(list(ids))
    rt.announce_initial_placement()
    assert seen == [list(range(NUM_LAYERS))]
