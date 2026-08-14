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
    _record_physical_load,
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


@pytest.mark.parametrize("algorithm", ["static", "dynamic"])
def test_routing_preserves_logical_experts(algorithm):
    """The accuracy-preserving invariant, stated as a test.

    Mapping each returned physical slot back through physical_to_logical_map
    must reproduce the router's original logical choice exactly.
    """
    rt = _runtime(algorithm)
    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    weights = torch.rand(NUM_TOKENS, TOPK)

    physical = rt.route(logical, weights, state, None)

    assert physical.shape == logical.shape
    assert int(physical.min()) >= 0 and int(physical.max()) < NUM_PHYSICAL
    back = rt.physical_to_logical_map[0][physical]
    assert torch.equal(back, logical), (
        "MLB routing changed the logical expert selection"
    )


def test_routing_uses_redundant_replicas():
    """A replica policy that never picks the second copy is a no-op; make sure
    the redundant slots actually receive traffic (this is exactly the failure
    mode found earlier in SGLang's qwen3_moe path)."""
    rt = _runtime("dynamic")
    state = _layer_state(rt)
    torch.manual_seed(0)
    # Only route to logical experts that have two replicas.
    logical = torch.randint(
        0, NUM_PHYSICAL - NUM_LOGICAL, (512, TOPK), dtype=torch.int64
    )
    physical = rt.route(logical, torch.rand(512, TOPK), state, None)
    redundant_hits = int((physical >= NUM_LOGICAL).sum())
    assert redundant_hits > 0, "redundant replicas received zero traffic"


def test_load_recording_matches_fused_kernel_semantics():
    """MLB takes over mapping, so recording is done separately; it must keep
    the fused kernel's semantics: physical counting, gated by record_enabled,
    and masking padded tokens."""
    load = torch.zeros(NUM_PHYSICAL, dtype=torch.int64)
    ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
    enabled = torch.tensor(1, dtype=torch.int32)

    _record_physical_load(ids, load, enabled, None)
    expected = torch.zeros(NUM_PHYSICAL, dtype=torch.int64)
    expected[:6] = 1
    assert torch.equal(load, expected)

    # Gate off: nothing recorded.
    load.zero_()
    _record_physical_load(ids, load, torch.tensor(0, dtype=torch.int32), None)
    assert int(load.sum()) == 0

    # Padding mask: only the first two tokens are real.
    load.zero_()
    _record_physical_load(ids, load, enabled, torch.tensor(2, dtype=torch.int32))
    assert int(load.sum()) == 4
    assert int(load[4]) == 0 and int(load[5]) == 0

    # Negative ids (vLLM marks padded rows with -1) must not be counted.
    load.zero_()
    _record_physical_load(
        torch.tensor([[-1, -1], [2, 3]], dtype=torch.int64), load, enabled, None
    )
    assert int(load.sum()) == 2


def test_recorded_load_totals_match_routing():
    rt = _runtime("dynamic")
    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    rt.route(logical, torch.rand(NUM_TOKENS, TOPK), state, None)
    assert int(state.expert_load_view.sum()) == NUM_TOKENS * TOPK


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
    physical = rt.route(logical, torch.rand(NUM_TOKENS, TOPK), state, None)
    back = rt.physical_to_logical_map[0][physical]
    assert torch.equal(back, logical), (
        "routing broke after a committed placement change"
    )


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
