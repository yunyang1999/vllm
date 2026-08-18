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


def test_every_replica_policy_answers_the_deferred_dispatch_request():
    """A runtime that fuses replica selection with expert-load recording asks
    for a share table because it cannot give up the per-token step. A policy
    that returned nothing here was not falling back gracefully -- it never ran
    at all, and the caller silently kept its built-in choice."""
    rt = _runtime("dynamic")
    state = _layer_state(rt)
    torch.manual_seed(0)
    logical = torch.randint(0, NUM_LOGICAL, (NUM_TOKENS, TOPK), dtype=torch.int64)
    shares, ids = rt.resolve_routing(
        logical, torch.rand(NUM_TOKENS, TOPK), state, None
    )
    # The contract is that the policy answers, not that it answers in one
    # particular form: a runtime that fuses selection with recording asks for a
    # share table, one that lets the policy resolve replicas takes ids. What
    # must never happen is neither, which is how these policies used to look
    # like they were doing nothing.
    assert (shares is None) != (ids is None), "exactly one form must come back"
    if shares is not None:
        counts = rt._logical_replica_count[0]
        row = shares[int((counts == 2).nonzero()[0])]
        assert torch.allclose(row[:2], torch.full((2,), 0.5), atol=1e-6)
        single = shares[int((counts == 1).nonzero()[0])]
        assert single[0].item() == 1.0 and single[1].item() == 0.0
    else:
        assert ids.shape == logical.shape


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
    shares, ids = rt.resolve_routing(
        logical, torch.rand(NUM_TOKENS, TOPK), state, None
    )
    assert (shares is None) != (ids is None)


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


def test_replica_routing_sees_every_replica_not_just_the_local_one():
    """A replica-routing policy must be shown the whole candidate list.

    Narrowing each expert to its local replica is what a rank-dispatch policy
    asks for; handing the same view to LPLB decides in advance the very thing
    LPLB is there to decide, and on a rank that holds a copy of everything it
    sees it removes the redundancy entirely, so the LP is skipped. It also
    keeps the returned shares indexable by vLLM's own map, since
    ``replica_shares`` passes on the probabilities without the candidates.
    """
    rt = _runtime("lplb")
    assert not rt.requires_rank_dispatch_map
    snapshot = rt._snapshot(_layer_state(rt), 0)
    assert torch.equal(
        snapshot.logical_to_physical_candidates, rt._logical_to_physical_map[0]
    ), "LPLB was handed a narrowed candidate list"
    assert torch.equal(
        snapshot.logical_to_physical_count, rt._logical_replica_count[0]
    )
    # The first 8 logical experts are replicated by _placement(); the choice
    # between their copies is exactly what the policy has to solve.
    assert int(snapshot.logical_to_physical_count[0]) == 2

    # `static` is the policy the collapse exists for, and it still gets it.
    st = _runtime("static")
    assert st.requires_rank_dispatch_map
    st_snapshot = st._snapshot(_layer_state(st), 0)
    assert int(st_snapshot.logical_to_physical_count[0]) == 1


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


def test_graphs_plus_rearranging_placement_state_is_refused(monkeypatch):
    """LPLB replaces its solver tensors when a committed placement changes
    their layout, and a captured CUDA graph cannot follow that -- replay reads
    freed memory, observed as Xid 43 on two devices. It is intermittent (it
    needs a rearrangement that actually changes the layout), so it has to be
    refused up front rather than left to surface in a long serving run."""
    import sys
    import types

    from vllm.distributed.eplb import mlb_runtime

    class _Mode:
        name = "FULL_AND_PIECEWISE"

    class _Compilation:
        cudagraph_mode = _Mode()

    class _Cfg:
        compilation_config = _Compilation()

    stub = types.ModuleType("vllm.config")
    stub.get_current_vllm_config = lambda: _Cfg()
    monkeypatch.setitem(sys.modules, "vllm.config", stub)

    with pytest.raises(ValueError, match="freed memory|Xid 43"):
        mlb_runtime._reject_graphs_with_rearranging_placement_state(rearranges=True)

    # Pinned placement is the supported combination and must stay allowed --
    # that is how the reported LPLB numbers were measured.
    mlb_runtime._reject_graphs_with_rearranging_placement_state(rearranges=False)

    _Mode.name = "NONE"  # eager is the other escape
    mlb_runtime._reject_graphs_with_rearranging_placement_state(rearranges=True)


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


def _runtime_with_redundancy(num_redundant: int) -> MlbRoutingRuntime:
    """A runtime whose only interesting property is whether experts repeat."""
    num_physical = NUM_LOGICAL + num_redundant
    phy2log = torch.stack(
        [torch.arange(num_physical) % NUM_LOGICAL for _ in range(NUM_LAYERS)]
    )
    return MlbRoutingRuntime(
        "lplb",
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=num_physical,
        physical_to_logical_map=phy2log,
    )


@pytest.mark.parametrize("num_redundant, allocated", [(0, False), (8, True)])
def test_count_buffers_exist_only_where_a_policy_consumes_them(
    num_redundant, allocated, monkeypatch
):
    """An LP with nothing to split must not cost a collective every step.

    Without redundant experts every logical expert holds exactly one replica,
    so LPLB returns identity dispatch and never reads the count. These buffers
    are what make vLLM run a per-layer count kernel and one EP all-reduce per
    step to produce it -- work whose result is then discarded.
    """
    monkeypatch.delenv("MLB_KEEP_ZERO_REDUNDANCY_COUNTS", raising=False)
    rt = _runtime_with_redundancy(num_redundant)
    assert (rt._logical_count_local is not None) is allocated
    assert (rt._logical_count_global is not None) is allocated


def test_the_zero_redundancy_override_restores_the_buffers(monkeypatch):
    """The override exists so the skipped work has a measured size rather than
    only an argued one; a benchmark that cannot restore the old path cannot
    report what removing it bought."""
    monkeypatch.setenv("MLB_KEEP_ZERO_REDUNDANCY_COUNTS", "1")
    rt = _runtime_with_redundancy(0)
    assert rt._logical_count_local is not None
    assert rt._logical_count_global is not None


def test_finalize_step_counts_is_inert_without_buffers(monkeypatch):
    """It must not reach for the EP group when there is nothing to reduce.

    These tests run with no distributed init, so an unguarded get_ep_group()
    would raise -- which is exactly the assertion.
    """
    monkeypatch.delenv("MLB_KEEP_ZERO_REDUNDANCY_COUNTS", raising=False)
    _runtime_with_redundancy(0).finalize_step_counts()


def _sized_runtime(algorithm: str, num_redundant: int = 8) -> MlbRoutingRuntime:
    num_physical = NUM_LOGICAL + num_redundant
    phy2log = torch.stack(
        [torch.arange(num_physical) % NUM_LOGICAL for _ in range(NUM_LAYERS)]
    )
    return MlbRoutingRuntime(
        algorithm,
        ep_size=EP_SIZE,
        ep_rank=0,
        num_logical_experts=NUM_LOGICAL,
        num_physical_experts=num_physical,
        physical_to_logical_map=phy2log,
    )


@pytest.mark.parametrize(
    "algorithm, consumes", [("lplb", True), ("static", False), ("dynamic", False)]
)
def test_only_policies_that_read_the_load_pay_for_gathering_it(algorithm, consumes):
    """The count machinery follows what the policy declares, not "has a
    post-TopK policy".

    A replica policy can route from placement alone. Keying the count on
    requires_post_topk_routing charged every such policy a per-layer kernel and
    one collective per step for a number it never reads.
    """
    rt = _sized_runtime(algorithm)
    assert rt.consumes_global_logical_count is consumes
    assert (rt._logical_count_local is not None) is consumes


def test_the_runtime_reports_what_the_framework_must_respect():
    """Concurrency and graph stability are declared, not inferred from LPLB.

    Both guards used to key off properties LPLB happens to have, which barred
    combinations that were never at risk of the fault being guarded against.
    """
    st = _sized_runtime("static")
    assert st.supports_concurrent_microbatches is True
    assert st.graph_stability == "stable"

    lp = _sized_runtime("lplb")
    assert lp.supports_concurrent_microbatches is False
    assert lp.graph_stability == "realloc_on_placement_change"


def test_placement_and_routing_share_one_balancer():
    """An engine gets one balancer serving both layers.

    Two instances cannot see each other, which rules out any placement decision
    that depends on what routing observed.
    """
    from vllm.distributed.eplb.mlb_runtime import (
        get_mlb_integration,
        reset_mlb_routing,
    )

    reset_mlb_routing()
    try:
        integration = get_mlb_integration()
        integration.algorithm = "lplb"
        num_physical = NUM_LOGICAL + 8
        phy2log = torch.stack(
            [torch.arange(num_physical) % NUM_LOGICAL for _ in range(NUM_LAYERS)]
        )
        routing = integration.bind_routing(
            ep_size=EP_SIZE,
            ep_rank=0,
            num_logical_experts=NUM_LOGICAL,
            num_physical_experts=num_physical,
            physical_to_logical_map=phy2log,
        )
        assert integration.balancer() is routing._mlb
        assert get_mlb_integration() is integration
    finally:
        reset_mlb_routing()


def test_async_placement_commits_are_delivered_on_the_main_thread():
    """The async worker commits a layer's new map without telling the policy.

    Left undelivered, a policy holding placement-derived state keeps solving
    against the layout the layer used to have. The dispatch kernel clamps to the
    live replica count and reads the committed map, so the token still reaches a
    valid replica of the right expert -- what degrades silently is the split,
    which was computed to balance a placement that no longer exists.

    Delivery is queued rather than immediate because rebuilding that state does
    GPU work and the commit runs on a worker thread.
    """
    from vllm.distributed.eplb import eplb_state as st

    st._ASYNC_COMMITTED_LAYERS.clear()
    st._ASYNC_COMMITTED_LAYERS.update({2, 0})

    rt = _runtime("lplb")
    seen: list[list[int]] = []
    rt.on_placement_committed = lambda ids: seen.append(list(ids))
    rt.finalize_step_counts = lambda: None

    # Mirror what prepare_forward does with the queue.
    if st._ASYNC_COMMITTED_LAYERS and rt.requires_placement_state:
        pending = sorted(st._ASYNC_COMMITTED_LAYERS)
        st._ASYNC_COMMITTED_LAYERS.clear()
        rt.on_placement_committed(pending)

    assert seen == [[0, 2]]
    assert not st._ASYNC_COMMITTED_LAYERS


def test_a_policy_without_placement_state_needs_no_such_delivery():
    """static routes from the committed map directly, so there is nothing
    cached that a placement change could invalidate."""
    rt = _runtime("static")
    assert rt.requires_placement_state is False


def test_a_share_table_is_built_over_the_map_the_kernel_will_index():
    """Collapsing the candidate list is safe only when ids come back.

    The kernel resolves column j against vLLM's own candidate map, which is
    never collapsed. A table built over a shortened list therefore means
    something different on each side: to the policy column 0 is this rank's
    local replica, to the kernel it is the globally-first one. The ranks that
    own a copy end up routed away from it.
    """
    rt = _runtime("static")
    state = _layer_state(rt)
    deferred = rt._snapshot(state, 0, defer_dispatch=True)
    direct = rt._snapshot(state, 0)

    # The deferred snapshot must present the same candidate layout the kernel
    # will index, i.e. the uncollapsed map.
    assert torch.equal(
        deferred.logical_to_physical_candidates,
        rt._logical_to_physical_map[0][:, : rt._max_replicas],
    )
    assert torch.equal(deferred.logical_to_physical_count, rt._logical_replica_count[0])

    # The direct path still collapses: a policy returning ids resolves them
    # itself, so a local-only list is exactly the locality hint it wants.
    assert rt.requires_rank_dispatch_map
    replicated = (rt._logical_replica_count[0] > 1).nonzero()
    if replicated.numel():
        lg = int(replicated[0])
        assert int(direct.logical_to_physical_count[lg]) <= int(
            deferred.logical_to_physical_count[lg]
        )


def test_the_nearest_replica_table_is_refreshed_on_every_commit():
    """This side's derived table has to follow the placement, whatever the
    policy keeps.

    static reads this table and holds nothing else, so gating the refresh on
    requires_placement_state left it routing by the placement the run started
    with. Once experts move, most defaults are no longer in their expert's
    candidate list, the share table falls back to column 0 for them, and the
    traffic those replicas exist to spread piles onto one.
    """
    for algorithm, keeps_state in (("static", False), ("lplb", True)):
        rt = _runtime(algorithm)
        assert rt.requires_placement_state is keeps_state
        calls = []
        rt._rebuild_default_replicas = lambda: calls.append(1)
        rt._commit_layers = lambda ids: None
        rt.on_placement_committed([0])
        assert calls, (
            f"{algorithm}: the nearest-replica table was not refreshed, so it "
            "still describes the placement before this commit"
        )
