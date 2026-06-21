from __future__ import annotations

import importlib.util
from pathlib import Path

from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def _residents_oldest_to_newest(policy: FIFOReinsertionPolicy):
    return list(policy._order.keys())


def test_fifo_reinsertion_hit_sets_visited_without_moving():
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "A"])
    policy = FIFOReinsertionPolicy()
    policy.reset(3, pages)

    for req in requests[:3]:
        policy.on_request(req)
    assert _residents_oldest_to_newest(policy) == ["A", "B", "C"]
    assert policy._visited == {"A": False, "B": False, "C": False}

    event = policy.on_request(requests[3])  # second request for "A" -> hit
    assert event.hit is True
    # Visited bit is set, but queue order is unchanged (no move-to-head on hit).
    assert policy._visited["A"] is True
    assert _residents_oldest_to_newest(policy) == ["A", "B", "C"]


def test_fifo_reinsertion_new_insertions_go_to_the_head():
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C"])
    policy = FIFOReinsertionPolicy()
    policy.reset(3, pages)
    for req in requests:
        policy.on_request(req)
    assert _residents_oldest_to_newest(policy)[-1] == "C"
    assert policy._visited["C"] is False


def test_fifo_reinsertion_evicts_unvisited_tail_without_reinsertion():
    # Cache: A, B, C (capacity 3), none re-accessed -> all visited=False.
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "D"])
    policy = FIFOReinsertionPolicy()
    policy.reset(3, pages)
    for req in requests[:3]:
        policy.on_request(req)

    event = policy.on_request(requests[3])  # D: miss, full -> tail is A, unvisited
    assert event.evicted == "A"
    assert _residents_oldest_to_newest(policy) == ["B", "C", "D"]


def test_fifo_reinsertion_reinserts_visited_survivors_at_head_then_evicts_first_unvisited():
    # Cache: A, B, C (capacity 3). Re-access A and B (visited=True), leave C untouched.
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "A", "B", "D"])
    policy = FIFOReinsertionPolicy()
    policy.reset(3, pages)

    for req in requests[:5]:  # A, B, C (miss), A, B (hits)
        policy.on_request(req)
    assert policy._visited == {"A": True, "B": True, "C": False}

    event = policy.on_request(requests[5])  # D: miss, cache full -> triggers scan
    # Tail-first scan: A.visited=1 -> clear bit, reinsert A at head; B.visited=1 ->
    # clear bit, reinsert B at head; C.visited=0 -> evict C. Unlike SIEVE, A and B
    # are physically moved to the head (mixed with the newly-inserted D), not left
    # in place behind a hand pointer.
    assert event.evicted == "C"
    assert policy._visited["A"] is False
    assert policy._visited["B"] is False
    assert "C" not in policy._visited
    assert _residents_oldest_to_newest(policy) == ["A", "B", "D"]


def test_fifo_reinsertion_capacity_one():
    requests, pages = build_requests_from_lists(page_ids=["A", "A", "B", "C"])
    policy = FIFOReinsertionPolicy()
    policy.reset(1, pages)

    e0 = policy.on_request(requests[0])  # A: miss, insert
    assert e0.hit is False and e0.evicted is None

    e1 = policy.on_request(requests[1])  # A again: hit
    assert e1.hit is True

    e2 = policy.on_request(requests[2])  # B: miss, full -> A visited, reinsert+clear, then evict A
    assert e2.hit is False and e2.evicted == "A"

    e3 = policy.on_request(requests[3])  # C: miss, full -> evict B (never re-accessed)
    assert e3.hit is False and e3.evicted == "B"

    assert _residents_oldest_to_newest(policy) == ["C"]


def test_fifo_reinsertion_runs_via_run_policy_helper():
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "A", "D", "B", "E"])
    result = run_policy(FIFOReinsertionPolicy(), requests, pages, capacity=3)
    assert result.policy_name == "fifo_reinsertion"
    assert result.total_hits + result.total_misses == len(requests)


def test_runner_policies_dict_accepts_fifo_reinsertion():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_policy_comparison_wulver_v1.py"
    spec = importlib.util.spec_from_file_location("run_policy_comparison_wulver_v1", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "fifo_reinsertion" in module.POLICIES
    instance = module.POLICIES["fifo_reinsertion"]("")
    assert isinstance(instance, FIFOReinsertionPolicy)
    assert instance.name == "fifo_reinsertion"


def test_runner_policy_registry_accepts_fifo_reinsertion():
    from lafc.runner.run_policy import POLICY_REGISTRY

    assert "fifo_reinsertion" in POLICY_REGISTRY
    assert POLICY_REGISTRY["fifo_reinsertion"].name == "fifo_reinsertion"
