from __future__ import annotations

import importlib.util
from pathlib import Path

from lafc.policies.sieve import SievePolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def _residents_oldest_to_newest(policy: SievePolicy):
    return list(policy._order.keys())


def test_sieve_hit_sets_visited_without_moving():
    # Fill a capacity-3 cache with A, B, C (insertion order = SIEVE queue order).
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "A"])
    policy = SievePolicy()
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


def test_sieve_new_insertions_go_to_the_head():
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C"])
    policy = SievePolicy()
    policy.reset(3, pages)
    for req in requests:
        policy.on_request(req)
    # Head = most-recently-inserted end = last key in insertion order.
    assert _residents_oldest_to_newest(policy)[-1] == "C"
    assert policy._visited["C"] is False


def test_sieve_eviction_skips_and_clears_visited_then_evicts_first_unvisited():
    # Cache: A, B, C (capacity 3). Re-access A and B (visited=True), leave C untouched.
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "A", "B", "D"])
    policy = SievePolicy()
    policy.reset(3, pages)

    for req in requests[:5]:  # A, B, C (miss), A, B (hits)
        policy.on_request(req)
    assert policy._visited == {"A": True, "B": True, "C": False}
    assert policy._hand is None

    event = policy.on_request(requests[5])  # D: miss, cache full -> triggers scan
    # Hand starts at tail (A): A.visited=1 -> clear, advance; B.visited=1 -> clear,
    # advance; C.visited=0 -> evict C (the first unvisited object the hand finds),
    # exactly per Algorithm 1 lines 5-14.
    assert event.evicted == "C"
    assert policy._visited["A"] is False
    assert policy._visited["B"] is False
    assert "C" not in policy._visited
    assert _residents_oldest_to_newest(policy) == ["A", "B", "D"]


def test_sieve_eviction_removes_first_unvisited_item_found_by_hand():
    # Cache: A, B, C (capacity 3), none re-accessed -> all visited=False.
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "D"])
    policy = SievePolicy()
    policy.reset(3, pages)
    for req in requests[:3]:
        policy.on_request(req)

    event = policy.on_request(requests[3])  # D: miss, full -> hand starts at tail (A)
    # No object is visited, so the scan stops immediately at the tail: evict A.
    assert event.evicted == "A"
    assert _residents_oldest_to_newest(policy) == ["B", "C", "D"]


def test_sieve_capacity_one():
    requests, pages = build_requests_from_lists(page_ids=["A", "A", "B", "C"])
    policy = SievePolicy()
    policy.reset(1, pages)

    e0 = policy.on_request(requests[0])  # A: miss, insert
    assert e0.hit is False and e0.evicted is None

    e1 = policy.on_request(requests[1])  # A again: hit
    assert e1.hit is True

    e2 = policy.on_request(requests[2])  # B: miss, full -> evict A
    assert e2.hit is False and e2.evicted == "A"

    e3 = policy.on_request(requests[3])  # C: miss, full -> evict B
    assert e3.hit is False and e3.evicted == "B"

    assert _residents_oldest_to_newest(policy) == ["C"]


def test_sieve_runs_via_run_policy_helper():
    requests, pages = build_requests_from_lists(page_ids=["A", "B", "C", "A", "D", "B", "E"])
    result = run_policy(SievePolicy(), requests, pages, capacity=3)
    assert result.policy_name == "sieve"
    assert result.total_hits + result.total_misses == len(requests)


def test_runner_policies_dict_accepts_sieve():
    # Mirrors the canonical heavy_r1 pipeline's integration point: confirms
    # `--policies sieve` resolves to a working SievePolicy instance without
    # running any trace (zero-compute check on the registry wiring itself).
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_policy_comparison_wulver_v1.py"
    spec = importlib.util.spec_from_file_location("run_policy_comparison_wulver_v1", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "sieve" in module.POLICIES
    instance = module.POLICIES["sieve"]("")
    assert isinstance(instance, SievePolicy)
    assert instance.name == "sieve"


def test_runner_policy_registry_accepts_sieve():
    from lafc.runner.run_policy import POLICY_REGISTRY

    assert "sieve" in POLICY_REGISTRY
    assert POLICY_REGISTRY["sieve"].name == "sieve"
