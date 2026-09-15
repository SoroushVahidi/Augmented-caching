from __future__ import annotations

import importlib.util
from pathlib import Path

from lafc.policies.lfu import LFUPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def _run(page_ids: list[str], capacity: int):
    requests, pages = build_requests_from_lists(page_ids=page_ids)
    return run_policy(LFUPolicy(), requests, pages, capacity)


def test_lfu_capacity_one():
    result = _run(["A", "A", "B", "B"], capacity=1)
    assert [event.evicted for event in result.events] == [None, None, "A", None]
    assert result.total_hits == 2
    assert result.total_misses == 2


def test_lfu_repeated_hot_key_survives():
    result = _run(["A", "B", "A", "A", "C"], capacity=2)
    assert result.events[-1].evicted == "B"


def test_lfu_scan_evicts_old_single_use_items():
    result = _run(["A", "B", "C", "D"], capacity=2)
    assert [event.evicted for event in result.events] == [None, None, "A", "B"]
    assert result.total_hits == 0


def test_lfu_equal_frequency_tie_uses_oldest_last_touch():
    result = _run(["A", "B", "C"], capacity=2)
    assert result.events[-1].evicted == "A"


def test_lfu_eviction_after_frequency_update():
    result = _run(["A", "B", "A", "C", "B"], capacity=2)
    assert result.events[3].evicted == "B"
    assert result.events[4].evicted == "C"


def test_lfu_deterministic_repeatability():
    trace = ["A", "B", "A", "C", "B", "D", "A", "E", "D", "F"]
    first = _run(trace, capacity=3)
    second = _run(trace, capacity=3)
    assert [(e.hit, e.evicted) for e in first.events] == [(e.hit, e.evicted) for e in second.events]
    assert first.total_misses == second.total_misses


def test_lfu_capacity_32_plus_behavior():
    trace = [f"k{i}" for i in range(32)] + ["k0"] * 4 + ["new"]
    result = _run(trace, capacity=32)
    assert result.events[-1].evicted == "k1"
    assert result.total_hits == 4
    assert result.total_misses == 33


def test_runner_policies_dict_accepts_lfu():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_policy_comparison_wulver_v1.py"
    spec = importlib.util.spec_from_file_location("run_policy_comparison_wulver_v1", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "lfu" in module.POLICIES
    assert isinstance(module.POLICIES["lfu"](""), LFUPolicy)


def test_runner_policy_registry_accepts_lfu():
    from lafc.runner.run_policy import POLICY_REGISTRY

    assert "lfu" in POLICY_REGISTRY
    assert POLICY_REGISTRY["lfu"].name == "lfu"
