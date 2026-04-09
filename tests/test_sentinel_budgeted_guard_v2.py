from __future__ import annotations

from lafc.policies.sentinel_budgeted_guard_v2 import SentinelBudgetedGuardV2Policy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def test_sentinel_budgeted_guard_v2_registered_and_runs() -> None:
    assert "sentinel_budgeted_guard_v2" in POLICY_REGISTRY

    reqs, pages = build_requests_from_lists(
        ["A", "B", "C", "A", "D", "A", "B", "E", "A", "F", "A", "G", "A"]
    )
    reqs = attach_predicted_caches(reqs, capacity=3)

    policy = SentinelBudgetedGuardV2Policy(warmup_steps=2, override_budget_total=2)
    result = run_policy(policy, reqs, pages, capacity=3)

    diag = (result.extra_diagnostics or {}).get("sentinel_budgeted_guard_v2", {})
    summary = diag.get("summary", {})
    step_log = diag.get("step_log", [])

    assert len(step_log) == len(reqs)
    assert summary.get("robust_steps", 0.0) > 0.0
    assert 0.0 <= float(summary.get("predictor_coverage", 0.0)) <= 1.0
    assert step_log[0]["chosen_line"] == "robust"


def test_sentinel_budgeted_guard_v2_finite_budget_caps_overrides() -> None:
    reqs, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "D", "A", "B", "E", "A", "B"])
    reqs = attach_predicted_caches(reqs, capacity=2)

    policy = SentinelBudgetedGuardV2Policy(
        warmup_steps=0,
        risk_threshold=1.0,
        override_budget_total=1,
        reentry_stable_steps=0,
        reentry_memory_threshold=10,
        guard_trigger_threshold=99,
    )
    result = run_policy(policy, reqs, pages, capacity=2)
    diag = (result.extra_diagnostics or {}).get("sentinel_budgeted_guard_v2", {})
    summary = diag.get("summary", {})

    predictor_steps = float(summary.get("predictor_steps", 0.0))
    remaining_budget = float(summary.get("remaining_override_budget", -1.0))
    assert predictor_steps <= 1.0
    assert 0.0 <= remaining_budget <= 1.0
    if predictor_steps >= 1.0:
        assert remaining_budget == 0.0


def test_sentinel_budgeted_guard_v2_reentry_blocks_predictor_until_stable() -> None:
    reqs, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "A", "C", "A", "D", "A", "C"])
    reqs = attach_predicted_caches(reqs, capacity=2)

    policy = SentinelBudgetedGuardV2Policy(
        warmup_steps=0,
        risk_threshold=1.0,
        override_budget_total=10,
        guard_duration=1,
        guard_trigger_threshold=1,
        reentry_stable_steps=3,
        reentry_memory_threshold=0,
    )
    result = run_policy(policy, reqs, pages, capacity=2)

    diag = (result.extra_diagnostics or {}).get("sentinel_budgeted_guard_v2", {})
    summary = diag.get("summary", {})

    assert summary.get("guard_triggers", 0.0) >= 0.0
    assert summary.get("reentry_blocks", 0.0) > 0.0
