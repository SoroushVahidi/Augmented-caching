from __future__ import annotations

from lafc.policies.sentinel_robust_tripwire_v1 import SentinelRobustTripwireV1Policy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def test_sentinel_tripwire_registered_and_runs():
    assert "sentinel_robust_tripwire_v1" in POLICY_REGISTRY

    reqs, pages = build_requests_from_lists(
        ["A", "B", "C", "A", "D", "A", "B", "E", "A", "F", "A", "G", "A"]
    )
    reqs = attach_predicted_caches(reqs, capacity=3)

    policy = SentinelRobustTripwireV1Policy(warmup_steps=2, budget_init=1, budget_max=1)
    result = run_policy(policy, reqs, pages, capacity=3)

    assert result.total_misses >= 0

    diag = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
    summary = diag.get("summary", {})
    step_log = diag.get("step_log", [])

    assert len(step_log) == len(reqs)
    assert summary.get("robust_steps", 0.0) > 0.0
    assert 0.0 <= float(summary.get("predictor_coverage", 0.0)) <= 1.0
    # Warmup enforces robust-first behavior at the beginning.
    assert step_log[0]["chosen_line"] == "robust"


def test_sentinel_tripwire_only_overrides_on_disagreement():
    # With capacity=1, robust and predictor shadows cannot disagree on eviction:
    # there is only one possible victim when full.
    reqs, pages = build_requests_from_lists(["A", "B", "A", "B", "A", "B"])
    reqs = attach_predicted_caches(reqs, capacity=1)

    policy = SentinelRobustTripwireV1Policy(
        warmup_steps=0,
        risk_threshold=1.0,
        budget_init=10,
        budget_max=10,
    )
    result = run_policy(policy, reqs, pages, capacity=1)
    diag = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
    summary = diag.get("summary", {})

    assert summary.get("predictor_steps", 0.0) == 0.0
