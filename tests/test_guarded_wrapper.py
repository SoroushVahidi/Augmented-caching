from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import Ridge

from lafc.evict_value_dataset_v1 import EvictValueDatasetV1Config, build_evict_value_examples_v1
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.policies.guard_wrapper import EvictValueV1GuardedPolicy
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _fit_bad_model(model_path: Path) -> None:
    reqs, _pages = load_trace("data/example_atlas_v1.json")
    rows = [
        r
        for r in build_evict_value_examples_v1(
            reqs, capacity=2, trace_name="toy", cfg=EvictValueDatasetV1Config(horizons=(8,))
        )
        if int(r["horizon"]) == 8
    ]
    x = [[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows]
    # Deliberately anti-safe target: evict hotter pages (lower predicted loss).
    y = [-float(r["recent_candidate_request_rate"]) for r in rows]
    est = Ridge(alpha=1.0)
    est.fit(x, y)
    EvictValueV1Model(
        model_name="guard_test_bad_model",
        estimator=est,
        feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS),
    ).save(model_path)


def test_guarded_policy_registered():
    assert "evict_value_v1_guarded" in POLICY_REGISTRY


def test_guarded_policy_triggers_and_uses_fallback(tmp_path: Path):
    model_path = tmp_path / "guard_bad.pkl"
    _fit_bad_model(model_path)

    # Repeated reuse of A can expose bad evictions quickly.
    reqs, pages = build_requests_from_lists(["A", "B", "A", "C", "A", "D", "A", "E", "A", "F", "A"])

    policy = EvictValueV1GuardedPolicy(
        model_path=str(model_path),
        fallback_policy="lru",
        early_return_window=2,
        trigger_threshold=1,
        trigger_window=6,
        guard_duration=3,
    )
    result = run_policy(policy, reqs, pages, capacity=2)
    assert result.total_misses >= 0

    diag = (result.extra_diagnostics or {}).get("evict_value_v1_guarded", {})
    summary = diag.get("summary", {})
    step_log = diag.get("step_log", [])

    assert summary.get("guard_triggers", 0) >= 1
    assert summary.get("fallback_time_steps", 0) >= 1
    assert any(bool(s.get("guard_triggered")) for s in step_log)
    assert any(s.get("mode_before") == "fallback" for s in step_log)


def test_guarded_policy_deterministic(tmp_path: Path):
    model_path = tmp_path / "guard_bad.pkl"
    _fit_bad_model(model_path)

    reqs, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "A", "E", "A", "F", "A"])

    p1 = EvictValueV1GuardedPolicy(model_path=str(model_path), trigger_threshold=1, guard_duration=2)
    p2 = EvictValueV1GuardedPolicy(model_path=str(model_path), trigger_threshold=1, guard_duration=2)

    r1 = run_policy(p1, reqs, pages, capacity=2)
    r2 = run_policy(p2, reqs, pages, capacity=2)

    assert r1.total_misses == r2.total_misses
    assert [e.evicted for e in r1.events] == [e.evicted for e in r2.events]


def test_guarded_policy_cache_invariant(tmp_path: Path):
    model_path = tmp_path / "guard_bad.pkl"
    _fit_bad_model(model_path)

    reqs, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "A", "E", "A"])
    pol = EvictValueV1GuardedPolicy(model_path=str(model_path), trigger_threshold=1, guard_duration=2)
    pol.reset(2, pages)

    for req in reqs:
        event = pol.on_request(req)
        assert len(pol.current_cache()) <= 2
        if event.evicted is not None:
            assert event.evicted != req.page_id
