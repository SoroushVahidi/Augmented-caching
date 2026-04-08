from __future__ import annotations

import json

from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy, save_results
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _trace_for_policy_style_checks():
    page_ids = ["A", "B", "C", "A", "B", "D", "A", "B"]
    prediction_records = [
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 1, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def _trace_for_mismatch_decay():
    # At t=2, with A/B cached, evict B if predictor-led (B has larger bucket).
    # B immediately returns at t=3 => fast mismatch signal for predictor-dominated decision.
    page_ids = ["A", "B", "C", "B", "A", "D", "B", "A", "E", "B", "A"]
    prediction_records = [
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def _extract_evictions(result):
    return [e.evicted for e in result.events]


def test_atlas_v2_runs_on_tiny_trace():
    requests, pages = _trace_for_policy_style_checks()
    result = run_policy(AtlasV2Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0
    assert result.extra_diagnostics is not None
    assert "atlas_v2" in result.extra_diagnostics


def test_atlas_v2_gamma_high_keeps_predictor_led_decisions():
    requests, pages = _trace_for_policy_style_checks()
    result = run_policy(
        AtlasV2Policy(atlas_initial_gamma=1.0, atlas_rho=0.2),
        requests,
        pages,
        capacity=2,
    )
    summary = result.extra_diagnostics["atlas_v2"]["summary"]
    assert summary["gamma_final"] > 0.6
    assert summary["fraction_predictor_dominated"] >= 0.5


def test_atlas_v2_mismatch_accumulation_decreases_gamma():
    requests, pages = _trace_for_mismatch_decay()
    result = run_policy(
        AtlasV2Policy(
            atlas_initial_gamma=1.0,
            atlas_rho=0.8,
            atlas_mismatch_threshold=2,
            atlas_window=8,
        ),
        requests,
        pages,
        capacity=2,
    )
    summary = result.extra_diagnostics["atlas_v2"]["summary"]
    assert summary["rolling_mismatch_rate"] > 0.0
    assert summary["gamma_final"] < 1.0


def test_atlas_v2_low_gamma_is_more_lru_like():
    requests, pages = _trace_for_policy_style_checks()
    v2 = run_policy(
        AtlasV2Policy(atlas_initial_gamma=0.01, atlas_rho=1.0),
        requests,
        pages,
        capacity=2,
    )
    lru = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert _extract_evictions(v2) == _extract_evictions(lru)


def test_atlas_v2_is_deterministic():
    requests, pages = _trace_for_mismatch_decay()
    p1 = AtlasV2Policy()
    p2 = AtlasV2Policy()
    r1 = run_policy(p1, requests, pages, capacity=2)
    r2 = run_policy(p2, requests, pages, capacity=2)
    assert _extract_evictions(r1) == _extract_evictions(r2)


def test_atlas_v2_backward_compatibility_old_trace_format(tmp_path):
    trace = {
        "requests": ["A", "B", "C", "A"],
        "predictions": [3, 4, 10, 99],
    }
    path = tmp_path / "legacy_trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    requests, pages = load_trace(str(path))
    result = run_policy(AtlasV2Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0


def test_atlas_v2_diagnostics_saved(tmp_path):
    requests, pages = _trace_for_policy_style_checks()
    result = run_policy(AtlasV2Policy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    diag_path = tmp_path / "atlas_v2_diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "decision_log" in payload
    assert "time_series" in payload


def test_atlas_v2_confidence_sensitive_tie_breaking_changes_choice():
    # Two eviction decisions tie in blended score at gamma=1:
    # A: base=1 pred=0 conf=0.8 => 0.2
    # B: base=0 pred=1 conf=0.2 => 0.2
    page_ids = ["A", "B", "C"]
    prediction_records = [
        {"bucket": 0, "confidence": 0.8},
        {"bucket": 1, "confidence": 0.2},
        {"bucket": 1, "confidence": 1.0},
    ]
    requests, pages = build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)

    high = run_policy(
        AtlasV2Policy(atlas_initial_gamma=1.0, tie_confidence_threshold=0.5),
        requests,
        pages,
        capacity=2,
    )
    low = run_policy(
        AtlasV2Policy(atlas_initial_gamma=0.2, tie_confidence_threshold=0.5),
        requests,
        pages,
        capacity=2,
    )

    assert high.events[2].evicted == "B"
    assert low.events[2].evicted == "A"


def test_atlas_v1_still_runs_unchanged():
    requests, pages = _trace_for_policy_style_checks()
    result = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    assert result.total_misses >= 0
