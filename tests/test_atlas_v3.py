from __future__ import annotations

import json

from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy, save_results
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _evictions(result):
    return [e.evicted for e in result.events]


def _trace_init_contexts():
    page_ids = ["A", "B", "C", "A", "B", "D"]
    prediction_records = [
        {"bucket": 0, "confidence": 0.2},
        {"bucket": 2, "confidence": 0.9},
        {"bucket": 3, "confidence": 0.9},
        {"bucket": 0, "confidence": 0.2},
        {"bucket": 2, "confidence": 0.9},
        {"bucket": 3, "confidence": 0.9},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def _trace_bad_returns():
    # Predictor-led eviction of B should return quickly and be marked bad.
    page_ids = ["A", "B", "C", "B", "A", "D", "B"]
    prediction_records = [
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def _trace_good_returns():
    # Predictor-led eviction happens, but evicted page does not return within tolerated horizon.
    page_ids = ["A", "B", "C", "A", "D", "E", "B"]
    prediction_records = [
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 1, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 2, "confidence": 1.0},
        {"bucket": 1, "confidence": 1.0},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def test_atlas_v3_smoke_run():
    requests, pages = _trace_init_contexts()
    result = run_policy(AtlasV3Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0
    assert result.extra_diagnostics is not None
    assert "atlas_v3" in result.extra_diagnostics


def test_atlas_v3_local_trust_initialization():
    requests, pages = _trace_init_contexts()
    policy = AtlasV3Policy(atlas_initial_local_trust=0.61)
    result = run_policy(policy, requests, pages, capacity=2)
    table = result.extra_diagnostics["atlas_v3"]["summary"]["local_trust_table"]
    assert table
    assert any(abs(v - 0.61) < 1e-9 for v in table.values())


def test_atlas_v3_local_trust_decreases_after_bad_outcome():
    requests, pages = _trace_bad_returns()
    result = run_policy(
        AtlasV3Policy(atlas_initial_local_trust=0.8, atlas_eta_neg=0.2, atlas_eta_pos=0.01),
        requests,
        pages,
        capacity=2,
    )
    summary = result.extra_diagnostics["atlas_v3"]["summary"]
    assert sum(summary["context_bad_counts"].values()) >= 1
    assert min(summary["local_trust_table"].values()) < 0.8


def test_atlas_v3_local_trust_increases_after_good_outcome():
    requests, pages = _trace_good_returns()
    result = run_policy(
        AtlasV3Policy(atlas_initial_local_trust=1.0, atlas_eta_pos=0.05, atlas_eta_neg=0.2, bucket_horizon=1),
        requests,
        pages,
        capacity=2,
    )
    summary = result.extra_diagnostics["atlas_v3"]["summary"]
    assert sum(summary["context_good_counts"].values()) >= 1
    assert max(summary["local_trust_table"].values()) == 1.0


def test_bad_context_does_not_globally_collapse_other_contexts():
    requests, pages = _trace_bad_returns()
    result = run_policy(
        AtlasV3Policy(atlas_initial_local_trust=0.8, atlas_eta_neg=0.3, atlas_eta_pos=0.01),
        requests,
        pages,
        capacity=2,
    )
    table = result.extra_diagnostics["atlas_v3"]["summary"]["local_trust_table"]
    values = list(table.values())
    assert len(values) >= 2
    assert max(values) - min(values) > 1e-6


def test_atlas_v3_deterministic_reproducibility():
    requests, pages = _trace_bad_returns()
    p1 = AtlasV3Policy()
    p2 = AtlasV3Policy()
    r1 = run_policy(p1, requests, pages, capacity=2)
    r2 = run_policy(p2, requests, pages, capacity=2)
    assert _evictions(r1) == _evictions(r2)


def test_atlas_v3_diagnostics_presence(tmp_path):
    requests, pages = _trace_init_contexts()
    result = run_policy(AtlasV3Policy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    diag_path = tmp_path / "atlas_v3_diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "decision_log" in payload
    assert "time_series" in payload


def test_atlas_v3_backward_compatibility_old_trace_format(tmp_path):
    trace = {"requests": ["A", "B", "C", "A"], "predictions": [3, 4, 10, 99]}
    path = tmp_path / "legacy_trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    requests, pages = load_trace(str(path))
    result = run_policy(AtlasV3Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0


def test_atlas_v1_v2_non_regression_smoke():
    requests, pages = _trace_init_contexts()
    r1 = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    r2 = run_policy(AtlasV2Policy(default_confidence=0.5), requests, pages, capacity=2)
    lru = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert r1.total_misses >= 0
    assert r2.total_misses >= 0
    assert lru.total_misses >= 0


def test_atlas_v3_context_mode_bucket_only_runs():
    requests, pages = _trace_init_contexts()
    result = run_policy(
        AtlasV3Policy(atlas_context_mode="bucket_only"),
        requests,
        pages,
        capacity=2,
    )
    assert result.total_misses >= 0


def test_atlas_v3_adaptive_tie_coef_runs():
    requests, pages = _trace_init_contexts()
    result = run_policy(
        AtlasV3Policy(atlas_adaptive_tie_coef=0.5),
        requests,
        pages,
        capacity=2,
    )
    assert result.total_misses >= 0
