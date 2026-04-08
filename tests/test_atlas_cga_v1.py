from __future__ import annotations

import json

from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
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
    page_ids = ["A", "B", "C", "A", "D", "E", "B", "F", "G"]
    prediction_records = [{"bucket": 3, "confidence": 1.0}] * len(page_ids)
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def test_atlas_cga_v1_smoke_run():
    requests, pages = _trace_init_contexts()
    result = run_policy(AtlasCGAV1Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0
    assert result.extra_diagnostics is not None
    assert "atlas_cga_v1" in result.extra_diagnostics


def test_calibration_stats_initialization_uses_prior():
    policy = AtlasCGAV1Policy(atlas_calibration_prior_a=2.0, atlas_calibration_prior_b=1.0)
    _, pages = _trace_init_contexts()
    policy.reset(capacity=2, pages=pages)
    empirical, posterior, wcal, pcal = policy._calibration_stats_for_context(("bucket=0", "bin_0"))
    assert abs(empirical - (2.0 / 3.0)) < 1e-9
    assert abs(posterior - (2.0 / 3.0)) < 1e-9
    assert wcal == 0.0
    assert abs(pcal - (2.0 / 3.0)) < 1e-9


def test_calibrated_probability_increases_after_safe_outcomes():
    policy = AtlasCGAV1Policy(atlas_calibration_shrinkage=0.0, atlas_calibration_min_support=0)
    _, pages = _trace_good_returns()
    policy.reset(capacity=2, pages=pages)
    ctx = ("bucket=3", "bin_2")
    for _ in range(6):
        policy._apply_outcome_update(ctx, calibrated_signal=0.8, is_bad=False, trust_eligible=False)
    _, _, _, pcal = policy._calibration_stats_for_context(ctx)
    assert pcal > 0.8


def test_calibrated_probability_decreases_after_unsafe_outcomes():
    policy = AtlasCGAV1Policy(atlas_calibration_shrinkage=0.0, atlas_calibration_min_support=0)
    _, pages = _trace_bad_returns()
    policy.reset(capacity=2, pages=pages)
    ctx = ("bucket=3", "bin_2")
    for _ in range(6):
        policy._apply_outcome_update(ctx, calibrated_signal=0.8, is_bad=True, trust_eligible=False)
    _, _, _, pcal = policy._calibration_stats_for_context(ctx)
    assert pcal < 0.2


def test_low_support_contexts_shrunk_toward_prior():
    requests, pages = _trace_init_contexts()
    prior = 0.5
    result = run_policy(
        AtlasCGAV1Policy(
            atlas_calibration_prior_a=1.0,
            atlas_calibration_prior_b=1.0,
            atlas_calibration_shrinkage=50.0,
            atlas_calibration_min_support=10,
        ),
        requests,
        pages,
        capacity=2,
    )
    table = result.extra_diagnostics["atlas_cga_v1"]["summary"]["calibration_table"]
    assert all(abs(v["pcal"] - prior) < 0.2 for v in table.values())


def test_atlas_cga_v1_deterministic_reproducibility():
    requests, pages = _trace_bad_returns()
    p1 = AtlasCGAV1Policy()
    p2 = AtlasCGAV1Policy()
    r1 = run_policy(p1, requests, pages, capacity=2)
    r2 = run_policy(p2, requests, pages, capacity=2)
    assert _evictions(r1) == _evictions(r2)


def test_atlas_cga_v1_backward_compatibility_old_trace_format(tmp_path):
    trace = {"requests": ["A", "B", "C", "A"], "predictions": [3, 4, 10, 99]}
    path = tmp_path / "legacy_trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    requests, pages = load_trace(str(path))
    result = run_policy(AtlasCGAV1Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0


def test_atlas_cga_v1_diagnostics_presence(tmp_path):
    requests, pages = _trace_init_contexts()
    result = run_policy(AtlasCGAV1Policy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    diag_path = tmp_path / "atlas_cga_v1_diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "decision_log" in payload
    assert "time_series" in payload


def test_atlas_v1_v2_v3_non_regression_smoke():
    requests, pages = _trace_init_contexts()
    r1 = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    r2 = run_policy(AtlasV2Policy(default_confidence=0.5), requests, pages, capacity=2)
    r3 = run_policy(AtlasV3Policy(default_confidence=0.5), requests, pages, capacity=2)
    lru = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert r1.total_misses >= 0
    assert r2.total_misses >= 0
    assert r3.total_misses >= 0
    assert lru.total_misses >= 0
