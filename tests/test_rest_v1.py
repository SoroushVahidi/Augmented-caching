from __future__ import annotations

import json

from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
from lafc.policies.atlas_cga_v2 import AtlasCGAV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.rest_v1 import RestV1Policy
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
    # Trusted eviction in this context should be penalized (quick return within H).
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
    # Trusted eviction should be rewarded (no return within H).
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


def _trace_mode_choice():
    # At t=2 with cache [A,B], predictor prefers B (bucket=3), LRU prefers A.
    page_ids = ["A", "B", "C"]
    prediction_records = [
        {"bucket": 0, "confidence": 1.0},
        {"bucket": 3, "confidence": 1.0},
        {"bucket": 0, "confidence": 1.0},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def test_rest_v1_smoke_run():
    requests, pages = _trace_init_contexts()
    result = run_policy(RestV1Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0
    assert "rest_v1" in (result.extra_diagnostics or {})


def test_rest_v1_deterministic_reproducibility():
    requests, pages = _trace_bad_returns()
    r1 = run_policy(RestV1Policy(), requests, pages, capacity=2)
    r2 = run_policy(RestV1Policy(), requests, pages, capacity=2)
    assert _evictions(r1) == _evictions(r2)


def test_rest_v1_trust_initialization():
    requests, pages = _trace_init_contexts()
    result = run_policy(RestV1Policy(rest_initial_trust=0.61), requests, pages, capacity=2)
    table = result.extra_diagnostics["rest_v1"]["summary"]["trust_table"]
    assert table
    assert any(abs(v - 0.61) < 1e-9 for v in table.values())


def test_rest_v1_trust_increases_after_good_outcome():
    requests, pages = _trace_good_returns()
    result = run_policy(
        RestV1Policy(rest_initial_trust=1.0, rest_eta_pos=0.05, rest_eta_neg=0.2, rest_horizon=1),
        requests,
        pages,
        capacity=2,
    )
    summary = result.extra_diagnostics["rest_v1"]["summary"]
    assert sum(summary["context_good_counts"].values()) >= 1
    assert max(summary["trust_table"].values()) == 1.0


def test_rest_v1_trust_decreases_after_bad_outcome():
    requests, pages = _trace_bad_returns()
    result = run_policy(
        RestV1Policy(rest_initial_trust=0.8, rest_eta_neg=0.2, rest_eta_pos=0.01, rest_horizon=2),
        requests,
        pages,
        capacity=2,
    )
    summary = result.extra_diagnostics["rest_v1"]["summary"]
    assert sum(summary["context_bad_counts"].values()) >= 1
    assert min(summary["trust_table"].values()) < 0.8


def test_rest_v1_contexts_remain_isolated():
    requests, pages = _trace_bad_returns()
    result = run_policy(
        RestV1Policy(rest_initial_trust=0.8, rest_eta_neg=0.3, rest_eta_pos=0.01),
        requests,
        pages,
        capacity=2,
    )
    table = result.extra_diagnostics["rest_v1"]["summary"]["trust_table"]
    values = list(table.values())
    assert len(values) >= 2
    assert max(values) - min(values) > 1e-6


def test_rest_v1_backward_compatibility_old_trace_format(tmp_path):
    trace = {"requests": ["A", "B", "C", "A"], "predictions": [3, 4, 10, 99]}
    path = tmp_path / "legacy_trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    requests, pages = load_trace(str(path))
    result = run_policy(RestV1Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0


def test_rest_v1_diagnostics_presence(tmp_path):
    requests, pages = _trace_init_contexts()
    result = run_policy(RestV1Policy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    diag_path = tmp_path / "rest_v1_diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "decision_log" in payload
    assert "time_series" in payload


def test_atlas_cga_non_regression_smoke():
    requests, pages = _trace_init_contexts()
    r1 = run_policy(AtlasV3Policy(), requests, pages, capacity=2)
    r2 = run_policy(AtlasCGAV1Policy(), requests, pages, capacity=2)
    r3 = run_policy(AtlasCGAV2Policy(), requests, pages, capacity=2)
    assert r1.total_misses >= 0
    assert r2.total_misses >= 0
    assert r3.total_misses >= 0


def test_rest_v1_explicit_mode_choice_on_tiny_trace():
    requests, pages = _trace_mode_choice()

    trust_result = run_policy(
        RestV1Policy(rest_initial_trust=1.0, rest_trust_threshold=0.5),
        requests,
        pages,
        capacity=2,
    )
    abstain_result = run_policy(
        RestV1Policy(rest_initial_trust=0.0, rest_trust_threshold=0.5),
        requests,
        pages,
        capacity=2,
    )

    # Last request triggers first eviction.
    assert trust_result.events[-1].evicted == "B"
    assert abstain_result.events[-1].evicted == "A"

    trust_log = trust_result.extra_diagnostics["rest_v1"]["decision_log"]
    abstain_log = abstain_result.extra_diagnostics["rest_v1"]["decision_log"]
    assert trust_log[-1]["mode"] == "TRUST"
    assert abstain_log[-1]["mode"] == "ABSTAIN"
