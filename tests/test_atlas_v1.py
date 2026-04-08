from __future__ import annotations

import json

from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy, save_results
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _atlas_trace(conf_a: float, conf_b: float):
    page_ids = ["A", "B", "C", "A", "B"]
    prediction_records = [
        {"bucket": 0, "confidence": conf_a},
        {"bucket": 3, "confidence": conf_b},
        {"bucket": 2, "confidence": 0.5},
        {"bucket": 1, "confidence": conf_a},
        {"bucket": 2, "confidence": conf_b},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def test_atlas_v1_smoke_runs_and_is_deterministic():
    requests, pages = _atlas_trace(conf_a=0.6, conf_b=0.7)
    p1 = AtlasV1Policy(default_confidence=0.5)
    p2 = AtlasV1Policy(default_confidence=0.5)

    r1 = run_policy(p1, requests, pages, capacity=2)
    r2 = run_policy(p2, requests, pages, capacity=2)

    assert r1.total_misses >= 0
    assert [e.evicted for e in r1.events] == [e.evicted for e in r2.events]


def test_atlas_v1_confidence_one_prefers_prediction():
    requests, pages = _atlas_trace(conf_a=1.0, conf_b=1.0)
    result = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    # At t=2 request C causes first eviction; prediction says B is more evictable.
    assert result.events[2].evicted == "B"


def test_atlas_v1_confidence_zero_reduces_to_lru_behavior():
    requests, pages = _atlas_trace(conf_a=0.0, conf_b=0.0)
    atlas = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    lru = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert [e.evicted for e in atlas.events] == [e.evicted for e in lru.events]


def test_atlas_v1_mixed_confidence_mixes_trust_levels():
    requests, pages = _atlas_trace(conf_a=0.2, conf_b=0.9)
    result = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    atlas_diag = result.extra_diagnostics["atlas_v1"]["summary"]
    assert 0.0 < atlas_diag["average_lambda"] < 1.0
    assert atlas_diag["fraction_low_confidence_decisions"] > 0.0


def test_backward_compatibility_old_trace_format_for_old_policies(tmp_path):
    trace = {
        "requests": ["A", "B", "C", "A"],
        "predictions": [3, 4, 10, 99],
    }
    path = tmp_path / "legacy_trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    requests, pages = load_trace(str(path))
    result = run_policy(BlindOraclePolicy(), requests, pages, capacity=2)
    assert result.total_misses >= 0


def test_runner_outputs_include_atlas_diagnostics(tmp_path):
    requests, pages = _atlas_trace(conf_a=0.2, conf_b=0.9)
    result = run_policy(AtlasV1Policy(default_confidence=0.5), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    diag_path = tmp_path / "atlas_v1_diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "decision_log" in payload
