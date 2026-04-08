from __future__ import annotations

import json

from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
from lafc.policies.atlas_cga_v2 import AtlasCGAV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.runner.run_policy import run_policy, save_results
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _evictions(result):
    return [e.evicted for e in result.events]


def _trace():
    page_ids = ["A", "B", "C", "A", "B", "D", "E", "A"]
    prediction_records = [
        {"bucket": 0, "confidence": 0.2},
        {"bucket": 2, "confidence": 0.9},
        {"bucket": 3, "confidence": 0.9},
        {"bucket": 0, "confidence": 0.2},
        {"bucket": 2, "confidence": 0.9},
        {"bucket": 3, "confidence": 0.9},
        {"bucket": 2, "confidence": 0.6},
        {"bucket": 0, "confidence": 0.2},
    ]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)


def test_atlas_cga_v2_smoke_run():
    requests, pages = _trace()
    result = run_policy(AtlasCGAV2Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0
    assert "atlas_cga_v2" in (result.extra_diagnostics or {})


def test_hierarchical_counts_initialization():
    _, pages = _trace()
    policy = AtlasCGAV2Policy(atlas_hier_global_prior_a=2.0, atlas_hier_global_prior_b=1.0)
    policy.reset(capacity=2, pages=pages)
    assert policy._n_global == 0
    assert policy._s_global == 0
    assert policy._posterior(0, 0) == 2.0 / 3.0


def test_low_support_borrows_strength_from_coarse_levels():
    _, pages = _trace()
    policy = AtlasCGAV2Policy(atlas_hier_shrink_strength=10.0, atlas_hier_min_support=5)
    policy.reset(capacity=2, pages=pages)

    ctx = ("bucket=2", "bin_2")
    for _ in range(20):
        policy._apply_outcome_update(ctx, "bucket=2", "bin_2", calibrated_signal=0.8, is_bad=False, trust_eligible=False)
    sparse_ctx = ("bucket=2", "bin_0")
    policy._apply_outcome_update(sparse_ctx, "bucket=2", "bin_0", calibrated_signal=0.8, is_bad=True, trust_eligible=False)

    p_ctx, p_bucket, p_conf, p_global, _, _, _, p_shared = policy._shared_calibration(sparse_ctx)
    assert abs(p_shared - p_bucket) < abs(p_ctx - p_bucket)
    assert p_global > 0.5


def test_high_support_relies_more_on_context():
    _, pages = _trace()
    policy = AtlasCGAV2Policy(atlas_hier_shrink_strength=1.0, atlas_hier_min_support=2)
    policy.reset(capacity=2, pages=pages)

    ctx = ("bucket=3", "bin_2")
    for _ in range(30):
        policy._apply_outcome_update(ctx, "bucket=3", "bin_2", calibrated_signal=0.9, is_bad=False, trust_eligible=False)
    w_ctx, w_bucket, w_conf, w_global = policy._hier_weights(policy._n_ctx[ctx], policy._n_bucket["bucket=3"], policy._n_conf["bin_2"])
    assert w_ctx >= 0.3
    assert (w_ctx + w_bucket + w_conf + w_global) == 1.0


def test_atlas_cga_v2_deterministic_reproducibility():
    requests, pages = _trace()
    r1 = run_policy(AtlasCGAV2Policy(), requests, pages, capacity=2)
    r2 = run_policy(AtlasCGAV2Policy(), requests, pages, capacity=2)
    assert _evictions(r1) == _evictions(r2)


def test_atlas_cga_v2_backward_compatibility_old_trace_format(tmp_path):
    trace = {"requests": ["A", "B", "C", "A"], "predictions": [3, 4, 10, 99]}
    path = tmp_path / "legacy_trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    requests, pages = load_trace(str(path))
    result = run_policy(AtlasCGAV2Policy(), requests, pages, capacity=2)
    assert result.total_misses >= 0


def test_atlas_cga_v2_diagnostics_presence(tmp_path):
    requests, pages = _trace()
    result = run_policy(AtlasCGAV2Policy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))
    diag_path = tmp_path / "atlas_cga_v2_diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "decision_log" in payload
    assert "time_series" in payload


def test_atlas_cga_v1_v3_non_regression_smoke():
    requests, pages = _trace()
    r1 = run_policy(AtlasCGAV1Policy(), requests, pages, capacity=2)
    r2 = run_policy(AtlasV3Policy(), requests, pages, capacity=2)
    assert r1.total_misses >= 0
    assert r2.total_misses >= 0
