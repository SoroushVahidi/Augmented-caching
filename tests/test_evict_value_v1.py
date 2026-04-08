from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from lafc.evict_value_dataset_v1 import EvictValueDatasetV1Config, build_evict_value_examples_v1
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


def test_evict_value_v1_dataset_smoke():
    reqs, _ = load_trace("data/example_atlas_v1.json")
    rows = build_evict_value_examples_v1(reqs, capacity=2, trace_name="toy", cfg=EvictValueDatasetV1Config(horizons=(4, 8, 16)))
    assert rows
    assert {4, 8, 16}.issubset({int(r["horizon"]) for r in rows})
    sample = rows[0]
    assert "decision_id" in sample and "candidate_page_id" in sample
    assert "y_loss" in sample and "y_value" in sample
    for c in EVICT_VALUE_V1_FEATURE_COLUMNS:
        assert c in sample


def test_evict_value_v1_feature_determinism():
    candidates = ["A", "B", "C"]
    kwargs = dict(
        request_bucket=1,
        request_confidence=0.8,
        candidates=candidates,
        candidate="B",
        bucket_by_page={"A": 0, "B": 2, "C": 1},
        confidence_by_page={"A": 0.3, "B": 0.8, "C": 0.6},
        recent_request_rate=0.25,
        recent_hit_rate=0.2,
    )
    f1 = compute_candidate_features_v1(**kwargs).as_dict()
    f2 = compute_candidate_features_v1(**kwargs).as_dict()
    assert f1 == f2


def test_evict_value_v1_training_smoke_and_reproducibility(tmp_path: Path):
    out_dir = tmp_path / "derived"
    analysis_dir = tmp_path / "analysis"
    model_dir = tmp_path / "models"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_value_dataset_v1.py",
            "--sample-only",
            "--capacities",
            "2",
            "--horizons",
            "8",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )
    train_path = out_dir / "evict_value_v1_train.csv"
    assert train_path.exists()

    cmd = [
        sys.executable,
        "scripts/train_evict_value_v1.py",
        "--data-dir",
        str(out_dir),
        "--horizon",
        "8",
        "--models-dir",
        str(model_dir),
        "--metrics-json",
        str(analysis_dir / "m1.json"),
        "--comparison-csv",
        str(analysis_dir / "c1.csv"),
    ]
    subprocess.run(cmd, check=True)
    cmd2 = list(cmd)
    cmd2[cmd2.index("--comparison-csv") + 1] = str(analysis_dir / "c2.csv")
    subprocess.run(cmd2, check=True)

    c1 = list(csv.DictReader((analysis_dir / "c1.csv").open("r", encoding="utf-8")))
    c2 = list(csv.DictReader((analysis_dir / "c2.csv").open("r", encoding="utf-8")))
    assert c1 == c2


def test_evict_value_v1_policy_smoke_and_diagnostics(tmp_path: Path):
    reqs, pages = load_trace("data/example_atlas_v1.json")
    rows = [
        r
        for r in build_evict_value_examples_v1(reqs, capacity=2, trace_name="toy", cfg=EvictValueDatasetV1Config(horizons=(8,)))
        if int(r["horizon"]) == 8
    ]
    x = [[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows]
    y = [float(r["y_loss"]) for r in rows]

    from sklearn.ensemble import RandomForestRegressor

    est = RandomForestRegressor(n_estimators=25, random_state=7)
    est.fit(x, y)
    artifact = EvictValueV1Model(model_name="rf_test", estimator=est, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS))
    model_path = tmp_path / "evict_value.pkl"
    artifact.save(model_path)

    policy = EvictValueV1Policy(model_path=str(model_path))
    result = run_policy(policy, reqs, pages, capacity=2)
    assert result.total_misses >= 0

    summary = policy.diagnostics_summary()
    assert summary["model_name"] == "rf_test"
    assert summary["evictions"] >= 0


def test_evict_value_v1_backward_compatibility_with_existing_policies():
    reqs, pages = load_trace("data/example_unweighted.json")
    from lafc.policies.rest_v1 import RestV1Policy

    result = run_policy(RestV1Policy(), reqs, pages, capacity=3)
    assert result.total_misses >= 0
