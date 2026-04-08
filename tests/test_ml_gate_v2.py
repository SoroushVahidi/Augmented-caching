from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from lafc.learned_gate.dataset_v2 import GateDatasetV2Config, build_gate_examples_v2
from lafc.learned_gate.features_v2 import ML_GATE_V2_FEATURE_COLUMNS
from lafc.learned_gate.model_v2 import LearnedGateV2Model
from lafc.policies.ml_gate_v2 import MLGateV2Policy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


def test_v2_dataset_smoke_and_labels():
    reqs, _ = load_trace("data/example_atlas_v1.json")
    rows = build_gate_examples_v2(reqs, capacity=2, trace_name="toy", cfg=GateDatasetV2Config(horizons=(4, 8)))
    assert rows
    assert {4, 8}.issubset({int(r["horizon"]) for r in rows})
    for r in rows[:3]:
        assert "y_reg" in r and "y_cls" in r
        for c in ML_GATE_V2_FEATURE_COLUMNS:
            assert c in r


def test_v2_label_determinism():
    reqs, _ = load_trace("data/example_atlas_v1.json")
    cfg = GateDatasetV2Config(horizons=(4, 8, 16), margin=0.0)
    r1 = build_gate_examples_v2(reqs, capacity=3, trace_name="toy", cfg=cfg)
    r2 = build_gate_examples_v2(reqs, capacity=3, trace_name="toy", cfg=cfg)
    assert r1 == r2


def test_v2_model_training_and_policy_inference(tmp_path: Path):
    reqs, pages = load_trace("data/example_atlas_v1.json")
    rows = [r for r in build_gate_examples_v2(reqs, capacity=2, trace_name="toy", cfg=GateDatasetV2Config(horizons=(8,))) if int(r["horizon"]) == 8]
    x = [[float(r[c]) for c in ML_GATE_V2_FEATURE_COLUMNS] for r in rows]
    y = [int(r["y_cls"]) for r in rows]

    est = RandomForestClassifier(n_estimators=20, random_state=7)
    est.fit(x, y)
    artifact = LearnedGateV2Model(model_name="rf_test", estimator=est, feature_columns=list(ML_GATE_V2_FEATURE_COLUMNS), threshold=0.5)
    path = tmp_path / "ml_gate_v2_rf.pkl"
    artifact.save(path)

    pol = MLGateV2Policy(model_path=str(path))
    res = run_policy(pol, reqs, pages, capacity=2)
    assert res.total_misses >= 0
    assert (res.extra_diagnostics or {}).get("ml_gate_v2") is not None
