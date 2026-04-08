from __future__ import annotations

from pathlib import Path

from lafc.learned_gate.dataset import GateDatasetConfig, build_gate_examples
from lafc.learned_gate.features import ML_GATE_FEATURE_COLUMNS
from lafc.learned_gate.model import LearnedGateModel
from lafc.policies.ml_gate_v1 import MLGateV1Policy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


def test_dataset_extraction_smoke():
    requests, _ = load_trace("data/example_atlas_v1.json")
    rows = build_gate_examples(requests, capacity=2, cfg=GateDatasetConfig(horizon=3), trace_name="toy")
    assert rows
    assert "y" in rows[0]
    for c in ML_GATE_FEATURE_COLUMNS:
        assert c in rows[0]


def test_feature_generation_determinism():
    requests, _ = load_trace("data/example_atlas_v1.json")
    cfg = GateDatasetConfig(horizon=3)
    r1 = build_gate_examples(requests, capacity=3, cfg=cfg, trace_name="toy")
    r2 = build_gate_examples(requests, capacity=3, cfg=cfg, trace_name="toy")
    assert r1 == r2


def test_model_training_smoke(tmp_path: Path):
    requests, _ = load_trace("data/example_atlas_v1.json")
    rows = build_gate_examples(requests, capacity=2, cfg=GateDatasetConfig(horizon=3), trace_name="toy")
    x = [{c: float(r[c]) for c in ML_GATE_FEATURE_COLUMNS} for r in rows]
    y = [int(r["y"]) for r in rows]
    model = LearnedGateModel.new_logistic(random_state=7)
    model.fit(x, y)
    model_path = tmp_path / "m.pkl"
    model.save(model_path)
    reloaded = LearnedGateModel.load(model_path)
    assert reloaded.predict_one(x[0]) in {0, 1}


def test_policy_inference_smoke(tmp_path: Path):
    requests, pages = load_trace("data/example_atlas_v1.json")
    rows = build_gate_examples(requests, capacity=2, cfg=GateDatasetConfig(horizon=3), trace_name="toy")
    x = [{c: float(r[c]) for c in ML_GATE_FEATURE_COLUMNS} for r in rows]
    y = [int(r["y"]) for r in rows]
    model = LearnedGateModel.new_logistic(random_state=7)
    model.fit(x, y)
    model_path = tmp_path / "ml_gate_v1.pkl"
    model.save(model_path)

    pol = MLGateV1Policy(model_path=str(model_path))
    res = run_policy(pol, requests, pages, capacity=2)
    assert res.total_misses >= 0
    assert (res.extra_diagnostics or {}).get("ml_gate_v1") is not None


def test_reproducible_predictions_under_seed(tmp_path: Path):
    requests, _ = load_trace("data/example_atlas_v1.json")
    rows = build_gate_examples(requests, capacity=2, cfg=GateDatasetConfig(horizon=3), trace_name="toy")
    x = [{c: float(r[c]) for c in ML_GATE_FEATURE_COLUMNS} for r in rows]
    y = [int(r["y"]) for r in rows]
    m1 = LearnedGateModel.new_logistic(random_state=7)
    m2 = LearnedGateModel.new_logistic(random_state=7)
    m1.fit(x, y)
    m2.fit(x, y)
    p1 = [m1.predict_one(r) for r in x]
    p2 = [m2.predict_one(r) for r in x]
    assert p1 == p2
