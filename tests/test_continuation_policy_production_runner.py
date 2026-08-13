from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyRegressor

from lafc.continuation_policy_ablation import (
    ContinuationAblationConfig,
    FrozenPi1Provenance,
    _registry_self_hash,
    build_decision_aligned_continuation_rows,
)
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.simulator.request_trace import build_requests_from_lists
from scripts.experiments import run_continuation_policy_causal_ablation as runner


def _write_trace(path: Path, family: str, page_ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for page_id in page_ids:
        rows.append(
            {
                "item_id": str(page_id),
                "source_dataset": family,
                "metadata": {"bucket": 0, "confidence": 0.5},
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_model(path: Path) -> str:
    reqs, _pages = build_requests_from_lists(page_ids=["a", "b", "c", "d"])
    x = [[0.0 for _ in EVICT_VALUE_V1_FEATURE_COLUMNS] for _ in reqs]
    y = [0.0 for _ in reqs]
    est = DummyRegressor(strategy="constant", constant=0.0)
    est.fit(x, y)
    model = EvictValueV1Model(
        model_name="objective_eviction_loss_dummy",
        estimator=est,
        feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS),
    )
    model.save(path)
    return runner.sha256_of_file(path)


def _write_registry(tmp_path: Path, model_path: Path, model_sha: str) -> Path:
    registry = {
        "MODEL_SELECTION_FROZEN": True,
        "records": [
            {
                "objective": "objective_eviction_loss",
                "held_out_family": "heldout",
                "fold_id": "cross_family_v1_heldout",
                "training_families": ["trainfam"],
                "validation_family": "valfam",
                "protocol_id": "supervision_objective_ablation_v1",
                "model_artifact_path": str(model_path),
                "model_artifact_sha256": model_sha,
            }
        ],
    }
    registry["registry_sha256"] = _registry_self_hash(registry)
    path = tmp_path / "analysis" / "supervision_objective_ablation_v1" / "model_registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    folds = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds.mkdir(parents=True)
    data = tmp_path / "data"
    seq = ["a", "b", "c", "a", "d", "b", "e", "c", "f", "d", "g", "e", "h", "f"]
    _write_trace(data / "processed" / "heldout" / "trace.jsonl", "heldout", seq)
    _write_trace(data / "processed" / "trainfam" / "trace.jsonl", "trainfam", seq)
    _write_trace(data / "processed" / "valfam" / "trace.jsonl", "valfam", seq)
    fold = {
        "fold_id": "cross_family_v1_heldout",
        "test_family": "heldout",
        "test_trace_name": "heldout_trace",
        "test_trace_path": "processed/heldout/trace.jsonl",
        "history": [0, 0],
        "score": [0, 12],
        "validation_family": "valfam",
        "training_families": ["trainfam"],
        "train_manifest": str(folds / "heldout_train_manifest.csv"),
        "family_split_map": str(folds / "heldout_family_split_map.json"),
    }
    (folds / "heldout.json").write_text(json.dumps(fold), encoding="utf-8")
    (folds / "heldout_family_split_map.json").write_text(
        json.dumps({"trainfam": "train", "valfam": "val"}),
        encoding="utf-8",
    )
    (folds / "heldout_train_manifest.csv").write_text(
        "path,trace_name,trace_family\n"
        "processed/trainfam/trace.jsonl,train_trace,trainfam\n"
        "processed/valfam/trace.jsonl,val_trace,valfam\n",
        encoding="utf-8",
    )

    model_path = tmp_path / "models" / "supervision_objective_ablation_v1" / "objective_eviction_loss" / "heldout.pkl"
    model_sha = _write_model(model_path)
    registry = _write_registry(tmp_path, model_path, model_sha)
    config = {
        "protocol_id": "continuation_policy_causal_ablation_production_v1",
        "version": 1,
        "status": "TEST",
        "conditions": {
            "C0_BASELINE_LRU": {},
            "C1_LRU_CONTINUATION_LEARNED_PI1": {},
            "C2_PI1_CONTINUATION_LEARNED_PI2": {},
        },
        "folds": {"fold_dir": str(folds), "held_out_families": ["heldout"]},
        "capacities": [2],
        "horizon": 1,
        "feature_schema": "EVICT_VALUE_V1_FEATURE_COLUMNS",
        "seed": 0,
        "training_budget": {"max_train_rows": 20, "max_val_rows": 10},
        "evaluation_window": {"history_start": 0, "history_end": 0, "score_start": 0, "score_end": 12},
        "frozen_pi1_provenance": {
            "registry": str(registry),
            "required_objective": "objective_eviction_loss",
            "required_registry_flag": "MODEL_SELECTION_FROZEN=true",
        },
        "output": {
            "analysis_root": str(tmp_path / "analysis" / "continuation_policy_causal_ablation_production_v1"),
            "model_root": str(tmp_path / "models" / "continuation_policy_causal_ablation_production_v1"),
        },
    }
    config_path = tmp_path / "configs" / "continuation_policy_causal_ablation_production_v1.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, data, Path(config["output"]["analysis_root"]), Path(config["output"]["model_root"])


def _args(config_path: Path, data: Path, out: Path, models: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(config_path),
        "--families",
        "heldout",
        "--capacities",
        "2",
        "--data-read-root",
        str(data),
        "--output-root",
        str(out),
        "--model-root",
        str(models),
        "--max-train-decisions",
        "2",
        "--max-val-decisions",
        "1",
        "--score-start",
        "0",
        "--score-end",
        "12",
        *extra,
    ]


def test_production_config_parsing_and_preflight_only(tmp_path: Path):
    config_path, data, out, models = _fixture(tmp_path)

    rc = runner.main(_args(config_path, data, out, models, "--preflight-only"))

    assert rc == 0
    assert not (out / "policy_comparison.csv").exists()


def test_tiny_production_smoke_outputs_c0_c1_c2_and_artifacts(tmp_path: Path):
    config_path, data, out, models = _fixture(tmp_path)

    rc = runner.main(_args(config_path, data, out, models))

    assert rc == 0
    policies = list(csv_row["condition"] for csv_row in runner._read_csv(out / "policy_comparison.csv"))
    assert policies == [
        "C0_BASELINE_LRU",
        "C1_LRU_CONTINUATION_LEARNED_PI1",
        "C2_PI1_CONTINUATION_LEARNED_PI2",
    ]
    assert len(runner._read_csv(out / "label_agreement.csv")) == 1
    training = runner._read_csv(out / "training_summary.csv")
    assert len(training) == 1
    assert training[0]["best_model_name"] in {"ridge", "random_forest", "hist_gb"}
    manifest = json.loads((out / "unit_completion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed_units"] == 1
    assert (models / "heldout" / "cap2" / "pi2.pkl").exists()


def test_resume_skips_completed_unit_without_duplicate_rows(tmp_path: Path):
    config_path, data, out, models = _fixture(tmp_path)
    assert runner.main(_args(config_path, data, out, models)) == 0

    rc = runner.main(_args(config_path, data, out, models, "--resume"))

    assert rc == 0
    assert len(runner._read_csv(out / "policy_comparison.csv")) == 3
    assert len(runner._read_csv(out / "label_agreement.csv")) == 1
    assert len(runner._read_csv(out / "training_summary.csv")) == 1


def test_partial_temp_artifact_is_not_treated_as_complete(tmp_path: Path):
    config_path, data, out, models = _fixture(tmp_path)
    config = runner._load_json(config_path)
    runner._atomic_write_json(out / "config_snapshot.json", config)
    temp_unit = out / "units" / ".heldout_cap2.tmp.fake"
    temp_unit.mkdir(parents=True)
    (temp_unit / "unit_summary.json").write_text("{}", encoding="utf-8")

    rc = runner.main(_args(config_path, data, out, models, "--resume"))

    assert rc == 0
    assert (out / "units" / "heldout_cap2" / "unit_summary.json").exists()
    assert temp_unit.exists()


def test_incompatible_existing_output_fails_closed(tmp_path: Path):
    config_path, data, out, models = _fixture(tmp_path)
    assert runner.main(_args(config_path, data, out, models)) == 0
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["horizon"] = 2
    bad_config = tmp_path / "configs" / "bad_config.json"
    bad_config.write_text(json.dumps(cfg), encoding="utf-8")

    assert runner.main(_args(bad_config, data, out, models, "--resume")) == 2


def test_resume_reuses_frozen_source_identity_without_changing_protocol(tmp_path: Path):
    config = runner._load_json(Path("configs/continuation_policy_causal_ablation_production_v1.json"))
    repo_root = runner._repo_root()
    source_sha = runner._git_sha(repo_root)
    paths = runner.Paths(
        repo_root=repo_root,
        config_path=repo_root / "configs/continuation_policy_causal_ablation_production_v1.json",
        output_root=tmp_path / "output",
        model_root=tmp_path / "models",
        data_read_root=tmp_path / "data",
        fold_dir=tmp_path / "folds",
        registry_path=tmp_path / "registry.json",
    )
    paths.output_root.mkdir()
    snapshot = dict(config)
    snapshot["source_sha_at_runner_start"] = source_sha
    runner._atomic_write_json(paths.output_root / "config_snapshot.json", snapshot)

    assert runner._source_sha_for_run(paths, config, resume=True) == source_sha


def test_same_example_alignment_gate_rejects_duplicate_candidate_key():
    reqs, _pages = build_requests_from_lists(page_ids=["a", "b", "c", "a", "d", "b"])
    rows = build_decision_aligned_continuation_rows(
        requests=reqs,
        capacity=2,
        trace_name="toy",
        trace_family="trainfam",
        cfg=ContinuationAblationConfig(horizon=1),
        pi1_model=type("M", (), {"predict_loss_batch": lambda self, rows: [0.0 for _ in rows]})(),
        pi1_provenance=FrozenPi1Provenance(
            held_out_family="heldout",
            validation_family="valfam",
            training_families=("trainfam",),
            model_path="m.pkl",
            model_sha256="a" * 64,
            registry_path="r.json",
            registry_sha256="b" * 64,
        ),
    )
    rows.append(dict(rows[0]))

    with pytest.raises(runner.ProtocolError, match="duplicate"):
        runner._validate_same_example_gate(rows, unit=runner.Unit("heldout", 2))


def test_model_identity_gate_rejects_wrong_pi1_hash(tmp_path: Path):
    config_path, data, out, models = _fixture(tmp_path)
    config = runner._load_json(config_path)
    paths = runner._make_paths(
        config,
        runner.build_arg_parser().parse_args(_args(config_path, data, out, models)),
        runner._repo_root(),
    )
    reqs, _pages = build_requests_from_lists(page_ids=["a", "b", "c", "a", "d", "b"])
    prov = FrozenPi1Provenance(
        held_out_family="heldout",
        validation_family="valfam",
        training_families=("trainfam",),
        model_path="m.pkl",
        model_sha256="a" * 64,
        registry_path="r.json",
        registry_sha256="b" * 64,
    )
    rows = build_decision_aligned_continuation_rows(
        requests=reqs,
        capacity=2,
        trace_name="toy",
        trace_family="trainfam",
        cfg=ContinuationAblationConfig(horizon=1),
        pi1_model=type("M", (), {"predict_loss_batch": lambda self, rows: [0.0 for _ in rows]})(),
        pi1_provenance=prov,
    )
    rows[0]["pi1_hash"] = "bad"

    with pytest.raises(runner.ProtocolError, match="wrong pi1 hash"):
        runner._validate_leakage_gate(
            paths=paths,
            unit=runner.Unit("heldout", 2),
            train_rows=rows,
            val_rows=[],
            pi1_provenance=prov,
            cfg=ContinuationAblationConfig(horizon=1),
            score_start=0,
            score_end=12,
        )
