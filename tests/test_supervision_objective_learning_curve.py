from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_EXPERIMENTS_DIR = str((Path("scripts") / "experiments").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = _EXPERIMENTS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _EXPERIMENTS_DIR)
    yield
    if inserted and _EXPERIMENTS_DIR in sys.path:
        sys.path.remove(_EXPERIMENTS_DIR)


def _import_module():
    import run_supervision_objective_learning_curve as m

    return m


def _base_config(tmp_path: Path) -> dict:
    return {
        "protocol_id": "supervision_objective_learning_curve_v1",
        "source_protocol_id": "supervision_objective_ablation_v1",
        "fractions": [0.5, 1.0],
        "held_out_families": ["brightkite"],
        "capacities": [32],
        "seed": 0,
        "horizon": 4,
        "pairwise_max_pairs_per_decision": 6,
        "pairwise_sample_seed": 0,
        "validation_decision_fraction": 1.0,
        "conditions": {
            "eviction_loss_scalar": {"fixed_model_family_by_fold": {"brightkite": "ridge"}},
            "eviction_loss_pairwise": {},
        },
        "dataset_repo_root": str(tmp_path / "objective_repo"),
        "dataset_root": str(tmp_path / "objective_repo" / "data" / "derived" / "supervision_objective_ablation_v1"),
        "data_read_root": str(tmp_path / "primary_repo"),
        "output_dir": "analysis/supervision_objective_learning_curve_v1",
        "models_dir": "models/supervision_objective_learning_curve_v1",
        "max_wall_hours_default": 1.0,
    }


def _write_trace(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx in range(40):
        lines.append(
            json.dumps(
                {
                    "item_id": f"p{idx}",
                    "source_dataset": "toy",
                    "metadata": {"bucket": idx % 4, "confidence": 0.5},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fold_and_dataset(tmp_path: Path) -> dict:
    cfg = _base_config(tmp_path)
    experiments_root = tmp_path / "repo"
    objective_repo = Path(cfg["dataset_repo_root"])
    primary_repo = Path(cfg["data_read_root"])
    fold_dir = experiments_root / "configs" / "fair_cross_family_v1" / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_trace = primary_repo / "data" / "processed" / "cloudphysics" / "trace.jsonl"
    val_trace = primary_repo / "data" / "processed" / "citibike" / "trace.jsonl"
    test_trace = primary_repo / "data" / "processed" / "brightkite" / "trace.jsonl"
    train_sha = _write_trace(train_trace)
    val_sha = _write_trace(val_trace)
    test_sha = _write_trace(test_trace)

    fold = {
        "fold_id": "cross_family_v1_brightkite",
        "test_family": "brightkite",
        "validation_family": "citibike",
        "training_families": ["cloudphysics"],
        "test_trace_name": "brightkite_trace",
        "test_trace_path": str(test_trace),
        "test_trace_sha256": test_sha,
    }
    (fold_dir / "brightkite.json").write_text(json.dumps(fold), encoding="utf-8")

    data_root = Path(cfg["dataset_root"]) / "brightkite"
    scalar_dir = data_root / "scalar" / "shards"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    shard = scalar_dir / "toy.part0000.csv"
    fieldnames = [
        "example_id",
        "trace_name",
        "trace_family",
        "split",
        "capacity",
        "horizon",
        "decision_id",
        "decision_t",
        "candidate_page_id",
        "eviction_loss_label",
        "next_arrival_label_raw",
        "next_arrival_label_censored",
        "next_arrival_censored_flag",
        "reuse_distance_label_raw",
        "reuse_distance_label_censored",
        "reuse_distance_censored_flag",
        "incoming_bucket_norm",
        "incoming_confidence",
        "candidate_bucket_norm",
        "candidate_confidence",
        "candidate_rank_norm",
        "candidate_count_norm",
        "bucket_gap_norm",
        "confidence_gap",
        "recent_req_rate",
        "recent_hit_rate",
    ]
    with shard.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        rows = [
            ("train", "d1", "A", 1.0),
            ("train", "d1", "B", 3.0),
            ("train", "d2", "A", 2.0),
            ("train", "d2", "C", 4.0),
            ("val", "v1", "A", 1.0),
            ("val", "v1", "B", 2.0),
        ]
        for split, decision_id, candidate, loss in rows:
            writer.writerow(
                {
                    "example_id": f"{decision_id}|{candidate}",
                    "trace_name": "toy_train" if split == "train" else "toy_val",
                    "trace_family": "cloudphysics" if split == "train" else "citibike",
                    "split": split,
                    "capacity": 32,
                    "horizon": 4,
                    "decision_id": decision_id,
                    "decision_t": 1,
                    "candidate_page_id": candidate,
                    "eviction_loss_label": loss,
                    "next_arrival_label_raw": loss + 1,
                    "next_arrival_label_censored": loss + 1,
                    "next_arrival_censored_flag": 0,
                    "reuse_distance_label_raw": loss,
                    "reuse_distance_label_censored": loss,
                    "reuse_distance_censored_flag": 0,
                    "incoming_bucket_norm": 0.0,
                    "incoming_confidence": 0.5,
                    "candidate_bucket_norm": 0.1 if candidate == "A" else 0.2,
                    "candidate_confidence": 0.3 if candidate == "A" else 0.4,
                    "candidate_rank_norm": 0.0,
                    "candidate_count_norm": 1.0,
                    "bucket_gap_norm": 0.0,
                    "confidence_gap": 0.0,
                    "recent_req_rate": 0.0,
                    "recent_hit_rate": 0.0,
                }
            )

    manifest = {
        "format": "supervision_objective_ablation_v1_candidate_csv_shards",
        "protocol_id": "supervision_objective_ablation_v1",
        "held_out_family": "brightkite",
        "fold_id": "cross_family_v1_brightkite",
        "training_families": ["cloudphysics"],
        "validation_family": "citibike",
        "horizon": 4,
        "capacities": [32],
        "trace_stats": [
            {
                "trace_name": "toy_train",
                "trace_family": "cloudphysics",
                "split": "train",
                "path": str(train_trace),
                "trace_sha256": train_sha,
                "request_count": 1,
            },
            {
                "trace_name": "toy_val",
                "trace_family": "citibike",
                "split": "val",
                "path": str(val_trace),
                "trace_sha256": val_sha,
                "request_count": 1,
            },
        ],
        "scalar_shards": [{"path": "data/derived/supervision_objective_ablation_v1/brightkite/scalar/shards/toy.part0000.csv", "row_count": 6}],
        "pairwise_shards": [],
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"config": cfg, "repo_root": experiments_root}


def test_nested_subset_validation_rejects_non_nested():
    m = _import_module()
    with pytest.raises(m.ProtocolBlocked):
        m._validate_nested_subsets({0.5: ("d2",), 1.0: ("d1",)}, [0.5, 1.0])


def test_regret_pairwise_labels_match_sign():
    m = _import_module()
    rows = [
        {
            "decision_id": "d1",
            "trace_name": "toy",
            "trace_family": "toyfam",
            "decision_t": 1,
            "capacity": 32,
            "horizon": 4,
            "candidate_page_id": "A",
            "eviction_loss_label": 1.0,
            **{c: 0.0 for c in m.FEATURES},
        },
        {
            "decision_id": "d1",
            "trace_name": "toy",
            "trace_family": "toyfam",
            "decision_t": 1,
            "capacity": 32,
            "horizon": 4,
            "candidate_page_id": "B",
            "eviction_loss_label": 3.0,
            **{c: 1.0 for c in m.FEATURES},
        },
    ]
    pairs = m.build_pairwise_rows(rows, source="regret")
    m._validate_pairwise_same_target_rows(rows, pairs)
    assert len(pairs) == 1
    assert pairs[0]["label_i_preferred"] in (0, 1)


def test_plan_units_reports_expected_counts(tmp_path, monkeypatch):
    payload = _write_fold_and_dataset(tmp_path)
    cfg = payload["config"]
    repo_root = payload["repo_root"]
    monkeypatch.chdir(repo_root)
    (repo_root / "configs" / "supervision_objective_learning_curve_v1.json").parent.mkdir(parents=True, exist_ok=True)
    units = _import_module().plan_units(
        config=cfg,
        held_out_families=["brightkite"],
        fractions=[0.5, 1.0],
    )
    assert len(units) == 2
    assert units[0]["train_decision_count"] == 4
    assert units[1]["train_decision_count"] == 8


def test_campaign_state_marks_completed_units(tmp_path):
    m = _import_module()
    state_path = tmp_path / "campaign_state.json"
    m._mark_completed_unit(state_path, "brightkite|0.010000", 12.5)
    payload = json.loads(state_path.read_text())
    assert payload["completed_units"] == ["brightkite|0.010000"]
    assert payload["unit_seconds"]["brightkite|0.010000"] == 12.5


def test_incremental_writer_rejects_duplicate_key_via_already_done(tmp_path):
    m = _import_module()
    writer = m.IncrementalCsvWriter(tmp_path / "policy.csv", m.FIELDNAMES, m.KEY_FIELDS)
    row = {field: "" for field in m.FIELDNAMES}
    row.update(
        {
            "condition": "eviction_loss_scalar",
            "fraction": "0.01",
            "held_out_family": "brightkite",
            "capacity": "32",
        }
    )
    key = {
        "condition": "eviction_loss_scalar",
        "fraction": "0.01",
        "held_out_family": "brightkite",
        "capacity": 32,
    }
    assert not writer.already_done(key)
    writer.write_row(row)
    writer.close()

    writer2 = m.IncrementalCsvWriter(tmp_path / "policy.csv", m.FIELDNAMES, m.KEY_FIELDS)
    assert writer2.already_done(key)
    writer2.close()
