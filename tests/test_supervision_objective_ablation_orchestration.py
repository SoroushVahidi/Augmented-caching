"""Fail-closed tests for the supervision-objective-ablation orchestration:
scripts/build_supervision_objective_ablation_dataset.py,
scripts/build_supervision_objective_ablation_registry.py, and
scripts/experiments/run_supervision_objective_ablation.py.

Follows the same pattern as tests/test_evict_value_v1_cross_family_eval.py:
import the runner scripts as modules and unit-test their fail-closed
helper functions directly against fake/tampered data, rather than running
full simulations (fast, and exercises the actual rejection logic).
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path("scripts").resolve())
_EXPERIMENTS_DIR = str(Path("scripts/experiments").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = []
    for d in (_SCRIPTS_DIR, _EXPERIMENTS_DIR):
        if d not in sys.path:
            sys.path.insert(0, d)
            inserted.append(d)
    yield
    for d in inserted:
        if d in sys.path:
            sys.path.remove(d)


def _import_eval_runner():
    import run_supervision_objective_ablation as m
    return m


def _import_registry_builder():
    import build_supervision_objective_ablation_registry as m
    return m


def _import_dataset_builder():
    import build_supervision_objective_ablation_dataset as m
    return m


# ---------------------------------------------------------------------
# Eval runner: registry loading / freeze gate
# ---------------------------------------------------------------------

def test_unfrozen_registry_rejected(tmp_path):
    m = _import_eval_runner()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"MODEL_SELECTION_FROZEN": False, "missing_models": ["x/y"]}))
    with pytest.raises(m.WrongFoldError, match="MODEL_SELECTION_FROZEN"):
        m._load_registry(registry_path)


def test_missing_registry_file_rejected(tmp_path):
    m = _import_eval_runner()
    with pytest.raises(FileNotFoundError):
        m._load_registry(tmp_path / "does_not_exist.json")


def test_incomplete_registry_treated_as_unfrozen(tmp_path):
    m = _import_eval_runner()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"MODEL_SELECTION_FROZEN": False, "actual_model_count": 27, "expected_model_count": 28}))
    with pytest.raises(m.WrongFoldError):
        m._load_registry(registry_path)


# ---------------------------------------------------------------------
# Eval runner: record lookup / wrong-fold protection
# ---------------------------------------------------------------------

def test_missing_record_for_objective_family_rejected():
    m = _import_eval_runner()
    registry = {"records": [{"objective": "objective_eviction_loss", "held_out_family": "brightkite"}]}
    with pytest.raises(m.WrongFoldError, match="No registry record"):
        m._find_record(registry, "objective_pairwise", "brightkite")


def test_wrong_fold_naming_mismatch_rejected(tmp_path):
    m = _import_eval_runner()
    # A record whose artifact path names a DIFFERENT family than the one
    # being evaluated must never be silently accepted.
    record = {
        "objective": "objective_eviction_loss",
        "model_artifact_path": "models/supervision_objective_ablation_v1/objective_eviction_loss/citibike.pkl",
        "model_artifact_sha256": "irrelevant",
    }
    with pytest.raises(m.WrongFoldError, match="expected naming convention"):
        m._verify_model(record, family="brightkite", objective="objective_eviction_loss")


def test_wrong_objective_directory_mismatch_rejected():
    m = _import_eval_runner()
    record = {
        "objective": "objective_eviction_loss",
        "model_artifact_path": "models/supervision_objective_ablation_v1/objective_next_arrival/brightkite.pkl",
        "model_artifact_sha256": "irrelevant",
    }
    with pytest.raises(m.WrongFoldError, match="expected naming convention"):
        m._verify_model(record, family="brightkite", objective="objective_eviction_loss")


def test_missing_model_file_raises(tmp_path, monkeypatch):
    m = _import_eval_runner()
    monkeypatch.chdir(tmp_path)
    record = {
        "objective": "objective_eviction_loss",
        "model_artifact_path": "models/supervision_objective_ablation_v1/objective_eviction_loss/brightkite.pkl",
        "model_artifact_sha256": "irrelevant",
    }
    with pytest.raises(FileNotFoundError):
        m._verify_model(record, family="brightkite", objective="objective_eviction_loss")


def test_model_hash_mismatch_rejected(tmp_path, monkeypatch):
    m = _import_eval_runner()
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "models" / "supervision_objective_ablation_v1" / "objective_eviction_loss" / "brightkite.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"some model bytes")
    record = {
        "objective": "objective_eviction_loss",
        "model_artifact_path": str(model_path.relative_to(tmp_path)),
        "model_artifact_sha256": "0" * 64,  # deliberately wrong
    }
    with pytest.raises(m.WrongFoldError, match="hash mismatch"):
        m._verify_model(record, family="brightkite", objective="objective_eviction_loss")


def test_model_hash_matching_passes(tmp_path, monkeypatch):
    m = _import_eval_runner()
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "models" / "supervision_objective_ablation_v1" / "objective_eviction_loss" / "brightkite.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"some model bytes")
    from lafc.experiments.external_baseline_common import sha256_of_file
    record = {
        "objective": "objective_eviction_loss",
        "model_artifact_path": str(model_path.relative_to(tmp_path)),
        "model_artifact_sha256": sha256_of_file(model_path),
    }
    result = m._verify_model(record, family="brightkite", objective="objective_eviction_loss")
    assert result == Path(record["model_artifact_path"])


def test_incremental_csv_writer_key_fields_prevent_duplicate_rows(tmp_path):
    from lafc.experiments.external_baseline_common import IncrementalCsvWriter

    m = _import_eval_runner()
    out = tmp_path / "policy_comparison.csv"
    writer = IncrementalCsvWriter(out, m.FIELDNAMES, m.KEY_FIELDS)
    row = {f: "" for f in m.FIELDNAMES}
    row.update({"objective": "objective_eviction_loss", "held_out_family": "brightkite", "capacity": "32"})
    key = {"objective": "objective_eviction_loss", "held_out_family": "brightkite", "capacity": 32}
    assert not writer.already_done(key)
    writer.write_row(row)
    writer.close()

    writer2 = IncrementalCsvWriter(out, m.FIELDNAMES, m.KEY_FIELDS)
    assert writer2.already_done(key)  # resume must recognize the existing row
    writer2.close()


def test_frozen_history_and_score_windows_match_protocol():
    m = _import_eval_runner()
    assert m.HISTORY_START == 0
    assert m.HISTORY_END == 10000
    assert m.SCORE_START == 10000
    assert m.SCORE_END == 50000


def test_expected_84_row_key_space_has_no_duplicates():
    m = _import_eval_runner()
    caps = [32, 64, 128]
    keys = set()
    for family in m.FAMILIES:
        for objective in m.ALL_OBJECTIVES:
            for cap in caps:
                keys.add((objective, family, cap))
    assert len(keys) == 84
    assert len(m.FAMILIES) == 7
    assert len(m.ALL_OBJECTIVES) == 4


# ---------------------------------------------------------------------
# Registry builder: hash mismatch / incomplete gating
# ---------------------------------------------------------------------

def test_registry_builder_refuses_frozen_write_when_models_missing(tmp_path, monkeypatch, capsys):
    m = _import_registry_builder()
    monkeypatch.chdir(tmp_path)
    # No models directory / metrics at all -> everything missing.
    (tmp_path / "configs" / "fair_cross_family_v1" / "folds").mkdir(parents=True)
    for fam in m.FAMILIES:
        fold = {
            "fold_id": f"cross_family_v1_{fam}",
            "test_family": fam,
            "training_families": [f for f in m.FAMILIES if f != fam][:5],
            "validation_family": [f for f in m.FAMILIES if f != fam][5] if len([f for f in m.FAMILIES if f != fam]) > 5 else "x",
        }
        (tmp_path / "configs" / "fair_cross_family_v1" / "folds" / f"{fam}.json").write_text(json.dumps(fold))

    sys.argv = ["prog", "--models-dir", str(tmp_path / "models"), "--metrics-dir", str(tmp_path / "metrics"),
                "--out", str(tmp_path / "registry.json")]
    with pytest.raises(SystemExit):
        m.main()
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert not (tmp_path / "registry.json").exists()


def test_registry_builder_detects_tampered_model_artifact(tmp_path, monkeypatch):
    m = _import_registry_builder()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "fair_cross_family_v1" / "folds").mkdir(parents=True)
    fold = {
        "fold_id": "cross_family_v1_brightkite", "test_family": "brightkite",
        "training_families": ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"],
        "validation_family": "citibike",
    }
    (tmp_path / "configs" / "fair_cross_family_v1" / "folds" / "brightkite.json").write_text(json.dumps(fold))

    model_path = tmp_path / "models" / "objective_eviction_loss" / "brightkite.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"original bytes")

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "brightkite.json").write_text(json.dumps({
        "objectives": {"objective_eviction_loss": {"best_model_name": "ridge", "model_sha256": "stale-hash-from-training-time"}}
    }))

    sys.argv = ["prog", "--models-dir", str(tmp_path / "models"), "--metrics-dir", str(metrics_dir),
                "--out", str(tmp_path / "registry.json"), "--families", "brightkite", "--objectives", "objective_eviction_loss"]
    with pytest.raises(ValueError, match="hash mismatch"):
        m.main()


# ---------------------------------------------------------------------
# Dataset builder: held-out-family isolation assertions
# ---------------------------------------------------------------------

def test_dataset_builder_rejects_held_out_family_in_split_map(tmp_path, monkeypatch):
    m = _import_dataset_builder()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "FAIRNESS_WORKTREE_FOLDS_DIR", tmp_path / "nonexistent_ref_dir")
    folds_dir = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds_dir.mkdir(parents=True)
    fold = {
        "fold_id": "cross_family_v1_brightkite", "test_family": "brightkite",
        "training_families": ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"],
        "validation_family": "citibike",
        "train_manifest": str(folds_dir / "brightkite_train_manifest.csv"),
    }
    (folds_dir / "brightkite.json").write_text(json.dumps(fold))
    # BUG: split map wrongly includes the held-out family itself.
    bad_map = {"citibike": "val", "cloudphysics": "train", "metacdn": "train",
               "metakv": "train", "twemcache": "train", "wiki2018": "train", "brightkite": "train"}
    (folds_dir / "brightkite_family_split_map.json").write_text(json.dumps(bad_map))
    with open(folds_dir / "brightkite_train_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "trace_name", "trace_family"])

    sys.argv = ["prog", "--held-out-family", "brightkite", "--data-read-root", str(tmp_path)]
    with pytest.raises(ValueError, match="own family_split_map"):
        m.main()


def test_dataset_builder_rejects_held_out_family_in_train_manifest(tmp_path, monkeypatch):
    m = _import_dataset_builder()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "FAIRNESS_WORKTREE_FOLDS_DIR", tmp_path / "nonexistent_ref_dir")
    folds_dir = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds_dir.mkdir(parents=True)
    fold = {
        "fold_id": "cross_family_v1_brightkite", "test_family": "brightkite",
        "training_families": ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"],
        "validation_family": "citibike",
        "train_manifest": str(folds_dir / "brightkite_train_manifest.csv"),
    }
    (folds_dir / "brightkite.json").write_text(json.dumps(fold))
    good_map = {"citibike": "val", "cloudphysics": "train", "metacdn": "train",
                "metakv": "train", "twemcache": "train", "wiki2018": "train"}
    (folds_dir / "brightkite_family_split_map.json").write_text(json.dumps(good_map))
    # BUG: train manifest leaks a row from the held-out family.
    with open(folds_dir / "brightkite_train_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "trace_name", "trace_family"])
        w.writerow(["data/processed/brightkite/trace.jsonl", "brightkite_50k", "brightkite"])

    sys.argv = ["prog", "--held-out-family", "brightkite", "--data-read-root", str(tmp_path)]
    with pytest.raises(ValueError, match="own train manifest"):
        m.main()


def test_dataset_builder_rejects_wrong_horizon(tmp_path, monkeypatch):
    m = _import_dataset_builder()
    monkeypatch.chdir(tmp_path)
    sys.argv = ["prog", "--held-out-family", "brightkite", "--horizon", "8"]
    with pytest.raises(ValueError, match="frozen protocol"):
        m.main()


def test_dataset_builder_fold_manifest_matches_fairness_worktree_copy():
    m = _import_dataset_builder()
    # Real committed fold (not tampered) must pass the byte-identity check
    # against the fairness worktree's copy (both exist in this environment).
    fold = m._verify_fold_identical_to_fairness_worktree("brightkite")
    assert fold["test_family"] == "brightkite"
