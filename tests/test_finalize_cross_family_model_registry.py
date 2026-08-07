"""Fail-closed tests for scripts/experiments/finalize_cross_family_model_registry.py.

Follows the same pattern as tests/test_evict_value_v1_cross_family_eval.py:
import the script as a module and exercise its per-fold check function
against small on-disk fixtures under tmp_path, plus the CLI's fail-closed
exit behavior with an incomplete fold set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path("scripts/experiments").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = _SCRIPTS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _SCRIPTS_DIR)
    yield
    if inserted and _SCRIPTS_DIR in sys.path:
        sys.path.remove(_SCRIPTS_DIR)


def _import_module():
    import finalize_cross_family_model_registry as m
    return m


def _write_fold(tmp_path: Path, family: str, validation_family: str, training_families: list) -> None:
    folds_dir = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    fold = {
        "fold_id": f"cross_family_v1_{family}",
        "test_family": family,
        "validation_family": validation_family,
        "training_families": training_families,
        "model_output_path": f"models/evict_value_v1_cross_family_v1_{family}.pkl",
        "dataset_output_root": f"data/derived/evict_value_v1_cross_family_v1/{family}/",
    }
    (folds_dir / f"{family}.json").write_text(json.dumps(fold))


def _write_protocol_config(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "reviewer_fairness_cross_family_v1.json").write_text(json.dumps({"protocol_id": "reviewer_fair_cross_family_v1"}))


def _write_manifest(tmp_path: Path, family: str, input_families) -> None:
    manifest_dir = tmp_path / "data" / "derived" / "evict_value_v1_cross_family_v1" / family
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"preflight": {"trace_stats": [{"trace_family": f} for f in input_families]}}
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest))


def _write_model_and_metrics(tmp_path: Path, family: str, selected="hist_gb", winner="hist_gb") -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"evict_value_v1_cross_family_v1_{family}.pkl").write_bytes(b"fake model bytes")

    metrics_dir = tmp_path / "analysis" / "reviewer_fairness_cross_family_v1" / family
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "train_metrics.json").write_text(json.dumps({"best_overall": {"model": winner, "val_mean_regret": 0.01}}))
    (metrics_dir / "best_config.json").write_text(json.dumps({"model": selected}))


TRAIN5 = ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]


def test_missing_dataset_manifest_raises(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    with pytest.raises(FileNotFoundError, match="Stage-1 dataset manifest"):
        m._check_fold("brightkite", "deadbeef")


def test_training_family_overlap_rejected(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    bad_training = ["citibike", "metacdn", "metakv", "twemcache", "wiki2018"]  # overlaps validation_family
    _write_fold(tmp_path, "brightkite", "citibike", bad_training)
    with pytest.raises(ValueError, match="overlaps held-out/validation"):
        m._check_fold("brightkite", "deadbeef")


def test_held_out_family_leaked_into_own_manifest_rejected(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    _write_manifest(tmp_path, "brightkite", input_families=["brightkite"] + TRAIN5)
    with pytest.raises(ValueError, match="fold isolation failed upstream"):
        m._check_fold("brightkite", "deadbeef")


def test_missing_final_model_raises(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5)
    with pytest.raises(FileNotFoundError, match="promoted final model not found"):
        m._check_fold("brightkite", "deadbeef")


def test_selected_model_mismatch_with_validation_winner_rejected(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5)
    _write_model_and_metrics(tmp_path, "brightkite", selected="ridge", winner="hist_gb")
    with pytest.raises(ValueError, match="does not match the validation-selected winner"):
        m._check_fold("brightkite", "deadbeef")


def test_fully_valid_fold_passes(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5)
    _write_model_and_metrics(tmp_path, "brightkite", selected="hist_gb", winner="hist_gb")
    record = m._check_fold("brightkite", "deadbeef")
    assert record["held_out_family"] == "brightkite"
    assert record["model_artifact_sha256"]
    assert record["protocol_config_sha256"] == "deadbeef"


def test_main_fails_closed_with_incomplete_folds_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_protocol_config(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5)
    _write_model_and_metrics(tmp_path, "brightkite", selected="hist_gb", winner="hist_gb")
    # citibike fold intentionally left incomplete (no manifest/model at all).
    _write_fold(tmp_path, "citibike", "cloudphysics", ["brightkite", "metacdn", "metakv", "twemcache", "wiki2018"])

    m = _import_module()
    monkeypatch.setattr(sys, "argv", ["finalize_cross_family_model_registry.py", "--families", "brightkite,citibike"])
    with pytest.raises(SystemExit) as exc_info:
        m.main()
    assert exc_info.value.code == 1
    out_path = tmp_path / "analysis" / "reviewer_fairness_cross_family_v1" / "model_registry.json"
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.out


def test_main_dry_run_does_not_write_even_when_complete(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_protocol_config(tmp_path)
    _write_fold(tmp_path, "brightkite", "citibike", TRAIN5)
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5)
    _write_model_and_metrics(tmp_path, "brightkite", selected="hist_gb", winner="hist_gb")

    m = _import_module()
    monkeypatch.setattr(sys, "argv", ["finalize_cross_family_model_registry.py", "--families", "brightkite", "--dry-run"])
    m.main()
    out_path = tmp_path / "analysis" / "reviewer_fairness_cross_family_v1" / "model_registry.json"
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert "FROZEN=True" in captured.out
