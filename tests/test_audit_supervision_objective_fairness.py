"""Fixture tests for scripts/experiments/audit_supervision_objective_fairness.py."""

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
    import audit_supervision_objective_fairness as m
    return m


TRAIN5 = ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]


def _write_fold(tmp_path: Path, family="brightkite", validation="citibike", training=TRAIN5) -> None:
    folds_dir = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    fold = {"fold_id": f"cross_family_v1_{family}", "test_family": family,
            "validation_family": validation, "training_families": training}
    (folds_dir / f"{family}.json").write_text(json.dumps(fold))


def _scalar_obj(best_model="ridge", n_train=150000, n_val=30000, direction="min", tamper_best_model=None):
    rows = [
        {"model": "ridge", "val_mean_regret": 0.004, "val_mae": 1.0, "val_rmse": 1.0},
        {"model": "random_forest", "val_mean_regret": 0.02, "val_mae": 1.5, "val_rmse": 1.5},
        {"model": "hist_gb", "val_mean_regret": 0.01, "val_mae": 1.2, "val_rmse": 1.2},
    ]
    actual_best = min(rows, key=lambda r: (r["val_mean_regret"], r["val_mae"], r["val_rmse"]))["model"]
    return {
        "direction": direction, "comparison_rows": rows,
        "best_model_name": tamper_best_model or actual_best,
        "n_train_rows": n_train, "n_val_rows": n_val,
    }


def _write_metrics(tmp_path: Path, family: str, fold_id: str, objectives: dict) -> None:
    d = tmp_path / "analysis" / "supervision_objective_ablation_v1" / "training"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{family}.json").write_text(json.dumps({
        "held_out_family": family, "fold_id": fold_id, "objectives": objectives,
    }))


def _default_objectives():
    return {
        "objective_eviction_loss": _scalar_obj(),
        "objective_next_arrival": _scalar_obj(),
        "objective_reuse_distance": _scalar_obj(),
        "objective_pairwise": {"label_source": "next_arrival", "n_train_pairs": 5000,
                                "model_path": "x", "model_sha256": "y", "seed": 0},
    }


def test_not_built_reports_not_built(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    report = m.audit_fold("brightkite")
    assert report["status"] == "NOT_BUILT"


def test_clean_fold_passes(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    _write_metrics(tmp_path, "brightkite", "cross_family_v1_brightkite", _default_objectives())
    report = m.audit_fold("brightkite")
    assert report["status"] == "PASS"
    assert report["checks"]["pairwise_difference_documented"] == "DOCUMENTED_DIFFERENT_SETUP"


def test_mismatched_request_budget_detected(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    objs = _default_objectives()
    objs["objective_next_arrival"] = _scalar_obj(n_train=99999)  # differs from the other two
    _write_metrics(tmp_path, "brightkite", "cross_family_v1_brightkite", objs)
    report = m.audit_fold("brightkite")
    assert report["status"] == "FAIL"
    assert "FAIL" in report["checks"]["scalar_request_budget_identical"]


def test_selection_not_matching_val_optimal_detected(tmp_path, monkeypatch):
    """A best_model_name that doesn't match the recorded min(val_mean_regret,
    val_mae, val_rmse) rule must be caught -- this is the exact bug class
    the audit exists to find (e.g. held-out-informed or manually-overridden
    selection slipping in unrecorded)."""
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    objs = _default_objectives()
    objs["objective_eviction_loss"] = _scalar_obj(tamper_best_model="random_forest")
    _write_metrics(tmp_path, "brightkite", "cross_family_v1_brightkite", objs)
    report = m.audit_fold("brightkite")
    assert report["status"] == "FAIL"
    assert "FAIL" in report["checks"]["model_selection_used_validation_only"]
    assert "random_forest" in report["checks"]["model_selection_used_validation_only"]


def test_direction_field_does_not_affect_selection_check(tmp_path, monkeypatch):
    """Regression test for the real bug caught during development: model
    SELECTION is always min(val_mean_regret, val_mae, val_rmse) regardless
    of the objective's own "direction" field (which governs how regret is
    computed internally, not how models are ranked against each other)."""
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    objs = _default_objectives()
    objs["objective_next_arrival"] = _scalar_obj(direction="max")  # direction="max" but selection still min-regret
    _write_metrics(tmp_path, "brightkite", "cross_family_v1_brightkite", objs)
    report = m.audit_fold("brightkite")
    assert report["status"] == "PASS"


def test_main_blocked_without_partial_audit_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, family="brightkite")
    m = _import_module()
    monkeypatch.setattr(sys, "argv", ["audit_supervision_objective_fairness.py", "--families", "brightkite"])
    with pytest.raises(SystemExit) as exc_info:
        m.main()
    assert exc_info.value.code == 1
    out_path = tmp_path / "analysis" / "supervision_objective_ablation_v1" / "fairness_audit.json"
    assert not out_path.exists()
