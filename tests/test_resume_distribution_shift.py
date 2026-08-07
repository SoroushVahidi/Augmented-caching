"""Fixture tests for scripts/experiments/resume_distribution_shift.py."""

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
    import resume_distribution_shift as m
    return m


def _setup(tmp_path: Path, config: dict, snapshot: dict | None = None, completed_folds=None):
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "distribution_shift_ablation_v1.json").write_text(json.dumps(config))
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        (out_dir / "protocol_snapshot.json").write_text(json.dumps(snapshot))
    if completed_folds is not None:
        (out_dir / "campaign_state.json").write_text(json.dumps({"completed_folds": completed_folds}))
    return out_dir


def test_no_snapshot_means_never_launched(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _setup(tmp_path, {"protocol_id": "x"})
    with pytest.raises(m.ResumeBlocked, match="never launched"):
        m.plan(tmp_path / "analysis" / "distribution_shift_ablation_v1")


def test_protocol_drift_blocked(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = _setup(tmp_path, {"protocol_id": "x", "v": 2}, snapshot={"protocol_id": "x", "v": 1}, completed_folds=[])
    with pytest.raises(m.ResumeBlocked, match="Protocol drift"):
        m.plan(out_dir)


def test_clean_resume_plan_reports_actual_state(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    cfg = {"protocol_id": "x"}
    out_dir = _setup(tmp_path, cfg, snapshot=cfg, completed_folds=["brightkite", "citibike"])
    p = m.plan(out_dir)
    assert p["n_completed_folds"] == 2
    assert p["next_fold"] == "cloudphysics"
    assert p["remaining_folds_in_order"] == ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
    assert p["all_artifacts_clean"] is True  # no CSVs written yet -> vacuously clean


def test_duplicate_rows_flagged_dirty(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    cfg = {"protocol_id": "x"}
    out_dir = _setup(tmp_path, cfg, snapshot=cfg, completed_folds=["brightkite"])
    csv_path = out_dir / "policy_comparison.csv"
    csv_path.write_text(
        "condition,held_out_family,capacity,status\n"
        "OFF_POLICY_LRU,brightkite,32,ok\n"
        "OFF_POLICY_LRU,brightkite,32,ok\n"  # duplicate key
    )
    p = m.plan(out_dir)
    integrity = p["artifact_integrity"]["policy_comparison.csv"]
    assert integrity["duplicate_keys"] == 1
    assert integrity["clean"] is False
    assert p["all_artifacts_clean"] is False


def test_failed_status_row_flagged_dirty(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    cfg = {"protocol_id": "x"}
    out_dir = _setup(tmp_path, cfg, snapshot=cfg, completed_folds=[])
    csv_path = out_dir / "policy_comparison.csv"
    csv_path.write_text("condition,held_out_family,capacity,status\nOFF_POLICY_LRU,brightkite,32,failed\n")
    p = m.plan(out_dir)
    integrity = p["artifact_integrity"]["policy_comparison.csv"]
    assert integrity["non_ok_status_rows"] == 1
    assert integrity["clean"] is False


def test_nan_value_flagged_dirty(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    cfg = {"protocol_id": "x"}
    out_dir = _setup(tmp_path, cfg, snapshot=cfg, completed_folds=[])
    csv_path = out_dir / "policy_comparison.csv"
    csv_path.write_text("condition,held_out_family,capacity,status,state_shift_index\nOFF_POLICY_LRU,brightkite,32,ok,nan\n")
    p = m.plan(out_dir)
    integrity = p["artifact_integrity"]["policy_comparison.csv"]
    assert integrity["nan_or_inf_values"] == 1
    assert integrity["clean"] is False


def test_all_folds_complete_reports_none_next(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    cfg = {"protocol_id": "x"}
    out_dir = _setup(tmp_path, cfg, snapshot=cfg, completed_folds=list(m.FAMILIES))
    p = m.plan(out_dir)
    assert p["next_fold"] is None
    assert p["remaining_folds_in_order"] == []


def test_main_dry_run_never_launches_subprocess(tmp_path, monkeypatch, capsys):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    cfg = {"protocol_id": "x"}
    _setup(tmp_path, cfg, snapshot=cfg, completed_folds=["brightkite"])

    called = []
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: called.append(a))
    monkeypatch.setattr(sys, "argv", ["resume_distribution_shift.py", "--dry-run"])
    m.main()
    assert called == []
    assert "NOT launching" in capsys.readouterr().out
