"""Fixture tests for scripts/experiments/audit_distribution_shift_completion.py."""

from __future__ import annotations

import csv
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
    import audit_distribution_shift_completion as m
    return m


POLICY_FIELDS = ["condition", "held_out_family", "capacity", "status", "score_start", "score_end", "scored_requests"]


def _full_valid_rows(m):
    rows = []
    for family in m.FAMILIES:
        for cap in m.CAPACITIES:
            for cond in m.CONDITIONS:
                rows.append({
                    "condition": cond, "held_out_family": family, "capacity": cap, "status": "ok",
                    "score_start": 10000, "score_end": 50000, "scored_requests": 40000,
                })
    return rows


def _write_csv(path: Path, fields: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_shift_and_traj(m, out_dir: Path) -> None:
    shift_rows = [{"condition": r["condition"], "held_out_family": r["held_out_family"], "capacity": r["capacity"]}
                  for r in _full_valid_rows(m)]
    _write_csv(out_dir / "state_shift_metrics.csv", ["condition", "held_out_family", "capacity"], shift_rows)
    traj_rows = [
        {"held_out_family": f, "capacity": c, "reference_condition": "OFF_POLICY_LRU", "other_condition": "DAGGER_ITER1"}
        for f in m.FAMILIES for c in m.CAPACITIES
    ]
    _write_csv(out_dir / "trajectory_divergence.csv",
               ["held_out_family", "capacity", "reference_condition", "other_condition"], traj_rows)


def _write_protocol(tmp_path: Path, out_dir: Path, cfg=None) -> None:
    cfg = cfg or {"protocol_id": "distribution_shift_ablation_v1"}
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "distribution_shift_ablation_v1.json").write_text(json.dumps(cfg))
    (out_dir / "protocol_snapshot.json").write_text(json.dumps(cfg))


def test_no_csv_yet_is_incomplete(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    result = m.audit(out_dir)
    assert result["classification"] == "INCOMPLETE"
    assert result["primary_rows"] == 0


def test_partial_rows_is_incomplete_no_deep_checks(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    _write_csv(out_dir / "policy_comparison.csv", POLICY_FIELDS, _full_valid_rows(m)[:18])
    result = m.audit(out_dir)
    assert result["classification"] == "INCOMPLETE"
    assert result["primary_rows"] == 18
    assert result["checks"] == {}


def test_full_valid_campaign_classified_complete_valid(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    _write_csv(out_dir / "policy_comparison.csv", POLICY_FIELDS, _full_valid_rows(m))
    _write_shift_and_traj(m, out_dir)
    _write_protocol(tmp_path, out_dir)
    result = m.audit(out_dir)
    assert result["classification"] == "COMPLETE_VALID"
    assert all(v == "PASS" for v in result["checks"].values())


def test_full_but_duplicate_rows_classified_complete_invalid(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    rows = _full_valid_rows(m)
    rows[-1] = dict(rows[-2])  # duplicate the last key, still 42 rows total
    _write_csv(out_dir / "policy_comparison.csv", POLICY_FIELDS, rows)
    _write_shift_and_traj(m, out_dir)
    _write_protocol(tmp_path, out_dir)
    result = m.audit(out_dir)
    assert result["classification"] == "COMPLETE_INVALID"
    assert "FAIL" in result["checks"]["no_duplicate_keys"]


def test_full_but_protocol_drift_classified_complete_invalid(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    _write_csv(out_dir / "policy_comparison.csv", POLICY_FIELDS, _full_valid_rows(m))
    _write_shift_and_traj(m, out_dir)
    _write_protocol(tmp_path, out_dir, cfg={"protocol_id": "distribution_shift_ablation_v1"})
    # Drift the live config after the snapshot was taken.
    (tmp_path / "configs" / "distribution_shift_ablation_v1.json").write_text(
        json.dumps({"protocol_id": "distribution_shift_ablation_v1", "changed": True})
    )
    result = m.audit(out_dir)
    assert result["classification"] == "COMPLETE_INVALID"
    assert "FAIL" in result["checks"]["frozen_protocol_unchanged"]


def test_full_but_wrong_scoring_window_classified_complete_invalid(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    rows = _full_valid_rows(m)
    rows[0]["score_end"] = 45000  # wrong window
    _write_csv(out_dir / "policy_comparison.csv", POLICY_FIELDS, rows)
    _write_shift_and_traj(m, out_dir)
    _write_protocol(tmp_path, out_dir)
    result = m.audit(out_dir)
    assert result["classification"] == "COMPLETE_INVALID"
    assert "FAIL" in result["checks"]["scoring_window_consistent"]


def test_missing_trajectory_coverage_classified_complete_invalid(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    _write_csv(out_dir / "policy_comparison.csv", POLICY_FIELDS, _full_valid_rows(m))
    shift_rows = [{"condition": r["condition"], "held_out_family": r["held_out_family"], "capacity": r["capacity"]}
                  for r in _full_valid_rows(m)]
    _write_csv(out_dir / "state_shift_metrics.csv", ["condition", "held_out_family", "capacity"], shift_rows)
    # trajectory_divergence.csv intentionally omitted -> missing coverage
    _write_protocol(tmp_path, out_dir)
    result = m.audit(out_dir)
    assert result["classification"] == "COMPLETE_INVALID"
    assert "FAIL" in result["checks"]["trajectory_diagnostic_coverage"]
