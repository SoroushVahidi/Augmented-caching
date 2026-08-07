"""Fixture tests for scripts/revision_status.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path("scripts").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = _SCRIPTS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _SCRIPTS_DIR)
    yield
    if inserted and _SCRIPTS_DIR in sys.path:
        sys.path.remove(_SCRIPTS_DIR)


def _import_module():
    import revision_status as m
    return m


def test_parse_worktree_porcelain():
    m = _import_module()
    porcelain = (
        "worktree /home/soroush/Augmented-caching\n"
        "HEAD abc123\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /home/soroush/Augmented-caching-fairness\n"
        "HEAD def456\n"
        "branch refs/heads/feat/reviewer-fairness-protocol\n"
    )

    class FakeResult:
        stdout = porcelain

    import subprocess as real_subprocess
    orig_run = real_subprocess.run

    def fake_run(cmd, **kwargs):
        if "worktree" in cmd:
            return FakeResult()
        return orig_run(cmd, **kwargs)

    import unittest.mock
    with unittest.mock.patch.object(m.subprocess, "run", side_effect=fake_run):
        worktrees = m.find_worktrees(Path("/fake"))
    assert worktrees["main"] == Path("/home/soroush/Augmented-caching")
    assert worktrees["feat/reviewer-fairness-protocol"] == Path("/home/soroush/Augmented-caching-fairness")


def _write_csv(path: Path, n_data_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b"])
        for i in range(n_data_rows):
            w.writerow([i, i])


def test_concern_1_status_counts_datasets_and_models(tmp_path):
    m = _import_module()
    data_root = tmp_path / "data" / "derived" / "evict_value_v1_cross_family_v1"
    for f in ["brightkite", "citibike"]:
        (data_root / f).mkdir(parents=True)
        (data_root / f / "manifest.json").write_text("{}")
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "evict_value_v1_cross_family_v1_brightkite.pkl").write_bytes(b"x")

    status = m.concern_1_status(tmp_path)
    assert status["datasets_done"] == 2
    assert status["models_done"] == 1
    assert status["registry_frozen"] is False
    assert status["status"] == "RUNNING"


def test_concern_1_status_ready_for_next_stage_when_all_models_done_unfrozen(tmp_path):
    m = _import_module()
    models_root = tmp_path / "models"
    models_root.mkdir()
    for f in m.FAMILIES:
        (models_root / f"evict_value_v1_cross_family_v1_{f}.pkl").write_bytes(b"x")
    status = m.concern_1_status(tmp_path)
    assert status["models_done"] == 7
    assert status["status"] == "TRAINING COMPLETE — READY FOR NEXT STAGE"


def test_concern_2_status_counts_across_4_objectives(tmp_path):
    m = _import_module()
    models_root = tmp_path / "models" / "supervision_objective_ablation_v1"
    for o in m.OBJECTIVES:
        (models_root / o).mkdir(parents=True)
    (models_root / "objective_eviction_loss" / "brightkite.pkl").write_bytes(b"x")
    (models_root / "objective_pairwise" / "brightkite.pkl").write_bytes(b"x")
    status = m.concern_2_status(tmp_path)
    assert status["models_done"] == 2
    assert status["models_total"] == 28


def test_concern_3_status_reads_campaign_state(tmp_path):
    m = _import_module()
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "policy_comparison.csv", 18)
    (out_dir / "campaign_state.json").write_text(json.dumps({"completed_folds": ["brightkite", "citibike", "cloudphysics"]}))
    status = m.concern_3_status(tmp_path)
    assert status["primary_rows"] == 18
    assert status["folds_complete"] == 3
    assert status["status"] == "STOPPED_CLEANLY_PARTIAL"


def test_concern_3_status_complete_when_42_rows(tmp_path):
    m = _import_module()
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "policy_comparison.csv", 42)
    status = m.concern_3_status(tmp_path)
    assert status["status"] == "COMPLETE"


def test_count_csv_rows_missing_file_is_zero(tmp_path):
    m = _import_module()
    assert m._count_csv_rows(tmp_path / "nope.csv") == 0
