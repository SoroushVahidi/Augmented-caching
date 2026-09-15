"""Synthetic resume-failure scenarios (Task 10 of the final launch gate).

All states here are fabricated locally -- no real Slurm job, no
computation. Confirms resume_planner.py distinguishes NOT_RUN / INCOMPLETE
/ FAILED / DONE correctly and never recommends relaunching a task that
already succeeded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "pe_long_horizon"


def _manifest():
    return json.loads((REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json").read_text())


def _mark_done(run_root: Path, task: dict) -> None:
    out_dir = run_root / "raw" / task["task_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "COMPLETE.json").write_text(json.dumps({"task_id": task["task_id"]}), encoding="utf-8")
    (out_dir / "validated_summary.json").write_text(
        json.dumps({"task_id": task["task_id"], "validation_result": "PASS"}), encoding="utf-8"
    )


def _mark_incomplete_timeout(run_root: Path, task: dict) -> None:
    """Simulates a TIMEOUT: output dir exists (partial shards), no COMPLETE.json."""
    out_dir = run_root / "raw" / task["task_id"]
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "shards" / "part0000.csv").write_text("trace_name\n", encoding="utf-8")
    # deliberately no COMPLETE.json


def _mark_validation_failed(run_root: Path, task: dict) -> None:
    """Generation succeeded but the validator found a real defect."""
    out_dir = run_root / "raw" / task["task_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "COMPLETE.json").write_text(json.dumps({"task_id": task["task_id"]}), encoding="utf-8")
    (out_dir / "validated_summary.json").write_text(
        json.dumps({"task_id": task["task_id"], "validation_result": "FAIL", "duplicate_candidate_keys": 3}),
        encoding="utf-8",
    )


def _run_resume_planner(run_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "resume_planner.py"),
         "--manifest", str(REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"),
         "--run-root", str(run_root)],
        capture_output=True, text=True,
    )


def test_one_cap32_task_missing(tmp_path: Path) -> None:
    manifest = _manifest()
    run_root = tmp_path / "run_root_missing_cap32"
    missing_task = next(t for t in manifest["tasks"] if t["family"] == "cloudphysics" and t["capacity"] == 32)
    for t in manifest["tasks"]:
        if t is missing_task:
            continue
        _mark_done(run_root, t)

    result = _run_resume_planner(run_root)
    assert result.returncode == 0
    assert "NOT_RUN: 1" in result.stdout
    assert missing_task["task_id"] in result.stdout
    assert f"--array={missing_task['array_index']}" in result.stdout


def test_one_cap256_task_timeout(tmp_path: Path) -> None:
    manifest = _manifest()
    run_root = tmp_path / "run_root_timeout_cap256"
    timeout_task = next(t for t in manifest["tasks"] if t["family"] == "wiki2018" and t["capacity"] == 256)
    for t in manifest["tasks"]:
        if t is timeout_task:
            _mark_incomplete_timeout(run_root, t)
        else:
            _mark_done(run_root, t)

    result = _run_resume_planner(run_root)
    assert result.returncode == 0
    assert "INCOMPLETE: 1" in result.stdout
    assert timeout_task["task_id"] in result.stdout
    # must be flagged for resubmit consideration, distinct from DONE
    assert f"--array={timeout_task['array_index']}" in result.stdout


def test_one_validation_failure(tmp_path: Path) -> None:
    manifest = _manifest()
    run_root = tmp_path / "run_root_validation_fail"
    failed_task = next(t for t in manifest["tasks"] if t["family"] == "metakv" and t["capacity"] == 128)
    for t in manifest["tasks"]:
        if t is failed_task:
            _mark_validation_failed(run_root, t)
        else:
            _mark_done(run_root, t)

    result = _run_resume_planner(run_root)
    assert result.returncode == 0
    assert "FAILED: 1" in result.stdout
    assert failed_task["task_id"] in result.stdout


def test_19_of_20_production_complete(tmp_path: Path) -> None:
    manifest = _manifest()
    run_root = tmp_path / "run_root_19_of_20"
    for t in manifest["tasks"][:19]:
        _mark_done(run_root, t)
    # task 20 (index 19) left NOT_RUN entirely

    result = _run_resume_planner(run_root)
    assert "DONE: 19" in result.stdout
    assert "NOT_RUN: 1" in result.stdout


def test_resume_planner_never_recommends_relaunching_done_tasks(tmp_path: Path) -> None:
    manifest = _manifest()
    run_root = tmp_path / "run_root_all_done"
    for t in manifest["tasks"]:
        _mark_done(run_root, t)

    result = _run_resume_planner(run_root)
    assert "DONE: 20" in result.stdout
    assert "All 20 production tasks DONE and validated PASS." in result.stdout
    # no --array= resubmit spec should appear when nothing needs resubmitting
    assert "--array=" not in result.stdout


def test_aggregator_refuses_at_59_of_60_cells_single_task_defect(tmp_path: Path) -> None:
    """59/60 logical cells: 19 fully-valid tasks (57 cells) + 1 task that
    completed generation but validated FAIL (contributes 0 of its 3 cells) --
    demonstrates the shortfall is not exactly 59 from this angle, so also
    directly assert the aggregator's own hard gate at the boundary condition
    the manifest structure actually allows: any non-PASS task blocks
    aggregation entirely, regardless of how many of its 3 horizons might
    individually have been fine.
    """
    manifest = _manifest()
    run_root = tmp_path / "run_root_59_cells"
    for t in manifest["tasks"][:19]:
        _mark_done(run_root, t)
    _mark_validation_failed(run_root, manifest["tasks"][19])

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "aggregate.py"),
         "--manifest", str(REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"),
         "--run-root", str(run_root),
         "--out-dir", str(tmp_path / "summaries")],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "AGGREGATION_REFUSED" in result.stdout
    assert not (tmp_path / "summaries" / "table_A_family_capacity_horizon.csv").exists()
