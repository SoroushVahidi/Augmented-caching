"""Dry-run tests for the PE long-horizon production package (Task 14).

No Wulver compute. Exercises the manifest builder, launch guard, per-cell
validator (against synthetic data), aggregator (against synthetic
summaries), and resume planner, all locally.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "pe_long_horizon"


def run(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)


def test_manifest_has_60_cells_20_tasks_no_brightkite_no_citibike_no_h16() -> None:
    result = run([str(SCRIPTS / "build_manifest.py")])
    assert result.returncode == 0, result.stderr
    manifest_path = REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["num_production_tasks"] == 20
    assert manifest["num_logical_cells"] == 60
    assert len(manifest["tasks"]) == 20
    assert len(manifest["cells"]) == 60

    families = {t["family"] for t in manifest["tasks"]}
    assert "brightkite" not in families
    assert "citibike" not in families
    assert families == {"cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"}

    horizons_seen = {c["horizon"] for c in manifest["cells"]}
    assert 16 not in horizons_seen
    assert horizons_seen == {32, 64, 128}


def test_array_index_mapping_is_deterministic_and_unique() -> None:
    manifest_path = REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = [t["array_index"] for t in manifest["tasks"]]
    assert sorted(indices) == list(range(20))  # exactly 0..19, each once

    # re-running the builder must reproduce the identical mapping (pure function)
    result = run([str(SCRIPTS / "build_manifest.py")])
    assert result.returncode == 0
    manifest2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest2["tasks"] == manifest["tasks"]
    assert manifest2["cells"] == manifest["cells"]


def test_no_path_collision_across_tasks() -> None:
    manifest_path = REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dirs = [t["out_dir"] for t in manifest["tasks"]]
    assert len(out_dirs) == len(set(out_dirs)), "duplicate out_dir across production tasks"


def test_launch_guard_detects_placeholders_via_find_placeholders() -> None:
    # As of the final launch-gate pass, resource_params.env has all
    # placeholders resolved (by design -- that file is meant to be launch-
    # ready). This test checks the placeholder-DETECTION logic in isolation
    # (not the live file's current state) so it stays meaningful regardless.
    import importlib.util

    spec = importlib.util.spec_from_file_location("launch_guard", SCRIPTS / "launch_guard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    fake_params = {"CPUS_PER_TASK": "__TBD__", "MEMORY_PER_TASK": "4G", "X": "__WAIT_FOR_1288047__"}
    assert set(mod.find_placeholders(fake_params)) == {"CPUS_PER_TASK", "X"}


def test_launch_guard_passes_on_the_live_resolved_config() -> None:
    result = run([str(SCRIPTS / "launch_guard.py")])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Production launch gate: OPEN" in result.stdout


def test_launch_guard_passes_once_placeholders_resolved(tmp_path: Path) -> None:
    resolved = tmp_path / "resource_params.env"
    resolved.write_text(
        "CPUS_PER_TASK=2\nMEMORY_PER_TASK=16G\nWALLTIME_PER_TASK=06:00:00\nARRAY_CONCURRENCY=10\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("launch_guard", SCRIPTS / "launch_guard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    params = mod.load_params(resolved)
    assert mod.find_placeholders(params) == []


def _write_synthetic_cell(out_dir: Path, family: str, capacity: int, horizons: list[int], *, n_decisions: int = 3) -> None:
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trace_name", "trace_family", "dataset_source", "capacity", "horizon",
        "decision_id", "decision_t", "decision_chunk_id", "candidate_page_id",
        "split", "y_loss", "y_value",
    ]
    rows = []
    for h in horizons:
        for d in range(n_decisions):
            for cand in range(2):
                y_loss = float((d + cand) % 3)
                rows.append(
                    {
                        "trace_name": f"{family}_trace", "trace_family": family, "dataset_source": family,
                        "capacity": capacity, "horizon": h, "decision_id": f"dec{d}", "decision_t": d,
                        "decision_chunk_id": 0, "candidate_page_id": f"P{cand}", "split": "train",
                        "y_loss": y_loss, "y_value": -y_loss,
                    }
                )
    with (shards_dir / "part0000.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    completion = {
        "task_id": f"{family}__cap{capacity}", "family": family, "capacity": capacity,
        "horizons": horizons, "start_ts": "2026-09-15T00:00:00", "end_ts": "2026-09-15T00:01:00",
        "slurm_array_job_id": "0", "slurm_array_task_id": "0", "hostname": "test",
        "git_sha": "3a04a25094de1e8c57a603e2662b321bcf1d9e88", "command": ["python", "synthetic"],
    }
    (out_dir / "COMPLETE.json").write_text(json.dumps(completion), encoding="utf-8")


def test_validator_accepts_wellformed_synthetic_cell(tmp_path: Path) -> None:
    out_dir = tmp_path / "cloudphysics__cap32"
    _write_synthetic_cell(out_dir, "cloudphysics", 32, [32, 64, 128])
    result = run(
        [
            str(SCRIPTS / "validate_cell.py"), str(out_dir),
            "--family", "cloudphysics", "--capacity", "32", "--horizons", "32,64,128",
            "--expected-sha", "3a04a25094de1e8c57a603e2662b321bcf1d9e88",
        ]
    )
    assert "VALIDATION_RESULT: PASS" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0
    summary = json.loads((out_dir / "validated_summary.json").read_text(encoding="utf-8"))
    assert summary["validation_result"] == "PASS"
    assert set(summary["per_horizon"].keys()) == {"32", "64", "128"}


def test_validator_rejects_malformed_synthetic_output_missing_horizon(tmp_path: Path) -> None:
    out_dir = tmp_path / "metacdn__cap64"
    # only write horizons 32 and 64 -- 128 is missing -> should FAIL
    _write_synthetic_cell(out_dir, "metacdn", 64, [32, 64])
    result = run(
        [
            str(SCRIPTS / "validate_cell.py"), str(out_dir),
            "--family", "metacdn", "--capacity", "64", "--horizons", "32,64,128",
        ]
    )
    assert result.returncode == 1
    assert "VALIDATION_RESULT: FAIL" in result.stdout
    summary = json.loads((out_dir / "validated_summary.json").read_text(encoding="utf-8"))
    assert summary["missing_horizons"] == [128]


def test_validator_rejects_duplicate_candidate_keys(tmp_path: Path) -> None:
    out_dir = tmp_path / "metakv__cap128"
    _write_synthetic_cell(out_dir, "metakv", 128, [32, 64, 128])
    # duplicate one row's exact key by appending an identical row
    shard = out_dir / "shards" / "part0000.csv"
    lines = shard.read_text(encoding="utf-8").splitlines()
    lines.append(lines[1])  # duplicate the first data row
    shard.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = run(
        [
            str(SCRIPTS / "validate_cell.py"), str(out_dir),
            "--family", "metakv", "--capacity", "128", "--horizons", "32,64,128",
        ]
    )
    assert result.returncode == 1
    summary = json.loads((out_dir / "validated_summary.json").read_text(encoding="utf-8"))
    assert summary["duplicate_candidate_keys"] >= 1
    assert summary["validation_result"] == "FAIL"


def _make_synthetic_run_root(tmp_path: Path, manifest: dict, *, n_complete: int) -> Path:
    run_root = tmp_path / "run_root"
    for i, task in enumerate(manifest["tasks"]):
        out_dir = run_root / "raw" / task["task_id"]
        if i < n_complete:
            _write_synthetic_cell(out_dir, task["family"], task["capacity"], task["horizons"])
            subprocess.run(
                [
                    sys.executable, str(SCRIPTS / "validate_cell.py"), str(out_dir),
                    "--family", task["family"], "--capacity", str(task["capacity"]),
                    "--horizons", ",".join(str(h) for h in task["horizons"]),
                ],
                capture_output=True, text=True,
            )
    return run_root


def test_aggregator_rejects_59_of_60_cells(tmp_path: Path) -> None:
    manifest = json.loads((REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json").read_text())
    run_root = _make_synthetic_run_root(tmp_path, manifest, n_complete=19)  # 19/20 tasks -> 57/60 cells
    result = run(
        [
            str(SCRIPTS / "aggregate.py"),
            "--manifest", str(REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"),
            "--run-root", str(run_root),
            "--out-dir", str(tmp_path / "summaries"),
        ]
    )
    assert result.returncode == 1
    assert "AGGREGATION_REFUSED" in result.stdout
    assert not (tmp_path / "summaries" / "table_A_family_capacity_horizon.csv").exists()


def test_aggregator_accepts_60_of_60_synthetic_valid_cells(tmp_path: Path) -> None:
    manifest = json.loads((REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json").read_text())
    run_root = _make_synthetic_run_root(tmp_path, manifest, n_complete=20)  # all 20 tasks -> 60/60 cells
    result = run(
        [
            str(SCRIPTS / "aggregate.py"),
            "--manifest", str(REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"),
            "--run-root", str(run_root),
            "--out-dir", str(tmp_path / "summaries"),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AGGREGATION_OK" in result.stdout
    table_a = (tmp_path / "summaries" / "table_A_family_capacity_horizon.csv").read_text()
    # 20 tasks x 3 horizons = 60 data rows + 1 header
    assert len(table_a.strip().splitlines()) == 61


def test_resume_planner_identifies_missing_cells(tmp_path: Path) -> None:
    manifest = json.loads((REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json").read_text())
    run_root = _make_synthetic_run_root(tmp_path, manifest, n_complete=18)  # 2 tasks NOT_RUN
    result = run(
        [
            str(SCRIPTS / "resume_planner.py"),
            "--manifest", str(REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"),
            "--run-root", str(run_root),
        ]
    )
    assert result.returncode == 0
    assert "NOT_RUN: 2" in result.stdout
    assert "DONE: 18" in result.stdout
    assert "--array=" in result.stdout
