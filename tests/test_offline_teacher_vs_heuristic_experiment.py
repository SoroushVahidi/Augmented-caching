from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_offline_teacher_vs_heuristic_experiment_runs(tmp_path):
    out_dir = tmp_path / "exp"
    cmd = [
        sys.executable,
        "scripts/run_offline_teacher_vs_heuristic_experiment.py",
        "--trace-glob",
        "data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json",
        "--capacities",
        "2",
        "--horizon",
        "8",
        "--max-requests-per-trace",
        "40",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.md").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "results" in summary
    assert "disagreement" in summary
    assert "gain_concentration" in summary

    label_sources = {row["label_source"] for row in summary["results"]}
    assert "heuristic" in label_sources
    assert "offline_teacher" in label_sources
