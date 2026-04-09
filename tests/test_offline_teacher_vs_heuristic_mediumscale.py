from __future__ import annotations

import json
import subprocess
import sys


def test_offline_teacher_vs_heuristic_mediumscale_runs(tmp_path):
    out_dir = tmp_path / "medium"
    cmd = [
        sys.executable,
        "scripts/run_offline_teacher_vs_heuristic_mediumscale.py",
        "--trace-glob",
        "data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json",
        "--capacities",
        "2,3",
        "--horizon",
        "10",
        "--max-requests-per-trace",
        "50",
        "--seeds",
        "0,1",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "downstream_results.csv").exists()
    assert (out_dir / "disagreement_by_family.csv").exists()
    assert (out_dir / "disagreement_by_capacity.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.md").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "aggregate_downstream" in summary
    assert "teacher_vs_heuristic" in summary
    assert "gain_concentration" in summary
    assert summary["teacher_vs_heuristic"]["workloads_compared"] >= 1
