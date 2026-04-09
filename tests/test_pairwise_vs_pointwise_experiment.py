from __future__ import annotations

import json
import subprocess
import sys


def test_pairwise_vs_pointwise_experiment_runs(tmp_path):
    out_dir = tmp_path / "pairwise_vs_pointwise"
    cmd = [
        sys.executable,
        "scripts/run_pairwise_vs_pointwise_experiment.py",
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
        "--supervision-style",
        "both",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "downstream_results.csv").exists()
    assert (out_dir / "disagreement_analysis.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.md").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    styles = {row["supervision_style"] for row in summary["aggregate_downstream"]}
    assert "pointwise" in styles
    assert "pairwise" in styles
    assert "pairwise_vs_pointwise" in summary
    assert "wins" in summary["pairwise_vs_pointwise"]
