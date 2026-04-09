from __future__ import annotations

import csv
import json
import subprocess
import sys


def test_hybrid_fallback_experiment_runs(tmp_path):
    out_dir = tmp_path / "hybrid_fallback"
    cmd = [
        sys.executable,
        "scripts/run_hybrid_fallback_experiment.py",
        "--trace-glob",
        "data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json",
        "--capacities",
        "2,3",
        "--horizon",
        "8",
        "--max-requests-per-trace",
        "60",
        "--seeds",
        "0,1",
        "--margin-thresholds",
        "0.00,0.03,0.08",
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    expected = [
        "results.csv",
        "downstream_results.csv",
        "threshold_selection.csv",
        "trigger_analysis.csv",
        "summary.json",
        "report.md",
    ]
    for name in expected:
        assert (out_dir / name).exists(), f"missing {name}"

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "hybrid_vs_pointwise" in summary
    assert "wins" in summary["hybrid_vs_pointwise"]

    rows = list(csv.DictReader((out_dir / "results.csv").open("r", encoding="utf-8")))
    policies = {r["policy"] for r in rows}
    assert {"pointwise", "hybrid", "lru"}.issubset(policies)

    threshold_rows = list(csv.DictReader((out_dir / "threshold_selection.csv").open("r", encoding="utf-8")))
    assert threshold_rows

    trigger_rows = list(csv.DictReader((out_dir / "trigger_analysis.csv").open("r", encoding="utf-8")))
    assert trigger_rows
    assert "trigger_frequency" in trigger_rows[0]
