from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from lafc.evict_decision_aligned_v1 import (
    DecisionAlignedEvictConfig,
    build_evict_pairwise_examples_v1,
    build_evict_regret_examples_v1,
)
from lafc.simulator.request_trace import load_trace


def test_decision_aligned_dataset_builders_smoke():
    requests, _pages = load_trace("data/example_atlas_v1.json")
    cfg = DecisionAlignedEvictConfig(horizon=8)

    regret_rows = build_evict_regret_examples_v1(requests=requests, capacity=2, trace_name="toy", cfg=cfg)
    assert regret_rows
    assert all(float(r["y_regret"]) >= 0.0 for r in regret_rows)

    pairwise_rows = build_evict_pairwise_examples_v1(requests=requests, capacity=2, trace_name="toy", cfg=cfg)
    assert pairwise_rows
    assert {"label_i_better", "delta_request_bucket", "regret_i", "regret_j"}.issubset(pairwise_rows[0].keys())


def test_decision_aligned_scripts_smoke(tmp_path: Path):
    data_dir = tmp_path / "derived"
    regret_out = tmp_path / "regret_analysis"
    pair_out = tmp_path / "pair_analysis"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_regret_dataset_v1.py",
            "--trace-glob",
            "data/example_*.json",
            "--capacities",
            "2",
            "--horizon",
            "8",
            "--output-dir",
            str(data_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_pairwise_dataset_v1.py",
            "--trace-glob",
            "data/example_*.json",
            "--capacities",
            "2",
            "--horizon",
            "8",
            "--output-dir",
            str(data_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_regret_first_check.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(regret_out),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_pairwise_first_check.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(pair_out),
        ],
        check=True,
    )

    assert (regret_out / "summary.json").exists()
    assert (pair_out / "summary.json").exists()

    metrics = list(csv.DictReader((pair_out / "metrics.csv").open("r", encoding="utf-8")))
    assert metrics
    assert "pairwise_accuracy" in metrics[0]
