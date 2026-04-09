from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_decision_aligned_target_scripts_smoke(tmp_path: Path):
    cand_dir = tmp_path / "cand"
    pair_dir = tmp_path / "pair"
    cand_analysis = tmp_path / "cand_analysis"
    pair_analysis = tmp_path / "pair_analysis"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_value_decision_aligned_dataset.py",
            "--trace-glob",
            "data/example_*.json",
            "--capacities",
            "2",
            "--horizons",
            "4,8",
            "--continuation-policy",
            "lru",
            "--max-rows",
            "5000",
            "--output-dir",
            str(cand_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_value_pairwise_dataset.py",
            "--candidate-csv",
            str(cand_dir / "candidate_rows.csv"),
            "--output-dir",
            str(pair_dir),
            "--max-rows",
            "10000",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_value_decision_aligned_first_check.py",
            "--candidate-csv",
            str(cand_dir / "candidate_rows.csv"),
            "--output-dir",
            str(cand_analysis),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_value_pairwise_first_check.py",
            "--pairwise-csv",
            str(pair_dir / "pairwise_rows.csv"),
            "--output-dir",
            str(pair_analysis),
        ],
        check=True,
    )

    assert (cand_dir / "dataset_summary.json").exists()
    assert (pair_dir / "dataset_summary.json").exists()
    assert (cand_analysis / "summary.json").exists()
    assert (pair_analysis / "summary.json").exists()

    cand_rows = list(csv.DictReader((cand_dir / "candidate_rows.csv").open("r", encoding="utf-8")))
    assert cand_rows
    required_cols = {
        "trace",
        "request_t",
        "capacity",
        "horizon",
        "candidate_page_id",
        "rollout_loss_h",
        "rollout_regret_h",
        "candidate_is_rollout_optimal",
        "candidate_rank",
        "candidate_count",
    }
    assert required_cols.issubset(cand_rows[0].keys())

    metrics = list(csv.DictReader((cand_analysis / "metrics.csv").open("r", encoding="utf-8")))
    assert metrics
    assert "rmse_rollout_loss" in metrics[0]

    pair_metrics = list(csv.DictReader((pair_analysis / "metrics.csv").open("r", encoding="utf-8")))
    assert pair_metrics
    assert "pairwise_accuracy" in pair_metrics[0]
