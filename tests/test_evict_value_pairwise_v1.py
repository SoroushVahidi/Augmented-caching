from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pairwise_v1_training_and_first_check_smoke(tmp_path: Path):
    cand_dir = tmp_path / "cand"
    pair_dir = tmp_path / "pair"
    analysis_dir = tmp_path / "analysis"
    model_path = tmp_path / "pairwise_best.pkl"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_value_decision_aligned_dataset.py",
            "--trace-glob",
            "data/example_*.json",
            "--capacities",
            "2",
            "--horizons",
            "4",
            "--max-rows",
            "4000",
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
            "5000",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_evict_value_pairwise_v1.py",
            "--pairwise-csv",
            str(pair_dir / "pairwise_rows.csv"),
            "--metrics-json",
            str(tmp_path / "pair_metrics.json"),
            "--comparison-csv",
            str(tmp_path / "pair_cmp.csv"),
            "--best-model",
            str(model_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_value_pairwise_first_check.py",
            "--candidate-csv",
            str(cand_dir / "candidate_rows.csv"),
            "--pairwise-csv",
            str(pair_dir / "pairwise_rows.csv"),
            "--pairwise-model",
            str(model_path),
            "--output-dir",
            str(analysis_dir),
            "--summary-md",
            str(tmp_path / "summary.md"),
        ],
        check=True,
    )

    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    assert "mean_misses" in summary
    assert (analysis_dir / "offline_metrics.csv").exists()
    assert model_path.exists()
