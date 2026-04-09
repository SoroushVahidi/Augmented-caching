from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from lafc.evict_value_v2_rollout import (
    EvictValueV2RolloutConfig,
    build_pairwise_rows_from_candidate_rows,
    build_rollout_candidate_rows_v2,
)
from lafc.simulator.request_trace import load_trace


def test_v2_rollout_and_pairwise_builders_smoke():
    reqs, _pages = load_trace("data/example_atlas_v1.json")
    cfg = EvictValueV2RolloutConfig(horizons=(4, 8), reference_policy="lru")

    candidate_rows = build_rollout_candidate_rows_v2(
        requests=reqs,
        capacity=2,
        trace_name="toy",
        trace_family="examples",
        cfg=cfg,
    )
    assert candidate_rows
    assert {4, 8}.issubset({int(r["horizon"]) for r in candidate_rows})
    assert all(float(r["rollout_regret_h"]) >= 0.0 for r in candidate_rows)

    pairwise_rows = build_pairwise_rows_from_candidate_rows(candidate_rows)
    assert pairwise_rows
    assert {"label_i_better", "rollout_regret_i", "rollout_regret_j", "delta_request_bucket"}.issubset(pairwise_rows[0].keys())


def test_v2_scripts_smoke(tmp_path: Path):
    rollout_dir = tmp_path / "rollout"
    pairwise_dir = tmp_path / "pairwise"
    rollout_analysis = tmp_path / "rollout_analysis"
    pairwise_analysis = tmp_path / "pairwise_analysis"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_value_v2_rollout_dataset.py",
            "--trace-glob",
            "data/example_*.json",
            "--dataset",
            "examples",
            "--capacities",
            "2",
            "--horizons",
            "4,8",
            "--reference-policy",
            "lru",
            "--max-rows",
            "5000",
            "--output-dir",
            str(rollout_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_evict_value_v2_pairwise_dataset.py",
            "--candidate-csv",
            str(rollout_dir / "candidate_rows.csv"),
            "--output-dir",
            str(pairwise_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_value_v2_rollout_first_check.py",
            "--candidate-csv",
            str(rollout_dir / "candidate_rows.csv"),
            "--output-dir",
            str(rollout_analysis),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_evict_value_v2_pairwise_first_check.py",
            "--pairwise-csv",
            str(pairwise_dir / "pairwise_rows.csv"),
            "--output-dir",
            str(pairwise_analysis),
        ],
        check=True,
    )

    assert (rollout_analysis / "summary.json").exists()
    assert (pairwise_analysis / "summary.json").exists()

    rows = list(csv.DictReader((rollout_analysis / "per_horizon_metrics.csv").open("r", encoding="utf-8")))
    assert rows
    assert "horizon" in rows[0]
