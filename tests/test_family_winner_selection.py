from __future__ import annotations

import csv
import json
import subprocess
import sys


def test_family_winner_selection_runs(tmp_path):
    out_dir = tmp_path / "winner_selection"
    model_path = out_dir / "models" / "ml_gate_v2_lightweight.pkl"
    cmd = [
        sys.executable,
        "scripts/run_family_winner_selection.py",
        "--capacities",
        "2,3",
        "--max-requests",
        "80",
        "--regimes",
        "clean,noisy",
        "--ml-gate-horizon",
        "8",
        "--ml-gate-model-path",
        str(model_path),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "winner_selection_results.csv").exists()
    assert (out_dir / "winner_selection_summary.json").exists()
    assert (out_dir / "winner_selection_report.md").exists()

    rows = list(csv.DictReader((out_dir / "winner_selection_results.csv").open("r", encoding="utf-8")))
    policies = {r["policy"] for r in rows}
    assert {
        "evict_value_v1",
        "rest_v1",
        "atlas_v3",
        "ml_gate_v2",
        "trust_and_doubt",
        "robust_ftp_d_marker",
        "blind_oracle_lru_combiner",
        "lru",
    }.issubset(policies)

    summary = json.loads((out_dir / "winner_selection_summary.json").read_text(encoding="utf-8"))
    assert summary["strongest_family_candidate"] in policies
    assert summary["unpromising_family_candidate"] in policies
