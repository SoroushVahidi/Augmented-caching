from __future__ import annotations

import csv
import subprocess
import sys


def test_evict_value_failure_slice_audit_smoke(tmp_path):
    out_csv = tmp_path / "audit.csv"
    out_md = tmp_path / "summary.md"
    cmd = [
        sys.executable,
        "scripts/run_evict_value_failure_slice_audit.py",
        "--trace-manifest",
        "",
        "--trace-glob",
        "data/example_unweighted.json,data/example_atlas_v1.json",
        "--max-traces",
        "2",
        "--capacities",
        "2",
        "--max-requests-per-trace",
        "80",
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True)

    assert out_csv.exists()
    assert out_md.exists()

    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    assert rows, "expected at least one aligned eviction row"
    expected_cols = {
        "trace_name",
        "trace_family",
        "capacity",
        "t",
        "request_page",
        "evict_value_v1_victim",
        "predictive_marker_victim",
        "trust_and_doubt_victim",
        "rest_v1_victim",
        "lru_victim",
    }
    assert expected_cols.issubset(rows[0].keys())

    md = out_md.read_text(encoding="utf-8")
    assert "Overall comparison counts" in md
    assert "Per-family breakdown" in md
