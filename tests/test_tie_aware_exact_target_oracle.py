from __future__ import annotations

import csv

import pytest

from scripts.experiments.run_tie_aware_exact_target_oracle import (
    CORE_CSV_FIELDS,
    DIAGNOSTIC_CSV_FIELDS,
    aggregate_completed_units,
    summary_csv_fieldnames,
    write_summary_csv,
)


CORE = {
    "family": "brightkite",
    "capacity": 32,
    "seed": "",
    "misses": 1,
    "miss_ratio": 0.1,
    "delta_vs_LRU": 0,
    "delta_vs_current_exact": -1,
    "trace_sha256": "abc",
}

DIAG = {
    "fraction_tied_decisions": 1.0,
    "fraction_all_tied": 0.5,
    "mean_optimal_set_fraction": 0.9,
}


def _lru_row():
    return {**CORE, "tie_policy": "LRU_REFERENCE"}


def _tie_row():
    return {**CORE, "tie_policy": "CURRENT_DETERMINISTIC", "misses": 2, "miss_ratio": 0.2, "delta_vs_LRU": 1, "delta_vs_current_exact": 0, **DIAG}


def test_heterogeneous_rows_fail_with_first_row_fieldnames(tmp_path):
    rows = [_lru_row(), _tie_row()]
    with pytest.raises(ValueError, match="dict contains fields not in fieldnames"):
        with (tmp_path / "bad.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def test_summary_csv_union_writes_lru_reference_without_diagnostics(tmp_path):
    rows = [_lru_row(), _tie_row()]
    assert summary_csv_fieldnames(rows) == CORE_CSV_FIELDS + DIAGNOSTIC_CSV_FIELDS
    path = tmp_path / "summary.csv"
    write_summary_csv(path, rows)
    text = path.read_text()
    parsed = list(csv.DictReader(text.splitlines()))
    assert parsed[0]["tie_policy"] == "LRU_REFERENCE"
    assert parsed[0]["fraction_tied_decisions"] == ""
    assert parsed[0]["fraction_all_tied"] == ""
    assert parsed[0]["mean_optimal_set_fraction"] == ""
    assert parsed[1]["fraction_tied_decisions"] == "1.0"
    assert parsed[1]["fraction_all_tied"] == "0.5"
    assert parsed[1]["mean_optimal_set_fraction"] == "0.9"
    header = text.splitlines()[0].split(",")
    assert header[:9] == CORE_CSV_FIELDS
    assert header[9:] == DIAGNOSTIC_CSV_FIELDS


def test_aggregate_completed_units_rejects_incomplete_campaign(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing unit summary"):
        aggregate_completed_units(tmp_path)
