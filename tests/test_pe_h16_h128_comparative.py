from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pe_h16_h128_comparative.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pe_h16_h128_comparative", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def tiny_h16_rows() -> list[dict[str, str]]:
    return [
        {
            "trace_family": "cloudphysics",
            "capacity": "32",
            "horizon": "16",
            "n_decisions": "2",
            "all_tied_fraction": "0.5",
            "unique_winner_fraction": "0.0",
            "mean_optimal_set_fraction": "0.75",
            "median_optimal_set_fraction": "1.0",
            "random_optimal_probability": "0.75",
            "mean_random_regret": "0.125",
        },
        {
            "trace_family": "metacdn",
            "capacity": "32",
            "horizon": "16",
            "n_decisions": "4",
            "all_tied_fraction": "0.25",
            "unique_winner_fraction": "0.0",
            "mean_optimal_set_fraction": "0.5",
            "median_optimal_set_fraction": "0.5",
            "random_optimal_probability": "0.5",
            "mean_random_regret": "0.25",
        },
    ]


def test_h16_schema_conversion_and_family_mapping():
    mod = load_module()
    rows = mod.convert_h16_cells(tiny_h16_rows())
    first = rows[0]
    assert first["family"] == "alibaba-block"
    assert first["internal_family"] == "cloudphysics"
    assert first["horizon"] == 16
    assert first["decision_count"] == 2
    assert first["candidate_row_count"] == 64
    assert first["all_tied_decision_count"] == 1
    assert first["discriminative_decision_count"] == 1
    assert first["discriminative_fraction"] == 0.5


def test_horizon_identity_and_duplicate_detection():
    mod = load_module()
    rows = []
    for family in mod.EXPECTED_INTERNAL_FAMILIES:
        for capacity in mod.EXPECTED_CAPACITIES:
            for horizon in mod.EXPECTED_HORIZONS:
                rows.append(
                    {
                        "family": mod.READER_FAMILY[family],
                        "internal_family": family,
                        "capacity": capacity,
                        "horizon": horizon,
                        "decision_count": 10,
                        "candidate_row_count": 10 * capacity,
                        "all_tied_decision_count": 5,
                        "discriminative_decision_count": 5,
                        "all_tied_fraction": 0.5,
                        "discriminative_fraction": 0.5,
                        "mean_optimal_set_fraction": 0.75,
                        "random_optimal_probability": 0.75,
                        "mean_random_regret": 0.1,
                    }
                )
    assert mod.validate_cells(rows)["status"] == "PASS"
    bad = rows + [dict(rows[0])]
    validation = mod.validate_cells(bad)
    assert validation["status"] == "FAIL"
    assert any("duplicate" in err for err in validation["errors"])


def test_macro_and_decision_micro_weighting_are_distinct():
    mod = load_module()
    rows = [
        {
            "family": "alibaba-block",
            "internal_family": "cloudphysics",
            "capacity": 32,
            "horizon": 16,
            "decision_count": 1,
            "candidate_row_count": 32,
            "all_tied_fraction": 1.0,
            "discriminative_fraction": 0.0,
            "mean_optimal_set_fraction": 1.0,
            "random_optimal_probability": 1.0,
            "mean_random_regret": 0.0,
        },
        {
            "family": "metacdn",
            "internal_family": "metacdn",
            "capacity": 32,
            "horizon": 16,
            "decision_count": 3,
            "candidate_row_count": 96,
            "all_tied_fraction": 0.0,
            "discriminative_fraction": 1.0,
            "mean_optimal_set_fraction": 0.5,
            "random_optimal_probability": 0.5,
            "mean_random_regret": 0.25,
        },
    ]
    micro = mod.summarize_micro(rows)[0]
    macro = mod.summarize_macro(rows)[0]
    assert micro["all_tied_fraction"] == 0.25
    assert macro["all_tied_fraction_mean"] == 0.5
    assert micro["mean_random_regret"] == 0.1875
    assert macro["mean_random_regret_mean"] == 0.125


def test_all_tied_plus_discriminative_consistency_is_enforced():
    mod = load_module()
    row = {
        "family": "alibaba-block",
        "internal_family": "cloudphysics",
        "capacity": 32,
        "horizon": 16,
        "decision_count": 10,
        "candidate_row_count": 320,
        "all_tied_fraction": 0.5,
        "discriminative_fraction": 0.6,
        "mean_optimal_set_fraction": 0.75,
        "random_optimal_probability": 0.75,
        "mean_random_regret": 0.1,
    }
    validation = mod.validate_cells([row])
    assert validation["status"] == "FAIL"
    assert any("all_tied + discriminative" in err for err in validation["errors"])


def test_deterministic_end_to_end_aggregation(tmp_path: Path):
    mod = load_module()
    h16 = tmp_path / "h16.csv"
    long_root = tmp_path / "long"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    h16_fields = [
        "trace_family",
        "capacity",
        "horizon",
        "n_decisions",
        "all_tied_fraction",
        "unique_winner_fraction",
        "mean_optimal_set_fraction",
        "median_optimal_set_fraction",
        "random_optimal_probability",
        "mean_random_regret",
    ]
    with h16.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=h16_fields)
        writer.writeheader()
        for family in mod.EXPECTED_INTERNAL_FAMILIES:
            for capacity in mod.EXPECTED_CAPACITIES:
                writer.writerow(
                    {
                        "trace_family": family,
                        "capacity": capacity,
                        "horizon": 16,
                        "n_decisions": 10,
                        "all_tied_fraction": 0.5,
                        "unique_winner_fraction": 0.0,
                        "mean_optimal_set_fraction": 0.75,
                        "median_optimal_set_fraction": 1.0,
                        "random_optimal_probability": 0.75,
                        "mean_random_regret": 0.1,
                    }
                )
    h16_report = tmp_path / "decision_view_report.json"
    h16_script = tmp_path / "script.py"
    h16_report.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    h16_script.write_text("print('adapter source')\n", encoding="utf-8")

    summaries = long_root / "summaries"
    manifests = long_root / "manifests"
    summaries.mkdir(parents=True)
    manifests.mkdir(parents=True)
    long_fields = [
        "family",
        "capacity",
        "horizon",
        "decisions",
        "candidate_rows",
        "all_tied_fraction",
        "discriminativeness",
        "mean_optimal_set_fraction",
        "random_optimal_probability",
        "mean_random_regret",
    ]
    with (summaries / "table_A_family_capacity_horizon.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=long_fields)
        writer.writeheader()
        for family in mod.EXPECTED_INTERNAL_FAMILIES:
            for capacity in mod.EXPECTED_CAPACITIES:
                for horizon in [32, 64, 128]:
                    writer.writerow(
                        {
                            "family": family,
                            "capacity": capacity,
                            "horizon": horizon,
                            "decisions": 10,
                            "candidate_rows": 10 * capacity,
                            "all_tied_fraction": 0.4,
                            "discriminativeness": 0.6,
                            "mean_optimal_set_fraction": 0.7,
                            "random_optimal_probability": 0.7,
                            "mean_random_regret": 0.2,
                        }
                    )
    (summaries / "table_B_horizon_pooled.json").write_text(json.dumps({"32": {}}), encoding="utf-8")
    (manifests / "campaign_manifest.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    args = ["--h16-source", str(h16), "--h16-report", str(h16_report), "--h16-script", str(h16_script), "--long-root", str(long_root)]
    assert mod.main([*args, "--out-dir", str(out1)]) == 0
    assert mod.main([*args, "--out-dir", str(out2)]) == 0
    assert (out1 / "combined_family_capacity_horizon.csv").read_text() == (
        out2 / "combined_family_capacity_horizon.csv"
    ).read_text()
    assert json.loads((out1 / "validation.json").read_text())["status"] == "PASS"
