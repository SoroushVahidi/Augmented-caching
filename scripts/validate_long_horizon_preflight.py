"""Scientific validation for a long-horizon candidate-row preflight output directory.

Stdlib-only (csv/json), deliberately avoids pandas so it runs unmodified in
Wulver's dataset-generation venv (which does not carry pandas). Reads every
CSV shard, groups rows by (horizon, decision_id), and reports per-horizon
correctness/discriminativeness/regret metrics plus schema/duplicate/
missing-key checks.

Usage:
    python scripts/validate_long_horizon_preflight.py <out_dir> [--horizons 16,32,64,128]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_BASE_COLUMNS = [
    "trace_name",
    "trace_family",
    "dataset_source",
    "capacity",
    "horizon",
    "decision_id",
    "decision_t",
    "decision_chunk_id",
    "candidate_page_id",
    "split",
    "y_loss",
    "y_value",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--horizons", default="16,32,64,128")
    args = ap.parse_args()

    expected_horizons = sorted(int(x) for x in args.horizons.split(","))
    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    shards = sorted(shards_dir.glob("*.csv"))
    if not shards:
        print(f"FAIL: no shard CSVs found under {shards_dir}")
        return 1

    seen_keys: set[tuple[str, str, str]] = set()
    dup_count = 0
    schema_ok = True
    y_value_mismatches = 0
    row_count_by_horizon: dict[int, int] = defaultdict(int)
    losses_by_decision: dict[tuple[int, str], list[float]] = defaultdict(list)
    decisions_by_horizon: dict[int, set[str]] = defaultdict(set)
    all_decision_ids: set[str] = set()
    decision_horizons_seen: dict[str, set[int]] = defaultdict(set)

    for shard in shards:
        with shard.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            for col in EXPECTED_BASE_COLUMNS:
                if col not in fieldnames:
                    print(f"FAIL: shard {shard.name} missing expected column {col}")
                    schema_ok = False
            for row in reader:
                h = int(row["horizon"])
                did = row["decision_id"]
                cand = row["candidate_page_id"]
                y_loss = float(row["y_loss"])
                y_value = float(row["y_value"])

                key = (did, cand, str(h))
                if key in seen_keys:
                    dup_count += 1
                else:
                    seen_keys.add(key)

                if abs(y_value - (-y_loss)) > 1e-9:
                    y_value_mismatches += 1

                row_count_by_horizon[h] += 1
                losses_by_decision[(h, did)].append(y_loss)
                decisions_by_horizon[h].add(did)
                all_decision_ids.add(did)
                decision_horizons_seen[did].add(h)

    unexpected_horizons = sorted(set(row_count_by_horizon) - set(expected_horizons))
    missing_horizons = sorted(set(expected_horizons) - set(row_count_by_horizon))

    # No silently dropped horizon per decision: every decision_id must have
    # rows at every expected horizon.
    decisions_missing_a_horizon = [
        did for did in all_decision_ids if decision_horizons_seen[did] != set(expected_horizons)
    ]

    print("=== SCHEMA / INTEGRITY ===")
    print(f"schema_ok={schema_ok}")
    print(f"duplicate_candidate_keys={dup_count}")
    print(f"y_value_eq_neg_y_loss_violations={y_value_mismatches}")
    print(f"unexpected_horizons={unexpected_horizons}")
    print(f"missing_horizons={missing_horizons}")
    print(f"decisions_missing_a_horizon={len(decisions_missing_a_horizon)} (of {len(all_decision_ids)} total decisions)")

    print()
    print("=== PER-HORIZON SCIENTIFIC METRICS ===")
    results = {}
    for h in expected_horizons:
        decs = decisions_by_horizon.get(h, set())
        total_decs = len(decs)
        candidate_rows = row_count_by_horizon.get(h, 0)

        tied_decs = 0
        optimal_set_fractions = []
        random_regrets = []
        for did in decs:
            losses = losses_by_decision[(h, did)]
            n = len(losses)
            mn = min(losses)
            mx = max(losses)
            if mn == mx:
                tied_decs += 1
            n_optimal = sum(1 for v in losses if v == mn)
            optimal_set_fractions.append(n_optimal / n)
            mean_loss = sum(losses) / n
            random_regrets.append(mean_loss - mn)

        all_tied_fraction = tied_decs / total_decs if total_decs else 0.0
        discriminativeness = 1.0 - all_tied_fraction
        mean_optimal_set_fraction = (
            sum(optimal_set_fractions) / len(optimal_set_fractions) if optimal_set_fractions else 0.0
        )
        # random-optimal probability: probability a uniformly-random candidate
        # choice hits an optimal candidate, pooled over all rows (row-weighted).
        total_optimal_rows = sum(
            sum(1 for v in losses_by_decision[(h, did)] if v == min(losses_by_decision[(h, did)]))
            for did in decs
        )
        random_optimal_probability = total_optimal_rows / candidate_rows if candidate_rows else 0.0
        mean_random_regret = sum(random_regrets) / len(random_regrets) if random_regrets else 0.0

        results[h] = {
            "candidate_rows": candidate_rows,
            "decisions": total_decs,
            "all_tied_fraction": all_tied_fraction,
            "discriminativeness": discriminativeness,
            "mean_optimal_set_fraction": mean_optimal_set_fraction,
            "random_optimal_probability": random_optimal_probability,
            "mean_random_regret": mean_random_regret,
        }
        print(
            f"H={h:<4} rows={candidate_rows:<8} decisions={total_decs:<7} "
            f"all_tied_fraction={all_tied_fraction:.4f} discriminativeness={discriminativeness:.4f} "
            f"mean_optimal_set_fraction={mean_optimal_set_fraction:.4f} "
            f"random_optimal_probability={random_optimal_probability:.4f} "
            f"mean_random_regret={mean_random_regret:.4f}"
        )

    out_path = out_dir / "preflight_validation_report.json"
    report = {
        "schema_ok": schema_ok,
        "duplicate_candidate_keys": dup_count,
        "y_value_eq_neg_y_loss_violations": y_value_mismatches,
        "unexpected_horizons": unexpected_horizons,
        "missing_horizons": missing_horizons,
        "decisions_missing_a_horizon": len(decisions_missing_a_horizon),
        "per_horizon": results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")

    ok = (
        schema_ok
        and dup_count == 0
        and y_value_mismatches == 0
        and not unexpected_horizons
        and not missing_horizons
        and not decisions_missing_a_horizon
    )
    print()
    print("VALIDATION_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
