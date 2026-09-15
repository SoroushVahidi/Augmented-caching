"""Validate ONE completed PE long-horizon production task (family x capacity).

Out-of-core: streams each shard CSV row-by-row (stdlib csv, Welford-style
running stats) rather than loading the whole cell into memory -- this
matters at production scale (cap256 alone was measured at ~9-12GB per
family; a full task directory can be larger).

Usage:
    python scripts/pe_long_horizon/validate_cell.py <task_out_dir> \
        --family cloudphysics --capacity 32 --horizons 32,64,128 \
        --expected-sha 3a04a25094de1e8c57a603e2662b321bcf1d9e88

Writes <task_out_dir>/validated_summary.json (one compact summary per cell,
keyed by horizon) and prints VALIDATION_RESULT: PASS/FAIL. Never mutates or
deletes raw shard data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_BASE_COLUMNS = [
    "trace_name", "trace_family", "dataset_source", "capacity", "horizon",
    "decision_id", "decision_t", "decision_chunk_id", "candidate_page_id",
    "split", "y_loss", "y_value",
]


class RunningStats:
    """Streaming mean/std (Welford) -- avoids holding all y_loss values in memory."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_v = math.inf
        self.max_v = -math.inf

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.m2 += d * d2
        self.min_v = min(self.min_v, x)
        self.max_v = max(self.max_v, x)

    def as_dict(self) -> dict:
        var = self.m2 / self.n if self.n else 0.0
        return {
            "mean": self.mean if self.n else None,
            "std": math.sqrt(var) if self.n else None,
            "min": self.min_v if self.n else None,
            "max": self.max_v if self.n else None,
            "n": self.n,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_out_dir")
    ap.add_argument("--family", required=True)
    ap.add_argument("--capacity", type=int, required=True)
    ap.add_argument("--horizons", default="32,64,128")
    ap.add_argument("--expected-sha", default=None)
    args = ap.parse_args()

    out_dir = Path(args.task_out_dir)
    expected_horizons = sorted(int(x) for x in args.horizons.split(","))

    completion_marker = out_dir / "COMPLETE.json"
    if not out_dir.exists():
        print(f"FAIL: output dir does not exist: {out_dir}")
        return 1
    if not completion_marker.exists():
        print(f"FAIL: completion marker missing: {completion_marker}")
        return 1

    provenance = json.loads(completion_marker.read_text(encoding="utf-8"))
    if args.expected_sha and provenance.get("git_sha") != args.expected_sha:
        print(f"FAIL: provenance SHA {provenance.get('git_sha')} != expected {args.expected_sha}")
        return 1
    if provenance.get("family") != args.family:
        print(f"FAIL: provenance family {provenance.get('family')} != expected {args.family}")
        return 1
    if int(provenance.get("capacity", -1)) != args.capacity:
        print(f"FAIL: provenance capacity {provenance.get('capacity')} != expected {args.capacity}")
        return 1

    shards_dir = out_dir / "shards"
    shards = sorted(shards_dir.glob("*.csv"))
    if not shards:
        print(f"FAIL: no shard CSVs under {shards_dir}")
        return 1

    seen_keys: set[tuple[str, str, int]] = set()
    dup_count = 0
    schema_ok = True
    y_value_mismatches = 0
    nan_inf_count = 0
    malformed_key_count = 0
    row_count_by_horizon: dict[int, int] = defaultdict(int)
    decisions_by_horizon: dict[int, set[str]] = defaultdict(set)
    losses_by_decision: dict[tuple[int, str], list[float]] = defaultdict(list)
    stats_by_horizon: dict[int, RunningStats] = defaultdict(RunningStats)
    sha256 = hashlib.sha256()
    total_bytes = 0

    for shard in shards:
        total_bytes += shard.stat().st_size
        with shard.open("rb") as fh_bytes:
            for chunk in iter(lambda: fh_bytes.read(1 << 20), b""):
                sha256.update(chunk)
        with shard.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            for col in EXPECTED_BASE_COLUMNS:
                if col not in fieldnames:
                    print(f"FAIL: shard {shard.name} missing column {col}")
                    schema_ok = False
            for row in reader:
                try:
                    h = int(row["horizon"])
                    did = row["decision_id"]
                    cand = row["candidate_page_id"]
                    y_loss = float(row["y_loss"])
                    y_value = float(row["y_value"])
                except (KeyError, ValueError):
                    malformed_key_count += 1
                    continue

                if not did or not cand:
                    malformed_key_count += 1
                    continue
                if math.isnan(y_loss) or math.isinf(y_loss) or math.isnan(y_value) or math.isinf(y_value):
                    nan_inf_count += 1
                    continue

                key = (did, cand, h)
                if key in seen_keys:
                    dup_count += 1
                else:
                    seen_keys.add(key)

                if abs(y_value - (-y_loss)) > 1e-9:
                    y_value_mismatches += 1

                row_count_by_horizon[h] += 1
                decisions_by_horizon[h].add(did)
                losses_by_decision[(h, did)].append(y_loss)
                stats_by_horizon[h].push(y_loss)

    unexpected_horizons = sorted(set(row_count_by_horizon) - set(expected_horizons))
    missing_horizons = sorted(set(expected_horizons) - set(row_count_by_horizon))
    all_decision_ids = set().union(*decisions_by_horizon.values()) if decisions_by_horizon else set()
    decision_horizons_seen: dict[str, set[int]] = defaultdict(set)
    for h, decs in decisions_by_horizon.items():
        for d in decs:
            decision_horizons_seen[d].add(h)
    decisions_missing_a_horizon = [
        d for d in all_decision_ids if decision_horizons_seen[d] != set(expected_horizons)
    ]

    per_horizon_summary = {}
    for h in expected_horizons:
        decs = decisions_by_horizon.get(h, set())
        total_decs = len(decs)
        candidate_rows = row_count_by_horizon.get(h, 0)
        tied_decs = 0
        opt_fractions = []
        random_regrets = []
        total_optimal_rows = 0
        for did in decs:
            losses = losses_by_decision[(h, did)]
            n = len(losses)
            mn, mx = min(losses), max(losses)
            if mn == mx:
                tied_decs += 1
            n_opt = sum(1 for v in losses if v == mn)
            opt_fractions.append(n_opt / n)
            total_optimal_rows += n_opt
            random_regrets.append(sum(losses) / n - mn)

        all_tied_fraction = tied_decs / total_decs if total_decs else 0.0
        per_horizon_summary[str(h)] = {
            "candidate_rows": candidate_rows,
            "decisions": total_decs,
            "all_tied_fraction": all_tied_fraction,
            "discriminativeness": 1.0 - all_tied_fraction,
            "mean_optimal_set_fraction": sum(opt_fractions) / len(opt_fractions) if opt_fractions else 0.0,
            "random_optimal_probability": total_optimal_rows / candidate_rows if candidate_rows else 0.0,
            "mean_random_regret": sum(random_regrets) / len(random_regrets) if random_regrets else 0.0,
            "y_loss": stats_by_horizon[h].as_dict(),
        }

    ok = (
        schema_ok
        and dup_count == 0
        and y_value_mismatches == 0
        and nan_inf_count == 0
        and malformed_key_count == 0
        and not unexpected_horizons
        and not missing_horizons
        and not decisions_missing_a_horizon
    )

    summary = {
        "task_id": provenance.get("task_id"),
        "family": args.family,
        "capacity": args.capacity,
        "expected_horizons": expected_horizons,
        "schema_ok": schema_ok,
        "duplicate_candidate_keys": dup_count,
        "malformed_key_rows_skipped": malformed_key_count,
        "nan_inf_rows_skipped": nan_inf_count,
        "y_value_eq_neg_y_loss_violations": y_value_mismatches,
        "unexpected_horizons": unexpected_horizons,
        "missing_horizons": missing_horizons,
        "decisions_missing_a_horizon": len(decisions_missing_a_horizon),
        "output_bytes": total_bytes,
        "output_sha256": sha256.hexdigest(),
        "provenance": provenance,
        "per_horizon": per_horizon_summary,
        "validation_result": "PASS" if ok else "FAIL",
    }

    out_path = out_dir / "validated_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("VALIDATION_RESULT:", summary["validation_result"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
