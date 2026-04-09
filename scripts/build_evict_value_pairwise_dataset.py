from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_v2_rollout import build_pairwise_rows_from_candidate_rows


def _read_candidate_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows: List[Dict[str, object]] = []
    int_cols = {"request_t", "capacity", "horizon", "candidate_rank", "candidate_count"}
    float_cols = {
        "rollout_loss_h",
        "rollout_regret_h",
        "candidate_is_rollout_optimal",
        "request_bucket",
        "request_confidence",
        "candidate_bucket",
        "candidate_confidence",
        "candidate_conf_bucket_agreement",
        "candidate_is_farthest_bucket",
        "candidate_is_closest_bucket",
        "candidate_recency_rank",
        "cache_unique_bucket_count",
        "cache_bucket_mean",
        "cache_bucket_std",
        "recent_request_rate",
        "recent_hit_rate",
    }
    for row in raw:
        parsed: Dict[str, object] = dict(row)
        for c in int_cols:
            if c in parsed:
                parsed[c] = int(float(parsed[c]))
        for c in float_cols:
            if c in parsed:
                parsed[c] = float(parsed[c])
        rows.append(parsed)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pairwise decision-aligned eviction dataset")
    ap.add_argument("--candidate-csv", default="data/derived/evict_value_decision_aligned/candidate_rows.csv")
    ap.add_argument("--output-dir", default="data/derived/evict_value_pairwise")
    ap.add_argument("--include-ties", action="store_true")
    ap.add_argument("--max-rows", type=int, default=500000)
    ap.add_argument("--sample-seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.sample_seed)
    candidate_rows = _read_candidate_rows(Path(args.candidate_csv))
    pairwise_rows = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=args.include_ties)
    rng.shuffle(pairwise_rows)
    if len(pairwise_rows) > args.max_rows:
        pairwise_rows = pairwise_rows[: args.max_rows]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise_csv = out_dir / "pairwise_rows.csv"
    _write_csv(pairwise_csv, pairwise_rows)

    summary = {
        "candidate_csv": args.candidate_csv,
        "rows_total": len(pairwise_rows),
        "decision_count": len({str(r["decision_id"]) for r in pairwise_rows}),
        "include_ties": args.include_ties,
        "sample_seed": args.sample_seed,
        "max_rows": args.max_rows,
        "output_pairwise_csv": str(pairwise_csv),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
