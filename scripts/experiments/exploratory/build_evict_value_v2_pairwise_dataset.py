from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_v2_rollout import build_pairwise_rows_from_candidate_rows


def _read_candidate_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    out: List[Dict[str, object]] = []
    for row in raw:
        parsed: Dict[str, object] = dict(row)
        for key in [
            "request_t",
            "capacity",
            "horizon",
            "candidate_rank",
            "candidate_count",
            "request_bucket",
            "candidate_bucket",
            "candidate_recency_rank",
            "cache_unique_bucket_count",
        ]:
            if key in parsed:
                parsed[key] = int(float(parsed[key]))
        for key, value in list(parsed.items()):
            if key.startswith("candidate_") or key.startswith("cache_") or key.startswith("request_") or key.startswith("recent_"):
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    pass
        for k in ["rollout_loss_h", "rollout_regret_h", "candidate_is_rollout_optimal"]:
            parsed[k] = float(parsed[k])
        out.append(parsed)
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pairwise ranking dataset from rollout-labeled candidate rows")
    ap.add_argument("--candidate-csv", default="data/derived/evict_value_v2_rollout/candidate_rows.csv")
    ap.add_argument("--output-dir", default="data/derived/evict_value_v2_pairwise")
    ap.add_argument("--include-ties", action="store_true")
    args = ap.parse_args()

    candidate_rows = _read_candidate_rows(Path(args.candidate_csv))
    pairwise_rows = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=args.include_ties)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairwise_csv = out_dir / "pairwise_rows.csv"
    _write_csv(pairwise_csv, pairwise_rows)

    summary = {
        "candidate_csv": args.candidate_csv,
        "rows_total": len(pairwise_rows),
        "decision_count": len({str(r["decision_id"]) for r in pairwise_rows}),
        "include_ties": args.include_ties,
        "output_pairwise_csv": str(pairwise_csv),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
