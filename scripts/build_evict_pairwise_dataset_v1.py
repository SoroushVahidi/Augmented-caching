from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List

from lafc.evict_decision_aligned_v1 import DecisionAlignedEvictConfig, build_evict_pairwise_examples_v1
from lafc.evict_value_dataset_v1 import _split_by_trace_and_capacity
from lafc.simulator.request_trace import load_trace


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_trace_paths(trace_glob: str) -> List[str]:
    patterns = [p.strip() for p in trace_glob.split(",") if p.strip()]
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob(pattern)))
    deduped = sorted(set(paths))
    if not deduped:
        raise ValueError(f"No traces matched --trace-glob={trace_glob}")
    return deduped


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count_decisions(rows: Iterable[Dict[str, object]]) -> int:
    return len({str(r["decision_id"]) for r in rows})


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pairwise eviction dataset (v1)")
    ap.add_argument("--trace-glob", default="data/example_*.json")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=300000)
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--include-ties", action="store_true")
    ap.add_argument("--output-dir", default="data/derived")
    args = ap.parse_args()

    cfg = DecisionAlignedEvictConfig(horizon=args.horizon, include_ties=args.include_ties)
    capacities = _parse_csv_ints(args.capacities)
    trace_paths = _resolve_trace_paths(args.trace_glob)

    rows: List[Dict[str, object]] = []
    trace_row_counts: Dict[str, int] = {}
    for trace_path in trace_paths:
        requests, _pages = load_trace(trace_path)
        if args.max_requests_per_trace > 0:
            requests = requests[: args.max_requests_per_trace]

        before = len(rows)
        for capacity in capacities:
            rows.extend(
                build_evict_pairwise_examples_v1(
                    requests=requests,
                    capacity=capacity,
                    trace_name=trace_path,
                    cfg=cfg,
                )
            )
            if len(rows) >= args.max_rows:
                rows = rows[: args.max_rows]
                break
        trace_row_counts[trace_path] = len(rows) - before
        if len(rows) >= args.max_rows:
            break

    splits = _split_by_trace_and_capacity(rows)
    out = Path(args.output_dir)
    _write_csv(out / "evict_pairwise_v1_train.csv", splits["train"] or rows[:1])
    _write_csv(out / "evict_pairwise_v1_val.csv", splits["val"] or rows[:1])
    _write_csv(out / "evict_pairwise_v1_test.csv", splits["test"] or rows[:1])

    summary = {
        "dataset": "evict_pairwise_v1",
        "horizon": args.horizon,
        "capacities": capacities,
        "trace_glob": args.trace_glob,
        "include_ties": args.include_ties,
        "rows_total": len(rows),
        "decisions_total": _count_decisions(rows),
        "rows_by_split": {k: len(v) for k, v in splits.items()},
        "decisions_by_split": {k: _count_decisions(v) for k, v in splits.items()},
        "trace_row_counts": trace_row_counts,
    }
    (out / "evict_pairwise_v1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
