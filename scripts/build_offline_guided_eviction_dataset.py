from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List

from lafc.offline.trace_inputs import load_trace_with_sizes
from lafc.offline_teacher_supervision import (
    OfflineTeacherLabelConfig,
    build_offline_teacher_candidate_rows,
    build_offline_teacher_pairwise_rows,
)
from lafc.simulator.request_trace import load_trace


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_trace_paths(trace_glob: str) -> List[str]:
    patterns = [p.strip() for p in trace_glob.split(",") if p.strip()]
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob(pattern)))
    out = sorted(set(paths))
    if not out:
        raise ValueError(f"No traces matched --trace-glob={trace_glob}")
    return out


def _trace_family(path: str) -> str:
    p = Path(path)
    return p.parent.name if p.parent.name else "unknown"


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count_decisions(rows: Iterable[Dict[str, object]]) -> int:
    return len({str(r["decision_id"]) for r in rows})


def _count_teacher(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        key = str(r.get("teacher_type", "unknown"))
        out[key] = out.get(key, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build offline-teacher supervised eviction dataset")
    ap.add_argument("--trace-glob", default="data/example_*.json,data/example_general_caching.json")
    ap.add_argument("--capacities", default="2,3")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--output-dir", default="data/derived/offline_teacher_supervision")
    args = ap.parse_args()

    capacities = _parse_csv_ints(args.capacities)
    cfg = OfflineTeacherLabelConfig(horizon=args.horizon)

    rows: List[Dict[str, object]] = []
    trace_paths = _resolve_trace_paths(args.trace_glob)
    for trace_path in trace_paths:
        family = _trace_family(trace_path)
        try:
            requests, pages, sizes = load_trace_with_sizes(trace_path)
        except Exception:
            requests, pages = load_trace(trace_path)
            sizes = {pid: 1.0 for pid in pages}

        if args.max_requests_per_trace > 0:
            requests = requests[: args.max_requests_per_trace]

        for cap in capacities:
            chunk = build_offline_teacher_candidate_rows(
                requests=requests,
                pages=pages,
                page_sizes=sizes,
                capacity=float(cap),
                trace_name=trace_path,
                trace_family=family,
                cfg=cfg,
            )
            rows.extend(chunk)
            if len(rows) >= args.max_rows:
                rows = rows[: args.max_rows]
                break
        if len(rows) >= args.max_rows:
            break

    pairwise_rows = build_offline_teacher_pairwise_rows(rows, include_ties=cfg.include_pairwise_ties)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "candidate_rows.csv", rows)
    if pairwise_rows:
        _write_csv(out / "pairwise_rows.csv", pairwise_rows)

    summary = {
        "trace_glob": args.trace_glob,
        "capacities": capacities,
        "horizon": args.horizon,
        "rows_total": len(rows),
        "decisions_total": _count_decisions(rows),
        "pairwise_rows_total": len(pairwise_rows),
        "teacher_row_counts": _count_teacher(rows),
        "outputs": {
            "candidate_rows_csv": str(out / "candidate_rows.csv"),
            "pairwise_rows_csv": str(out / "pairwise_rows.csv"),
        },
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
