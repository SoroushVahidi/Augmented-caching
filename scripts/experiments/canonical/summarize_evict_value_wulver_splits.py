from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_wulver_v1 import assign_split, materialize_summary, update_summary_maps


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize or re-partition Wulver evict_value_v1 dataset shards.")
    ap.add_argument("--manifest", required=True, help="Path to manifest.json from dataset generation.")
    ap.add_argument("--split-mode", default=None, choices=["trace_chunk", "source_family"], help="If set, reassign split labels before summary.")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--split-train-pct", type=int, default=70)
    ap.add_argument("--split-val-pct", type=int, default=15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out-summary-csv", default=None)
    ap.add_argument("--write-partitioned-dir", default=None, help="Optional output dir to write split-specific shards.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths = [Path(s["path"]) for s in payload.get("shards", [])]

    rows_by_key: Dict[str, int] = {}
    decisions_by_key: Dict[str, set[str]] = {}
    split_buffers: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}

    for spath in shard_paths:
        for row in _read_rows(spath):
            if args.split_mode:
                row["split"] = assign_split(
                    split_mode=args.split_mode,
                    trace_name=str(row["trace_name"]),
                    dataset_source=str(row["dataset_source"]),
                    trace_family=str(row["trace_family"]),
                    t=int(row["decision_t"]),
                    chunk_size=args.chunk_size,
                    train_pct=args.split_train_pct,
                    val_pct=args.split_val_pct,
                    seed=args.split_seed,
                )
            update_summary_maps(row, rows_by_key=rows_by_key, decisions_by_key=decisions_by_key)
            if args.write_partitioned_dir:
                split_buffers[str(row["split"])].append(row)

    if args.write_partitioned_dir:
        out_root = Path(args.write_partitioned_dir)
        for split, buf in split_buffers.items():
            if buf:
                _write_rows(out_root / f"evict_value_v1_wulver_{split}.csv", buf)

    summary_rows = materialize_summary(rows_by_key=rows_by_key, decisions_by_key=decisions_by_key)
    out_summary = Path(args.out_summary_csv) if args.out_summary_csv else manifest_path.parent / "split_summary_recomputed.csv"
    _write_rows(out_summary, summary_rows)
    print(f"Wrote split summary: {out_summary}")


if __name__ == "__main__":
    main()
