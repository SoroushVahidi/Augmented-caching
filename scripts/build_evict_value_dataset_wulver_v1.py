from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_wulver_v1 import (
    WulverDatasetConfig,
    dataset_columns,
    iter_candidate_rows,
    load_trace_from_any,
    materialize_summary,
    parse_trace_manifest,
    update_summary_maps,
)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Wulver-scale evict_value_v1 candidate dataset shards.")
    ap.add_argument("--trace-manifest", default=None, help="CSV with path,trace_name,dataset_source,trace_family")
    ap.add_argument("--trace-glob", action="append", default=["data/processed/*/trace.jsonl"], help="Glob(s) used when manifest is absent.")
    ap.add_argument("--capacities", default="64,128,256")
    ap.add_argument("--horizons", default="8,16,32")
    ap.add_argument("--split-mode", default="trace_chunk", choices=["trace_chunk", "source_family"])
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--split-train-pct", type=int, default=70)
    ap.add_argument("--split-val-pct", type=int, default=15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--history-window", type=int, default=64)
    ap.add_argument("--max-rows-per-shard", type=int, default=500000)
    ap.add_argument("--max-traces", type=int, default=None)
    ap.add_argument("--min-requests-per-trace", type=int, default=1000)
    ap.add_argument("--min-unique-pages-per-trace", type=int, default=100)
    ap.add_argument("--allow-tiny-traces", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument(
        "--max-requests-per-trace",
        type=int,
        default=None,
        help="Optional cap on request length per trace (prefix) to bound CPU for multi-family runs.",
    )
    ap.add_argument("--out-dir", default="data/derived/evict_value_v1_wulver")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    cfg = WulverDatasetConfig(
        horizons=horizons,
        history_window=args.history_window,
        split_mode=args.split_mode,
        chunk_size=args.chunk_size,
        split_train_pct=args.split_train_pct,
        split_val_pct=args.split_val_pct,
        split_seed=args.split_seed,
    )
    specs = parse_trace_manifest(args.trace_manifest, args.trace_glob)
    if args.max_traces is not None:
        specs = specs[: args.max_traces]
    if not specs:
        raise ValueError("No traces discovered. Provide --trace-manifest or --trace-glob.")

    trace_stats: List[Dict[str, object]] = []
    valid_specs = []
    for spec in specs:
        reqs, _pages, inferred_source = load_trace_from_any(spec.path)
        req_count = len(reqs)
        unique_pages = len({r.page_id for r in reqs})
        supports_any_capacity = any((req_count > c and unique_pages > c) for c in capacities)
        is_tiny = (
            req_count < args.min_requests_per_trace
            or unique_pages < args.min_unique_pages_per_trace
            or not supports_any_capacity
        )
        trace_stats.append(
            {
                "path": spec.path,
                "trace_name": spec.trace_name,
                "dataset_source": spec.dataset_source if spec.dataset_source != "unknown" else inferred_source,
                "request_count": req_count,
                "unique_page_count": unique_pages,
                "supports_any_capacity": supports_any_capacity,
                "is_tiny": is_tiny,
            }
        )
        if not is_tiny:
            valid_specs.append(spec)

    print(
        f"[preflight] discovered={len(specs)} valid={len(valid_specs)} "
        f"min_requests={args.min_requests_per_trace} min_unique_pages={args.min_unique_pages_per_trace}"
    )
    for stat in sorted(trace_stats, key=lambda x: int(x["request_count"]), reverse=True)[:20]:
        print(
            "[preflight] "
            f"path={stat['path']} requests={stat['request_count']} unique_pages={stat['unique_page_count']} "
            f"supports_capacity={stat['supports_any_capacity']} tiny={stat['is_tiny']}"
        )

    if not valid_specs and not args.allow_tiny_traces:
        raise ValueError(
            "Only tiny/insufficient traces were discovered. "
            "Provide real processed traces via --trace-manifest/--trace-glob, or override with --allow-tiny-traces."
        )
    if valid_specs:
        specs = valid_specs
    if args.preflight_only:
        return

    out_root = Path(args.out_dir)
    shards_dir = out_root / "shards"
    logs_dir = out_root / "logs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = dataset_columns()

    rows_by_key: Dict[str, int] = {}
    decisions_by_key: Dict[str, set[str]] = {}
    manifest_items: List[Dict[str, object]] = []

    for spec in specs:
        for cap in capacities:
            safe_trace = spec.trace_name.replace("/", "__").replace(":", "_")
            shard_prefix = f"{safe_trace}__cap{cap}"
            done_marker = logs_dir / f"{shard_prefix}.done.json"
            if done_marker.exists() and not args.overwrite:
                print(f"[skip] {shard_prefix} already completed")
                payload = json.loads(done_marker.read_text(encoding="utf-8"))
                manifest_items.extend(list(payload.get("shards", [])))
                continue

            reqs, _pages, inferred_source = load_trace_from_any(spec.path)
            if args.max_requests_per_trace is not None:
                reqs = reqs[: args.max_requests_per_trace]
            dataset_source = spec.dataset_source if spec.dataset_source != "unknown" else inferred_source
            trace_family = spec.trace_family if spec.trace_family != "unknown" else dataset_source
            shard_rows: List[Dict[str, object]] = []
            shard_index = 0
            built_shards: List[Dict[str, object]] = []

            for row in iter_candidate_rows(
                requests=reqs,
                capacity=cap,
                trace_name=spec.trace_name,
                dataset_source=dataset_source,
                trace_family=trace_family,
                cfg=cfg,
            ):
                update_summary_maps(row, rows_by_key=rows_by_key, decisions_by_key=decisions_by_key)
                shard_rows.append(row)
                if len(shard_rows) >= args.max_rows_per_shard:
                    shard_path = shards_dir / f"{shard_prefix}.part{shard_index:04d}.csv"
                    _write_csv(shard_path, shard_rows, fieldnames)
                    built_shards.append({"path": str(shard_path), "row_count": len(shard_rows)})
                    shard_rows = []
                    shard_index += 1

            if shard_rows:
                shard_path = shards_dir / f"{shard_prefix}.part{shard_index:04d}.csv"
                _write_csv(shard_path, shard_rows, fieldnames)
                built_shards.append({"path": str(shard_path), "row_count": len(shard_rows)})

            done_payload = {
                "trace_path": spec.path,
                "trace_name": spec.trace_name,
                "dataset_source": dataset_source,
                "trace_family": trace_family,
                "capacity": cap,
                "shards": built_shards,
            }
            done_marker.write_text(json.dumps(done_payload, indent=2), encoding="utf-8")
            manifest_items.extend(built_shards)
            print(f"[done] {shard_prefix} shards={len(built_shards)}")

    summary_rows = materialize_summary(rows_by_key=rows_by_key, decisions_by_key=decisions_by_key)
    _write_csv(out_root / "split_summary.csv", summary_rows, ["split", "trace_family", "capacity", "horizon", "row_count", "decision_count"])

    manifest = {
        "format": "evict_value_v1_wulver_candidate_csv_shards",
        "split_mode": args.split_mode,
        "chunk_size": args.chunk_size,
        "capacities": capacities,
        "horizons": list(horizons),
        "max_requests_per_trace": args.max_requests_per_trace,
        "trace_count": len(specs),
        "shard_count": len(manifest_items),
        "preflight": {
            "min_requests_per_trace": args.min_requests_per_trace,
            "min_unique_pages_per_trace": args.min_unique_pages_per_trace,
            "trace_stats": trace_stats,
        },
        "shards": manifest_items,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest={out_root / 'manifest.json'} summary={out_root / 'split_summary.csv'}")


if __name__ == "__main__":
    main()
