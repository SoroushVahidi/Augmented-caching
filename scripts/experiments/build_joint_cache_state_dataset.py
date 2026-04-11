from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from lafc.experiments.joint_cache_state_dataset import (
    JointCacheStateDatasetConfig,
    build_joint_cache_state_examples,
    to_jsonl_lines,
)
from lafc.simulator.request_trace import load_trace


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_paths(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = to_jsonl_lines(rows)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_report(summary: Dict[str, object], sample_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Joint cache-state decision dataset (experimental)")
    lines.append("")
    lines.append("This run constructs decision-state examples only (no model training).")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total_decisions: {summary['total_decisions']}")
    lines.append(f"- capacities: {summary['capacities']}")
    lines.append(f"- horizon: {summary['horizon']}")
    lines.append(f"- traces: {len(summary['trace_paths'])}")
    lines.append(f"- split_counts: {summary['split_counts']}")
    lines.append(f"- mean_candidates_per_decision: {summary['mean_candidates_per_decision']:.3f}")
    lines.append("")
    lines.append("## Sample decisions")
    for row in sample_rows:
        lines.append(
            f"- `{row['decision_id']}` incoming={row['incoming_request']['page_id']} "
            f"oracle_victim={row['oracle_victim']} candidates={row['cache_residents']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build experimental joint cache-state decision dataset (data construction only)")
    ap.add_argument(
        "--trace-paths",
        default="data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json",
        help="Comma-separated repository trace JSON files",
    )
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--history-window", type=int, default=64)
    ap.add_argument("--max-decisions", type=int, default=0)
    ap.add_argument("--out-data-dir", default="data/derived/experiments/joint_cache_state_decision_v1")
    ap.add_argument("--out-analysis-dir", default="analysis/joint_cache_state_decision_v1")
    args = ap.parse_args()

    cfg = JointCacheStateDatasetConfig(horizon=int(args.horizon), history_window=int(args.history_window))
    capacities = _parse_csv_ints(args.capacities)
    trace_paths = _parse_csv_paths(args.trace_paths)

    rows: List[Dict[str, object]] = []
    trace_counts: Dict[str, int] = {}
    for trace_path in trace_paths:
        requests, _pages = load_trace(trace_path)
        before = len(rows)
        for cap in capacities:
            rows.extend(
                build_joint_cache_state_examples(requests=requests, capacity=cap, trace_name=trace_path, cfg=cfg)
            )
            if args.max_decisions > 0 and len(rows) >= args.max_decisions:
                rows = rows[: args.max_decisions]
                break
        trace_counts[trace_path] = len(rows) - before
        if args.max_decisions > 0 and len(rows) >= args.max_decisions:
            break

    split_counts = {
        "train": sum(1 for r in rows if str(r.get("split")) == "train"),
        "val": sum(1 for r in rows if str(r.get("split")) == "val"),
        "test": sum(1 for r in rows if str(r.get("split")) == "test"),
    }
    candidates_per_decision = [len(r.get("cache_residents", [])) for r in rows]

    data_dir = Path(args.out_data_dir)
    analysis_dir = Path(args.out_analysis_dir)
    _write_jsonl(data_dir / "decision_states.jsonl", rows)

    summary: Dict[str, object] = {
        "total_decisions": len(rows),
        "trace_paths": trace_paths,
        "trace_decision_counts": trace_counts,
        "capacities": capacities,
        "horizon": int(args.horizon),
        "history_window": int(args.history_window),
        "split_counts": split_counts,
        "mean_candidates_per_decision": float(mean(candidates_per_decision)) if candidates_per_decision else 0.0,
        "outputs": {
            "decision_states_jsonl": str(data_dir / "decision_states.jsonl"),
            "report_md": str(analysis_dir / "report.md"),
            "dataset_summary_json": str(analysis_dir / "dataset_summary.json"),
            "sample_json": str(analysis_dir / "sample_decisions.json"),
        },
        "note": "Experimental/non-canonical artifact; no training executed.",
    }

    sample_rows = rows[: min(5, len(rows))]
    _write_json(analysis_dir / "dataset_summary.json", summary)
    _write_json(analysis_dir / "sample_decisions.json", sample_rows)
    (analysis_dir / "report.md").write_text(_build_report(summary, sample_rows), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
