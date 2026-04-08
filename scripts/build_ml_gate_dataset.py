from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from lafc.learned_gate.dataset import GateDatasetConfig, build_gate_examples, split_by_trace
from lafc.simulator.request_trace import load_trace

DEFAULT_TRACES = [
    "data/example_unweighted.json",
    "data/example_atlas_v1.json",
    "data/examples/brightkite_sample.jsonl",
]


def _discover_traces() -> List[str]:
    found = ["data/example_unweighted.json", "data/example_atlas_v1.json"]
    for p in Path("data/examples").glob("*.jsonl"):
        found.append(str(p))
    return sorted(set(found))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learned-gate v1 training dataset")
    parser.add_argument("--capacities", default="2,3,4")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--regret-window", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--out-dir", default="data/derived")
    args = parser.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    cfg = GateDatasetConfig(horizon=args.horizon, regret_window=args.regret_window)

    traces = _discover_traces() if not args.sample_only else ["data/example_unweighted.json", "data/example_atlas_v1.json"]
    rows: List[Dict[str, object]] = []

    for trace in traces:
        if not Path(trace).exists():
            continue
        if Path(trace).suffix.lower() != ".json":
            continue
        requests, _ = load_trace(trace)
        for cap in capacities:
            rows.extend(build_gate_examples(requests=requests, capacity=cap, cfg=cfg, trace_name=trace))
            if len(rows) >= args.max_rows:
                rows = rows[: args.max_rows]
                break
        if len(rows) >= args.max_rows:
            break

    splits = split_by_trace(rows)
    out_dir = Path(args.out_dir)
    train_rows = list(splits["train"])
    val_rows = list(splits["val"])
    test_rows = list(splits["test"])
    if (not val_rows or not test_rows) and train_rows:
        val_rows = [r for i, r in enumerate(train_rows) if i % 5 == 1]
        test_rows = [r for i, r in enumerate(train_rows) if i % 5 == 2]
        keep = [r for i, r in enumerate(train_rows) if i % 5 not in {1, 2}]
        train_rows = keep or train_rows
    _write_csv(out_dir / "ml_gate_train.csv", train_rows)
    _write_csv(out_dir / "ml_gate_val.csv", val_rows or train_rows[:1])
    _write_csv(out_dir / "ml_gate_test.csv", test_rows or train_rows[:1])

    print(f"rows_total={len(rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")


if __name__ == "__main__":
    main()
