from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from lafc.learned_gate.dataset_v2 import GateDatasetV2Config, _split_by_trace_and_capacity, build_gate_examples_v2
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _make_stress_trace(page_ids: List[str], buckets: List[int], confs: List[float]):
    recs = [{"bucket": b, "confidence": c} for b, c in zip(buckets, confs)]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=recs)


def _stress_traces():
    return {
        "stress::predictor_good_lru_bad": _make_stress_trace(
            ["A", "B", "C", "A", "D", "A", "B", "C", "A", "D"],
            [0, 3, 3, 0, 3, 0, 3, 3, 0, 3],
            [1.0] * 10,
        ),
        "stress::predictor_bad_lru_good": _make_stress_trace(
            ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"],
            [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
            [1.0] * 10,
        ),
        "stress::mixed_regime": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "C", "B", "D"],
            [0, 3, 1, 0, 2, 3, 0, 1, 2, 3],
            [0.9, 0.9, 0.3, 0.9, 0.7, 0.3, 0.9, 0.3, 0.7, 0.3],
        ),
    }


def _iter_traces(sample_only: bool):
    base = ["data/example_unweighted.json", "data/example_atlas_v1.json"]
    for p in base:
        yield p, load_trace(p)
    if not sample_only:
        for name, payload in _stress_traces().items():
            yield name, payload


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stronger v2 learned-gate dataset")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizons", default="4,8,16")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--out-dir", default="data/derived")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    cfg = GateDatasetV2Config(horizons=horizons, margin=float(args.margin))

    rows: List[Dict[str, object]] = []
    for trace_name, (requests, _pages) in _iter_traces(sample_only=args.sample_only):
        for cap in capacities:
            rows.extend(build_gate_examples_v2(requests=requests, capacity=cap, trace_name=trace_name, cfg=cfg))
            if len(rows) >= args.max_rows:
                rows = rows[: args.max_rows]
                break
        if len(rows) >= args.max_rows:
            break

    splits = _split_by_trace_and_capacity(rows)
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    if (not val or not test) and train:
        val = [r for i, r in enumerate(train) if i % 7 == 1]
        test = [r for i, r in enumerate(train) if i % 7 == 2]
        train = [r for i, r in enumerate(train) if i % 7 not in {1, 2}] or train

    out = Path(args.out_dir)
    _write_csv(out / "ml_gate_v2_train.csv", train)
    _write_csv(out / "ml_gate_v2_val.csv", val or train[:1])
    _write_csv(out / "ml_gate_v2_test.csv", test or train[:1])
    print(f"rows_total={len(rows)} train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
