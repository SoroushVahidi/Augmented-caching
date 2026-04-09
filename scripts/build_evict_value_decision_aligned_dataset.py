from __future__ import annotations

import argparse
import csv
import json
import random
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List

from lafc.evict_value_v2_rollout import EvictValueV2RolloutConfig, build_rollout_candidate_rows_v2
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
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count_decisions(rows: Iterable[Dict[str, object]]) -> int:
    return len({str(r["decision_id"]) for r in rows})


def _family_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_family: Dict[str, Dict[str, float]] = {}
    for r in rows:
        family = str(r.get("family", "unknown"))
        slot = by_family.setdefault(family, {"rows": 0.0, "regret_sum": 0.0, "decisions": set()})
        slot["rows"] += 1.0
        slot["regret_sum"] += float(r["rollout_regret_h"])
        slot["decisions"].add(str(r["decision_id"]))

    out: List[Dict[str, object]] = []
    for family, payload in sorted(by_family.items()):
        rows_n = int(payload["rows"])
        out.append(
            {
                "family": family,
                "rows": rows_n,
                "decisions": len(payload["decisions"]),
                "mean_rollout_regret": (float(payload["regret_sum"]) / float(rows_n)) if rows_n else 0.0,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build decision-aligned eviction dataset (loss/regret labels)")
    ap.add_argument("--trace-glob", default="data/example_*.json")
    ap.add_argument("--dataset", default="mixed")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizons", default="4,8,16,32")
    ap.add_argument("--continuation-policy", choices=["lru", "blind_oracle"], default="lru")
    ap.add_argument("--discount-gamma", type=float, default=0.0, help="Optional gamma in [0,1] for per-row discounted targets; 0 disables.")
    ap.add_argument("--max-rows", type=int, default=250000)
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--max-decisions-per-trace", type=int, default=0)
    ap.add_argument("--output-dir", default="data/derived/evict_value_decision_aligned")
    ap.add_argument("--sample-seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.sample_seed)
    capacities = _parse_csv_ints(args.capacities)
    horizons = tuple(_parse_csv_ints(args.horizons))

    cfg = EvictValueV2RolloutConfig(horizons=horizons, reference_policy=args.continuation_policy)
    trace_paths = _resolve_trace_paths(args.trace_glob)
    rng.shuffle(trace_paths)

    rows: List[Dict[str, object]] = []
    per_trace_rows: Dict[str, int] = {}
    per_trace_decisions: Dict[str, set[str]] = {}

    for trace_path in trace_paths:
        requests, _pages = load_trace(trace_path)
        if args.max_requests_per_trace > 0:
            requests = requests[: args.max_requests_per_trace]

        before = len(rows)
        family = _trace_family(trace_path)
        for capacity in capacities:
            candidate_rows = build_rollout_candidate_rows_v2(
                requests=requests,
                capacity=capacity,
                trace_name=trace_path,
                trace_family=family,
                cfg=cfg,
            )
            if args.max_decisions_per_trace > 0:
                decision_ids = sorted({str(r["decision_id"]) for r in candidate_rows})
                rng.shuffle(decision_ids)
                keep = set(decision_ids[: args.max_decisions_per_trace])
                candidate_rows = [r for r in candidate_rows if str(r["decision_id"]) in keep]

            for row in candidate_rows:
                if args.discount_gamma > 0.0:
                    h = int(row["horizon"])
                    w = float(args.discount_gamma ** max(0, h - 1))
                    row["discount_weight"] = w
                    row["rollout_loss_discounted"] = float(row["rollout_loss_h"]) * w
                    row["rollout_regret_discounted"] = float(row["rollout_regret_h"]) * w
                if len(rows) >= args.max_rows:
                    break
                rows.append(row)
                per_trace_decisions.setdefault(trace_path, set()).add(str(row["decision_id"]))
            if len(rows) >= args.max_rows:
                break

        per_trace_rows[trace_path] = len(rows) - before
        if len(rows) >= args.max_rows:
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = out_dir / "candidate_rows.csv"
    _write_csv(candidate_csv, rows)
    family_rows = _family_summary(rows)
    _write_csv(out_dir / "family_summary.csv", family_rows)

    summary = {
        "dataset": args.dataset,
        "trace_glob": args.trace_glob,
        "capacities": capacities,
        "horizons": list(horizons),
        "continuation_policy": args.continuation_policy,
        "discount_gamma": args.discount_gamma,
        "sample_seed": args.sample_seed,
        "rows_total": len(rows),
        "decisions_total": _count_decisions(rows),
        "trace_row_counts": per_trace_rows,
        "trace_decision_counts": {k: len(v) for k, v in per_trace_decisions.items()},
        "outputs": {
            "candidate_rows_csv": str(candidate_csv),
            "family_summary_csv": str(out_dir / "family_summary.csv"),
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
