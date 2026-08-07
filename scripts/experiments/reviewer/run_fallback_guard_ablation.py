"""Targeted, non-canonical ablation for the evict_value_v1 guard/fallback mechanism.

This script is intentionally separate from scripts/run_policy_comparison_wulver_v1.py
(the canonical KBS heavy_r1 sweep). It exists only to answer R3-Issue6 / R3-Rec5:
whether the guard wrapper (src/lafc/policies/guard_wrapper.py) ever triggers, how
often, and whether it changes miss ratio relative to running evict_value_v1 without
it. It must never be confused with, or overwrite, canonical cap32/cap64/cap128
outputs -- all outputs use the `fallback_guard_ablation_` prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.guard_wrapper import EvictValueV1GuardedPolicy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy

TRACE_PATHS: Dict[str, str] = {
    "brightkite": "data/processed/brightkite/trace.jsonl",
    "citibike": "data/processed/citibike/trace.jsonl",
    "metacdn": "data/processed/metacdn/trace.jsonl",
    "metakv": "data/processed/metakv/trace.jsonl",
}

MODEL_PATH = "models/evict_value_wulver_v1_best_heavy_r1.pkl"

GUARD_VARIANTS = {
    "guarded_default": dict(early_return_window=2, trigger_threshold=2, trigger_window=16, guard_duration=8),
    "guarded_sensitive_m1": dict(early_return_window=2, trigger_threshold=1, trigger_window=16, guard_duration=8),
    "guarded_long_duration_d32": dict(early_return_window=2, trigger_threshold=2, trigger_window=16, guard_duration=32),
}


def run_one(condition: str, requests, pages, capacity: int) -> Dict[str, object]:
    if condition == "lru":
        policy = LRUPolicy()
    elif condition == "evict_value_v1":
        policy = EvictValueV1Policy(model_path=MODEL_PATH)
    elif condition in GUARD_VARIANTS:
        policy = EvictValueV1GuardedPolicy(
            model_path=MODEL_PATH,
            fallback_policy="lru",
            **GUARD_VARIANTS[condition],
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    t0 = time.time()
    result = run_policy(policy, requests, pages, capacity)
    elapsed = time.time() - t0

    n = len(requests)
    miss_ratio = result.total_misses / n
    row: Dict[str, object] = {
        "condition": condition,
        "total_requests": n,
        "total_hits": result.total_hits,
        "total_misses": result.total_misses,
        "miss_ratio": miss_ratio,
        "hit_rate": 1.0 - miss_ratio,
        "elapsed_seconds": round(elapsed, 3),
    }

    if isinstance(policy, EvictValueV1GuardedPolicy):
        diag = policy.diagnostics_summary()
        base_steps = diag["base_time_steps"]
        fb_steps = diag["fallback_time_steps"]
        denom = base_steps + fb_steps
        row.update(
            {
                "guard_triggers": diag["guard_triggers"],
                "base_time_steps": base_steps,
                "fallback_time_steps": fb_steps,
                "fallback_active_fraction": (fb_steps / denom) if denom else 0.0,
                "early_return_events": diag["early_return_events"],
                "base_evictions": diag["base_evictions"],
                "fallback_evictions": diag["fallback_evictions"],
                "early_return_window_W": diag["early_return_window"],
                "trigger_threshold_M": diag["trigger_threshold"],
                "trigger_window_T": diag["trigger_window"],
                "guard_duration_D": diag["guard_duration"],
                "fallback_policy_name": diag["fallback_policy_name"],
                "guard_trigger_times": json.dumps(diag["guard_trigger_times"]),
            }
        )
    else:
        row.update(
            {
                "guard_triggers": "",
                "base_time_steps": "",
                "fallback_time_steps": "",
                "fallback_active_fraction": "",
                "early_return_events": "",
                "base_evictions": "",
                "fallback_evictions": "",
                "early_return_window_W": "",
                "trigger_threshold_M": "",
                "trigger_window_T": "",
                "guard_duration_D": "",
                "fallback_policy_name": "",
                "guard_trigger_times": "",
            }
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", default="brightkite,citibike,metacdn,metakv")
    ap.add_argument("--capacities", default="128")
    ap.add_argument("--max-requests", type=int, default=20000)
    ap.add_argument("--conditions", default="lru,evict_value_v1,guarded_default,guarded_sensitive_m1,guarded_long_duration_d32")
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    args = ap.parse_args()

    trace_names = [t.strip() for t in args.traces.split(",") if t.strip()]
    capacities = [int(c.strip()) for c in args.capacities.split(",") if c.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    all_rows: List[Dict[str, object]] = []
    for trace_name in trace_names:
        path = TRACE_PATHS[trace_name]
        print(f"[fallback_guard_ablation] loading {trace_name} from {path}", flush=True)
        requests, pages, _fmt = load_trace_from_any(path)
        for capacity in capacities:
            n = min(args.max_requests, len(requests))
            sub = requests[:n]
            for condition in conditions:
                print(
                    f"[fallback_guard_ablation] running trace={trace_name} capacity={capacity} "
                    f"condition={condition} n_requests={n}",
                    flush=True,
                )
                row = run_one(condition, sub, pages, capacity)
                row["trace"] = trace_name
                row["capacity"] = capacity
                all_rows.append(row)
                print(
                    f"[fallback_guard_ablation]   -> misses={row['total_misses']} "
                    f"miss_ratio={row['miss_ratio']:.5f} elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    fieldnames = [
        "trace",
        "capacity",
        "condition",
        "total_requests",
        "total_hits",
        "total_misses",
        "miss_ratio",
        "hit_rate",
        "elapsed_seconds",
        "guard_triggers",
        "base_time_steps",
        "fallback_time_steps",
        "fallback_active_fraction",
        "early_return_events",
        "base_evictions",
        "fallback_evictions",
        "early_return_window_W",
        "trigger_threshold_M",
        "trigger_window_T",
        "guard_duration_D",
        "fallback_policy_name",
        "guard_trigger_times",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[fallback_guard_ablation] wrote {args.output_csv}", flush=True)

    lines = [
        "# Fallback/guard ablation -- raw results (non-canonical)",
        "",
        f"traces={trace_names} capacities={capacities} max_requests={args.max_requests} conditions={conditions}",
        "",
        "| trace | capacity | condition | misses | miss_ratio | guard_triggers | fallback_active_fraction | elapsed_s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in all_rows:
        fa = row["fallback_active_fraction"]
        fa_str = f"{fa:.5f}" if isinstance(fa, float) else "-"
        gt = row["guard_triggers"] if row["guard_triggers"] != "" else "-"
        lines.append(
            f"| {row['trace']} | {row['capacity']} | {row['condition']} | {row['total_misses']} | "
            f"{row['miss_ratio']:.5f} | {gt} | {fa_str} | {row['elapsed_seconds']:.1f} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[fallback_guard_ablation] wrote {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
