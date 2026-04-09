from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from lafc.metrics.cost import hit_rate
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.guard_wrapper import EvictValueV1GuardedPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


OUT_DIR = Path("analysis")
OUT_CSV = OUT_DIR / "guarded_evict_value_first_check.csv"
OUT_MD = OUT_DIR / "guarded_evict_value_first_check.md"
OUT_JSON = OUT_DIR / "guarded_evict_value_first_check.json"


@dataclass
class Row:
    trace: str
    capacity: int
    policy: str
    total_cost: float
    total_misses: int
    hit_rate: float
    guard_triggers: float
    guard_time_steps: float



def _run(trace: str, capacity: int, model_path: str) -> List[Row]:
    requests, pages = load_trace(trace)

    base = run_policy(EvictValueV1Policy(model_path=model_path), requests, pages, capacity)
    guarded = run_policy(
        EvictValueV1GuardedPolicy(
            model_path=model_path,
            fallback_policy="lru",
            early_return_window=2,
            trigger_threshold=2,
            trigger_window=16,
            guard_duration=8,
        ),
        requests,
        pages,
        capacity,
    )

    gsum = (guarded.extra_diagnostics or {}).get("evict_value_v1_guarded", {}).get("summary", {})
    return [
        Row(
            trace=trace,
            capacity=capacity,
            policy="evict_value_v1",
            total_cost=base.total_cost,
            total_misses=base.total_misses,
            hit_rate=hit_rate(base.events),
            guard_triggers=0.0,
            guard_time_steps=0.0,
        ),
        Row(
            trace=trace,
            capacity=capacity,
            policy="evict_value_v1_guarded",
            total_cost=guarded.total_cost,
            total_misses=guarded.total_misses,
            hit_rate=hit_rate(guarded.events),
            guard_triggers=float(gsum.get("guard_triggers", 0.0)),
            guard_time_steps=float(gsum.get("guard_time_steps", 0.0)),
        ),
    ]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="First-check comparison for evict_value_v1 vs guarded wrapper")
    ap.add_argument("--model-path", default="models/evict_value_v1_hist_gb.pkl")
    ap.add_argument("--traces", default="data/example_atlas_v1.json,data/example_unweighted.json")
    ap.add_argument("--capacities", default="2,3")
    args = ap.parse_args()

    traces = [t.strip() for t in args.traces.split(",") if t.strip()]
    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]

    rows: List[Row] = []
    for trace in traces:
        for cap in capacities:
            rows.extend(_run(trace, cap, args.model_path))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(Row.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    payload: Dict[str, object] = {
        "model_path": args.model_path,
        "traces": traces,
        "capacities": capacities,
        "rows": [r.__dict__ for r in rows],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Guarded evict_value first check",
        "",
        f"- model_path: `{args.model_path}`",
        "",
        "| trace | cap | policy | cost | misses | hit_rate | guard_triggers | guard_time_steps |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.trace} | {r.capacity} | {r.policy} | {r.total_cost:.2f} | {r.total_misses} | "
            f"{r.hit_rate:.2%} | {r.guard_triggers:.0f} | {r.guard_time_steps:.0f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
