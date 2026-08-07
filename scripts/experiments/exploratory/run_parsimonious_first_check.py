"""First-check script for Baseline 5 (Im et al. 2022 AdaptiveQuery-b)."""

from __future__ import annotations

import csv
from pathlib import Path

from lafc.metrics.cost import hit_rate
from lafc.policies.adaptive_query import AdaptiveQueryPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


OUT_DIR = Path("analysis")
OUT_CSV = OUT_DIR / "parsimonious_first_check.csv"
OUT_MD = OUT_DIR / "parsimonious_first_check.md"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trace = "data/example_unweighted.json"
    capacity = 3
    requests, pages = load_trace(trace)

    runs = [
        ("lru", LRUPolicy()),
        ("marker", MarkerPolicy()),
        ("predictive_marker", PredictiveMarkerPolicy()),
        ("adaptive_query_b2", AdaptiveQueryPolicy(b=2, seed=0)),
        ("adaptive_query_b4", AdaptiveQueryPolicy(b=4, seed=0)),
    ]

    rows = []
    for label, policy in runs:
        result = run_policy(policy, requests, pages, capacity=capacity)
        aq_summary = (result.extra_diagnostics or {}).get("adaptive_query", {}).get("summary", {})
        rows.append(
            {
                "policy": label,
                "trace": trace,
                "capacity": capacity,
                "total_cost": result.total_cost,
                "total_misses": result.total_misses,
                "hit_rate": hit_rate(result.events),
                "queries_used": aq_summary.get("queries_used", 0.0),
                "fraction_misses_queried": aq_summary.get("fraction_misses_queried", 0.0),
                "fraction_misses_fallback_random": aq_summary.get(
                    "fraction_misses_fallback_random", 0.0
                ),
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Parsimonious Baseline First Check",
        "",
        "Command:",
        "",
        "- `python scripts/run_parsimonious_first_check.py`",
        "",
        "| policy | cost | misses | hit_rate | queries_used | frac_miss_queried | frac_miss_fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['total_cost']:.2f} | {r['total_misses']} | "
            f"{r['hit_rate']:.2%} | {r['queries_used']:.1f} | "
            f"{r['fraction_misses_queried']:.2%} | {r['fraction_misses_fallback_random']:.2%} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
