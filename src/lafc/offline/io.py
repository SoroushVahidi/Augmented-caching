"""Export helpers for offline baseline outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lafc.offline.types import OfflineSimulationResult


def save_offline_results(result: OfflineSimulationResult, output_dir: str) -> None:
    """Write standard text outputs for an offline baseline run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "solver_name": result.solver_name,
        "capacity": result.capacity,
        "total_requests": result.total_requests,
        "total_hits": result.total_hits,
        "total_misses": result.total_misses,
        "total_cost": result.total_cost,
        "hit_rate": result.hit_rate,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out / "per_step_decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "t",
                "page_id",
                "hit",
                "cost",
                "evicted",
                "evicted_next_use",
                "evicted_next_use_distance",
                "evicted_never_used_again",
                "tie_size",
                "inserted",
                "bypassed",
                "cache_occupancy",
            ]
        )
        for d in result.decisions:
            writer.writerow(
                [
                    d.t,
                    d.page_id,
                    d.hit,
                    d.cost,
                    d.evicted or "",
                    "" if d.evicted_next_use is None else d.evicted_next_use,
                    "" if d.evicted_next_use_distance is None else d.evicted_next_use_distance,
                    d.evicted_never_used_again,
                    d.tie_size,
                    d.inserted,
                    d.bypassed,
                    "" if d.cache_occupancy is None else d.cache_occupancy,
                ]
            )

    (out / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2), encoding="utf-8"
    )

    report = (
        "# Offline Caching Run\n\n"
        f"- Solver: `{result.solver_name}`\n"
        f"- Capacity: `{result.capacity}`\n"
        f"- Requests: `{result.total_requests}`\n"
        f"- Hits: `{result.total_hits}`\n"
        f"- Misses: `{result.total_misses}`\n"
        f"- Total cost: `{result.total_cost:.6f}`\n"
        f"- Hit rate: `{result.hit_rate:.6f}`\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
