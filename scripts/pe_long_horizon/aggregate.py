"""Global aggregator for PE long-horizon production.

Consumes ONLY validated per-cell summaries (validated_summary.json, written
by validate_cell.py with VALIDATION_RESULT: PASS). Refuses to produce final
scientific results unless all 60 expected logical cells (from the manifest)
are present and PASSed.

Produces:
  A. per family x capacity x horizon table
  B. horizon-level pooled summaries (H32, H64, H128)
  C. H16->H32->H64->H128 deltas and saturation diagnostics (descriptive only
     -- no inferential claims, no invented thresholds)

For H16, reads the existing canonical validated H16 summary (path supplied
via --h16-summary) rather than regenerating anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_validated_summaries(manifest: dict, run_root: Path) -> tuple[dict, list[str]]:
    """Returns (summaries_by_task_id, missing_or_failed_task_ids)."""
    summaries = {}
    missing = []
    for task in manifest["tasks"]:
        task_dir = run_root / "raw" / task["task_id"]
        summary_path = task_dir / "validated_summary.json"
        if not summary_path.exists():
            missing.append(task["task_id"])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("validation_result") != "PASS":
            missing.append(task["task_id"])
            continue
        summaries[task["task_id"]] = summary
    return summaries, missing


def load_h16_canonical(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/pe_long_horizon_production/manifest.json")
    ap.add_argument("--run-root", default=None, help="Defaults to manifest['run_root']")
    ap.add_argument("--h16-summary", default=None, help="Path to existing canonical validated H16 summary JSON")
    ap.add_argument("--out-dir", default=None, help="Defaults to <run_root>/summaries")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    run_root = Path(args.run_root) if args.run_root else Path(manifest["run_root"])
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "summaries"

    summaries, missing = load_validated_summaries(manifest, run_root)

    if missing:
        print(f"AGGREGATION_REFUSED: {len(missing)}/{len(manifest['tasks'])} tasks missing or not PASS:")
        for t in missing:
            print(f"  - {t}")
        return 1

    h16 = load_h16_canonical(Path(args.h16_summary) if args.h16_summary else None)

    # A. family x capacity x horizon table
    rows_a = []
    for task in manifest["tasks"]:
        summary = summaries[task["task_id"]]
        for h_str, per_h in summary["per_horizon"].items():
            rows_a.append(
                {
                    "family": task["family"],
                    "capacity": task["capacity"],
                    "horizon": int(h_str),
                    "decisions": per_h["decisions"],
                    "candidate_rows": per_h["candidate_rows"],
                    "all_tied_fraction": per_h["all_tied_fraction"],
                    "discriminativeness": per_h["discriminativeness"],
                    "mean_optimal_set_fraction": per_h["mean_optimal_set_fraction"],
                    "random_optimal_probability": per_h["random_optimal_probability"],
                    "mean_random_regret": per_h["mean_random_regret"],
                }
            )

    # B. horizon-level pooled summaries (row-weighted mean discriminativeness etc.)
    pooled_by_horizon: dict[int, dict] = {}
    for h in manifest["production_horizons"]:
        cells = [r for r in rows_a if r["horizon"] == h]
        total_rows = sum(c["candidate_rows"] for c in cells)
        total_decisions = sum(c["decisions"] for c in cells)
        pooled_by_horizon[h] = {
            "total_candidate_rows": total_rows,
            "total_decisions": total_decisions,
            "mean_discriminativeness_unweighted": sum(c["discriminativeness"] for c in cells) / len(cells) if cells else None,
            "mean_random_regret_unweighted": sum(c["mean_random_regret"] for c in cells) / len(cells) if cells else None,
            "num_cells": len(cells),
        }

    # C. deltas / saturation diagnostics (descriptive, no thresholds invented here)
    deltas = []
    for r in rows_a:
        family, capacity = r["family"], r["capacity"]
        h16_key = f"{family}__cap{capacity}__H16"
        h16_disc = None
        if h16:
            h16_cell = h16.get("cells", {}).get(h16_key)
            h16_disc = h16_cell["discriminativeness"] if h16_cell else None
        deltas.append(
            {
                "family": family,
                "capacity": capacity,
                "horizon": r["horizon"],
                "discriminativeness": r["discriminativeness"],
                "h16_discriminativeness": h16_disc,
                "delta_vs_h16": (r["discriminativeness"] - h16_disc) if h16_disc is not None else None,
            }
        )
    # H32->H64, H64->H128 deltas per family/capacity
    by_fc: dict[tuple[str, int], dict[int, float]] = {}
    for r in rows_a:
        by_fc.setdefault((r["family"], r["capacity"]), {})[r["horizon"]] = r["discriminativeness"]
    stepwise = []
    for (family, capacity), by_h in by_fc.items():
        d32, d64, d128 = by_h.get(32), by_h.get(64), by_h.get(128)
        stepwise.append(
            {
                "family": family,
                "capacity": capacity,
                "delta_32_to_64": (d64 - d32) if d32 is not None and d64 is not None else None,
                "delta_64_to_128": (d128 - d64) if d64 is not None and d128 is not None else None,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_A_family_capacity_horizon.csv").write_text(
        "family,capacity,horizon,decisions,candidate_rows,all_tied_fraction,discriminativeness,"
        "mean_optimal_set_fraction,random_optimal_probability,mean_random_regret\n"
        + "\n".join(
            f"{r['family']},{r['capacity']},{r['horizon']},{r['decisions']},{r['candidate_rows']},"
            f"{r['all_tied_fraction']},{r['discriminativeness']},{r['mean_optimal_set_fraction']},"
            f"{r['random_optimal_probability']},{r['mean_random_regret']}"
            for r in rows_a
        ),
        encoding="utf-8",
    )
    (out_dir / "table_B_horizon_pooled.json").write_text(json.dumps(pooled_by_horizon, indent=2), encoding="utf-8")
    (out_dir / "table_C_h16_deltas.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    (out_dir / "table_C_stepwise_deltas.json").write_text(json.dumps(stepwise, indent=2), encoding="utf-8")

    print(f"AGGREGATION_OK: all {len(manifest['tasks'])}/{len(manifest['tasks'])} tasks validated PASS.")
    print(f"Wrote tables to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
