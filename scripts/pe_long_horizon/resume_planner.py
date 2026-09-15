"""List ONLY the missing/failed PE long-horizon production tasks.

Read-only: inspects the manifest and each task's output directory on
whatever filesystem it's pointed at (local mirror of the Wulver run_root, or
Wulver itself via an already-open connection outside this script's
responsibility). Never resubmits anything -- prints a ready-to-review
--array spec for a human to use in a future, explicitly separate resubmit.

A task counts as:
  DONE       - COMPLETE.json exists and validated_summary.json says PASS
  INCOMPLETE - output dir exists but no COMPLETE.json (partial/interrupted
               run; must NOT be silently reused -- see INCOMPLETE.json
               written by the production sbatch template)
  FAILED     - COMPLETE.json exists but validated_summary.json says FAIL
  NOT_RUN    - output dir does not exist at all

Usage:
    python scripts/pe_long_horizon/resume_planner.py --run-root <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def classify_task(task: dict, run_root: Path) -> str:
    out_dir = run_root / "raw" / task["task_id"]
    completion = out_dir / "COMPLETE.json"
    validated = out_dir / "validated_summary.json"
    if not out_dir.exists():
        return "NOT_RUN"
    if not completion.exists():
        return "INCOMPLETE"
    if validated.exists():
        summary = json.loads(validated.read_text(encoding="utf-8"))
        if summary.get("validation_result") != "PASS":
            return "FAILED"
        return "DONE"
    return "COMPLETE_UNVALIDATED"  # generation finished, validator hasn't run yet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/pe_long_horizon_production/manifest.json")
    ap.add_argument("--run-root", default=None)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run_root = Path(args.run_root) if args.run_root else Path(manifest["run_root"])

    by_status: dict[str, list[str]] = {}
    array_indices_to_resubmit: list[int] = []
    for task in manifest["tasks"]:
        status = classify_task(task, run_root)
        by_status.setdefault(status, []).append(task["task_id"])
        if status in ("NOT_RUN", "INCOMPLETE", "FAILED"):
            array_indices_to_resubmit.append(task["array_index"])

    print("=== TASK STATUS SUMMARY ===")
    for status in ("DONE", "COMPLETE_UNVALIDATED", "INCOMPLETE", "FAILED", "NOT_RUN"):
        ids = by_status.get(status, [])
        print(f"{status}: {len(ids)}")
        for i in ids:
            print(f"  - {i}")

    if array_indices_to_resubmit:
        array_indices_to_resubmit.sort()
        spec = ",".join(str(i) for i in array_indices_to_resubmit)
        print()
        print("Ready-to-review resubmit array spec (NOT submitted by this script):")
        print(f"  --array={spec}")
    else:
        print()
        print("All 20 production tasks DONE and validated PASS.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
