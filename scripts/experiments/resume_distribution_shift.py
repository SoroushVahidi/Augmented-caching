"""Safe resume wrapper for the distribution-shift ablation campaign
(scripts/experiments/run_distribution_shift_ablation.py,
configs/distribution_shift_ablation_v1.json).

The underlying runner is already checkpoint-resumable on its own (it skips
folds recorded in campaign_state.json's completed_folds and skips any
(condition, held_out_family, capacity) row already present in
policy_comparison.csv via IncrementalCsvWriter.already_done()) -- this
wrapper does NOT reimplement that logic. It adds the safety checks the bare
runner does not do for you:

  - protocol drift detection: the frozen config
    (configs/distribution_shift_ablation_v1.json) must still exactly match
    the protocol_snapshot.json recorded when the campaign FIRST launched --
    resuming after an unnoticed protocol edit would silently mix rows
    computed under two different protocols in one CSV;
  - existing-artifact integrity: policy_comparison.csv /
    state_shift_metrics.csv / trajectory_divergence.csv are checked for
    duplicate keys, non-"ok" status rows, and NaN/Inf before trusting them
    as a safe resume base;
  - concurrent-runner protection: refuses to resume if a
    run_distribution_shift_ablation.py process is already running (would
    race on the same output files);
  - reports exactly what --dry-run promises: completed folds, primary row
    count, and the next fold that would run -- all read from the actual
    on-disk state, never hardcoded.

Never launches anything by itself unless invoked without --dry-run, and
never deletes or rewrites existing rows (relies entirely on the underlying
runner's own append-only, skip-if-present writer).

Usage:
    python scripts/experiments/resume_distribution_shift.py --dry-run
    python scripts/experiments/resume_distribution_shift.py --max-wall-hours 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

CONFIG_PATH = Path("configs/distribution_shift_ablation_v1.json")
OUT_DIR = Path("analysis/distribution_shift_ablation_v1")
UNDERLYING_RUNNER = Path("scripts/experiments/run_distribution_shift_ablation.py")
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
CONDITIONS = ["OFF_POLICY_LRU", "DAGGER_ITER1"]
CAPACITIES = [32, 64, 128]
EXPECTED_PRIMARY_ROWS = len(FAMILIES) * len(CAPACITIES) * len(CONDITIONS)  # 42


class ResumeBlocked(RuntimeError):
    pass


def _check_protocol_drift(out_dir: Path) -> None:
    snapshot_path = out_dir / "protocol_snapshot.json"
    if not snapshot_path.exists():
        raise ResumeBlocked(f"No protocol_snapshot.json at {snapshot_path} -- campaign has never launched; "
                             "nothing to resume (use the underlying runner directly for a first launch).")
    if not CONFIG_PATH.exists():
        raise ResumeBlocked(f"Frozen protocol config not found: {CONFIG_PATH}")
    current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if current != snapshot:
        raise ResumeBlocked(
            f"Protocol drift detected: {CONFIG_PATH} no longer matches the protocol_snapshot.json "
            "recorded at first launch. Resuming would mix rows computed under two different protocols "
            "in one CSV -- refusing. Investigate the diff before resuming."
        )


def _check_csv_integrity(path: Path, key_fields: List[str]) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seen = set()
    duplicates = 0
    non_ok = 0
    nan_inf = 0
    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        if key in seen:
            duplicates += 1
        seen.add(key)
        if "status" in row and row["status"] and row["status"] != "ok":
            non_ok += 1
        for k, v in row.items():
            if v in ("nan", "inf", "-inf", "NaN", "Infinity"):
                nan_inf += 1
                continue
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    nan_inf += 1
            except (TypeError, ValueError):
                pass
    return {
        "path": str(path), "exists": True, "row_count": len(rows),
        "duplicate_keys": duplicates, "non_ok_status_rows": non_ok, "nan_or_inf_values": nan_inf,
        "clean": duplicates == 0 and non_ok == 0 and nan_inf == 0,
    }


def _check_no_concurrent_runner() -> None:
    try:
        out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResumeBlocked(f"Could not check for a concurrent runner via `ps`: {exc}")
    for line in out.splitlines():
        if str(UNDERLYING_RUNNER) in line and "grep" not in line and "resume_distribution_shift.py" not in line:
            raise ResumeBlocked(f"A run_distribution_shift_ablation.py process appears to already be running: "
                                 f"{line.strip()}. Refusing to launch a duplicate runner against the same output files.")


def plan(out_dir: Path = OUT_DIR) -> Dict[str, object]:
    """Read-only: compute completed folds / rows / next fold from actual
    on-disk state. Never launches anything."""
    _check_protocol_drift(out_dir)

    state_path = out_dir / "campaign_state.json"
    completed_folds = json.loads(state_path.read_text(encoding="utf-8")).get("completed_folds", []) if state_path.exists() else []
    remaining_folds = [f for f in FAMILIES if f not in completed_folds]

    policy_integrity = _check_csv_integrity(out_dir / "policy_comparison.csv", ["condition", "held_out_family", "capacity"])
    shift_integrity = _check_csv_integrity(out_dir / "state_shift_metrics.csv", ["condition", "held_out_family", "capacity"])
    traj_integrity = _check_csv_integrity(out_dir / "trajectory_divergence.csv",
                                           ["held_out_family", "capacity", "reference_condition", "other_condition"])

    primary_rows = policy_integrity.get("row_count", 0) if policy_integrity.get("exists") else 0
    return {
        "completed_folds": completed_folds,
        "n_completed_folds": len(completed_folds),
        "n_total_folds": len(FAMILIES),
        "remaining_folds_in_order": remaining_folds,
        "next_fold": remaining_folds[0] if remaining_folds else None,
        "primary_rows": primary_rows,
        "expected_primary_rows": EXPECTED_PRIMARY_ROWS,
        "artifact_integrity": {
            "policy_comparison.csv": policy_integrity,
            "state_shift_metrics.csv": shift_integrity,
            "trajectory_divergence.csv": traj_integrity,
        },
        "all_artifacts_clean": all(
            v.get("clean", True) for v in (policy_integrity, shift_integrity, traj_integrity) if v.get("exists")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--models-dir", type=Path, default=Path("models/distribution_shift_ablation_v1"))
    ap.add_argument("--max-wall-hours", type=float, default=2.0)
    ap.add_argument("--data-read-root", default="/home/soroush/Augmented-caching")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report the resume plan (completed folds, primary rows, next fold, artifact "
                     "integrity) and exit without launching anything.")
    args = ap.parse_args()

    try:
        p = plan(args.out_dir)
    except ResumeBlocked as exc:
        print(f"[BLOCKED] {exc}")
        raise SystemExit(1)

    print(f"[plan] completed folds = {p['n_completed_folds']}/{p['n_total_folds']}: {p['completed_folds']}")
    print(f"[plan] primary rows = {p['primary_rows']}/{p['expected_primary_rows']}")
    print(f"[plan] next fold = {p['next_fold']!r}")
    print(f"[plan] remaining fold order = {p['remaining_folds_in_order']}")
    for name, integrity in p["artifact_integrity"].items():
        if not integrity.get("exists"):
            print(f"[artifact] {name}: does not exist yet")
            continue
        status = "CLEAN" if integrity["clean"] else "DIRTY"
        print(f"[artifact] {name}: {integrity['row_count']} rows, {status} "
              f"(dup={integrity['duplicate_keys']} non_ok={integrity['non_ok_status_rows']} "
              f"nan_inf={integrity['nan_or_inf_values']})")

    if not p["all_artifacts_clean"]:
        print("[BLOCKED] one or more existing artifacts are not clean -- refusing to resume onto a "
              "possibly-corrupt base. Investigate before resuming.")
        raise SystemExit(1)

    if p["next_fold"] is None:
        print("[done] all 7 folds already completed -- nothing to resume. Run the completion audit instead.")
        return

    if args.dry_run:
        print("\n[dry-run] all checks passed; NOT launching (--dry-run).")
        return

    _check_no_concurrent_runner()

    cmd = [
        sys.executable, str(UNDERLYING_RUNNER),
        "--config", str(CONFIG_PATH),
        "--max-wall-hours", str(args.max_wall_hours),
        "--models-dir", str(args.models_dir),
        "--out-dir", str(args.out_dir),
        "--resume",
        "--data-read-root", args.data_read_root,
    ]
    print(f"\n[launch] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
