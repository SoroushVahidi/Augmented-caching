"""Completion audit for the distribution-shift ablation campaign
(analysis/distribution_shift_ablation_v1/, configs/distribution_shift_ablation_v1.json).

Purely structural/integrity checks -- classifies the campaign as one of:

  INCOMPLETE       -- fewer than 42/42 primary rows exist yet.
  COMPLETE_VALID    -- 42/42 primary rows exist and every integrity check
                       below passes.
  COMPLETE_INVALID  -- 42/42 primary rows exist but at least one check
                       fails (duplicate/failed/NaN rows, wrong window,
                       missing diagnostic coverage, or protocol drift).

Checks performed once 42/42 exists (never run scientific interpretation --
no comparison of miss ratios or shift indices across conditions here, that
is the frozen statistical analysis's job, gated separately):
  - exactly 2 conditions (OFF_POLICY_LRU, DAGGER_ITER1);
  - exactly 7 held-out families;
  - exactly 3 capacities (32, 64, 128);
  - 42 unique (condition, held_out_family, capacity) keys, zero duplicates;
  - zero non-"ok" status rows, zero NaN/Inf values;
  - scoring window identical on every row (score_start=10000,
    score_end=50000, scored_requests=40000);
  - state_shift_metrics.csv covers the same 42 keys;
  - trajectory_divergence.csv covers the expected 21 (held_out_family,
    capacity) pairs (one OFF_POLICY_LRU-vs-DAGGER_ITER1 comparison per
    fold x capacity -- see run_distribution_shift_ablation.py's
    `if condition == "DAGGER_ITER1"` gate);
  - the frozen protocol config still matches protocol_snapshot.json
    (no drift since launch).

Usage:
    python scripts/experiments/audit_distribution_shift_completion.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resume_distribution_shift import (  # noqa: E402
    CAPACITIES, CONDITIONS, CONFIG_PATH, EXPECTED_PRIMARY_ROWS, FAMILIES, OUT_DIR,
    _check_csv_integrity,
)

EXPECTED_TRAJ_ROWS = len(FAMILIES) * len(CAPACITIES)  # 21: one DAGGER_ITER1-vs-OFF_POLICY_LRU pair per fold x cap
EXPECTED_SCORE_START = 10000
EXPECTED_SCORE_END = 50000
EXPECTED_SCORED_REQUESTS = 40000


def _protocol_unchanged(out_dir: Path) -> bool:
    snapshot_path = out_dir / "protocol_snapshot.json"
    if not snapshot_path.exists() or not CONFIG_PATH.exists():
        return False
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8")) == json.loads(snapshot_path.read_text(encoding="utf-8"))


def audit(out_dir: Path = OUT_DIR) -> Dict[str, object]:
    policy_path = out_dir / "policy_comparison.csv"
    if not policy_path.exists():
        return {"classification": "INCOMPLETE", "primary_rows": 0, "expected_primary_rows": EXPECTED_PRIMARY_ROWS,
                "checks": {}}

    with policy_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if len(rows) < EXPECTED_PRIMARY_ROWS:
        return {"classification": "INCOMPLETE", "primary_rows": len(rows),
                "expected_primary_rows": EXPECTED_PRIMARY_ROWS, "checks": {}}

    checks: Dict[str, str] = {}

    conditions_seen = {r["condition"] for r in rows}
    checks["conditions"] = "PASS" if conditions_seen == set(CONDITIONS) else f"FAIL ({conditions_seen})"

    families_seen = {r["held_out_family"] for r in rows}
    checks["families"] = "PASS" if families_seen == set(FAMILIES) else f"FAIL ({families_seen})"

    caps_seen = {int(r["capacity"]) for r in rows}
    checks["capacities"] = "PASS" if caps_seen == set(CAPACITIES) else f"FAIL ({caps_seen})"

    policy_integrity = _check_csv_integrity(policy_path, ["condition", "held_out_family", "capacity"])
    checks["row_count_exact_42"] = "PASS" if policy_integrity["row_count"] == EXPECTED_PRIMARY_ROWS else \
        f"FAIL ({policy_integrity['row_count']} != {EXPECTED_PRIMARY_ROWS})"
    checks["no_duplicate_keys"] = "PASS" if policy_integrity["duplicate_keys"] == 0 else \
        f"FAIL ({policy_integrity['duplicate_keys']} duplicates)"
    checks["zero_failures"] = "PASS" if policy_integrity["non_ok_status_rows"] == 0 else \
        f"FAIL ({policy_integrity['non_ok_status_rows']} non-ok rows)"
    checks["zero_nan_inf"] = "PASS" if policy_integrity["nan_or_inf_values"] == 0 else \
        f"FAIL ({policy_integrity['nan_or_inf_values']} NaN/Inf values)"

    window_bad = [
        r for r in rows
        if int(r.get("score_start", -1)) != EXPECTED_SCORE_START
        or int(r.get("score_end", -1)) != EXPECTED_SCORE_END
        or int(r.get("scored_requests", -1)) != EXPECTED_SCORED_REQUESTS
    ]
    checks["scoring_window_consistent"] = "PASS" if not window_bad else f"FAIL ({len(window_bad)} row(s) off-window)"

    shift_integrity = _check_csv_integrity(out_dir / "state_shift_metrics.csv", ["condition", "held_out_family", "capacity"])
    checks["state_shift_coverage"] = (
        "PASS" if shift_integrity.get("exists") and shift_integrity["row_count"] == EXPECTED_PRIMARY_ROWS and shift_integrity["clean"]
        else f"FAIL ({shift_integrity})"
    )

    traj_integrity = _check_csv_integrity(
        out_dir / "trajectory_divergence.csv", ["held_out_family", "capacity", "reference_condition", "other_condition"]
    )
    checks["trajectory_diagnostic_coverage"] = (
        "PASS" if traj_integrity.get("exists") and traj_integrity["row_count"] == EXPECTED_TRAJ_ROWS and traj_integrity["clean"]
        else f"FAIL ({traj_integrity})"
    )

    checks["frozen_protocol_unchanged"] = "PASS" if _protocol_unchanged(out_dir) else "FAIL (protocol drift since launch)"

    all_pass = all(v == "PASS" for v in checks.values())
    return {
        "classification": "COMPLETE_VALID" if all_pass else "COMPLETE_INVALID",
        "primary_rows": len(rows),
        "expected_primary_rows": EXPECTED_PRIMARY_ROWS,
        "checks": checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "completion_audit.json")
    args = ap.parse_args()

    result = audit(args.out_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[audit] classification={result['classification']} "
          f"primary_rows={result['primary_rows']}/{result['expected_primary_rows']}")
    for name, status in result.get("checks", {}).items():
        print(f"  - {name}: {status}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
