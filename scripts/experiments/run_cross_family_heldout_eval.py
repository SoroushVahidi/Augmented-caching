"""Gated launcher for the evict_value_v1_cross_family_v1 held-out evaluation.

Thin wrapper around the existing, already-sound
scripts/experiments/run_evict_value_v1_cross_family_eval.py (which already
fails closed on missing/wrong-fold/tampered models -- see that script's
docstring). This wrapper adds the one gate it does NOT have: it refuses to
launch unless scripts/experiments/finalize_cross_family_model_registry.py
has already written a frozen registry
(analysis/reviewer_fairness_cross_family_v1/model_registry.json,
MODEL_SELECTION_FROZEN=true) covering all 7 folds, and re-verifies every
recorded model hash against the on-disk file before launching (tamper/
staleness detection independent of the underlying evaluator's own checks).

Never trains during evaluation; never falls back to a surrogate model; the
underlying evaluator is invoked exactly as-is, not reimplemented.

Usage:
    python scripts/experiments/run_cross_family_heldout_eval.py --dry-run
    python scripts/experiments/run_cross_family_heldout_eval.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lafc.experiments.external_baseline_common import sha256_of_file

REGISTRY_PATH = Path("analysis/reviewer_fairness_cross_family_v1/model_registry.json")
UNDERLYING_EVALUATOR = Path("scripts/experiments/run_evict_value_v1_cross_family_eval.py")
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
CAPACITIES = [32, 64, 128]


class GateBlocked(RuntimeError):
    pass


def check_gate(registry_path: Path) -> dict:
    """Raise GateBlocked with a human-readable reason, or return the
    validated registry dict if every check passes."""
    if not registry_path.exists():
        raise GateBlocked(
            f"No model registry at {registry_path}. Run "
            "scripts/experiments/finalize_cross_family_model_registry.py first."
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    if not registry.get("MODEL_SELECTION_FROZEN"):
        raise GateBlocked(
            f"Registry MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')} -- "
            f"missing folds: {registry.get('missing_folds')}"
        )
    if not registry.get("is_full_campaign_scope"):
        raise GateBlocked(
            f"Registry scope_families={registry.get('scope_families')} is not the full "
            "7-family campaign scope -- refusing to run the real held-out evaluation "
            "against a scoped/smoke-test registry."
        )
    records = registry.get("records", [])
    seen_families = {r["held_out_family"] for r in records}
    if seen_families != set(FAMILIES):
        raise GateBlocked(f"Registry covers {sorted(seen_families)}, expected all of {FAMILIES}")

    for rec in records:
        model_path = Path(rec["model_artifact_path"])
        if not model_path.exists():
            raise GateBlocked(f"Registry record for {rec['held_out_family']} points at a model that no "
                               f"longer exists on disk: {model_path}")
        actual_hash = sha256_of_file(model_path)
        if actual_hash != rec["model_artifact_sha256"]:
            raise GateBlocked(
                f"Model hash mismatch for {rec['held_out_family']}: registry says "
                f"{rec['model_artifact_sha256']}, on-disk file hashes to {actual_hash} -- "
                "artifact was modified after the registry was frozen. Re-run the registry "
                "finalizer before evaluating."
            )
    return registry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    ap.add_argument("--data-read-root", default="/home/soroush/Augmented-caching",
                     help="Root containing the canonical data/processed/ traces -- this worktree's "
                     "own data/processed/ is empty by design.")
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--out-dir", default="analysis/reviewer_fairness_cross_family_v1/evict_value_v1")
    ap.add_argument("--dry-run", action="store_true",
                     help="Run the readiness gate and print the exact evaluation plan, but do not "
                     "invoke the underlying evaluator.")
    args = ap.parse_args()

    try:
        registry = check_gate(args.registry)
    except GateBlocked as exc:
        print(f"[BLOCKED] {exc}")
        raise SystemExit(1)

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    n_family_cap = len(FAMILIES) * len(caps)
    print(f"[gate] registry={args.registry} FROZEN=true registry_sha256={registry.get('registry_sha256')}")
    print(f"[gate] {len(registry['records'])}/7 model hashes verified against on-disk artifacts")
    print(f"[plan] {len(FAMILIES)} families x {len(caps)} capacities = {n_family_cap} (family, capacity) units")
    print(f"[plan] underlying evaluator writes 2 rows per unit (deployment_full_stream, "
          f"primary_controlled_window) = {n_family_cap * 2} total rows, of which "
          f"{n_family_cap} are the primary comparison rows")
    print(f"[plan] history=[0,10000) score=[10000,50000), data_read_root={args.data_read_root}")
    print(f"[plan] output -> {args.out_dir}/policy_comparison.csv")
    for family in FAMILIES:
        rec = next(r for r in registry["records"] if r["held_out_family"] == family)
        print(f"  - {family}: model_sha256={rec['model_artifact_sha256'][:12]}... "
              f"training_families={rec['training_families']}")

    if args.dry_run:
        print("\n[dry-run] gate passed; NOT launching the evaluator (--dry-run).")
        return

    cmd = [
        sys.executable, str(UNDERLYING_EVALUATOR),
        "--capacities", args.capacities,
        "--data-read-root", args.data_read_root,
        "--out-dir", args.out_dir,
    ]
    print(f"\n[launch] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
