"""Fairness-alignment audit for the supervision-objective ablation
(docs/supervision_objective_ablation_protocol.md).

For each held-out-family fold, validates that the 4 objectives
(objective_eviction_loss, objective_next_arrival, objective_reuse_distance,
objective_pairwise) were compared under matched conditions, using
analysis/supervision_objective_ablation_v1/training/<family>.json (written
by scripts/train_supervision_objective_ablation.py) plus the shared fold
config (configs/fair_cross_family_v1/folds/<family>.json, reused unchanged
by every objective -- see docs/supervision_objective_ablation_protocol.md):

  - same training/validation/held-out family (trivially true by
    construction -- one shared fold config -- but asserted, not assumed);
  - same request budget for the 3 scalar objectives (n_train_rows,
    n_val_rows must match exactly -- they draw from the same shared scalar
    shard set, see audit_supervision_objective_examples.py);
  - same model search budget for the 3 scalar objectives (identical
    {ridge, random_forest, hist_gb} sweep -- no objective got a larger or
    smaller search than another);
  - pairwise's structurally different setup (n_train_pairs, single fixed
    architecture, no per-model sweep) is explicitly recorded as a
    documented difference, not silently hidden or treated as a fairness
    violation;
  - each scalar objective's best_model_name is actually the val-optimal
    model in its own comparison_rows (selection used validation data only
    -- this file has no visibility into held-out results at all, so this
    also structurally rules out held-out-informed selection).

Fails closed if a fold's training-metrics file is missing/malformed for a
requested family.

Usage:
    python scripts/experiments/audit_supervision_objective_fairness.py --partial-audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
METRICS_DIR = Path("analysis/supervision_objective_ablation_v1/training")
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
SCALAR_OBJECTIVES = ["objective_eviction_loss", "objective_next_arrival", "objective_reuse_distance"]
EXPECTED_MODEL_SWEEP = {"ridge", "random_forest", "hist_gb"}
# Model SELECTION is always min(val_mean_regret, val_mae, val_rmse) for every
# scalar objective, regardless of that objective's own "direction" field --
# "direction" governs how mean_regret_vs_oracle is computed internally (pick
# max- or min-label candidate as "best" when scoring regret), not how models
# are compared against each other afterward. See
# lafc.supervision_objective_ablation_train.train_scalar_objective, line:
#   best_row = min(comparison_rows, key=lambda r: (r["val_mean_regret"], r["val_mae"], r["val_rmse"]))


def _load_fold(family: str) -> Dict[str, object]:
    path = FOLDS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(family: str) -> Optional[Dict[str, object]]:
    path = METRICS_DIR / f"{family}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def audit_fold(family: str) -> Dict[str, object]:
    fold = _load_fold(family)
    metrics = _load_metrics(family)
    if metrics is None:
        return {"family": family, "status": "NOT_BUILT", "checks": {}}

    objectives = metrics.get("objectives", {})
    checks: Dict[str, str] = {}

    checks["fold_identity"] = (
        "PASS" if metrics.get("held_out_family") == family and metrics.get("fold_id") == fold.get("fold_id")
        else f"FAIL (metrics fold_id={metrics.get('held_out_family')}/{metrics.get('fold_id')} != "
             f"expected {family}/{fold.get('fold_id')})"
    )

    missing_objectives = [o for o in SCALAR_OBJECTIVES + ["objective_pairwise"] if o not in objectives]
    checks["all_objectives_present"] = "PASS" if not missing_objectives else f"FAIL (missing: {missing_objectives})"
    if missing_objectives:
        return {"family": family, "status": "FAIL", "checks": checks}

    n_train = {o: objectives[o].get("n_train_rows") for o in SCALAR_OBJECTIVES}
    n_val = {o: objectives[o].get("n_val_rows") for o in SCALAR_OBJECTIVES}
    checks["scalar_request_budget_identical"] = (
        "PASS" if len(set(n_train.values())) == 1 and len(set(n_val.values())) == 1
        else f"FAIL (n_train_rows={n_train}, n_val_rows={n_val} differ across scalar objectives)"
    )

    sweeps = {
        o: {r["model"] for r in objectives[o].get("comparison_rows", [])}
        for o in SCALAR_OBJECTIVES
    }
    checks["scalar_model_search_budget_identical"] = (
        "PASS" if all(s == EXPECTED_MODEL_SWEEP for s in sweeps.values())
        else f"FAIL (model sweeps: {sweeps}, expected {EXPECTED_MODEL_SWEEP} for all)"
    )

    pairwise = objectives["objective_pairwise"]
    checks["pairwise_difference_documented"] = "DOCUMENTED_DIFFERENT_SETUP" if (
        "n_train_pairs" in pairwise and "comparison_rows" not in pairwise
    ) else f"UNEXPECTED_SHAPE (pairwise keys: {sorted(pairwise.keys())})"

    selection_ok = True
    selection_detail = []
    for o in SCALAR_OBJECTIVES:
        obj = objectives[o]
        rows = obj.get("comparison_rows", [])
        if not rows or not all("val_mean_regret" in r and "val_mae" in r and "val_rmse" in r for r in rows):
            selection_ok = False
            selection_detail.append(f"{o}: comparison_rows missing val_mean_regret/val_mae/val_rmse")
            continue
        expected_winner = min(rows, key=lambda r: (r["val_mean_regret"], r["val_mae"], r["val_rmse"]))["model"]
        if expected_winner != obj.get("best_model_name"):
            selection_ok = False
            selection_detail.append(
                f"{o}: best_model_name={obj.get('best_model_name')!r} but the recorded selection rule "
                f"(min val_mean_regret, tie-break val_mae/val_rmse) picks {expected_winner!r}"
            )
    checks["model_selection_used_validation_only"] = "PASS" if selection_ok else f"FAIL ({'; '.join(selection_detail)})"

    material_checks = {k: v for k, v in checks.items() if k != "pairwise_difference_documented"}
    status = "PASS" if all(v == "PASS" for v in material_checks.values()) else "FAIL"
    return {"family": family, "status": status, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=None, help="Comma-separated subset (default: all 7).")
    ap.add_argument("--out", type=Path, default=Path("analysis/supervision_objective_ablation_v1/fairness_audit.json"))
    ap.add_argument("--partial-audit", action="store_true",
                     help="Required to run/write while fewer than 7 folds have training metrics. Sets "
                     "FINAL=false in the output.")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",")] if args.families else FAMILIES
    reports: List[Dict[str, object]] = [audit_fold(f) for f in families]

    built = [r for r in reports if r["status"] != "NOT_BUILT"]
    not_built = [r["family"] for r in reports if r["status"] == "NOT_BUILT"]
    is_final = len(not_built) == 0 and families == FAMILIES

    if not is_final and not args.partial_audit:
        print(f"[BLOCKED] {len(not_built)}/{len(families)} fold(s) have no training metrics yet: {not_built}")
        print("Pass --partial-audit to write a partial (FINAL=false) report anyway.")
        raise SystemExit(1)

    output = {
        "protocol_id": "supervision_objective_ablation_v1",
        "FINAL": is_final,
        "scope_families": families,
        "folds_built": [r["family"] for r in built],
        "folds_not_built": not_built,
        "overall": "PASS" if built and all(r["status"] == "PASS" for r in built) else (
            "FAIL" if any(r["status"] == "FAIL" for r in built) else "NO_DATA"
        ),
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}: FINAL={is_final} overall={output['overall']} "
          f"({len(built)}/{len(families)} folds built)")
    if not is_final:
        print("[note] This is a PARTIAL audit (FINAL=false) -- do not cite as the completed campaign's result.")


if __name__ == "__main__":
    main()
