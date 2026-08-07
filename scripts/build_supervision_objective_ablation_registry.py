"""Build and freeze the model registry for the supervision-objective
ablation (docs/supervision_objective_ablation_protocol.md).

Requires all 4 objectives x 7 folds = 28 models to exist before writing
MODEL_SELECTION_FROZEN=true. No held-out evaluation may begin before this
gate passes (Section 6 of the orchestration task).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from lafc.experiments.external_baseline_common import base_provenance, sha256_of_file

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
OBJECTIVES = ["objective_eviction_loss", "objective_next_arrival", "objective_reuse_distance", "objective_pairwise"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", type=Path, default=Path("models/supervision_objective_ablation_v1"))
    ap.add_argument("--metrics-dir", type=Path, default=Path("analysis/supervision_objective_ablation_v1/training"))
    ap.add_argument("--out", type=Path, default=Path("analysis/supervision_objective_ablation_v1/model_registry.json"))
    ap.add_argument("--allow-incomplete", action="store_true",
                     help="Write a registry with MODEL_SELECTION_FROZEN=false even if models are missing "
                     "(diagnostic use only -- the eval runner refuses to run against an unfrozen registry).")
    ap.add_argument("--families", default=None,
                     help="Comma-separated subset of families to require (default: all 7). Freezes only "
                     "over this scoped subset -- for smoke-testing the registry/eval-runner wiring, not "
                     "for the real campaign (which must use the default full set).")
    ap.add_argument("--objectives", default=None, help="Comma-separated subset of objectives (default: all 4).")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",")] if args.families else FAMILIES
    objectives = [o.strip() for o in args.objectives.split(",")] if args.objectives else OBJECTIVES

    records: List[Dict[str, object]] = []
    missing: List[str] = []

    for family in families:
        fold_path = FOLDS_DIR / f"{family}.json"
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        metrics_path = args.metrics_dir / f"{family}.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {"objectives": {}}

        for objective in objectives:
            model_path = args.models_dir / objective / f"{family}.pkl"
            if not model_path.exists():
                missing.append(f"{objective}/{family}")
                continue
            obj_metrics = metrics.get("objectives", {}).get(objective, {})
            record = {
                "objective": objective,
                "held_out_family": family,
                "fold_id": fold["fold_id"],
                "training_families": fold["training_families"],
                "validation_family": fold["validation_family"],
                "protocol_id": "supervision_objective_ablation_v1",
                "dataset_manifest_path": f"data/derived/supervision_objective_ablation_v1/{family}/manifest.json",
                "selected_hyperparameters": obj_metrics.get("best_model_name", "shared_weight_mlp_pairwise"),
                "validation_metric": (
                    next(
                        (r["val_mean_regret"] for r in obj_metrics.get("comparison_rows", [])
                         if r["model"] == obj_metrics.get("best_model_name")),
                        None,
                    )
                    if objective != "objective_pairwise" else None
                ),
                "random_seed": obj_metrics.get("seed", 0),
                "model_artifact_path": str(model_path),
                "model_artifact_sha256": sha256_of_file(model_path),
                "expected_model_sha256_from_training": obj_metrics.get("model_sha256"),
            }
            if record["expected_model_sha256_from_training"] is not None and (
                record["expected_model_sha256_from_training"] != record["model_artifact_sha256"]
            ):
                raise ValueError(
                    f"Model artifact hash mismatch for {objective}/{family}: training-time hash "
                    f"{record['expected_model_sha256_from_training']} != current file hash "
                    f"{record['model_artifact_sha256']} -- artifact was modified after training."
                )
            records.append(record)

    frozen = len(missing) == 0 and len(records) == len(objectives) * len(families)
    if not frozen and not args.allow_incomplete:
        print(f"[BLOCKED] {len(missing)} model(s) missing: {missing}")
        print("Refusing to write a frozen registry. Pass --allow-incomplete for a diagnostic partial registry.")
        raise SystemExit(1)

    is_full_campaign = families == FAMILIES and objectives == OBJECTIVES
    registry = {
        **base_provenance(),
        "protocol_id": "supervision_objective_ablation_v1",
        "scope_families": families,
        "scope_objectives": objectives,
        "is_full_campaign_scope": is_full_campaign,
        "expected_model_count": len(objectives) * len(families),
        "actual_model_count": len(records),
        "missing_models": missing,
        "MODEL_SELECTION_FROZEN": frozen,
        "records": records,
    }
    import hashlib

    registry_bytes = json.dumps(registry, sort_keys=True).encode("utf-8")
    registry["registry_sha256"] = hashlib.sha256(registry_bytes).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}: {len(records)}/{len(objectives) * len(families)} models "
          f"(full_campaign_scope={is_full_campaign}), FROZEN={frozen}")


if __name__ == "__main__":
    main()
