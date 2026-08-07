"""Validate and freeze the model registry for evict_value_v1_cross_family_v1
(docs/reviewer_fairness_cross_family_v1.md, configs/reviewer_fairness_cross_family_v1.json).

Requires all 7 leave-one-family-out folds to have a promoted final model
before writing MODEL_SELECTION_FROZEN=true. No held-out evaluation may begin
before this gate passes (mirrors the pattern already used by
build_supervision_objective_ablation_registry.py in the sibling
supervision-objective-ablation worktree).

Per-fold validations (all must pass for that fold to count toward the 7):
  - fold config's test_family matches the family being checked;
  - training_families has exactly 5 entries, disjoint from
    {test_family, validation_family};
  - Stage-1 dataset manifest exists (dataset build reached completion);
  - the held-out family does not appear among the dataset manifest's own
    input trace families (fold isolation didn't leak upstream);
  - the promoted final model exists at the canonical top-level path
    (models/evict_value_v1_cross_family_v1_<family>.pkl) -- NOT the
    Stage-2 staging path (models/cross_family_v1_staging/<family>/...),
    which would mean a temporary artifact was substituted for the real one;
  - the model actually selected (best_config.json's "model") matches the
    validation-selected winner recorded in train_metrics.json
    (best_overall.model) -- i.e. no manual override slipped in unrecorded.

Fails closed: refuses to write MODEL_SELECTION_FROZEN=true (or write
anything at all, without --allow-incomplete) while fewer than 7 folds pass.

Usage:
    python scripts/experiments/finalize_cross_family_model_registry.py
    python scripts/experiments/finalize_cross_family_model_registry.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from lafc.experiments.external_baseline_common import base_provenance, sha256_of_file

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
PROTOCOL_CONFIG = Path("configs/reviewer_fairness_cross_family_v1.json")
MODELS_DIR = Path("models")
STAGING_DIR = Path("models/cross_family_v1_staging")
METRICS_DIR = Path("analysis/reviewer_fairness_cross_family_v1")
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]


def _load_fold(family: str) -> Dict[str, object]:
    path = FOLDS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_fold(family: str, protocol_hash: str) -> Dict[str, object]:
    """Return a registry record for `family`, or raise a descriptive
    exception (caller catches it and adds `family` to `missing`)."""
    fold = _load_fold(family)
    if fold["test_family"] != family:
        raise ValueError(f"Fold file {family}.json has test_family={fold['test_family']!r}, expected {family!r}")

    training_families = fold["training_families"]
    validation_family = fold["validation_family"]
    if len(training_families) != 5:
        raise ValueError(f"Fold {family}: expected 5 training families, got {len(training_families)}")
    overlap = set(training_families) & {family, validation_family}
    if overlap:
        raise ValueError(f"Fold {family}: training_families overlaps held-out/validation family: {overlap}")

    dataset_manifest_path = Path(fold["dataset_output_root"]) / "manifest.json"
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"Fold {family}: Stage-1 dataset manifest not found at {dataset_manifest_path}")
    ds_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    seen_families = {
        s.get("trace_family") for s in ds_manifest.get("preflight", {}).get("trace_stats", [])
    }
    if family in seen_families:
        raise ValueError(
            f"Fold {family}: held-out family appears in its own Stage-1 dataset manifest's input "
            "traces -- fold isolation failed upstream."
        )

    model_path = Path(fold["model_output_path"])
    expected_name = f"evict_value_v1_cross_family_v1_{family}.pkl"
    if model_path.name != expected_name or model_path.parent != MODELS_DIR:
        raise ValueError(
            f"Fold {family}: model_output_path {model_path} is not the canonical top-level "
            f"promoted path models/{expected_name} -- refusing to guess."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Fold {family}: promoted final model not found at {model_path}")
    # model_path.parent == MODELS_DIR (checked above) already rules out the
    # Stage-2 staging path (models/cross_family_v1_staging/<family>/...),
    # whose parent is a subdirectory, not models/ itself.

    metrics_dir = METRICS_DIR / family
    train_metrics_path = metrics_dir / "train_metrics.json"
    best_config_path = metrics_dir / "best_config.json"
    if not train_metrics_path.exists() or not best_config_path.exists():
        raise FileNotFoundError(f"Fold {family}: train_metrics.json / best_config.json not found under {metrics_dir}")
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    best_config = json.loads(best_config_path.read_text(encoding="utf-8"))

    winner = train_metrics.get("best_overall", {}).get("model")
    selected = best_config.get("model")
    if winner is None or selected is None or winner != selected:
        raise ValueError(
            f"Fold {family}: selected model {selected!r} does not match the validation-selected "
            f"winner {winner!r} recorded in train_metrics.json -- refusing to freeze an "
            "unrecorded/overridden selection."
        )

    return {
        "held_out_family": family,
        "fold_id": fold["fold_id"],
        "validation_family": validation_family,
        "training_families": training_families,
        "protocol_id": "reviewer_fair_cross_family_v1",
        "protocol_config_sha256": protocol_hash,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_of_file(dataset_manifest_path),
        "selected_hyperparameters": selected,
        "validation_metric": train_metrics.get("best_overall", {}).get("val_mean_regret"),
        "model_artifact_path": str(model_path),
        "model_artifact_sha256": sha256_of_file(model_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--out", type=Path, default=METRICS_DIR / "model_registry.json")
    ap.add_argument("--allow-incomplete", action="store_true",
                     help="Write a registry with MODEL_SELECTION_FROZEN=false even if folds are missing "
                     "(diagnostic use only -- the eval launcher refuses to run against an unfrozen registry).")
    ap.add_argument("--families", default=None,
                     help="Comma-separated subset of families to require (default: all 7). Scoped freeze "
                     "for smoke-testing the registry/eval-launcher wiring only -- the real campaign must "
                     "use the default full set.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Run all checks and print what would be written, but do not write the registry file.")
    args = ap.parse_args()

    if not PROTOCOL_CONFIG.exists():
        raise FileNotFoundError(f"Frozen protocol config not found: {PROTOCOL_CONFIG}")
    protocol_hash = sha256_of_file(PROTOCOL_CONFIG)

    families = [f.strip() for f in args.families.split(",")] if args.families else FAMILIES

    records: List[Dict[str, object]] = []
    missing: List[str] = []
    for family in families:
        try:
            records.append(_check_fold(family, protocol_hash))
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{family}: {exc}")

    frozen = len(missing) == 0 and len(records) == len(families)
    if not frozen and not args.allow_incomplete:
        print(f"[BLOCKED] {len(missing)}/{len(families)} fold(s) not ready:")
        for m in missing:
            print(f"  - {m}")
        print("Refusing to write a frozen registry. Pass --allow-incomplete for a diagnostic partial "
              "registry, or --dry-run to see this report without side effects.")
        raise SystemExit(1)

    is_full_campaign = families == FAMILIES
    registry = {
        **base_provenance(),
        "protocol_id": "reviewer_fair_cross_family_v1",
        "protocol_config_sha256": protocol_hash,
        "scope_families": families,
        "is_full_campaign_scope": is_full_campaign,
        "expected_model_count": len(families),
        "actual_model_count": len(records),
        "missing_folds": missing,
        "MODEL_SELECTION_FROZEN": frozen,
        "records": records,
    }
    registry_bytes = json.dumps(registry, sort_keys=True).encode("utf-8")
    registry["registry_sha256"] = hashlib.sha256(registry_bytes).hexdigest()

    if args.dry_run:
        print(f"[dry-run] would write {args.out}: {len(records)}/{len(families)} folds "
              f"(full_campaign_scope={is_full_campaign}), FROZEN={frozen}")
        if missing:
            print("[dry-run] missing:")
            for m in missing:
                print(f"  - {m}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}: {len(records)}/{len(families)} folds "
          f"(full_campaign_scope={is_full_campaign}), FROZEN={frozen}")


if __name__ == "__main__":
    main()
