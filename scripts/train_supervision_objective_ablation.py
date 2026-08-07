"""Fold-specific training for the supervision-objective ablation
(docs/supervision_objective_ablation_protocol.md).

For ONE held-out family's fold, trains all 4 objectives (3 scalar +
1 pairwise) on that fold's already-built shards
(data/derived/supervision_objective_ablation_v1/<family>/{scalar,pairwise}/shards/),
using ONLY the "train" split rows for fitting and ONLY the "val" split
rows for scalar model selection -- never touching the held-out family,
which never appears in these shards at all (verified upstream by
scripts/build_supervision_objective_ablation_dataset.py's isolation
assertions).

Row loading is capped (--max-train-rows / --max-val-rows) with a SAFE,
non-None default: the canonical single-objective pipeline OOM-killed
(57GB RSS) when it loaded an uncapped multi-family manifest into memory
during this exact stage (see build script's module docstring for the
journalctl evidence). The cap is applied identically to all three scalar
objectives -- same budget, same seed -- so no objective gets a size
advantage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.experiments.external_baseline_common import base_provenance, sha256_of_file
from lafc.supervision_objective_ablation_train import (
    FEATURES,
    train_pairwise_objective,
    train_scalar_objective,
)

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")

SCALAR_OBJECTIVES = {
    "objective_eviction_loss": ("eviction_loss_label", "min"),
    "objective_next_arrival": ("next_arrival_label_censored", "max"),
    "objective_reuse_distance": ("reuse_distance_label_censored", "max"),
}

NUMERIC_SCALAR_COLUMNS = (
    ["capacity", "horizon", "decision_t"]
    + [
        "eviction_loss_label", "next_arrival_label_raw", "next_arrival_label_censored",
        "next_arrival_censored_flag", "reuse_distance_label_raw", "reuse_distance_label_censored",
        "reuse_distance_censored_flag",
    ]
    + list(EVICT_VALUE_V1_FEATURE_COLUMNS)
)
NUMERIC_PAIRWISE_COLUMNS = (
    ["capacity", "horizon", "value_i", "value_j", "label_i_preferred", "is_tie"]
    + [f"i_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
    + [f"j_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
    + [f"delta_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
)


def _load_shard_rows(
    shard_paths: List[str], split: str, numeric_cols: List[str], max_rows: int, seed: int
) -> List[Dict[str, object]]:
    """Stream shard CSVs, filter by split, reservoir-style cap via
    random.sample on a bounded buffer (matches the reference architecture's
    existing --max-train-rows subsampling pattern)."""
    rows: List[Dict[str, object]] = []
    rng = random.Random(seed)
    paths = list(shard_paths)
    rng.shuffle(paths)
    for sp in paths:
        p = Path(sp)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if r.get("split") != split:
                    continue
                row = dict(r)
                for c in numeric_cols:
                    row[c] = float(row[c])
                rows.append(row)
        if len(rows) >= max_rows * 3:
            break
    if len(rows) > max_rows:
        rng2 = random.Random(seed + 1)
        rows = rng2.sample(rows, max_rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--held-out-family", required=True)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=Path("models/supervision_objective_ablation_v1"))
    ap.add_argument("--metrics-dir", type=Path, default=Path("analysis/supervision_objective_ablation_v1/training"))
    ap.add_argument("--max-train-rows", type=int, default=150000,
                     help="Resource-safety cap, identical across all scalar objectives.")
    ap.add_argument("--max-val-rows", type=int, default=30000)
    ap.add_argument("--max-train-pairs", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    family = args.held_out_family
    fold = json.loads((FOLDS_DIR / f"{family}.json").read_text(encoding="utf-8"))
    dataset_root = args.dataset_root or Path(f"data/derived/supervision_objective_ablation_v1/{family}")
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["held_out_family"] != family:
        raise ValueError(f"Dataset manifest held_out_family={manifest['held_out_family']} != requested {family}")
    if manifest["fold_id"] != fold["fold_id"]:
        raise ValueError(f"Dataset manifest fold_id={manifest['fold_id']} != fold's own fold_id {fold['fold_id']}")

    scalar_shard_paths = [s["path"] for s in manifest["scalar_shards"]]
    pairwise_shard_paths = [s["path"] for s in manifest["pairwise_shards"]]

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, object] = {"held_out_family": family, "fold_id": fold["fold_id"], "objectives": {}}

    train_rows = _load_shard_rows(scalar_shard_paths, "train", NUMERIC_SCALAR_COLUMNS, args.max_train_rows, args.seed)
    val_rows = _load_shard_rows(scalar_shard_paths, "val", NUMERIC_SCALAR_COLUMNS, args.max_val_rows, args.seed + 1)
    if not train_rows or not val_rows:
        raise ValueError(f"Fold {family}: insufficient rows train={len(train_rows)} val={len(val_rows)}")

    for objective, (label_col, direction) in SCALAR_OBJECTIVES.items():
        result = train_scalar_objective(
            objective=objective, label_column=label_col, direction=direction,
            train_rows=train_rows, val_rows=val_rows, test_rows=[], seed=args.seed,
        )
        obj_dir = args.models_dir / objective
        obj_dir.mkdir(parents=True, exist_ok=True)
        model_path = obj_dir / f"{family}.pkl"
        result.best_model.save(model_path)
        model_hash = sha256_of_file(model_path)
        all_metrics["objectives"][objective] = {
            "label_column": label_col, "direction": direction,
            "best_model_name": result.best_model_name,
            "comparison_rows": result.comparison_rows,
            "n_train_rows": len(train_rows), "n_val_rows": len(val_rows),
            "model_path": str(model_path), "model_sha256": model_hash, "seed": args.seed,
        }
        print(f"[trained] {objective} fold={family}: best={result.best_model_name} model_hash={model_hash[:12]}")

    train_pairs = _load_shard_rows(
        pairwise_shard_paths, "train", NUMERIC_PAIRWISE_COLUMNS, args.max_train_pairs, args.seed + 2
    )
    if not train_pairs:
        raise ValueError(
            f"Fold {family}: zero pairwise training pairs (see label_statistics.json -- some families may "
            "have near-zero distinguishable next-arrival pairs at H=4, e.g. wiki2018's near-unique traffic)."
        )
    pw_result = train_pairwise_objective(objective="objective_pairwise", train_pairs=train_pairs, seed=args.seed)
    pw_dir = args.models_dir / "objective_pairwise"
    pw_dir.mkdir(parents=True, exist_ok=True)
    pw_model_path = pw_dir / f"{family}.pkl"
    pw_result.model.save(pw_model_path)
    pw_hash = sha256_of_file(pw_model_path)
    all_metrics["objectives"]["objective_pairwise"] = {
        "label_source": "next_arrival", "n_train_pairs": pw_result.n_train_pairs,
        "model_path": str(pw_model_path), "model_sha256": pw_hash, "seed": args.seed,
    }
    print(f"[trained] objective_pairwise fold={family}: n_pairs={pw_result.n_train_pairs} model_hash={pw_hash[:12]}")

    metrics_path = args.metrics_dir / f"{family}.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    provenance = {**base_provenance(), "held_out_family": family, "fold_id": fold["fold_id"]}
    (args.metrics_dir / f"{family}_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
