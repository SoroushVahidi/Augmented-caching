from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


DEFAULT_FEATURES = [
    "raw_score",
    "depth",
    "expansions",
    "verifications",
    "remaining_budget",
    "post_verify_score",
    "score_delta",
    "survived_pruning_steps",
]


def _load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_top1_accuracy(rows: List[Dict[str, object]], probs: np.ndarray, target_col: str) -> float:
    grouped: Dict[tuple[str, str], List[tuple[int, float]]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = (str(r["example_id"]), str(r["decision_step"]))
        grouped[key].append((i, float(r[target_col])))

    hits = 0
    total = 0
    for _, idx_targets in grouped.items():
        if len(idx_targets) < 2:
            continue
        total += 1
        pred_i = max(idx_targets, key=lambda x: probs[x[0]])[0]
        true_i = max(idx_targets, key=lambda x: x[1])[0]
        hits += int(pred_i == true_i)
    return float(hits / total) if total > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train v2 continuation-value branch scorer")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", default="target_expand_better_than_baseline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = _load_csv(Path(args.dataset))
    if not rows:
        raise SystemExit("Dataset was empty")

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    def xy(rs: List[Dict[str, object]]):
        X = np.array([[float(r.get(f, 0.0)) for f in DEFAULT_FEATURES] for r in rs], dtype=np.float64)
        y = np.array([int(float(r[args.target])) for r in rs], dtype=np.int64)
        return X, y

    X_train, y_train = xy(train_rows)
    X_val, y_val = xy(val_rows)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    model.fit(X_train, y_train)

    def _eval(split_rows: List[Dict[str, object]], name: str) -> Dict[str, float]:
        if not split_rows:
            return {"rows": 0}
        X, y = xy(split_rows)
        p = model.predict_proba(X)[:, 1]
        pred = (p >= 0.5).astype(np.int64)
        out: Dict[str, float] = {
            "rows": int(len(split_rows)),
            "positive_rate": float(np.mean(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "top1_by_decision_point": float(_group_top1_accuracy(split_rows, p, "target_expand_value")),
        }
        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, p))
            out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
        return out

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_spec = {
        "feature_names": DEFAULT_FEATURES,
        "target_name": args.target,
        "notes": "v2 continuation-value approximation with episode-level split",
    }
    (out / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")
    coeff_payload = {
        "intercept": float(model.intercept_[0]),
        "coefficients": {k: float(v) for k, v in zip(DEFAULT_FEATURES, model.coef_[0])},
        "class_order": [int(c) for c in model.classes_],
        "model_family": "sklearn LogisticRegression",
    }
    (out / "model_coefficients.json").write_text(json.dumps(coeff_payload, indent=2), encoding="utf-8")

    summary = {
        "dataset": str(args.dataset),
        "model_name": "branch_scorer_lr_v2_continuation_value",
        "target": args.target,
        "features": DEFAULT_FEATURES,
        "split_metrics": {
            "train": _eval(train_rows, "train"),
            "val": _eval(val_rows, "val"),
            "test": _eval(test_rows, "test"),
        },
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
