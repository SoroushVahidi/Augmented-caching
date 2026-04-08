from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lafc.learned_gate.features_v2 import ML_GATE_V2_FEATURE_COLUMNS
from lafc.learned_gate.model_v2 import LearnedGateV2Model


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: List[Dict[str, object]] = []
    for r in rows:
        row: Dict[str, object] = dict(r)
        for c in ML_GATE_V2_FEATURE_COLUMNS + ["y_reg"]:
            row[c] = float(row[c])
        row["y_cls"] = int(row["y_cls"])
        row["horizon"] = int(row["horizon"])
        out.append(row)
    return out


def _select_horizon(rows: List[Dict[str, object]], horizon: int) -> List[Dict[str, object]]:
    return [r for r in rows if int(r["horizon"]) == horizon]


def _xy(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in ML_GATE_V2_FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.asarray([int(r["y_cls"]) for r in rows], dtype=int)
    return x, y


def _metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= thr).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "class_balance": float(np.mean(y)) if len(y) else 0.0,
        "pred_positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
        "brier": float(brier_score_loss(y, prob)) if len(y) else 0.0,
    }
    out["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else 0.5
    return out


def _feature_importance(estimator: object) -> Dict[str, float]:
    if hasattr(estimator, "feature_importances_"):
        vals = getattr(estimator, "feature_importances_")
    elif hasattr(estimator, "named_steps") and hasattr(estimator.named_steps.get("clf"), "coef_"):
        vals = np.abs(estimator.named_steps["clf"].coef_[0])
    else:
        return {}
    return {c: float(v) for c, v in zip(ML_GATE_V2_FEATURE_COLUMNS, vals)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train model family sweep for ml_gate_v2")
    ap.add_argument("--data-dir", default="data/derived")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--metrics-json", default="analysis/ml_gate_v2_metrics.json")
    ap.add_argument("--comparison-csv", default="analysis/ml_gate_v2_model_comparison.csv")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train = _select_horizon(_read_csv(data_dir / "ml_gate_v2_train.csv"), args.horizon)
    val = _select_horizon(_read_csv(data_dir / "ml_gate_v2_val.csv"), args.horizon)
    test = _select_horizon(_read_csv(data_dir / "ml_gate_v2_test.csv"), args.horizon)

    x_train, y_train = _xy(train)
    x_val, y_val = _xy(val)
    x_test, y_test = _xy(test)

    models: Dict[str, object] = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
        ]),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=args.seed),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=args.seed),
        "hist_gb": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=200, random_state=args.seed),
    }

    comparison_rows: List[Dict[str, object]] = []
    payload: Dict[str, object] = {
        "horizon": args.horizon,
        "threshold": args.threshold,
        "models": {},
    }

    best_name = None
    best_val_f1 = -1.0

    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        p_train = estimator.predict_proba(x_train)[:, 1]
        p_val = estimator.predict_proba(x_val)[:, 1]
        p_test = estimator.predict_proba(x_test)[:, 1]

        m_train = _metrics(y_train, p_train, args.threshold)
        m_val = _metrics(y_val, p_val, args.threshold)
        m_test = _metrics(y_test, p_test, args.threshold)
        if m_val["f1"] > best_val_f1:
            best_val_f1 = m_val["f1"]
            best_name = name

        calib = {}
        if len(np.unique(y_test)) > 1:
            frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=5, strategy="uniform")
            calib = {"mean_pred": [float(x) for x in mean_pred], "frac_pos": [float(x) for x in frac_pos]}

        payload["models"][name] = {
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "feature_importance": _feature_importance(estimator),
            "calibration": calib,
        }

        comparison_rows.append({
            "model": name,
            "train_f1": m_train["f1"],
            "val_f1": m_val["f1"],
            "test_f1": m_test["f1"],
            "val_auc": m_val["roc_auc"],
            "test_auc": m_test["roc_auc"],
            "test_pred_positive_rate": m_test["pred_positive_rate"],
        })

        artifact = LearnedGateV2Model(model_name=name, estimator=estimator, feature_columns=list(ML_GATE_V2_FEATURE_COLUMNS), threshold=args.threshold)
        artifact.save(Path(args.models_dir) / f"ml_gate_v2_{name}.pkl")

    payload["winner_by_val_f1"] = best_name

    cmp_path = Path(args.comparison_csv)
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    with cmp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
