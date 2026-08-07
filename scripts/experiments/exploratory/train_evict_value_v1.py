from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: List[Dict[str, object]] = []
    for r in rows:
        row: Dict[str, object] = dict(r)
        for c in EVICT_VALUE_V1_FEATURE_COLUMNS + ["y_loss", "y_value"]:
            row[c] = float(row[c])
        row["horizon"] = int(row["horizon"])
        out.append(row)
    return out


def _select_horizon(rows: List[Dict[str, object]], horizon: int) -> List[Dict[str, object]]:
    return [r for r in rows if int(r["horizon"]) == horizon]


def _xy(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.asarray([float(r["y_loss"]) for r in rows], dtype=float)
    return x, y


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "pred_mean": float(np.mean(pred)) if len(pred) else 0.0,
        "target_mean": float(np.mean(y)) if len(y) else 0.0,
    }


def _ranking_metrics(rows: List[Dict[str, object]], preds: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, pred in zip(rows, preds):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(pred)))

    top1 = 0
    regrets: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x[0]["y_loss"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == best[0]["candidate_page_id"])
        regrets.append(float(chosen[0]["y_loss"]) - float(best[0]["y_loss"]))

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_eviction_match": float(top1 / denom),
        "mean_regret_vs_oracle": float(np.mean(regrets) if regrets else 0.0),
    }


def _feature_importance(estimator: object) -> Dict[str, float]:
    base = estimator
    if hasattr(estimator, "named_steps"):
        base = estimator.named_steps.get("reg", estimator)
    if hasattr(base, "feature_importances_"):
        vals = base.feature_importances_
    elif hasattr(base, "coef_"):
        vals = np.abs(base.coef_)
    else:
        return {}
    return {c: float(v) for c, v in zip(EVICT_VALUE_V1_FEATURE_COLUMNS, vals)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train model-family sweep for evict_value_v1")
    ap.add_argument("--data-dir", default="data/derived")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metrics-json", default="analysis/evict_value_v1_metrics.json")
    ap.add_argument("--comparison-csv", default="analysis/evict_value_v1_model_comparison.csv")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train = _select_horizon(_read_csv(data_dir / "evict_value_v1_train.csv"), args.horizon)
    val = _select_horizon(_read_csv(data_dir / "evict_value_v1_val.csv"), args.horizon)
    test = _select_horizon(_read_csv(data_dir / "evict_value_v1_test.csv"), args.horizon)
    x_train, y_train = _xy(train)
    x_val, y_val = _xy(val)
    x_test, y_test = _xy(test)

    models: Dict[str, object] = {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=4, random_state=args.seed),
        "hist_gb": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=args.seed),
    }

    comparison_rows: List[Dict[str, object]] = []
    payload: Dict[str, object] = {"horizon": args.horizon, "models": {}}

    best_name = None
    best_val_regret = float("inf")
    for name, est in models.items():
        est.fit(x_train, y_train)
        p_train = est.predict(x_train)
        p_val = est.predict(x_val)
        p_test = est.predict(x_test)

        m_train = _metrics(y_train, p_train)
        m_val = _metrics(y_val, p_val)
        m_test = _metrics(y_test, p_test)
        r_train = _ranking_metrics(train, p_train)
        r_val = _ranking_metrics(val, p_val)
        r_test = _ranking_metrics(test, p_test)

        if r_val["mean_regret_vs_oracle"] < best_val_regret:
            best_val_regret = r_val["mean_regret_vs_oracle"]
            best_name = name

        payload["models"][name] = {
            "train": m_train,
            "val": m_val,
            "test": m_test,
            "ranking_train": r_train,
            "ranking_val": r_val,
            "ranking_test": r_test,
            "feature_importance": _feature_importance(est),
        }

        comparison_rows.append(
            {
                "model": name,
                "val_mae": m_val["mae"],
                "test_mae": m_test["mae"],
                "val_rmse": m_val["rmse"],
                "test_rmse": m_test["rmse"],
                "val_top1_eviction_match": r_val["top1_eviction_match"],
                "test_top1_eviction_match": r_test["top1_eviction_match"],
                "val_mean_regret": r_val["mean_regret_vs_oracle"],
                "test_mean_regret": r_test["mean_regret_vs_oracle"],
            }
        )

        EvictValueV1Model(model_name=name, estimator=est, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS)).save(
            Path(args.models_dir) / f"evict_value_v1_{name}.pkl"
        )

    payload["winner_by_val_regret"] = best_name

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
