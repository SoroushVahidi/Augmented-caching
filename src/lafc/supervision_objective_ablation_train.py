"""Objective-agnostic model training for the supervision-objective ablation
(docs/supervision_objective_ablation_protocol.md,
configs/supervision_objective_ablation_v1.json).

Reuses the EXACT scalar model family / hyperparameter grid and selection
rule as scripts/train_evict_value_wulver_v1.py (the canonical
objective_eviction_loss reference architecture), generalized only over
which label column is the regression target and whether the eviction rule
is argmin or argmax -- per the protocol's hyperparameter-fairness
requirement, no objective receives a larger or smaller search budget than
another. The pairwise objective reuses src/lafc/halp_model.py's
shared-weight two-layer MLP unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.halp_model import HALPModel

FEATURES = list(EVICT_VALUE_V1_FEATURE_COLUMNS)

Direction = Literal["min", "max"]


def _scalar_model_family(seed: int) -> Dict[str, object]:
    return {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=80, max_depth=12, min_samples_leaf=4, random_state=seed, n_jobs=1
        ),
        "hist_gb": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=150, random_state=seed
        ),
    }


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def _ranking_metrics(
    rows: List[Dict[str, object]], preds: np.ndarray, label_column: str, direction: Direction
) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, pred in zip(rows, preds):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(pred)))

    pick = min if direction == "min" else max
    top1 = 0
    regrets: List[float] = []
    for items in grouped.values():
        chosen = pick(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = pick(items, key=lambda x: (float(x[0][label_column]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == best[0]["candidate_page_id"])
        chosen_true = float(chosen[0][label_column])
        best_true = float(best[0][label_column])
        regret = (chosen_true - best_true) if direction == "min" else (best_true - chosen_true)
        regrets.append(regret)
    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_eviction_match": float(top1 / denom),
        "mean_regret_vs_oracle": float(np.mean(regrets) if regrets else 0.0),
    }


def _xy(rows: List[Dict[str, object]], label_column: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in FEATURES] for r in rows], dtype=float)
    y = np.asarray([float(r[label_column]) for r in rows], dtype=float)
    return x, y


@dataclass(frozen=True)
class ScalarTrainResult:
    objective: str
    label_column: str
    direction: Direction
    best_model_name: str
    best_model: EvictValueV1Model
    comparison_rows: List[Dict[str, object]]


def train_scalar_objective(
    *,
    objective: str,
    label_column: str,
    direction: Direction,
    train_rows: List[Dict[str, object]],
    val_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    seed: int = 0,
) -> ScalarTrainResult:
    """Train and select among {ridge, random_forest, hist_gb} for a single
    scalar supervision objective. Selection rule (identical across all
    scalar objectives): minimize validation mean_regret_vs_oracle, tie-break
    lower val MAE then val RMSE.
    """
    x_train, y_train = _xy(train_rows, label_column)
    x_val, y_val = _xy(val_rows, label_column)
    test_rows_eff = test_rows or val_rows
    x_test, y_test = _xy(test_rows_eff, label_column)

    models = _scalar_model_family(seed)
    comparison_rows: List[Dict[str, object]] = []
    fitted: Dict[str, object] = {}

    for name, est in models.items():
        est.fit(x_train, y_train)
        fitted[name] = est
        p_val = est.predict(x_val)
        p_test = est.predict(x_test)
        m_val = _metrics(y_val, p_val)
        m_test = _metrics(y_test, p_test)
        r_val = _ranking_metrics(val_rows, p_val, label_column, direction)
        r_test = _ranking_metrics(test_rows_eff, p_test, label_column, direction)
        comparison_rows.append(
            {
                "objective": objective,
                "model": name,
                "val_mae": m_val["mae"],
                "val_rmse": m_val["rmse"],
                "test_mae": m_test["mae"],
                "test_rmse": m_test["rmse"],
                "val_top1": r_val["top1_eviction_match"],
                "test_top1": r_test["top1_eviction_match"],
                "val_mean_regret": r_val["mean_regret_vs_oracle"],
                "test_mean_regret": r_test["mean_regret_vs_oracle"],
            }
        )

    best_row = min(comparison_rows, key=lambda r: (r["val_mean_regret"], r["val_mae"], r["val_rmse"]))
    best_name = str(best_row["model"])
    best_model = EvictValueV1Model(
        model_name=f"{objective}_{best_name}", estimator=fitted[best_name], feature_columns=list(FEATURES)
    )
    return ScalarTrainResult(
        objective=objective,
        label_column=label_column,
        direction=direction,
        best_model_name=best_name,
        best_model=best_model,
        comparison_rows=comparison_rows,
    )


@dataclass(frozen=True)
class PairwiseTrainResult:
    objective: str
    model: HALPModel
    n_train_pairs: int


def train_pairwise_objective(
    *,
    objective: str,
    train_pairs: List[Dict[str, object]],
    seed: int = 0,
) -> PairwiseTrainResult:
    """Train the shared-weight pairwise preference model (reused unmodified
    from src/lafc/halp_model.py) on next-arrival-ordering pairwise labels
    produced by build_pairwise_rows(..., source="next_arrival").
    """
    pref_rows: List[List[float]] = []
    nonpref_rows: List[List[float]] = []
    for p in train_pairs:
        i_feats = [float(p[f"i_{c}"]) for c in FEATURES]
        j_feats = [float(p[f"j_{c}"]) for c in FEATURES]
        if int(p["label_i_preferred"]) == 1:
            pref_rows.append(i_feats)
            nonpref_rows.append(j_feats)
        else:
            pref_rows.append(j_feats)
            nonpref_rows.append(i_feats)

    model = HALPModel(seed=seed)
    if pref_rows:
        model.fit(np.asarray(pref_rows, dtype=float), np.asarray(nonpref_rows, dtype=float))
    return PairwiseTrainResult(objective=objective, model=model, n_train_pairs=len(pref_rows))


__all__ = [
    "FEATURES",
    "train_scalar_objective",
    "train_pairwise_objective",
    "ScalarTrainResult",
    "PairwiseTrainResult",
]
