from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from lafc.evict_value_pairwise_model_v1 import EvictValuePairwiseV1Model


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))

    rows: List[Dict[str, object]] = []
    numeric_cols = {
        "capacity",
        "request_t",
        "horizon",
        "label_i_better",
        "rollout_regret_i",
        "rollout_regret_j",
        "rollout_loss_i",
        "rollout_loss_j",
    }
    for row in raw:
        parsed: Dict[str, object] = dict(row)
        for col in list(parsed.keys()):
            if col.startswith(("delta_", "i_", "j_")) or col in numeric_cols:
                parsed[col] = float(parsed[col])
        rows.append(parsed)
    return rows


def _split(trace_name: str) -> str:
    bucket = int(hashlib.md5(trace_name.encode("utf-8")).hexdigest(), 16) % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def _xy(rows: List[Dict[str, object]], delta_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    return x, y


def _decision_metrics(rows: List[Dict[str, object]], p_i_better: np.ndarray) -> Dict[str, float]:
    decisions: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row, p in zip(rows, p_i_better):
        d = decisions.setdefault(str(row["decision_id"]), {"wins": {}, "regret": {}})
        ai = str(row["candidate_i_page_id"])
        bi = str(row["candidate_j_page_id"])
        d["wins"][ai] = float(d["wins"].get(ai, 0.0) + p)
        d["wins"][bi] = float(d["wins"].get(bi, 0.0) + (1.0 - p))
        d["regret"][ai] = float(row["rollout_regret_i"])
        d["regret"][bi] = float(row["rollout_regret_j"])

    top1 = 0
    regrets: List[float] = []
    for data in decisions.values():
        chosen = max(data["wins"].keys(), key=lambda c: (data["wins"][c], c))
        best = min(data["regret"].keys(), key=lambda c: (data["regret"][c], c))
        top1 += int(chosen == best)
        regrets.append(float(data["regret"][chosen]))
    denom = max(len(decisions), 1)
    return {
        "decision_count": float(len(decisions)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_regret": float(np.mean(regrets) if regrets else 0.0),
    }


def _metrics(rows: List[Dict[str, object]], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {
        "pairwise_accuracy": float(accuracy_score(y_true, y_pred) if len(y_true) else 0.0),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])) if len(y_true) else 0.0,
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = 0.5
    out.update(_decision_metrics(rows, y_prob))
    return out


def _positive_prob(estimator: object, x: np.ndarray) -> np.ndarray:
    probs = estimator.predict_proba(x)
    if probs.shape[1] == 1:
        cls = int(getattr(estimator, "classes_", [0])[0])
        return np.asarray([float(cls)] * len(x), dtype=float)
    return probs[:, 1]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train pairwise candidate-centric eviction model sweep")
    ap.add_argument("--pairwise-csv", default="data/derived/evict_value_pairwise/pairwise_rows.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metrics-json", default="analysis/evict_value_pairwise_v1_metrics.json")
    ap.add_argument("--comparison-csv", default="analysis/evict_value_pairwise_v1_model_comparison.csv")
    ap.add_argument("--best-model", default="models/evict_value_pairwise_v1_best.pkl")
    args = ap.parse_args()

    rows = _read_rows(Path(args.pairwise_csv))
    if not rows:
        raise ValueError("pairwise rows are empty")
    delta_cols = sorted(c for c in rows[0].keys() if c.startswith("delta_"))

    splits: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[_split(str(row["trace"]))].append(row)

    x_train, y_train = _xy(splits["train"], delta_cols)

    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=700, random_state=args.seed),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, random_state=args.seed),
        "hist_gb": HistGradientBoostingClassifier(max_depth=6, max_iter=250, random_state=args.seed),
    }

    payload: Dict[str, object] = {"pairwise_csv": args.pairwise_csv, "models": {}, "winner_by_val_top1": None}
    comparison_rows: List[Dict[str, object]] = []
    best_name = None
    best_score = -1.0
    best_model = None

    for model_name, model in models.items():
        if len(set(int(v) for v in y_train.tolist())) < 2:
            only = int(y_train[0]) if len(y_train) else 0
            model = DummyClassifier(strategy="constant", constant=only)
        model.fit(x_train, y_train)
        payload["models"][model_name] = {}

        for split in ["train", "val", "test"]:
            split_rows = splits[split]
            if not split_rows:
                continue
            x_split, y_split = _xy(split_rows, delta_cols)
            pred = model.predict(x_split)
            prob = _positive_prob(model, x_split)
            m = _metrics(split_rows, y_split, pred, prob)
            payload["models"][model_name][split] = m
            if split == "val":
                comparison_rows.append(
                    {
                        "model": model_name,
                        "val_pairwise_accuracy": m["pairwise_accuracy"],
                        "val_top1_candidate_accuracy": m["top1_candidate_accuracy"],
                        "val_mean_regret": m["mean_regret"],
                        "val_roc_auc": m["roc_auc"],
                    }
                )
                score = float(m["top1_candidate_accuracy"])
                if score > best_score:
                    best_score = score
                    best_name = model_name
                    best_model = model
            elif split == "train" and not splits["val"]:
                score = float(m["top1_candidate_accuracy"])
                if score > best_score:
                    best_score = score
                    best_name = model_name
                    best_model = model

    if best_model is None or best_name is None:
        raise RuntimeError("No winner selected for pairwise model sweep")

    payload["winner_by_val_top1"] = best_name
    EvictValuePairwiseV1Model(model_name=best_name, estimator=best_model, delta_feature_columns=delta_cols).save(args.best_model)

    _write_csv(Path(args.comparison_csv), comparison_rows)
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
