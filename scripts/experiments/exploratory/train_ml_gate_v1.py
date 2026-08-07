from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from lafc.learned_gate.features import ML_GATE_FEATURE_COLUMNS
from lafc.learned_gate.model import LearnedGateModel


def _read_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{k: float(v) if k in ML_GATE_FEATURE_COLUMNS or k in {"y"} else v for k, v in r.items()} for r in rows]


def _xy(rows: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[int]]:
    x = [{c: float(r[c]) for c in ML_GATE_FEATURE_COLUMNS} for r in rows]
    y = [int(r["y"]) for r in rows]
    return x, y


def _metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "class_balance": float(sum(y_true) / len(y_true)) if y_true else 0.0,
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = 0.5
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned gate v1")
    parser.add_argument("--data-dir", default="data/derived")
    parser.add_argument("--model-out", default="models/ml_gate_v1.pkl")
    parser.add_argument("--metrics-out", default="analysis/ml_gate_v1_metrics.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = _read_csv(data_dir / "ml_gate_train.csv")
    val = _read_csv(data_dir / "ml_gate_val.csv")
    test = _read_csv(data_dir / "ml_gate_test.csv")

    model = LearnedGateModel.new_logistic(random_state=args.seed)
    x_train, y_train = _xy(train)
    model.fit(x_train, y_train)

    payload: Dict[str, Dict[str, float]] = {}
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        x, y = _xy(rows)
        y_prob = [model.predict_proba_one(r) for r in x]
        y_pred = [int(p >= model.threshold) for p in y_prob]
        payload[split_name] = _metrics(y, y_pred, y_prob)

    clf = model.pipeline.named_steps["clf"]
    payload["feature_importance_logreg"] = {
        f: float(w) for f, w in zip(ML_GATE_FEATURE_COLUMNS, clf.coef_[0])
    }

    model.save(args.model_out)
    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
