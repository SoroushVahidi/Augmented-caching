from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS


def _load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    out: List[Dict[str, object]] = []
    for r in raw:
        row: Dict[str, object] = dict(r)
        for c in EVICT_VALUE_V1_FEATURE_COLUMNS + ["y_regret", "y_loss"]:
            row[c] = float(row[c])
        row["capacity"] = int(row["capacity"])
        row["t"] = int(row["t"])
        row["horizon"] = int(row["horizon"])
        out.append(row)
    return out


def _xy(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.asarray([float(r["y_regret"]) for r in rows], dtype=float)
    return x, y


def _group_decision_metrics(rows: List[Dict[str, object]], pred: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, pr in zip(rows, pred):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(pr)))

    top1 = 0
    chosen_regrets: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x[0]["y_regret"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == best[0]["candidate_page_id"])
        chosen_regrets.append(float(chosen[0]["y_regret"]))

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_regret_of_chosen": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
    }


def _evaluate(rows: List[Dict[str, object]], pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray([float(r["y_regret"]) for r in rows], dtype=float)
    metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }
    metrics.update(_group_decision_metrics(rows, pred))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="First-check baseline for regret-aligned eviction target")
    ap.add_argument("--data-dir", default="data/derived")
    ap.add_argument("--output-dir", default="analysis/evict_regret_first_check")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train = _load_csv(data_dir / "evict_regret_v1_train.csv")
    val = _load_csv(data_dir / "evict_regret_v1_val.csv")
    test = _load_csv(data_dir / "evict_regret_v1_test.csv")

    x_train, y_train = _xy(train)
    x_val, _ = _xy(val)
    x_test, _ = _xy(test)

    models: Dict[str, object] = {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=args.seed),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {"task": "evict_regret_v1", "models": {}}
    best_name = ""
    best_val_regret = float("inf")

    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        pred_train = estimator.predict(x_train)
        pred_val = estimator.predict(x_val)
        pred_test = estimator.predict(x_test)

        train_metrics = _evaluate(train, pred_train)
        val_metrics = _evaluate(val, pred_val)
        test_metrics = _evaluate(test, pred_test)

        for split, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            metric_rows.append({"model": name, "split": split, **m})

        summary["models"][name] = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        if val_metrics["mean_regret_of_chosen"] < best_val_regret:
            best_val_regret = val_metrics["mean_regret_of_chosen"]
            best_name = name

    summary["winner_by_val_mean_regret"] = best_name

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = output_dir / "experiment_report.md"
    lines = [
        "# Regret-aligned eviction first check",
        "",
        "This is an exploratory baseline; it does not claim a solved policy.",
        "",
        f"- Data directory: `{data_dir}`",
        f"- Models compared: {', '.join(models.keys())}",
        f"- Winner by validation mean regret of chosen candidate: `{best_name}`",
        "",
        "## Metrics",
        "",
        "| Model | Split | MAE | RMSE | Top-1 candidate accuracy | Mean regret of chosen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['top1_candidate_accuracy']:.4f} | {row['mean_regret_of_chosen']:.4f} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
