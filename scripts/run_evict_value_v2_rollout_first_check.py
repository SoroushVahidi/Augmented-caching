from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows: List[Dict[str, object]] = []
    for row in raw:
        parsed: Dict[str, object] = dict(row)
        parsed["request_t"] = int(parsed["request_t"])
        parsed["capacity"] = int(parsed["capacity"])
        parsed["horizon"] = int(parsed["horizon"])
        parsed["rollout_loss_h"] = float(parsed["rollout_loss_h"])
        parsed["rollout_regret_h"] = float(parsed["rollout_regret_h"])
        for c in EVICT_VALUE_V1_FEATURE_COLUMNS:
            parsed[c] = float(parsed[c])
        rows.append(parsed)
    return rows


def _trace_split(trace: str) -> str:
    h = int(hashlib.md5(trace.encode("utf-8")).hexdigest(), 16) % 10
    if h <= 5:
        return "train"
    if h <= 7:
        return "val"
    return "test"


def _xy(rows: List[Dict[str, object]], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] + [float(r["capacity"]), float(r["horizon"])] for r in rows],
        dtype=float,
    )
    y = np.asarray([float(r[target_col]) for r in rows], dtype=float)
    return x, y


def _decision_metrics(rows: List[Dict[str, object]], pred_regret: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, pr in zip(rows, pred_regret):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(pr)))

    top1 = 0
    chosen_regrets: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x[0]["rollout_regret_h"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == best[0]["candidate_page_id"])
        chosen_regrets.append(float(chosen[0]["rollout_regret_h"]))

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
        "mean_regret_vs_oracle": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
    }


def _evaluate(rows: List[Dict[str, object]], pred_loss: np.ndarray, pred_regret: np.ndarray) -> Dict[str, float]:
    y_loss = np.asarray([float(r["rollout_loss_h"]) for r in rows], dtype=float)
    y_regret = np.asarray([float(r["rollout_regret_h"]) for r in rows], dtype=float)
    out = {
        "mae_rollout_loss": float(mean_absolute_error(y_loss, pred_loss)),
        "mae_rollout_regret": float(mean_absolute_error(y_regret, pred_regret)),
    }
    out.update(_decision_metrics(rows, pred_regret))
    return out


def _slice_by(rows: List[Dict[str, object]], key: str) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="First-check for evict_value_v2 rollout-labeled supervision")
    ap.add_argument("--candidate-csv", default="data/derived/evict_value_v2_rollout/candidate_rows.csv")
    ap.add_argument("--output-dir", default="analysis/evict_value_v2_rollout_first_check")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = _read_rows(Path(args.candidate_csv))
    split_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split_rows[_trace_split(str(row["trace"]))].append(row)

    x_train_loss, y_train_loss = _xy(split_rows["train"], "rollout_loss_h")
    x_train_reg, y_train_reg = _xy(split_rows["train"], "rollout_regret_h")

    models: Dict[str, object] = {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=args.seed),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    per_h_rows: List[Dict[str, object]] = []
    per_f_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {"task": "evict_value_v2_rollout", "models": {}}
    best_name = ""
    best_val = float("inf")

    for name, model_template in models.items():
        loss_model = clone(model_template)
        reg_model = clone(model_template)

        loss_model.fit(x_train_loss, y_train_loss)
        reg_model.fit(x_train_reg, y_train_reg)

        summary["models"][name] = {}
        for split in ["train", "val", "test"]:
            split_data = split_rows[split]
            x_split, _ = _xy(split_data, "rollout_loss_h")
            pred_loss = loss_model.predict(x_split)
            pred_reg = reg_model.predict(x_split)
            metrics = _evaluate(split_data, pred_loss, pred_reg)
            metric_rows.append({"model": name, "split": split, **metrics})
            summary["models"][name][split] = metrics

            by_h = _slice_by(split_data, "horizon")
            for horizon, h_rows in by_h.items():
                hx, _ = _xy(h_rows, "rollout_loss_h")
                hm = _evaluate(h_rows, loss_model.predict(hx), reg_model.predict(hx))
                per_h_rows.append({"model": name, "split": split, "horizon": int(horizon), **hm})

            by_f = _slice_by(split_data, "family")
            for family, f_rows in by_f.items():
                fx, _ = _xy(f_rows, "rollout_loss_h")
                fm = _evaluate(f_rows, loss_model.predict(fx), reg_model.predict(fx))
                per_f_rows.append({"model": name, "split": split, "family": family, **fm})

        val_metric = summary["models"][name]["val"]["mean_chosen_regret"]
        if val_metric < best_val:
            best_val = val_metric
            best_name = name

    summary["winner_by_val_mean_chosen_regret"] = best_name

    def _write(path: Path, rows_to_write: List[Dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_to_write[0].keys()))
            writer.writeheader()
            writer.writerows(rows_to_write)

    _write(output_dir / "metrics.csv", metric_rows)
    _write(output_dir / "per_horizon_metrics.csv", per_h_rows)
    _write(output_dir / "per_family_metrics.csv", per_f_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# evict_value_v2 rollout first check",
        "",
        "Conservative first-check only. Finite-horizon rollouts are approximation targets.",
        "",
        f"- Candidate CSV: `{args.candidate_csv}`",
        f"- Winner by val mean chosen regret: `{best_name}`",
        "",
        "## Core metrics (per split)",
        "",
        "| Model | Split | MAE(loss) | MAE(regret) | Top-1 acc | Mean chosen regret |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['mae_rollout_loss']:.4f} | {row['mae_rollout_regret']:.4f} | "
            f"{row['top1_candidate_accuracy']:.4f} | {row['mean_chosen_regret']:.4f} |"
        )
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
