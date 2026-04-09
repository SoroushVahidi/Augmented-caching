from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
            parsed[col] = float(parsed[col])
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


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred))) if len(y) else 0.0


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y - pred) ** 2))) if len(y) else 0.0


def _decision_metrics(rows: List[Dict[str, object]], pred_score: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, s in zip(rows, pred_score):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(s)))

    top1 = 0
    chosen_regrets: List[float] = []
    chosen_loss_gaps: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best_regret = min(float(x[0]["rollout_regret_h"]) for x in items)
        best_loss = min(float(x[0]["rollout_loss_h"]) for x in items)
        chosen_regret = float(chosen[0]["rollout_regret_h"])
        chosen_loss = float(chosen[0]["rollout_loss_h"])
        oracle = min(items, key=lambda x: (float(x[0]["rollout_regret_h"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == oracle[0]["candidate_page_id"])
        chosen_regrets.append(chosen_regret)
        chosen_loss_gaps.append(chosen_loss - best_loss)

    denom = max(1, len(grouped))
    return {
        "decision_count": float(len(grouped)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
        "mean_regret_vs_best": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
        "mean_loss_gap_vs_best": float(np.mean(chosen_loss_gaps) if chosen_loss_gaps else 0.0),
    }


def _evaluate(rows: List[Dict[str, object]], pred_loss: np.ndarray, pred_regret: np.ndarray) -> Dict[str, float]:
    y_loss = np.asarray([float(r["rollout_loss_h"]) for r in rows], dtype=float)
    y_regret = np.asarray([float(r["rollout_regret_h"]) for r in rows], dtype=float)
    out = {
        "mae_rollout_loss": _mae(y_loss, pred_loss),
        "rmse_rollout_loss": _rmse(y_loss, pred_loss),
        "mae_rollout_regret": _mae(y_regret, pred_regret),
        "rmse_rollout_regret": _rmse(y_regret, pred_regret),
    }
    out.update(_decision_metrics(rows, pred_regret))
    return out


def _slice_by(rows: List[Dict[str, object]], key: str) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="First-check for decision-aligned candidate labels")
    ap.add_argument("--candidate-csv", default="data/derived/evict_value_decision_aligned/candidate_rows.csv")
    ap.add_argument("--output-dir", default="analysis/evict_value_decision_aligned_first_check")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = _read_rows(Path(args.candidate_csv))
    split_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split_rows[_trace_split(str(row["trace"]))].append(row)

    x_train_loss, y_train_loss = _xy(split_rows["train"], "rollout_loss_h")
    x_train_reg, y_train_reg = _xy(split_rows["train"], "rollout_regret_h")

    model_templates = {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=args.seed),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    per_h_rows: List[Dict[str, object]] = []
    per_f_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {"task": "evict_value_decision_aligned", "models": {}}

    best_model = ""
    best_val = float("inf")
    for model_name, template in model_templates.items():
        loss_model = clone(template)
        reg_model = clone(template)
        loss_model.fit(x_train_loss, y_train_loss)
        reg_model.fit(x_train_reg, y_train_reg)

        summary["models"][model_name] = {}
        for split in ["train", "val", "test"]:
            part = split_rows[split]
            if not part:
                continue
            x_split, _ = _xy(part, "rollout_loss_h")
            m = _evaluate(part, loss_model.predict(x_split), reg_model.predict(x_split))
            metric_rows.append({"model": model_name, "split": split, **m})
            summary["models"][model_name][split] = m

            for horizon, h_rows in _slice_by(part, "horizon").items():
                hx, _ = _xy(h_rows, "rollout_loss_h")
                hm = _evaluate(h_rows, loss_model.predict(hx), reg_model.predict(hx))
                per_h_rows.append({"model": model_name, "split": split, "horizon": int(horizon), **hm})

            for family, f_rows in _slice_by(part, "family").items():
                fx, _ = _xy(f_rows, "rollout_loss_h")
                fm = _evaluate(f_rows, loss_model.predict(fx), reg_model.predict(fx))
                per_f_rows.append({"model": model_name, "split": split, "family": family, **fm})

        val_regret = summary["models"].get(model_name, {}).get("val", {}).get("mean_chosen_regret", float("inf"))
        if val_regret < best_val:
            best_val = val_regret
            best_model = model_name

    summary["winner_by_val_mean_chosen_regret"] = best_model

    _write_csv(out_dir / "metrics.csv", metric_rows)
    _write_csv(out_dir / "per_horizon_metrics.csv", per_h_rows)
    _write_csv(out_dir / "per_family_metrics.csv", per_f_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Decision-aligned candidate first check",
        "",
        "Finite-horizon rollout labels are an approximation to long-run cost-to-go.",
        "",
        f"- Candidate CSV: `{args.candidate_csv}`",
        f"- Winner by val mean chosen regret: `{best_model}`",
        "",
        "| Model | Split | MAE(loss) | RMSE(loss) | MAE(regret) | RMSE(regret) | Top-1 acc | Mean chosen regret | Mean regret vs best |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {row['mae_rollout_loss']:.4f} | {row['rmse_rollout_loss']:.4f} | "
            f"{row['mae_rollout_regret']:.4f} | {row['rmse_rollout_regret']:.4f} | {row['top1_candidate_accuracy']:.4f} | "
            f"{row['mean_chosen_regret']:.4f} | {row['mean_regret_vs_best']:.4f} |"
        )
    (out_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
