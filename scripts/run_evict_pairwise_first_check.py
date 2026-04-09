from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    out: List[Dict[str, object]] = []
    for r in raw:
        row: Dict[str, object] = dict(r)
        for key, value in list(row.items()):
            if key.startswith(("delta_", "i_", "j_")) or key in {
                "loss_i",
                "loss_j",
                "regret_i",
                "regret_j",
                "label_i_better",
                "is_tie",
            }:
                row[key] = float(value)
        row["capacity"] = int(row["capacity"])
        row["t"] = int(row["t"])
        row["horizon"] = int(row["horizon"])
        out.append(row)
    return out


def _xy(rows: List[Dict[str, object]]):
    delta_cols = sorted([k for k in rows[0].keys() if k.startswith("delta_")])
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    return x, y, delta_cols


def _decision_metrics(rows: List[Dict[str, object]], proba_i_better: np.ndarray) -> Dict[str, float]:
    decisions: Dict[str, Dict[str, object]] = {}
    for row, prob_i in zip(rows, proba_i_better):
        decision_id = str(row["decision_id"])
        payload = decisions.setdefault(decision_id, {"scores": {}, "regrets": {}})
        cand_i = str(row["candidate_i_page_id"])
        cand_j = str(row["candidate_j_page_id"])

        payload["scores"][cand_i] = float(payload["scores"].get(cand_i, 0.0) + prob_i)
        payload["scores"][cand_j] = float(payload["scores"].get(cand_j, 0.0) + (1.0 - prob_i))

        payload["regrets"][cand_i] = float(row["regret_i"])
        payload["regrets"][cand_j] = float(row["regret_j"])

    top1 = 0
    chosen_regrets: List[float] = []
    for payload in decisions.values():
        scores = payload["scores"]
        regrets = payload["regrets"]
        chosen = min(scores.keys(), key=lambda c: (-scores[c], c))
        best = min(regrets.keys(), key=lambda c: (regrets[c], c))
        top1 += int(chosen == best)
        chosen_regrets.append(float(regrets[chosen]))

    denom = max(len(decisions), 1)
    return {
        "decision_count": float(len(decisions)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_regret_of_chosen": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
    }


def _evaluate(rows: List[Dict[str, object]], pred_label: np.ndarray, proba_i_better: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    out = {
        "pairwise_accuracy": float(accuracy_score(y_true, pred_label)),
    }
    out.update(_decision_metrics(rows, proba_i_better))
    return out


def _predict_with_single_class_fallback(
    *,
    clf: LogisticRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class predictions and P(i better) with a safe one-class fallback."""
    classes = set(int(v) for v in y_train.tolist())
    if len(classes) < 2:
        only = int(next(iter(classes))) if classes else 0
        pred = np.full(shape=(len(x_eval),), fill_value=only, dtype=int)
        prob = np.full(shape=(len(x_eval),), fill_value=float(only), dtype=float)
        return pred, prob

    clf.fit(x_train, y_train)
    return clf.predict(x_eval), clf.predict_proba(x_eval)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description="First-check baseline for pairwise eviction target")
    ap.add_argument("--data-dir", default="data/derived")
    ap.add_argument("--output-dir", default="analysis/evict_pairwise_first_check")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train = _load_csv(data_dir / "evict_pairwise_v1_train.csv")
    val = _load_csv(data_dir / "evict_pairwise_v1_val.csv")
    test = _load_csv(data_dir / "evict_pairwise_v1_test.csv")

    x_train, y_train, _cols = _xy(train)
    x_val, _y_val, _ = _xy(val)
    x_test, _y_test, _ = _xy(test)

    clf = LogisticRegression(max_iter=600, random_state=args.seed)
    pred_train, proba_train = _predict_with_single_class_fallback(
        clf=clf,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_train,
    )
    pred_val, proba_val = _predict_with_single_class_fallback(
        clf=clf,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_val,
    )
    pred_test, proba_test = _predict_with_single_class_fallback(
        clf=clf,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_test,
    )

    train_metrics = _evaluate(train, pred_train, proba_train)
    val_metrics = _evaluate(val, pred_val, proba_val)
    test_metrics = _evaluate(test, pred_test, proba_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = [
        {"split": "train", **train_metrics},
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "task": "evict_pairwise_v1",
        "model": "logistic_regression_pairwise_delta",
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Pairwise eviction first check",
        "",
        "This is an exploratory pairwise baseline with in-memory training only.",
        "",
        f"- Data directory: `{data_dir}`",
        "- Model: logistic regression over delta feature vector (candidate_i - candidate_j)",
        "",
        "## Metrics",
        "",
        "| Split | Pairwise accuracy | Top-1 candidate accuracy | Mean regret of chosen |",
        "|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['split']} | {row['pairwise_accuracy']:.4f} | {row['top1_candidate_accuracy']:.4f} | {row['mean_regret_of_chosen']:.4f} |"
        )
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
