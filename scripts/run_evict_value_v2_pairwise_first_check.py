from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    rows: List[Dict[str, object]] = []
    for row in raw:
        parsed: Dict[str, object] = dict(row)
        parsed["request_t"] = int(parsed["request_t"])
        parsed["capacity"] = int(parsed["capacity"])
        parsed["horizon"] = int(parsed["horizon"])
        for key, value in list(parsed.items()):
            if key.startswith(("delta_", "i_", "j_")) or key in {
                "rollout_loss_i",
                "rollout_loss_j",
                "rollout_regret_i",
                "rollout_regret_j",
                "rollout_regret_diff",
                "label_i_better",
                "is_tie",
            }:
                parsed[key] = float(value)
        rows.append(parsed)
    return rows


def _trace_split(trace: str) -> str:
    h = int(hashlib.md5(trace.encode("utf-8")).hexdigest(), 16) % 10
    if h <= 5:
        return "train"
    if h <= 7:
        return "val"
    return "test"


def _xy(rows: List[Dict[str, object]]):
    delta_cols = sorted(c for c in rows[0].keys() if c.startswith("delta_"))
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    return x, y


def _decision_metrics(rows: List[Dict[str, object]], p_i_better: np.ndarray) -> Dict[str, float]:
    decisions: Dict[str, Dict[str, object]] = {}
    for row, p_i in zip(rows, p_i_better):
        d = decisions.setdefault(str(row["decision_id"]), {"scores": {}, "regrets": {}})
        ci = str(row["candidate_i_page_id"])
        cj = str(row["candidate_j_page_id"])
        d["scores"][ci] = float(d["scores"].get(ci, 0.0) + p_i)
        d["scores"][cj] = float(d["scores"].get(cj, 0.0) + (1.0 - p_i))
        d["regrets"][ci] = float(row["rollout_regret_i"])
        d["regrets"][cj] = float(row["rollout_regret_j"])

    top1 = 0
    chosen_regrets: List[float] = []
    for d in decisions.values():
        chosen = min(d["scores"].keys(), key=lambda c: (-d["scores"][c], c))
        best = min(d["regrets"].keys(), key=lambda c: (d["regrets"][c], c))
        top1 += int(chosen == best)
        chosen_regrets.append(float(d["regrets"][chosen]))

    denom = max(len(decisions), 1)
    return {
        "decision_count": float(len(decisions)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
    }


def _evaluate(rows: List[Dict[str, object]], pred_y: np.ndarray, p_i_better: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    out = {
        "pairwise_accuracy": float(accuracy_score(y_true, pred_y)),
    }
    out.update(_decision_metrics(rows, p_i_better))
    return out


def _slice_by(rows: List[Dict[str, object]], key: str) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="First-check for evict_value_v2 pairwise ranking")
    ap.add_argument("--pairwise-csv", default="data/derived/evict_value_v2_pairwise/pairwise_rows.csv")
    ap.add_argument("--output-dir", default="analysis/evict_value_v2_pairwise_first_check")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = _read_rows(Path(args.pairwise_csv))
    splits: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[_trace_split(str(row["trace"]))].append(row)

    x_train, y_train = _xy(splits["train"])
    clf = LogisticRegression(max_iter=600, random_state=args.seed)
    clf.fit(x_train, y_train)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    per_h_rows: List[Dict[str, object]] = []
    per_f_rows: List[Dict[str, object]] = []

    summary: Dict[str, object] = {"task": "evict_value_v2_pairwise", "model": "logistic_regression_delta"}

    for split in ["train", "val", "test"]:
        x_split, _y_split = _xy(splits[split])
        pred = clf.predict(x_split)
        prob = clf.predict_proba(x_split)[:, 1]
        metrics = _evaluate(splits[split], pred, prob)
        metric_rows.append({"split": split, **metrics})
        summary[split] = metrics

        by_h = _slice_by(splits[split], "horizon")
        for horizon, h_rows in by_h.items():
            hx, _ = _xy(h_rows)
            hm = _evaluate(h_rows, clf.predict(hx), clf.predict_proba(hx)[:, 1])
            per_h_rows.append({"split": split, "horizon": int(horizon), **hm})

        by_f = _slice_by(splits[split], "family")
        for family, f_rows in by_f.items():
            fx, _ = _xy(f_rows)
            fm = _evaluate(f_rows, clf.predict(fx), clf.predict_proba(fx)[:, 1])
            per_f_rows.append({"split": split, "family": family, **fm})

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
        "# evict_value_v2 pairwise first check",
        "",
        "Conservative first-check only. Pairwise ranking is still a surrogate for final miss cost.",
        "",
        f"- Pairwise CSV: `{args.pairwise_csv}`",
        "",
        "| Split | Pairwise acc | Top-1 acc | Mean chosen regret |",
        "|---|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['split']} | {row['pairwise_accuracy']:.4f} | {row['top1_candidate_accuracy']:.4f} | {row['mean_chosen_regret']:.4f} |"
        )
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
