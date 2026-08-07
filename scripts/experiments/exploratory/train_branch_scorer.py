from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments.learned_scorer import FeatureSpec, LearnedBranchScorer


DEFAULT_FEATURES = [
    "raw_score",
    "depth",
    "expansions",
    "verifications",
    "remaining_budget",
    "post_verify_score",
    "score_delta",
    "survived_pruning_steps",
]


def _iter_trace_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("branch_traces.jsonl")))
        elif path.name.endswith(".jsonl"):
            files.append(path)
    return files


def _load_rows(trace_files: List[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tf in trace_files:
        with tf.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train lightweight learned branch scorer")
    ap.add_argument("--trace-input", nargs="+", required=True, help="Directories or branch_traces.jsonl files")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    trace_files = _iter_trace_files(args.trace_input)
    if not trace_files:
        raise SystemExit("No branch_traces.jsonl files found")

    rows = _load_rows(trace_files)
    if not rows:
        raise SystemExit("Trace files were empty")

    X = np.array([[float(r.get(f, 0.0)) for f in DEFAULT_FEATURES] for r in rows], dtype=np.float64)
    y = np.array([int(r["eventually_correct_branch"]) for r in rows], dtype=np.int64)
    positive_rate = float(np.mean(y))

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    model.fit(X, y)

    p = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")

    scorer = LearnedBranchScorer(model=model, feature_spec=FeatureSpec(feature_names=DEFAULT_FEATURES))
    scorer.save(args.out_dir)

    summary = {
        "trace_files": [str(x) for x in trace_files],
        "rows": int(len(rows)),
        "positive_rate": positive_rate,
        "train_auc": float(auc),
        "target": "eventually_correct_branch (proxy: branch_id == gold_branch_id)",
        "features": DEFAULT_FEATURES,
        "model": "sklearn LogisticRegression(class_weight='balanced')",
    }
    out = Path(args.out_dir)
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
