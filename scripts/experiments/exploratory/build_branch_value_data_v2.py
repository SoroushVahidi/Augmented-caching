from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
                row = json.loads(line)
                row["example_id"] = f"{row.get('method','unknown')}::ep{int(row['episode_id'])}"
                row["decision_step"] = int(row["action_idx"])
                rows.append(row)
    return rows


def _argmax_branch(items: List[Dict[str, object]], score_overrides: Dict[int, float] | None = None) -> int:
    best_key: Tuple[float, float, float, int] | None = None
    best_branch = -1
    score_overrides = score_overrides or {}
    for r in items:
        bid = int(r["branch_id"])
        score = float(score_overrides.get(bid, float(r.get("raw_score", 0.0))))
        key = (
            score,
            float(r.get("depth", 0.0)),
            float(r.get("expansions", 0.0)),
            -bid,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_branch = bid
    return best_branch


def _episode_split(example_ids: List[str], seed: int, train_frac: float, val_frac: float) -> Dict[str, str]:
    unique_ids = sorted(set(example_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    out: Dict[str, str] = {}
    for i, eid in enumerate(unique_ids):
        if i < n_train:
            out[eid] = "train"
        elif i < n_train + n_val:
            out[eid] = "val"
        else:
            out[eid] = "test"
    return out


def build_dataset(rows: List[Dict[str, object]], seed: int, train_frac: float, val_frac: float) -> List[Dict[str, object]]:
    by_episode: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_episode[(str(r.get("method", "unknown")), int(r["episode_id"]))].append(r)

    example_ids = [str(r["example_id"]) for r in rows]
    split_map = _episode_split(example_ids, seed=seed, train_frac=train_frac, val_frac=val_frac)

    out_rows: List[Dict[str, object]] = []
    for _, episode_rows in by_episode.items():
        by_step: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        by_branch: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for r in episode_rows:
            by_step[int(r["decision_step"])].append(r)
            by_branch[int(r["branch_id"])].append(r)

        for branch_id, seq in by_branch.items():
            seq.sort(key=lambda x: int(x["decision_step"]))

        for step, step_rows in by_step.items():
            baseline_pick = _argmax_branch(step_rows)
            gold = int(step_rows[0]["gold_branch_id"])
            baseline_correct = int(baseline_pick == gold)

            for r in step_rows:
                bid = int(r["branch_id"])
                seq = by_branch[bid]
                curr_idx = next(i for i, x in enumerate(seq) if int(x["decision_step"]) == step)
                next_obs = seq[curr_idx + 1] if curr_idx + 1 < len(seq) else None
                score_now = float(r.get("raw_score", 0.0))
                if next_obs is None:
                    score_after_expand = score_now
                    approx_flag = 0
                else:
                    score_after_expand = float(next_obs.get("raw_score", score_now))
                    approx_flag = 1

                expand_pick = _argmax_branch(step_rows, score_overrides={bid: score_after_expand})
                expand_correct = int(expand_pick == gold)
                target_delta = expand_correct - baseline_correct

                row = {
                    "example_id": str(r["example_id"]),
                    "decision_step": step,
                    "branch_id": bid,
                    "method": str(r.get("method", "unknown")),
                    "gold_branch_id": gold,
                    "baseline_pick_branch_id": baseline_pick,
                    "expanded_pick_branch_id": expand_pick,
                    "baseline_correct": baseline_correct,
                    "target_expand_value": float(expand_correct),
                    "target_expand_delta": float(target_delta),
                    "target_expand_better_than_baseline": int(target_delta > 0),
                    "target_expand_approx_from_future_observation": approx_flag,
                    "split": split_map[str(r["example_id"])],
                }
                for feat in DEFAULT_FEATURES:
                    row[feat] = float(r.get(feat, 0.0))
                out_rows.append(row)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build branch continuation-value style dataset (v2 approximation)")
    ap.add_argument("--trace-input", nargs="+", required=True)
    ap.add_argument("--out-dir", default="outputs/branch_value_data")
    ap.add_argument("--run-id", default="branch_scorer_lr_v2_continuation_value")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    trace_files = _iter_trace_files(args.trace_input)
    if not trace_files:
        raise SystemExit("No branch_traces.jsonl files found")

    rows = _load_rows(trace_files)
    ds = build_dataset(rows, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "branch_value_dataset.csv"
    fields = list(ds[0].keys()) if ds else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ds)

    meta = {
        "trace_files": [str(p) for p in trace_files],
        "rows": len(ds),
        "unique_examples": len({r["example_id"] for r in ds}),
        "split_counts": {
            "train": sum(1 for r in ds if r["split"] == "train"),
            "val": sum(1 for r in ds if r["split"] == "val"),
            "test": sum(1 for r in ds if r["split"] == "test"),
        },
        "target_notes": "Approximate one-step continuation value from next observed same-branch snapshot; fallback no-change if unavailable.",
    }
    (out_dir / "build_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(out_csv), **meta}, indent=2))


if __name__ == "__main__":
    main()
