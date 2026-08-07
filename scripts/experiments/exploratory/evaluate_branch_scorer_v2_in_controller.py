from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_pilot_ranking_ablation import run_pilot

DEFAULT_MODEL_PATH = "outputs/models/branch_scorer_lr_v2_continuation_value"
DEFAULT_METHODS = [
    "adaptive_raw_score",
    "adaptive_score_plus_progress",
    "adaptive_relative_rank",
    "adaptive_learned_branch_score",
]


def _format_metrics_table(metrics: Dict[str, Dict[str, float]], methods: List[str]) -> str:
    headers = [
        "method",
        "accuracy",
        "avg_actions",
        "avg_expansions",
        "avg_verifications",
        "budget_exhaustion_rate",
    ]
    rows = []
    for method in methods:
        m = metrics[method]
        rows.append(
            [
                method,
                f"{m['accuracy']:.3f}",
                f"{m['avg_actions']:.3f}",
                f"{m['avg_expansions']:.3f}",
                f"{m['avg_verifications']:.3f}",
                f"{m['budget_exhaustion_rate']:.3f}",
            ]
        )

    widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]

    def fmt_row(items: List[str]) -> str:
        return " | ".join(item.ljust(widths[i]) for i, item in enumerate(items))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _top1_agreement(trace_rows: List[Dict[str, object]], methods: List[str]) -> Dict[str, Dict[str, float]]:
    index: Dict[str, Dict[Tuple[int, int], int]] = {m: {} for m in methods}
    for row in trace_rows:
        method = str(row["method"])
        if method not in index:
            continue
        key = (int(row["episode_id"]), int(row["action_idx"]))
        index[method][key] = int(row["branch_id"])

    learned = "adaptive_learned_branch_score"
    learned_map = index[learned]
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        if method == learned:
            continue
        other = index[method]
        keys = sorted(set(learned_map.keys()) & set(other.keys()))
        agree = sum(1 for k in keys if learned_map[k] == other[k])
        out[method] = {
            "common_decisions": float(len(keys)),
            "top1_agreement_rate": (float(agree) / float(len(keys))) if keys else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained v2 branch scorer inside pilot controller.")
    ap.add_argument("--samples", type=int, default=80)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--branch-count", type=int, default=5)
    ap.add_argument("--min-expand", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260413)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out-dir", type=str, default="outputs/pilot_text_summaries")
    ap.add_argument("--write-csv", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    if not (model_dir / "model.joblib").exists():
        raise SystemExit(f"Expected trained model artifact at {model_dir / 'model.joblib'}")

    metrics, trace_rows = run_pilot(
        samples=args.samples,
        budget=args.budget,
        branch_count=args.branch_count,
        min_expand=args.min_expand,
        seed=args.seed,
        methods=DEFAULT_METHODS,
        learned_model_path=args.model_path,
    )

    agreement = _top1_agreement(trace_rows, DEFAULT_METHODS)

    print("# branch_scorer_v2 controller pilot eval")
    print(
        f"seed={args.seed} samples={args.samples} budget={args.budget} "
        f"branch_count={args.branch_count} min_expand={args.min_expand}"
    )
    print(_format_metrics_table(metrics, DEFAULT_METHODS))

    if agreement:
        print("\nTop-1 branch-choice agreement vs adaptive_learned_branch_score")
        for method in [m for m in DEFAULT_METHODS if m != "adaptive_learned_branch_score"]:
            row = agreement[method]
            print(
                f"- {method}: agreement={row['top1_agreement_rate']:.3f} "
                f"over {int(row['common_decisions'])} common decisions"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation": "branch_scorer_v2_in_controller",
        "seed": args.seed,
        "samples": args.samples,
        "budget": args.budget,
        "branch_count": args.branch_count,
        "min_expand": args.min_expand,
        "methods": DEFAULT_METHODS,
        "learned_model_path": args.model_path,
        "metrics": metrics,
        "top1_agreement_vs_learned": agreement,
    }

    summary_path = out_dir / "branch_scorer_v2_controller_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_csv:
        csv_path = out_dir / "branch_scorer_v2_controller_eval_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "method",
                "accuracy",
                "avg_actions",
                "avg_expansions",
                "avg_verifications",
                "budget_exhaustion_rate",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for method in DEFAULT_METHODS:
                row = {"method": method, **metrics[method]}
                writer.writerow(row)

    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
