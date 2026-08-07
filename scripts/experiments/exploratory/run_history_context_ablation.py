from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.experiments.evict_value_history_ablation import (
    HISTORY_AWARE_EXTRA_COLUMNS,
    HistoryAblationConfig,
    build_rows,
    ranking_metrics,
    replay_misses,
    split_rows,
    train_hist_gb,
)
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _select_horizon(rows: List[Dict[str, object]], h: int) -> List[Dict[str, object]]:
    return [r for r in rows if int(r["horizon"]) == h]


def _make_stress_trace(page_ids: List[str], buckets: List[int], confs: List[float]):
    recs = [{"bucket": b, "confidence": c} for b, c in zip(buckets, confs)]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=recs)


def _stress_traces() -> Dict[str, Tuple[list, dict]]:
    return {
        "stress::predictor_good_lru_bad": _make_stress_trace(
            ["A", "B", "C", "A", "D", "A", "B", "C", "A", "D"],
            [0, 3, 3, 0, 3, 0, 3, 3, 0, 3],
            [1.0] * 10,
        ),
        "stress::predictor_bad_lru_good": _make_stress_trace(
            ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"],
            [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
            [1.0] * 10,
        ),
        "stress::mixed_regime": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "C", "B", "D"],
            [0, 3, 1, 0, 2, 3, 0, 1, 2, 3],
            [0.9, 0.9, 0.3, 0.9, 0.7, 0.3, 0.9, 0.3, 0.7, 0.3],
        ),
    }


def _iter_repo_light_traces() -> List[Tuple[str, list, dict]]:
    traces: List[Tuple[str, list, dict]] = []
    for p in ["data/example_unweighted.json", "data/example_atlas_v1.json", "data/example_general_caching.json"]:
        reqs, pages = load_trace(p)
        traces.append((p, reqs, pages))
    for name, (reqs, pages) in _stress_traces().items():
        traces.append((name, reqs, pages))
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight ablation: richer history-aware eviction scorer")
    ap.add_argument("--trace-paths", default="repo_light")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-requests-per-trace", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="analysis/history_context_ablation_light")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    cfg = HistoryAblationConfig(horizons=(args.horizon,), history_window=64)
    trace_items = _iter_repo_light_traces()

    base_rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    loaded_traces = []

    for trace_name, reqs, _pages in trace_items:
        reqs = reqs[: args.max_requests_per_trace]
        loaded_traces.append({"trace": trace_name, "request_count": len(reqs)})
        for cap in capacities:
            rb, rh = build_rows(requests=reqs, capacity=cap, trace_name=trace_name, cfg=cfg)
            base_rows.extend(rb)
            hist_rows.extend(rh)

    base_splits = split_rows(_select_horizon(base_rows, args.horizon))
    hist_splits = split_rows(_select_horizon(hist_rows, args.horizon))

    base_cols = list(EVICT_VALUE_V1_FEATURE_COLUMNS)
    hist_cols = list(EVICT_VALUE_V1_FEATURE_COLUMNS) + list(HISTORY_AWARE_EXTRA_COLUMNS)

    base_model = train_hist_gb(base_splits["train"], base_cols, seed=args.seed)
    hist_model = train_hist_gb(hist_splits["train"], hist_cols, seed=args.seed)

    def _pred(rows: List[Dict[str, object]], cols: List[str], model: object) -> np.ndarray:
        x = np.asarray([[float(r[c]) for c in cols] for r in rows], dtype=float)
        return np.asarray(model.predict(x), dtype=float)

    base_val_rank = ranking_metrics(base_splits["val"], _pred(base_splits["val"], base_cols, base_model))
    base_test_rank = ranking_metrics(base_splits["test"], _pred(base_splits["test"], base_cols, base_model))
    hist_val_rank = ranking_metrics(hist_splits["val"], _pred(hist_splits["val"], hist_cols, hist_model))
    hist_test_rank = ranking_metrics(hist_splits["test"], _pred(hist_splits["test"], hist_cols, hist_model))

    replay_rows: List[Dict[str, object]] = []
    for trace_name, reqs, _pages in trace_items:
        reqs = reqs[: args.max_requests_per_trace]
        for cap in capacities:
            misses_base = replay_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=base_cols,
                model=base_model,
                history_window=cfg.history_window,
                history_aware=False,
            )
            misses_hist = replay_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=hist_cols,
                model=hist_model,
                history_window=cfg.history_window,
                history_aware=True,
            )
            replay_rows.append(
                {
                    "trace": trace_name,
                    "capacity": cap,
                    "base_misses": misses_base,
                    "history_aware_misses": misses_hist,
                    "miss_delta_base_minus_history": float(misses_base - misses_hist),
                }
            )

    comparison_rows = [
        {
            "scorer": "base_v1_features",
            "feature_count": len(base_cols),
            "val_decisions": int(base_val_rank["decision_count"]),
            "val_top1_eviction_match": base_val_rank["top1_eviction_match"],
            "val_mean_regret": base_val_rank["mean_regret_vs_oracle"],
            "test_decisions": int(base_test_rank["decision_count"]),
            "test_top1_eviction_match": base_test_rank["top1_eviction_match"],
            "test_mean_regret": base_test_rank["mean_regret_vs_oracle"],
        },
        {
            "scorer": "history_aware_v1",
            "feature_count": len(hist_cols),
            "val_decisions": int(hist_val_rank["decision_count"]),
            "val_top1_eviction_match": hist_val_rank["top1_eviction_match"],
            "val_mean_regret": hist_val_rank["mean_regret_vs_oracle"],
            "test_decisions": int(hist_test_rank["decision_count"]),
            "test_top1_eviction_match": hist_test_rank["top1_eviction_match"],
            "test_mean_regret": hist_test_rank["mean_regret_vs_oracle"],
        },
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "downstream_replay.csv", replay_rows)

    avg_base_misses = mean(float(r["base_misses"]) for r in replay_rows)
    avg_hist_misses = mean(float(r["history_aware_misses"]) for r in replay_rows)
    delta = float(avg_base_misses - avg_hist_misses)

    verdict = "neutral"
    if delta > 0:
        verdict = "helps"
    elif delta < 0:
        verdict = "hurts"

    summary = {
        "traces": loaded_traces,
        "capacities": capacities,
        "horizon": args.horizon,
        "model_family": "HistGradientBoostingRegressor",
        "base_feature_count": len(base_cols),
        "history_aware_feature_count": len(hist_cols),
        "history_aware_extra_columns": list(HISTORY_AWARE_EXTRA_COLUMNS),
        "ranking": {
            "base": {"val": base_val_rank, "test": base_test_rank},
            "history_aware": {"val": hist_val_rank, "test": hist_test_rank},
        },
        "downstream_replay": {
            "mean_base_misses": avg_base_misses,
            "mean_history_aware_misses": avg_hist_misses,
            "mean_miss_delta_base_minus_history": delta,
            "verdict": verdict,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# History-context lightweight ablation for `evict_value_v1`")
    lines.append("")
    lines.append("## Scope and isolation")
    lines.append("- Separate experimental path only (`src/lafc/experiments/`, `scripts/experiments/`, and this analysis directory).")
    lines.append("- Uses only repository-provided compact traces (`repo_light`) and identical capacities/horizon used in the prior lightweight incoming-file ablation.")
    lines.append("- Keeps model family fixed for both arms: `HistGradientBoostingRegressor`.")
    lines.append("- Does not modify or rerun canonical heavy_r1 manuscript pipelines/artifacts.")
    lines.append("")
    lines.append("## Base pipeline identification")
    lines.append("- Base feature arm is exactly `EVICT_VALUE_V1_FEATURE_COLUMNS` from `src/lafc/evict_value_features_v1.py`.")
    lines.append("- Base candidate-generation semantics follow the same candidate-level replay setup used by `build_evict_value_examples_v1`.")
    lines.append("")
    lines.append("## Added history-aware features")
    for c in HISTORY_AWARE_EXTRA_COLUMNS:
        lines.append(f"- `{c}`")
    lines.append("")
    lines.append("## Candidate-ranking quality")
    lines.append(f"- Base test top1 match: {base_test_rank['top1_eviction_match']:.4f}")
    lines.append(f"- History-aware test top1 match: {hist_test_rank['top1_eviction_match']:.4f}")
    lines.append(f"- Base test mean regret: {base_test_rank['mean_regret_vs_oracle']:.4f}")
    lines.append(f"- History-aware test mean regret: {hist_test_rank['mean_regret_vs_oracle']:.4f}")
    lines.append("")
    lines.append("## Downstream replay misses")
    lines.append(f"- Mean base misses: {avg_base_misses:.3f}")
    lines.append(f"- Mean history-aware misses: {avg_hist_misses:.3f}")
    lines.append(f"- Mean miss delta (base - history-aware): {delta:.3f}")
    lines.append(f"- Verdict on richer history features in this lightweight run: **{verdict}**.")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
