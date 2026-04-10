from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.evict_value_incoming_ablation import (
    INCOMING_AWARE_EXTRA_COLUMNS,
    IncomingAblationConfig,
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
    ap = argparse.ArgumentParser(description="Lightweight ablation: incoming-file-aware eviction scorer")
    ap.add_argument(
        "--trace-paths",
        default="repo_light",
        help="Comma-separated paths; use 'repo_light' for built-in compact subset.",
    )
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-requests-per-trace", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="analysis/incoming_file_aware_ablation_light")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    cfg = IncomingAblationConfig(horizons=(args.horizon,), history_window=64)

    if args.trace_paths.strip() == "repo_light":
        trace_items = _iter_repo_light_traces()
    else:
        trace_items = []
        for path in [x.strip() for x in args.trace_paths.split(",") if x.strip()]:
            reqs, pages, source = load_trace_from_any(path)
            trace_items.append((f"{source}:{Path(path).stem}", reqs, pages))

    base_rows: List[Dict[str, object]] = []
    aware_rows: List[Dict[str, object]] = []
    loaded_traces = []

    for trace_name, reqs, _pages in trace_items:
        reqs = reqs[: args.max_requests_per_trace]
        loaded_traces.append({"trace": trace_name, "request_count": len(reqs)})
        for cap in capacities:
            rb, ra = build_rows(requests=reqs, capacity=cap, trace_name=trace_name, cfg=cfg)
            base_rows.extend(rb)
            aware_rows.extend(ra)

    base_splits = split_rows(_select_horizon(base_rows, args.horizon))
    aware_splits = split_rows(_select_horizon(aware_rows, args.horizon))

    base_cols = list(EVICT_VALUE_V1_FEATURE_COLUMNS)
    aware_cols = list(EVICT_VALUE_V1_FEATURE_COLUMNS) + list(INCOMING_AWARE_EXTRA_COLUMNS)

    base_model = train_hist_gb(base_splits["train"], base_cols, seed=args.seed)
    aware_model = train_hist_gb(aware_splits["train"], aware_cols, seed=args.seed)

    def _pred(rows: List[Dict[str, object]], cols: List[str], model: object) -> np.ndarray:
        x = np.asarray([[float(r[c]) for c in cols] for r in rows], dtype=float)
        return np.asarray(model.predict(x), dtype=float)

    base_val_rank = ranking_metrics(base_splits["val"], _pred(base_splits["val"], base_cols, base_model))
    base_test_rank = ranking_metrics(base_splits["test"], _pred(base_splits["test"], base_cols, base_model))
    aware_val_rank = ranking_metrics(aware_splits["val"], _pred(aware_splits["val"], aware_cols, aware_model))
    aware_test_rank = ranking_metrics(aware_splits["test"], _pred(aware_splits["test"], aware_cols, aware_model))

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
                incoming_aware=False,
            )
            misses_aware = replay_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=aware_cols,
                model=aware_model,
                history_window=cfg.history_window,
                incoming_aware=True,
            )
            replay_rows.append(
                {
                    "trace": trace_name,
                    "capacity": cap,
                    "base_misses": misses_base,
                    "incoming_aware_misses": misses_aware,
                    "miss_delta_base_minus_aware": float(misses_base - misses_aware),
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
            "scorer": "incoming_file_aware_v1",
            "feature_count": len(aware_cols),
            "val_decisions": int(aware_val_rank["decision_count"]),
            "val_top1_eviction_match": aware_val_rank["top1_eviction_match"],
            "val_mean_regret": aware_val_rank["mean_regret_vs_oracle"],
            "test_decisions": int(aware_test_rank["decision_count"]),
            "test_top1_eviction_match": aware_test_rank["top1_eviction_match"],
            "test_mean_regret": aware_test_rank["mean_regret_vs_oracle"],
        },
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "downstream_replay.csv", replay_rows)

    avg_base_misses = mean(float(r["base_misses"]) for r in replay_rows)
    avg_aware_misses = mean(float(r["incoming_aware_misses"]) for r in replay_rows)
    summary = {
        "traces": loaded_traces,
        "capacities": capacities,
        "horizon": args.horizon,
        "model_family": "HistGradientBoostingRegressor",
        "base_feature_count": len(base_cols),
        "incoming_aware_feature_count": len(aware_cols),
        "ranking": {
            "base": {"val": base_val_rank, "test": base_test_rank},
            "incoming_aware": {"val": aware_val_rank, "test": aware_test_rank},
        },
        "downstream_replay": {
            "mean_base_misses": avg_base_misses,
            "mean_incoming_aware_misses": avg_aware_misses,
            "mean_miss_delta_base_minus_aware": float(avg_base_misses - avg_aware_misses),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Incoming-file-aware lightweight ablation for `evict_value_v1`")
    lines.append("")
    lines.append("## Scope and isolation")
    lines.append("- Experimental path only (`src/lafc/experiments/`, `scripts/experiments/`, and this analysis directory).")
    lines.append("- Uses existing repository traces only (no new datasets).")
    lines.append("- Keeps model family fixed: `HistGradientBoostingRegressor` for both scorers.")
    lines.append("- Does not invoke or modify the canonical heavy_r1 manuscript pipeline.")
    lines.append("")
    lines.append("## Existing baseline pipeline identified")
    lines.append("- Candidate-level builder: `src/lafc/evict_value_dataset_v1.py::build_evict_value_examples_v1`.")
    lines.append("- Feature pipeline: `src/lafc/evict_value_features_v1.py::compute_candidate_features_v1` and `EVICT_VALUE_V1_FEATURE_COLUMNS`.")
    lines.append("- Canonical lightweight train script: `scripts/train_evict_value_v1.py`.")
    lines.append("")
    lines.append("## Incoming-file-aware additions")
    lines.append("- Added incoming/current-request relation features:")
    for c in INCOMING_AWARE_EXTRA_COLUMNS:
        lines.append(f"  - `{c}`")
    lines.append("")
    lines.append("## Ranking quality (candidate-level)")
    lines.append(f"- Base test top1 match: {base_test_rank['top1_eviction_match']:.4f}")
    lines.append(f"- Incoming-aware test top1 match: {aware_test_rank['top1_eviction_match']:.4f}")
    lines.append(f"- Base test mean regret: {base_test_rank['mean_regret_vs_oracle']:.4f}")
    lines.append(f"- Incoming-aware test mean regret: {aware_test_rank['mean_regret_vs_oracle']:.4f}")
    lines.append("")
    lines.append("## Downstream replay misses")
    lines.append(f"- Mean base misses: {avg_base_misses:.3f}")
    lines.append(f"- Mean incoming-aware misses: {avg_aware_misses:.3f}")
    lines.append(f"- Mean miss delta (base - incoming-aware): {avg_base_misses - avg_aware_misses:.3f}")
    lines.append("")
    lines.append("## Interpretation")
    if avg_aware_misses < avg_base_misses:
        lines.append("- In this lightweight subset run, explicit incoming-file conditioning reduced replay misses on average.")
    elif avg_aware_misses > avg_base_misses:
        lines.append("- In this lightweight subset run, explicit incoming-file conditioning increased replay misses on average.")
    else:
        lines.append("- In this lightweight subset run, explicit incoming-file conditioning had neutral average replay misses.")
    lines.append("- Treat this as a controlled lightweight signal, not a heavy_r1 manuscript claim.")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
