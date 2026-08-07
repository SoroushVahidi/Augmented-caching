from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.experiments.evict_value_history_ablation import (
    HISTORY_AWARE_EXTRA_COLUMNS,
    HistoryAblationConfig,
    _history_extra_features,
    build_rows,
    replay_misses,
    split_rows,
    train_hist_gb,
)
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import PageId, Request


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


def _decision_groups(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    g: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        g.setdefault(str(r["decision_id"]), []).append(r)
    return g


def _ranking_from_chosen(rows: Sequence[Dict[str, object]], chosen_by_decision: Dict[str, str]) -> Dict[str, float]:
    groups = _decision_groups(rows)
    top1 = 0
    regrets: List[float] = []
    for did, items in groups.items():
        chosen_pid = chosen_by_decision.get(did)
        if chosen_pid is None:
            continue
        best = min(items, key=lambda x: (float(x["y_loss"]), str(x["candidate_page_id"])))
        chosen = next((x for x in items if str(x["candidate_page_id"]) == chosen_pid), None)
        if chosen is None:
            continue
        top1 += int(str(best["candidate_page_id"]) == chosen_pid)
        regrets.append(float(chosen["y_loss"]) - float(best["y_loss"]))
    denom = max(len(regrets), 1)
    return {
        "decision_count": float(len(regrets)),
        "top1_eviction_match": float(top1 / denom),
        "mean_regret_vs_oracle": float(np.mean(regrets) if regrets else 0.0),
    }


def _choose_by_regression(rows: Sequence[Dict[str, object]], feature_columns: List[str], model: object) -> Dict[str, str]:
    groups = _decision_groups(rows)
    out: Dict[str, str] = {}
    for did, items in groups.items():
        x = np.asarray([[float(r[c]) for c in feature_columns] for r in items], dtype=float)
        pred = np.asarray(model.predict(x), dtype=float)
        idx = min(range(len(items)), key=lambda i: (float(pred[i]), str(items[i]["candidate_page_id"])))
        out[did] = str(items[idx]["candidate_page_id"])
    return out


def _build_pairwise_data(rows: Sequence[Dict[str, object]], feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    for items in _decision_groups(rows).values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                ya = float(a["y_loss"])
                yb = float(b["y_loss"])
                if ya == yb:
                    continue
                diff = [float(a[c]) - float(b[c]) for c in feature_columns]
                label = 1 if ya < yb else 0
                x_rows.append(diff)
                y_rows.append(label)
                x_rows.append([-d for d in diff])
                y_rows.append(1 - label)
    if not x_rows:
        raise ValueError("No non-tied pairwise labels available")
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int)


def _fit_pairwise_classifier(rows: Sequence[Dict[str, object]], feature_columns: List[str], seed: int) -> HistGradientBoostingClassifier:
    x, y = _build_pairwise_data(rows, feature_columns)
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=seed)
    clf.fit(x, y)
    return clf


def _choose_by_pairwise(rows: Sequence[Dict[str, object]], feature_columns: List[str], clf: HistGradientBoostingClassifier) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for did, items in _decision_groups(rows).items():
        utils: Dict[str, float] = {}
        for i, a in enumerate(items):
            aid = str(a["candidate_page_id"])
            total = 0.0
            count = 0
            for j, b in enumerate(items):
                if i == j:
                    continue
                diff = np.asarray([[float(a[c]) - float(b[c]) for c in feature_columns]], dtype=float)
                p = float(clf.predict_proba(diff)[0][1])
                total += p
                count += 1
            utils[aid] = total / max(count, 1)
        out[did] = max(utils.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return out


def _rate(hist: Sequence[PageId], pid: PageId) -> float:
    if not hist:
        return 0.0
    return float(sum(1 for x in hist if x == pid) / len(hist))


def _replay_pairwise_misses(
    *,
    requests: Sequence[Request],
    capacity: int,
    feature_columns: List[str],
    clf: HistGradientBoostingClassifier,
    history_window: int,
) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: collections.deque[PageId] = collections.deque(maxlen=history_window)
    recent_hit_hist: collections.deque[PageId] = collections.deque(maxlen=history_window)
    misses = 0

    for req in requests:
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue

        misses += 1
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))

        feat_rows: List[Dict[str, float]] = []
        for c in candidates:
            base = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=c,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=_rate(recent_req_hist, c),
                recent_hit_rate=_rate(recent_hit_hist, c),
            ).as_dict()
            base.update(
                _history_extra_features(
                    candidate=c,
                    recent_req_hist=list(recent_req_hist),
                    recent_hit_hist=list(recent_hit_hist),
                    history_window=history_window,
                )
            )
            feat_rows.append(base)

        utils: Dict[str, float] = {}
        for i, c in enumerate(candidates):
            total = 0.0
            cnt = 0
            for j, d in enumerate(candidates):
                if i == j:
                    continue
                diff = np.asarray([[float(feat_rows[i][k]) - float(feat_rows[j][k]) for k in feature_columns]], dtype=float)
                total += float(clf.predict_proba(diff)[0][1])
                cnt += 1
            utils[str(c)] = total / max(cnt, 1)

        victim = max(utils.items(), key=lambda kv: (kv[1], kv[0]))[0]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses


def main() -> None:
    ap = argparse.ArgumentParser(description="History-feature objective ablation (lightweight)")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-requests-per-trace", type=int, default=2500)
    ap.add_argument("--out-dir", default="analysis/history_objective_ablation_light")
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

    base_reg = train_hist_gb(base_splits["train"], base_cols, seed=args.seed)
    hist_reg = train_hist_gb(hist_splits["train"], hist_cols, seed=args.seed)
    hist_pairwise = _fit_pairwise_classifier(hist_splits["train"], hist_cols, seed=args.seed)

    variants = {
        "base_regression": {
            "chosen_val": _choose_by_regression(base_splits["val"], base_cols, base_reg),
            "chosen_test": _choose_by_regression(base_splits["test"], base_cols, base_reg),
        },
        "history_regression": {
            "chosen_val": _choose_by_regression(hist_splits["val"], hist_cols, hist_reg),
            "chosen_test": _choose_by_regression(hist_splits["test"], hist_cols, hist_reg),
        },
        "history_pairwise": {
            "chosen_val": _choose_by_pairwise(hist_splits["val"], hist_cols, hist_pairwise),
            "chosen_test": _choose_by_pairwise(hist_splits["test"], hist_cols, hist_pairwise),
        },
    }

    comparison_rows: List[Dict[str, object]] = []
    for name, payload in variants.items():
        ref_rows = hist_splits if name.startswith("history") else base_splits
        rv = _ranking_from_chosen(ref_rows["val"], payload["chosen_val"])
        rt = _ranking_from_chosen(ref_rows["test"], payload["chosen_test"])
        comparison_rows.append(
            {
                "variant": name,
                "feature_count": len(hist_cols) if name.startswith("history") else len(base_cols),
                "objective": "pairwise_classifier" if name == "history_pairwise" else "replay_loss_regression",
                "val_decisions": int(rv["decision_count"]),
                "val_top1_eviction_match": rv["top1_eviction_match"],
                "val_mean_regret": rv["mean_regret_vs_oracle"],
                "test_decisions": int(rt["decision_count"]),
                "test_top1_eviction_match": rt["top1_eviction_match"],
                "test_mean_regret": rt["mean_regret_vs_oracle"],
            }
        )

    replay_rows: List[Dict[str, object]] = []
    for trace_name, reqs, _pages in trace_items:
        reqs = reqs[: args.max_requests_per_trace]
        for cap in capacities:
            miss_base = replay_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=base_cols,
                model=base_reg,
                history_window=cfg.history_window,
                history_aware=False,
            )
            miss_hist_reg = replay_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=hist_cols,
                model=hist_reg,
                history_window=cfg.history_window,
                history_aware=True,
            )
            miss_hist_pair = _replay_pairwise_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=hist_cols,
                clf=hist_pairwise,
                history_window=cfg.history_window,
            )
            replay_rows.append(
                {
                    "trace": trace_name,
                    "capacity": cap,
                    "base_regression_misses": miss_base,
                    "history_regression_misses": miss_hist_reg,
                    "history_pairwise_misses": miss_hist_pair,
                    "delta_base_minus_hist_reg": float(miss_base - miss_hist_reg),
                    "delta_hist_reg_minus_hist_pair": float(miss_hist_reg - miss_hist_pair),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "downstream_replay.csv", replay_rows)

    mean_base = mean(float(r["base_regression_misses"]) for r in replay_rows)
    mean_hist_reg = mean(float(r["history_regression_misses"]) for r in replay_rows)
    mean_hist_pair = mean(float(r["history_pairwise_misses"]) for r in replay_rows)

    def _row(variant: str) -> Dict[str, object]:
        return next(r for r in comparison_rows if r["variant"] == variant)

    base_test = _row("base_regression")
    hist_reg_test = _row("history_regression")
    hist_pair_test = _row("history_pairwise")

    cause = "unclear"
    if mean_hist_pair < mean_hist_reg and float(hist_pair_test["test_mean_regret"]) < float(hist_reg_test["test_mean_regret"]):
        cause = "objective_likely_bottleneck"
    elif mean_hist_pair >= mean_hist_reg and float(hist_pair_test["test_mean_regret"]) >= float(hist_reg_test["test_mean_regret"]):
        cause = "features_likely_bottleneck"
    else:
        cause = "mixed_feature_and_objective_effects"

    summary = {
        "traces": loaded_traces,
        "capacities": capacities,
        "horizon": args.horizon,
        "model_family": "HistGradientBoosting (regressor/classifier)",
        "base_feature_columns": list(EVICT_VALUE_V1_FEATURE_COLUMNS),
        "history_feature_columns": list(HISTORY_AWARE_EXTRA_COLUMNS),
        "model_comparison": comparison_rows,
        "downstream_replay": {
            "mean_base_regression_misses": mean_base,
            "mean_history_regression_misses": mean_hist_reg,
            "mean_history_pairwise_misses": mean_hist_pair,
        },
        "interpretation": {
            "previous_mixed_result_primary_cause": cause,
            "history_regression_vs_base_miss_delta": float(mean_base - mean_hist_reg),
            "history_pairwise_vs_hist_reg_miss_delta": float(mean_hist_reg - mean_hist_pair),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# History-feature objective ablation (lightweight)")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Reuses the existing history-aware experimental path and compact `repo_light` trace subset.")
    lines.append("- Base and history-aware feature pipelines are fixed.")
    lines.append("- Compared objectives on history-aware arm: replay-loss regression vs pairwise decision objective.")
    lines.append("- Model family kept aligned with histogram gradient boosting (regressor/classifier variants).")
    lines.append("")
    lines.append("## Candidate-ranking metrics (test)")
    for v in [base_test, hist_reg_test, hist_pair_test]:
        lines.append(
            f"- {v['variant']}: top1={float(v['test_top1_eviction_match']):.4f}, "
            f"mean_regret={float(v['test_mean_regret']):.4f}"
        )
    lines.append("")
    lines.append("## Downstream replay misses (mean)")
    lines.append(f"- base_regression: {mean_base:.3f}")
    lines.append(f"- history_regression: {mean_hist_reg:.3f}")
    lines.append(f"- history_pairwise: {mean_hist_pair:.3f}")
    lines.append("")
    lines.append("## Answer on prior mixed history-feature result")
    if cause == "objective_likely_bottleneck":
        lines.append("- Evidence in this lightweight run suggests the earlier mixed result was more due to objective/loss mismatch than features alone.")
    elif cause == "features_likely_bottleneck":
        lines.append("- Evidence in this lightweight run suggests richer history features themselves were the larger bottleneck under tested objectives.")
    else:
        lines.append("- Evidence in this lightweight run is mixed: both feature choice and objective formulation likely contribute.")
    lines.append("- This remains a lightweight result and should not be interpreted as heavy_r1 manuscript evidence.")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
