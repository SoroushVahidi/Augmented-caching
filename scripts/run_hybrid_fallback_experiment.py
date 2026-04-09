from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict, deque
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, List, Sequence, Tuple

from lafc.simulator.request_trace import load_trace
from lafc.types import PageId, Request

FEATURE_COLUMNS: List[str] = [
    "request_bucket",
    "request_confidence",
    "candidate_bucket",
    "candidate_confidence",
    "candidate_recency_rank",
    "candidate_age_norm",
    "candidate_predictor_score",
    "candidate_lru_score",
    "candidate_is_predictor_victim",
    "candidate_is_lru_victim",
    "score_gap_to_predictor_best",
    "score_gap_to_lru_victim",
    "bucket_gap_to_predictor_best",
    "bucket_gap_to_lru_victim",
    "confidence_gap_to_predictor_best",
    "confidence_gap_to_lru_victim",
    "cache_bucket_mean",
    "cache_bucket_std",
    "cache_bucket_min",
    "cache_bucket_max",
    "cache_unique_bucket_count",
    "cache_confidence_mean",
    "cache_confidence_std",
    "predictor_lru_disagree",
    "recent_candidate_request_rate",
    "recent_candidate_hit_rate",
]

SURROGATE_INTERCEPT = 0.25
SURROGATE_WEIGHTS: Dict[str, float] = {
    "candidate_predictor_score": 0.55,
    "candidate_lru_score": 0.15,
    "candidate_age_norm": 0.12,
    "candidate_is_predictor_victim": -0.20,
    "candidate_is_lru_victim": -0.08,
    "request_bucket": -0.05,
    "request_confidence": -0.18,
    "candidate_confidence": -0.12,
    "candidate_recency_rank": 0.08,
    "recent_candidate_request_rate": -0.20,
    "recent_candidate_hit_rate": -0.10,
    "score_gap_to_predictor_best": 0.08,
    "score_gap_to_lru_victim": 0.05,
}


def _resolve_trace_paths(trace_glob: str) -> List[str]:
    paths: List[str] = []
    for pattern in [p.strip() for p in trace_glob.split(",") if p.strip()]:
        if any(ch in pattern for ch in "*?["):
            paths.extend(sorted(str(p) for p in Path().glob(pattern) if p.exists()))
        else:
            p = Path(pattern)
            if p.exists():
                paths.append(str(p))
    out = sorted(set(paths))
    if not out:
        raise ValueError(f"No traces matched --trace-glob={trace_glob}")
    return out


def _trace_family(path: str) -> str:
    parent = Path(path).parent
    return parent.name if parent.name else "unknown"


def _split_for_seed(trace: str, seed: int) -> str:
    h = int(hashlib.md5(f"{trace}|{seed}".encode("utf-8")).hexdigest(), 16) % 10
    if h <= 5:
        return "train"
    if h <= 7:
        return "val"
    return "test"


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _compute_lru_scores(candidates: List[PageId]) -> Dict[PageId, float]:
    if len(candidates) == 1:
        return {candidates[0]: 1.0}
    denom = len(candidates) - 1
    return {p: 1.0 - (i / denom) for i, p in enumerate(candidates)}


def _compute_predictor_scores(candidates: List[PageId], bucket_by_page: Dict[PageId, int]) -> Dict[PageId, float]:
    if len(candidates) == 1:
        return {candidates[0]: 1.0}
    bucket_values = {p: int(bucket_by_page.get(p, 0)) for p in candidates}
    uniq = sorted(set(bucket_values.values()))
    if len(uniq) == 1:
        return {p: 0.5 for p in candidates}
    rank = {b: i for i, b in enumerate(uniq)}
    denom = len(uniq) - 1
    return {p: (rank[bucket_values[p]] / denom) ** 2 for p in candidates}


def _candidate_features(
    *,
    request_bucket: int,
    request_confidence: float,
    candidates: List[PageId],
    candidate: PageId,
    bucket_by_page: Dict[PageId, int],
    confidence_by_page: Dict[PageId, float],
    recent_request_rate: float,
    recent_hit_rate: float,
) -> Dict[str, float]:
    p_scores = _compute_predictor_scores(candidates, bucket_by_page)
    lru_scores = _compute_lru_scores(candidates)
    pred_victim = max(candidates, key=lambda x: (p_scores[x], -candidates.index(x)))
    lru_victim = max(candidates, key=lambda x: (lru_scores[x], -candidates.index(x)))

    buckets = [float(bucket_by_page.get(p, 0)) for p in candidates]
    confs = [float(confidence_by_page.get(p, 0.5)) for p in candidates]
    recency_rank = float(candidates.index(candidate))
    denom = max(len(candidates) - 1, 1)

    return {
        "request_bucket": float(request_bucket),
        "request_confidence": float(request_confidence),
        "candidate_bucket": float(bucket_by_page.get(candidate, 0)),
        "candidate_confidence": float(confidence_by_page.get(candidate, 0.5)),
        "candidate_recency_rank": recency_rank,
        "candidate_age_norm": recency_rank / float(denom),
        "candidate_predictor_score": float(p_scores[candidate]),
        "candidate_lru_score": float(lru_scores[candidate]),
        "candidate_is_predictor_victim": float(candidate == pred_victim),
        "candidate_is_lru_victim": float(candidate == lru_victim),
        "score_gap_to_predictor_best": float(p_scores[candidate] - p_scores[pred_victim]),
        "score_gap_to_lru_victim": float(lru_scores[candidate] - lru_scores[lru_victim]),
        "bucket_gap_to_predictor_best": float(bucket_by_page.get(candidate, 0) - bucket_by_page.get(pred_victim, 0)),
        "bucket_gap_to_lru_victim": float(bucket_by_page.get(candidate, 0) - bucket_by_page.get(lru_victim, 0)),
        "confidence_gap_to_predictor_best": float(confidence_by_page.get(candidate, 0.5) - confidence_by_page.get(pred_victim, 0.5)),
        "confidence_gap_to_lru_victim": float(confidence_by_page.get(candidate, 0.5) - confidence_by_page.get(lru_victim, 0.5)),
        "cache_bucket_mean": sum(buckets) / len(buckets),
        "cache_bucket_std": _std(buckets),
        "cache_bucket_min": min(buckets),
        "cache_bucket_max": max(buckets),
        "cache_unique_bucket_count": float(len(set(buckets))),
        "cache_confidence_mean": sum(confs) / len(confs),
        "cache_confidence_std": _std(confs),
        "predictor_lru_disagree": float(pred_victim != lru_victim),
        "recent_candidate_request_rate": float(recent_request_rate),
        "recent_candidate_hit_rate": float(recent_hit_rate),
    }


def _predict_surrogate_loss(row: Dict[str, float], intercept: float, weights: Dict[str, float]) -> float:
    val = float(intercept)
    for k, w in weights.items():
        val += float(w) * float(row.get(k, 0.0))
    return float(val)


def _safe_mean(vals: List[float]) -> float:
    return float(mean(vals)) if vals else 0.0


def _workload_keys(trace_paths: Sequence[str], capacities: Sequence[int], split: str, seed: int) -> List[Tuple[str, int]]:
    out = []
    for trace_path in trace_paths:
        if _split_for_seed(trace_path, seed) != split:
            continue
        for cap in capacities:
            out.append((trace_path, cap))
    return sorted(out)


def _simulate_policy(
    *,
    requests: Sequence[Request],
    trace_path: str,
    capacity: int,
    mode: str,
    margin_threshold: float,
    history_window: int,
    fragility_window: int,
    intercept: float,
    weights: Dict[str, float],
) -> Dict[str, object]:
    order: OrderedDict[PageId, None] = OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = deque(maxlen=history_window)
    recent_hit_hist: Deque[PageId] = deque(maxlen=history_window)

    misses = 0
    hits = 0
    evictions = 0
    triggers = 0

    early_reuse_triggered: List[float] = []
    early_reuse_nontriggered: List[float] = []

    decision_logs: List[Dict[str, object]] = []

    seq = list(requests)
    for idx, req in enumerate(seq):
        pid = str(req.page_id)
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            hits += 1
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue

        misses += 1
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        evictions += 1
        candidates = list(order.keys())
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))

        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            req_rate = (sum(1 for x in recent_req_hist if x == cand) / len(recent_req_hist)) if recent_req_hist else 0.0
            hit_rate = (sum(1 for x in recent_hit_hist if x == cand) / len(recent_hit_hist)) if recent_hit_hist else 0.0
            feat = _candidate_features(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=cand,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            )
            scored.append((str(cand), _predict_surrogate_loss(feat, intercept, weights)))
        scored = sorted(scored, key=lambda x: (x[1], x[0]))

        model_victim = scored[0][0]
        lru_victim = str(next(iter(order.keys())))
        margin = float(scored[1][1] - scored[0][1]) if len(scored) > 1 else 999.0
        norm_margin = float(margin / (abs(scored[1][1]) + abs(scored[0][1]) + 1e-9)) if len(scored) > 1 else 1.0

        if mode == "pointwise":
            triggered = False
        elif mode == "hybrid":
            triggered = margin < margin_threshold
        elif mode == "lru":
            triggered = True
        else:
            raise ValueError(f"Unknown mode={mode}")

        victim = lru_victim if triggered else model_victim
        triggers += int(triggered)

        future = [str(r.page_id) for r in seq[idx + 1 : idx + 1 + max(fragility_window, 0)]]
        early_reuse = 1.0 if victim in future else 0.0
        if triggered:
            early_reuse_triggered.append(early_reuse)
        else:
            early_reuse_nontriggered.append(early_reuse)

        decision_logs.append(
            {
                "trace": trace_path,
                "capacity": capacity,
                "request_t": int(req.t),
                "mode": mode,
                "triggered": int(triggered),
                "margin": margin,
                "normalized_margin": norm_margin,
                "model_victim": model_victim,
                "fallback_victim": lru_victim,
                "chosen_victim": victim,
                "victim_reused_within_window": int(early_reuse),
            }
        )

        order.pop(victim, None)
        order[pid] = None
        recent_req_hist.append(pid)

    total = misses + hits
    return {
        "misses": misses,
        "hits": hits,
        "hit_rate": float(hits / total) if total else 0.0,
        "evictions": evictions,
        "trigger_count": triggers,
        "trigger_frequency": float(triggers / evictions) if evictions else 0.0,
        "victim_reuse_rate_triggered": _safe_mean(early_reuse_triggered),
        "victim_reuse_rate_nontriggered": _safe_mean(early_reuse_nontriggered),
        "victim_reuse_rate_overall": _safe_mean(early_reuse_triggered + early_reuse_nontriggered),
        "decisions": decision_logs,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _avg_rank_by_workload(results_rows: List[Dict[str, object]]) -> Dict[str, float]:
    grouped: Dict[Tuple[int, str, int], List[Dict[str, object]]] = {}
    for r in results_rows:
        k = (int(r["seed"]), str(r["trace"]), int(r["capacity"]))
        grouped.setdefault(k, []).append(r)

    ranks: Dict[str, List[float]] = {}
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda x: (int(x["misses"]), str(x["policy"])))
        for i, row in enumerate(ordered, start=1):
            ranks.setdefault(str(row["policy"]), []).append(float(i))
    return {p: _safe_mean(v) for p, v in ranks.items()}


def run_experiment(
    *,
    trace_paths: Sequence[str],
    capacities: Sequence[int],
    seeds: Sequence[int],
    margin_thresholds: Sequence[float],
    max_requests_per_trace: int,
    history_window: int,
    fragility_window: int,
    lightweight_config_path: str | None,
) -> Dict[str, object]:
    intercept = SURROGATE_INTERCEPT
    weights = dict(SURROGATE_WEIGHTS)
    if lightweight_config_path:
        payload = json.loads(Path(lightweight_config_path).read_text(encoding="utf-8"))
        if "intercept" in payload:
            intercept = float(payload["intercept"])
        for k, v in dict(payload.get("weights", {})).items():
            weights[str(k)] = float(v)

    req_cache: Dict[str, Sequence[Request]] = {}

    def _reqs(trace_path: str) -> Sequence[Request]:
        if trace_path not in req_cache:
            reqs, _ = load_trace(trace_path)
            if max_requests_per_trace > 0:
                reqs = reqs[:max_requests_per_trace]
            req_cache[trace_path] = reqs
        return req_cache[trace_path]

    threshold_rows: List[Dict[str, object]] = []
    results_rows: List[Dict[str, object]] = []
    downstream_rows: List[Dict[str, object]] = []
    trigger_rows: List[Dict[str, object]] = []
    decision_rows: List[Dict[str, object]] = []

    for seed in seeds:
        val_keys = _workload_keys(trace_paths, capacities, "val", seed)
        test_keys = _workload_keys(trace_paths, capacities, "test", seed)
        if not val_keys:
            val_keys = _workload_keys(trace_paths, capacities, "train", seed)
        if not test_keys:
            test_keys = _workload_keys(trace_paths, capacities, "val", seed)
        if not val_keys or not test_keys:
            continue

        best_threshold = float(margin_thresholds[0])
        best_key: Tuple[int, float, float] | None = None

        for threshold in margin_thresholds:
            val_total_misses = 0
            val_evictions = 0
            val_triggers = 0
            for trace_path, cap in val_keys:
                sim = _simulate_policy(
                    requests=_reqs(trace_path),
                    trace_path=trace_path,
                    capacity=cap,
                    mode="hybrid",
                    margin_threshold=float(threshold),
                    history_window=history_window,
                    fragility_window=fragility_window,
                    intercept=intercept,
                    weights=weights,
                )
                val_total_misses += int(sim["misses"])
                val_evictions += int(sim["evictions"])
                val_triggers += int(sim["trigger_count"])

            val_trigger_freq = float(val_triggers / val_evictions) if val_evictions else 0.0
            threshold_rows.append(
                {
                    "seed": seed,
                    "threshold": float(threshold),
                    "val_total_misses": val_total_misses,
                    "val_trigger_frequency": val_trigger_freq,
                    "val_workloads": len(val_keys),
                }
            )
            key = (val_total_misses, val_trigger_freq, float(threshold))
            if best_key is None or key < best_key:
                best_key = key
                best_threshold = float(threshold)

        for trace_path, cap in test_keys:
            pointwise = _simulate_policy(
                requests=_reqs(trace_path),
                trace_path=trace_path,
                capacity=cap,
                mode="pointwise",
                margin_threshold=best_threshold,
                history_window=history_window,
                fragility_window=fragility_window,
                intercept=intercept,
                weights=weights,
            )
            hybrid = _simulate_policy(
                requests=_reqs(trace_path),
                trace_path=trace_path,
                capacity=cap,
                mode="hybrid",
                margin_threshold=best_threshold,
                history_window=history_window,
                fragility_window=fragility_window,
                intercept=intercept,
                weights=weights,
            )
            lru = _simulate_policy(
                requests=_reqs(trace_path),
                trace_path=trace_path,
                capacity=cap,
                mode="lru",
                margin_threshold=best_threshold,
                history_window=history_window,
                fragility_window=fragility_window,
                intercept=intercept,
                weights=weights,
            )

            for policy_name, out in (("pointwise", pointwise), ("hybrid", hybrid), ("lru", lru)):
                results_rows.append(
                    {
                        "seed": seed,
                        "split": "test",
                        "trace": trace_path,
                        "trace_family": _trace_family(trace_path),
                        "capacity": cap,
                        "policy": policy_name,
                        "selected_threshold": best_threshold,
                        "misses": int(out["misses"]),
                        "hits": int(out["hits"]),
                        "hit_rate": float(out["hit_rate"]),
                        "evictions": int(out["evictions"]),
                        "trigger_count": int(out["trigger_count"]),
                        "trigger_frequency": float(out["trigger_frequency"]),
                        "victim_reuse_rate_overall": float(out["victim_reuse_rate_overall"]),
                        "victim_reuse_rate_triggered": float(out["victim_reuse_rate_triggered"]),
                        "victim_reuse_rate_nontriggered": float(out["victim_reuse_rate_nontriggered"]),
                    }
                )
                decision_rows.extend(out["decisions"])

            downstream_rows.append(
                {
                    "seed": seed,
                    "trace": trace_path,
                    "trace_family": _trace_family(trace_path),
                    "capacity": cap,
                    "selected_threshold": best_threshold,
                    "pointwise_misses": int(pointwise["misses"]),
                    "hybrid_misses": int(hybrid["misses"]),
                    "lru_misses": int(lru["misses"]),
                    "hybrid_minus_pointwise": int(hybrid["misses"]) - int(pointwise["misses"]),
                    "hybrid_minus_lru": int(hybrid["misses"]) - int(lru["misses"]),
                    "hybrid_trigger_frequency": float(hybrid["trigger_frequency"]),
                }
            )
            trigger_rows.append(
                {
                    "seed": seed,
                    "trace": trace_path,
                    "trace_family": _trace_family(trace_path),
                    "capacity": cap,
                    "selected_threshold": best_threshold,
                    "trigger_frequency": float(hybrid["trigger_frequency"]),
                    "victim_reuse_rate_triggered": float(hybrid["victim_reuse_rate_triggered"]),
                    "victim_reuse_rate_nontriggered": float(hybrid["victim_reuse_rate_nontriggered"]),
                    "victim_reuse_gap_triggered_minus_nontriggered": float(hybrid["victim_reuse_rate_triggered"] - hybrid["victim_reuse_rate_nontriggered"]),
                }
            )

    by_policy: Dict[str, Dict[str, float]] = {}
    for r in results_rows:
        p = str(r["policy"])
        if p not in by_policy:
            by_policy[p] = {"misses": 0.0, "hits": 0.0, "evictions": 0.0, "triggers": 0.0, "workloads": 0.0}
        by_policy[p]["misses"] += float(r["misses"])
        by_policy[p]["hits"] += float(r["hits"])
        by_policy[p]["evictions"] += float(r["evictions"])
        by_policy[p]["triggers"] += float(r["trigger_count"])
        by_policy[p]["workloads"] += 1.0

    wtl = {"wins": 0, "ties": 0, "losses": 0}
    by_workload: Dict[Tuple[int, str, int], Dict[str, int]] = {}
    for r in results_rows:
        k = (int(r["seed"]), str(r["trace"]), int(r["capacity"]))
        by_workload.setdefault(k, {})[str(r["policy"])] = int(r["misses"])
    for scores in by_workload.values():
        if "hybrid" in scores and "pointwise" in scores:
            if scores["hybrid"] < scores["pointwise"]:
                wtl["wins"] += 1
            elif scores["hybrid"] > scores["pointwise"]:
                wtl["losses"] += 1
            else:
                wtl["ties"] += 1

    avg_rank = _avg_rank_by_workload(results_rows)

    summary = {
        "config": {
            "trace_paths": list(trace_paths),
            "capacities": [int(c) for c in capacities],
            "seeds": [int(s) for s in seeds],
            "margin_threshold_grid": [float(x) for x in margin_thresholds],
            "max_requests_per_trace": int(max_requests_per_trace),
            "history_window": int(history_window),
            "fragility_window": int(fragility_window),
            "fallback_policy": "lru",
            "trigger_rule": "top1_top2_margin_lt_threshold",
            "scorer_path": "evict_value_v1_lightweight_surrogate_compatible",
            "lightweight_config_path": lightweight_config_path,
        },
        "policy_totals": {
            p: {
                "total_misses": int(v["misses"]),
                "total_hits": int(v["hits"]),
                "hit_rate": float(v["hits"] / (v["hits"] + v["misses"])) if (v["hits"] + v["misses"]) else 0.0,
                "avg_trigger_frequency": float(v["triggers"] / v["evictions"]) if v["evictions"] else 0.0,
                "workloads": int(v["workloads"]),
            }
            for p, v in by_policy.items()
        },
        "hybrid_vs_pointwise": {
            **wtl,
            "average_rank": {
                "hybrid": float(avg_rank.get("hybrid", 0.0)),
                "pointwise": float(avg_rank.get("pointwise", 0.0)),
                "lru": float(avg_rank.get("lru", 0.0)),
            },
        },
    }

    return {
        "threshold_selection_rows": threshold_rows,
        "results_rows": results_rows,
        "downstream_rows": downstream_rows,
        "trigger_analysis_rows": trigger_rows,
        "decision_rows": decision_rows,
        "summary": summary,
    }


def _build_report(payload: Dict[str, object]) -> str:
    summary = payload["summary"]
    totals = summary.get("policy_totals", {})
    hvp = summary.get("hybrid_vs_pointwise", {})
    lines = [
        "# Hybrid fallback experiment report",
        "",
        "## Key question",
        "Can a simple confidence-aware fallback make the learned candidate-scoring policy better downstream than pointwise alone?",
        "",
        "## Protocol",
        "- Fallback trigger: top1-vs-top2 score margin < threshold.",
        "- Fallback policy: LRU.",
        "- Threshold selected on validation workloads only; held-out test used only for final comparison.",
        "",
        "## Held-out results",
        f"- Hybrid vs pointwise wins/ties/losses: {hvp.get('wins', 0)}/{hvp.get('ties', 0)}/{hvp.get('losses', 0)}",
        f"- Hybrid avg trigger frequency: {float(totals.get('hybrid', {}).get('avg_trigger_frequency', 0.0)):.4f}",
        "",
        "| policy | total_misses | hit_rate | avg_trigger_frequency |",
        "|---|---:|---:|---:|",
    ]
    for p in ["pointwise", "hybrid", "lru"]:
        row = totals.get(p, {})
        lines.append(
            f"| {p} | {row.get('total_misses', 0)} | {float(row.get('hit_rate', 0.0)):.4f} | {float(row.get('avg_trigger_frequency', 0.0)):.4f} |"
        )
    lines.extend([
        "",
        "## Fragility analysis",
        "- `trigger_analysis.csv` reports victim-reuse-within-window rates for triggered vs non-triggered decisions.",
        "- Higher triggered victim-reuse rate indicates low-margin decisions are indeed more fragile.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run hybrid confidence-aware fallback experiment (medium/local)")
    ap.add_argument("--trace-glob", default="data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json")
    ap.add_argument("--capacities", default="2,3,4,5")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--horizon", type=int, default=8, help="Unused compatibility argument.")
    ap.add_argument("--margin-thresholds", default="0.00,0.01,0.03,0.05,0.08,0.12")
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--history-window", type=int, default=64)
    ap.add_argument("--fragility-window", type=int, default=16)
    ap.add_argument("--lightweight-config-path", default=None)
    ap.add_argument("--output-dir", default="analysis/hybrid_fallback_experiment")
    args = ap.parse_args()

    payload = run_experiment(
        trace_paths=_resolve_trace_paths(args.trace_glob),
        capacities=[int(x.strip()) for x in args.capacities.split(",") if x.strip()],
        seeds=[int(x.strip()) for x in args.seeds.split(",") if x.strip()],
        margin_thresholds=[float(x.strip()) for x in args.margin_thresholds.split(",") if x.strip()],
        max_requests_per_trace=args.max_requests_per_trace,
        history_window=args.history_window,
        fragility_window=args.fragility_window,
        lightweight_config_path=args.lightweight_config_path,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "results.csv", payload["results_rows"])
    _write_csv(out_dir / "downstream_results.csv", payload["downstream_rows"])
    _write_csv(out_dir / "threshold_selection.csv", payload["threshold_selection_rows"])
    _write_csv(out_dir / "trigger_analysis.csv", payload["trigger_analysis_rows"])
    _write_csv(out_dir / "decision_logs.csv", payload["decision_rows"])
    (out_dir / "summary.json").write_text(json.dumps(payload["summary"], indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_build_report(payload), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
