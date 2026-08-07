from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_v2_rollout import EvictValueV2RolloutConfig, build_pairwise_rows_from_candidate_rows, build_rollout_candidate_rows_v2
from lafc.simulator.request_trace import load_trace
from lafc.types import PageId, Request


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


def _build_candidate_rows(
    *,
    trace_paths: Sequence[str],
    capacities: Sequence[int],
    horizon: int,
    max_requests_per_trace: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace_path in trace_paths:
        reqs, _pages = load_trace(trace_path)
        if max_requests_per_trace > 0:
            reqs = reqs[:max_requests_per_trace]
        family = _trace_family(trace_path)
        for cap in capacities:
            cfg = EvictValueV2RolloutConfig(horizons=(horizon,), reference_policy="lru")
            raw = build_rollout_candidate_rows_v2(
                requests=reqs,
                capacity=cap,
                trace_name=trace_path,
                trace_family=family,
                cfg=cfg,
            )
            for r in raw:
                row = dict(r)
                row["target_regret"] = float(row["rollout_regret_h"])
                t_val = int(row.get("request_t", row.get("t", 0)))
                cap_val = int(float(row["capacity"]))
                row["decision_key"] = f"{row['trace']}|c{cap_val}|t{t_val}"
                rows.append(row)
    return rows


def _split_rows_for_seed(rows: List[Dict[str, object]], seed: int) -> Dict[str, List[Dict[str, object]]]:
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[_split_for_seed(str(r["trace"]), seed)].append(r)
    return out


def _xy_pointwise(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] + [float(r["capacity"]), float(r["horizon"])]
            for r in rows
        ],
        dtype=float,
    )
    y = np.asarray([float(r["target_regret"]) for r in rows], dtype=float)
    return x, y


def _build_pairwise_rows(candidate_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    pairwise = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=False)
    out: List[Dict[str, object]] = []
    for row in pairwise:
        r = dict(row)
        t_val = int(r.get("request_t", r.get("t", 0)))
        cap_val = int(float(r["capacity"]))
        r["decision_key"] = f"{r['trace']}|c{cap_val}|t{t_val}"
        out.append(r)
    return out


def _xy_pairwise(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    delta_cols = sorted(k for k in rows[0].keys() if k.startswith("delta_"))
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    return x, y, delta_cols


def _reg_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)) if len(y) else 0.0,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))) if len(y) else 0.0,
    }


def _pointwise_decision_stats(rows: List[Dict[str, object]], pred: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = defaultdict(list)
    for r, p in zip(rows, pred):
        grouped[str(r["decision_key"])].append((r, float(p)))

    top1 = 0
    chosen_regrets: List[float] = []
    regret_gap: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x[0]["target_regret"]), str(x[0]["candidate_page_id"])))
        top1 += int(str(chosen[0]["candidate_page_id"]) == str(best[0]["candidate_page_id"]))
        best_regret = min(float(x[0]["target_regret"]) for x in items)
        chosen_regrets.append(float(chosen[0]["target_regret"]))
        regret_gap.append(float(chosen[0]["target_regret"]) - best_regret)

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
        "mean_regret_vs_best": float(np.mean(regret_gap) if regret_gap else 0.0),
    }


def _pairwise_decision_stats(
    candidate_rows: List[Dict[str, object]],
    pairwise_rows: List[Dict[str, object]],
    clf: LogisticRegression,
    delta_cols: List[str],
) -> Dict[str, float]:
    score_by_decision: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    regret_by_decision: Dict[str, Dict[str, float]] = defaultdict(dict)

    for p in pairwise_rows:
        d = str(p["decision_key"])
        ci = str(p["candidate_i_page_id"])
        cj = str(p["candidate_j_page_id"])
        x = np.asarray([[float(p[c]) for c in delta_cols]], dtype=float)
        prob_i = float(clf.predict_proba(x)[0, 1])
        score_by_decision[d][ci] += prob_i
        score_by_decision[d][cj] += 1.0 - prob_i

    for r in candidate_rows:
        d = str(r["decision_key"])
        regret_by_decision[d][str(r["candidate_page_id"])] = float(r["target_regret"])

    top1 = 0
    chosen_regrets: List[float] = []
    regret_gap: List[float] = []
    for d, scores in score_by_decision.items():
        regrets = regret_by_decision.get(d, {})
        if not regrets:
            continue
        chosen = max(scores.keys(), key=lambda c: (scores[c], c))
        best = min(regrets.keys(), key=lambda c: (regrets[c], c))
        top1 += int(chosen == best)
        best_regret = min(regrets.values())
        chosen_regrets.append(float(regrets[chosen]))
        regret_gap.append(float(regrets[chosen] - best_regret))

    denom = max(len(score_by_decision), 1)
    return {
        "decision_count": float(len(score_by_decision)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
        "mean_regret_vs_best": float(np.mean(regret_gap) if regret_gap else 0.0),
    }


def _run_lru_misses(requests: Sequence[Request], capacity: int) -> Tuple[int, int]:
    order: OrderedDict[PageId, None] = OrderedDict()
    misses = 0
    hits = 0
    for req in requests:
        pid = req.page_id
        if pid in order:
            order.move_to_end(pid)
            hits += 1
            continue
        misses += 1
        if len(order) >= capacity:
            order.popitem(last=False)
        order[pid] = None
    return misses, hits


def _candidate_feature_vectors(
    *,
    candidates: List[str],
    req_bucket: int,
    req_conf: float,
    bucket_by_page: Dict[PageId, int],
    conf_by_page: Dict[PageId, float],
    recent_req_hist: Deque[PageId],
    recent_hit_hist: Deque[PageId],
    capacity: int,
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cand in candidates:
        req_rate = (sum(1 for x in recent_req_hist if x == cand) / len(recent_req_hist)) if recent_req_hist else 0.0
        hit_rate = (sum(1 for x in recent_hit_hist if x == cand) / len(recent_hit_hist)) if recent_hit_hist else 0.0
        feat = compute_candidate_features_v1(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            candidate=cand,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_request_rate=req_rate,
            recent_hit_rate=hit_rate,
        ).as_dict()
        feat["capacity"] = float(capacity)
        feat["horizon"] = float(horizon)
        out[cand] = {k: float(v) for k, v in feat.items()}
    return out


def _run_pointwise_policy_misses(
    requests: Sequence[Request],
    capacity: int,
    model: Pipeline,
    horizon: int,
) -> Tuple[int, int]:
    order: OrderedDict[PageId, None] = OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = deque(maxlen=64)
    recent_hit_hist: Deque[PageId] = deque(maxlen=64)
    misses = 0
    hits = 0

    for req in requests:
        pid = req.page_id
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

        candidates = list(order.keys())
        feat_map = _candidate_feature_vectors(
            candidates=candidates,
            req_bucket=int(req.metadata.get("bucket", 0)),
            req_conf=float(req.metadata.get("confidence", 0.5)),
            bucket_by_page=bucket_by_page,
            conf_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
            capacity=capacity,
            horizon=horizon,
        )
        scores: List[Tuple[str, float]] = []
        for cand in candidates:
            x = np.asarray([[feat_map[cand][c] for c in EVICT_VALUE_V1_FEATURE_COLUMNS] + [float(capacity), float(horizon)]], dtype=float)
            scores.append((cand, float(model.predict(x)[0])))
        victim = min(scores, key=lambda x: (x[1], x[0]))[0]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses, hits


def _run_pairwise_policy_misses(
    requests: Sequence[Request],
    capacity: int,
    pair_model: LogisticRegression | None,
    horizon: int,
    delta_cols: List[str],
    constant_prob: float | None = None,
) -> Tuple[int, int]:
    order: OrderedDict[PageId, None] = OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = deque(maxlen=64)
    recent_hit_hist: Deque[PageId] = deque(maxlen=64)
    misses = 0
    hits = 0

    for req in requests:
        pid = req.page_id
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

        candidates = list(order.keys())
        feat_map = _candidate_feature_vectors(
            candidates=candidates,
            req_bucket=int(req.metadata.get("bucket", 0)),
            req_conf=float(req.metadata.get("confidence", 0.5)),
            bucket_by_page=bucket_by_page,
            conf_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
            capacity=capacity,
            horizon=horizon,
        )

        scores = {cand: 0.0 for cand in candidates}
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                ci = candidates[i]
                cj = candidates[j]
                delta_values = {f"delta_{col}": feat_map[ci][col] - feat_map[cj][col] for col in EVICT_VALUE_V1_FEATURE_COLUMNS}
                x = np.asarray([[delta_values[c] for c in delta_cols]], dtype=float)
                if constant_prob is not None:
                    prob_i = float(constant_prob)
                else:
                    assert pair_model is not None
                    prob_i = float(pair_model.predict_proba(x)[0, 1])
                scores[ci] += prob_i
                scores[cj] += 1.0 - prob_i

        victim = max(scores.keys(), key=lambda c: (scores[c], c))
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses, hits


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    trace_glob: str,
    capacities: List[int],
    horizon: int,
    max_requests_per_trace: int,
    seeds: List[int],
    output_dir: str,
    supervision_style: str,
) -> Dict[str, object]:
    trace_paths = _resolve_trace_paths(trace_glob)
    candidate_rows = _build_candidate_rows(
        trace_paths=trace_paths,
        capacities=capacities,
        horizon=horizon,
        max_requests_per_trace=max_requests_per_trace,
    )
    pairwise_rows = _build_pairwise_rows(candidate_rows)

    result_rows: List[Dict[str, object]] = []
    downstream_rows: List[Dict[str, object]] = []
    disagreement_rows: List[Dict[str, object]] = []

    for seed in seeds:
        split_candidates = _split_rows_for_seed(candidate_rows, seed)
        split_pairwise = _split_rows_for_seed(pairwise_rows, seed)

        if not split_candidates["train"] or not split_pairwise["train"]:
            continue

        x_train_pw, y_train_pw = _xy_pointwise(split_candidates["train"])
        pointwise_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        pointwise_model.fit(x_train_pw, y_train_pw)

        x_pair_train, y_pair_train, delta_cols = _xy_pairwise(split_pairwise["train"])
        pairwise_model = LogisticRegression(max_iter=600, random_state=seed)
        pairwise_single_class = None
        if len(set(int(v) for v in y_pair_train.tolist())) < 2:
            pairwise_single_class = int(y_pair_train[0]) if len(y_pair_train) else 0
        else:
            pairwise_model.fit(x_pair_train, y_pair_train)

        for split in ["train", "val", "test"]:
            eval_candidates = split_candidates[split]
            eval_pairwise = split_pairwise[split]
            if not eval_candidates:
                continue

            x_eval_pw, y_eval_pw = _xy_pointwise(eval_candidates)
            pw_pred = pointwise_model.predict(x_eval_pw)
            pw_metrics = {
                "seed": seed,
                "split": split,
                "supervision_style": "pointwise",
                **_reg_metrics(y_eval_pw, pw_pred),
                **_pointwise_decision_stats(eval_candidates, pw_pred),
            }
            result_rows.append(pw_metrics)

            if pairwise_single_class is not None:
                pair_metrics = _pointwise_decision_stats(
                    eval_candidates,
                    np.asarray([float(pairwise_single_class)] * len(eval_candidates), dtype=float),
                )
            else:
                pair_metrics = _pairwise_decision_stats(eval_candidates, eval_pairwise, pairwise_model, delta_cols)

            pair_metrics = {
                "seed": seed,
                "split": split,
                "supervision_style": "pairwise",
                "mae": 0.0,
                "rmse": 0.0,
                **pair_metrics,
            }
            result_rows.append(pair_metrics)

            if split == "test":
                # disagreement and hardness analyses on decision-level choices
                grouped_candidates: Dict[str, List[Dict[str, object]]] = defaultdict(list)
                for row in eval_candidates:
                    grouped_candidates[str(row["decision_key"])].append(row)

                for d, items in grouped_candidates.items():
                    feats = np.asarray(
                        [[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] + [float(r["capacity"]), float(r["horizon"])] for r in items],
                        dtype=float,
                    )
                    preds = pointwise_model.predict(feats)
                    point_choice = min(zip(items, preds), key=lambda x: (x[1], str(x[0]["candidate_page_id"])))[0]

                    scores = {str(r["candidate_page_id"]): 0.0 for r in items}
                    by_pid = {str(r["candidate_page_id"]): r for r in items}
                    pids = sorted(by_pid.keys())
                    for i in range(len(pids)):
                        for j in range(i + 1, len(pids)):
                            ri = by_pid[pids[i]]
                            rj = by_pid[pids[j]]
                            delta = np.asarray(
                                [[float(ri[c]) - float(rj[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS]],
                                dtype=float,
                            )
                            if pairwise_single_class is not None:
                                prob_i = float(pairwise_single_class)
                            else:
                                prob_i = float(pairwise_model.predict_proba(delta)[0, 1])
                            scores[pids[i]] += prob_i
                            scores[pids[j]] += 1.0 - prob_i
                    pair_choice_pid = max(scores.keys(), key=lambda c: (scores[c], c))
                    pair_choice = by_pid[pair_choice_pid]

                    true_regrets = sorted(float(r["target_regret"]) for r in items)
                    best_regret = true_regrets[0]
                    second_regret = true_regrets[1] if len(true_regrets) > 1 else true_regrets[0]
                    hard_tie = int(abs(second_regret - best_regret) <= 1e-9)

                    point_sorted = sorted(float(v) for v in preds)
                    point_margin = point_sorted[1] - point_sorted[0] if len(point_sorted) > 1 else 0.0

                    disagreement_rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "decision_key": d,
                            "family": str(items[0].get("family", "unknown")),
                            "capacity": int(float(items[0]["capacity"])),
                            "candidate_count": len(items),
                            "hard_tie": hard_tie,
                            "pointwise_top2_margin": float(point_margin),
                            "is_close_pointwise": int(point_margin <= 0.1),
                            "decision_disagreement": int(str(point_choice["candidate_page_id"]) != str(pair_choice["candidate_page_id"])),
                            "pointwise_regret": float(point_choice["target_regret"]),
                            "pairwise_regret": float(pair_choice["target_regret"]),
                            "gain_pairwise_minus_pointwise": float(point_choice["target_regret"] - pair_choice["target_regret"]),
                        }
                    )

        test_traces = [t for t in trace_paths if _split_for_seed(t, seed) == "test"]
        if supervision_style in {"pointwise", "both"}:
            styles_to_eval = ["pointwise"]
        elif supervision_style == "pairwise":
            styles_to_eval = ["pairwise"]
        else:
            styles_to_eval = ["pointwise", "pairwise"]
        if supervision_style == "both":
            styles_to_eval = ["pointwise", "pairwise"]

        for trace_path in test_traces:
            reqs, _ = load_trace(trace_path)
            if max_requests_per_trace > 0:
                reqs = reqs[:max_requests_per_trace]
            for cap in capacities:
                lru_misses, lru_hits = _run_lru_misses(reqs, cap)
                for style in styles_to_eval:
                    if style == "pointwise":
                        misses, hits = _run_pointwise_policy_misses(reqs, cap, pointwise_model, horizon)
                    else:
                        misses, hits = _run_pairwise_policy_misses(
                            reqs,
                            cap,
                            pairwise_model if pairwise_single_class is None else None,
                            horizon,
                            [f"delta_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS],
                            constant_prob=float(pairwise_single_class) if pairwise_single_class is not None else None,
                        )
                    downstream_rows.append(
                        {
                            "seed": seed,
                            "trace": trace_path,
                            "family": _trace_family(trace_path),
                            "capacity": cap,
                            "supervision_style": style,
                            "requests": len(reqs),
                            "policy_misses": misses,
                            "policy_hit_rate": (hits / len(reqs)) if reqs else 0.0,
                            "lru_misses": lru_misses,
                            "lru_hit_rate": (lru_hits / len(reqs)) if reqs else 0.0,
                            "delta_misses_vs_lru": misses - lru_misses,
                        }
                    )

    # aggregate downstream comparison
    agg_downstream: List[Dict[str, object]] = []
    ranks: List[Dict[str, object]] = []
    workload_rows = defaultdict(list)
    for row in downstream_rows:
        workload_rows[(int(row["seed"]), str(row["trace"]), int(row["capacity"]))].append(row)

    wins = ties = losses = 0
    for key, rows in workload_rows.items():
        by_style = {str(r["supervision_style"]): int(r["policy_misses"]) for r in rows}
        if "pointwise" in by_style and "pairwise" in by_style:
            pm = by_style["pointwise"]
            qm = by_style["pairwise"]
            if pm == qm:
                ranks.append({"supervision_style": "pointwise", "rank": 1})
                ranks.append({"supervision_style": "pairwise", "rank": 1})
            elif pm < qm:
                ranks.append({"supervision_style": "pointwise", "rank": 1})
                ranks.append({"supervision_style": "pairwise", "rank": 2})
            else:
                ranks.append({"supervision_style": "pairwise", "rank": 1})
                ranks.append({"supervision_style": "pointwise", "rank": 2})
        if len(rows) == 2:
            p = next(r for r in rows if r["supervision_style"] == "pointwise")
            q = next(r for r in rows if r["supervision_style"] == "pairwise")
            if int(q["policy_misses"]) < int(p["policy_misses"]):
                wins += 1
            elif int(q["policy_misses"]) > int(p["policy_misses"]):
                losses += 1
            else:
                ties += 1

    for style in sorted({str(r["supervision_style"]) for r in downstream_rows}):
        rows = [r for r in downstream_rows if str(r["supervision_style"]) == style]
        total_requests = sum(int(r["requests"]) for r in rows)
        total_misses = sum(int(r["policy_misses"]) for r in rows)
        total_lru = sum(int(r["lru_misses"]) for r in rows)
        agg_downstream.append(
            {
                "supervision_style": style,
                "workloads": len(rows),
                "total_requests": total_requests,
                "total_misses": total_misses,
                "hit_rate": 1.0 - (total_misses / total_requests if total_requests else 0.0),
                "delta_misses_vs_lru": total_misses - total_lru,
                "avg_rank": float(mean(r["rank"] for r in ranks if r["supervision_style"] == style)),
            }
        )

    # disagreement/hardness aggregations
    group_defs = {
        "all": lambda r: "all",
        "hard_tie": lambda r: f"hard_tie={int(r['hard_tie'])}",
        "pointwise_close": lambda r: f"pointwise_close={int(r['is_close_pointwise'])}",
        "family": lambda r: f"family={r['family']}",
        "capacity": lambda r: f"capacity={int(r['capacity'])}",
        "disagreement": lambda r: f"decision_disagreement={int(r['decision_disagreement'])}",
    }
    disagreement_analysis_rows: List[Dict[str, object]] = []
    for group_type, fn in group_defs.items():
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in disagreement_rows:
            grouped[fn(row)].append(row)
        for group_value, items in sorted(grouped.items()):
            disagreement_analysis_rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "decisions": len(items),
                    "mean_gain_pairwise_minus_pointwise": float(mean(float(x["gain_pairwise_minus_pointwise"]) for x in items)),
                    "pairwise_better_rate": float(sum(1 for x in items if float(x["gain_pairwise_minus_pointwise"]) > 0.0) / len(items)),
                    "pointwise_better_rate": float(sum(1 for x in items if float(x["gain_pairwise_minus_pointwise"]) < 0.0) / len(items)),
                    "tie_rate": float(sum(1 for x in items if float(x["gain_pairwise_minus_pointwise"]) == 0.0) / len(items)),
                    "decision_disagreement_rate": float(sum(int(x["decision_disagreement"]) for x in items) / len(items)),
                }
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "results.csv", result_rows)
    _write_csv(output / "downstream_results.csv", downstream_rows)
    _write_csv(output / "disagreement_analysis.csv", disagreement_analysis_rows)

    summary = {
        "trace_glob": trace_glob,
        "trace_count": len(trace_paths),
        "capacities": capacities,
        "horizon": horizon,
        "max_requests_per_trace": max_requests_per_trace,
        "seeds": seeds,
        "pointwise_model_family": "ridge_regression_with_standard_scaler",
        "pairwise_model_family": "logistic_regression_on_delta_features",
        "aggregate_downstream": agg_downstream,
        "pairwise_vs_pointwise": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "workloads_compared": wins + losses + ties,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    by_style = {r["supervision_style"]: r for r in agg_downstream}
    p = by_style.get("pointwise", {})
    q = by_style.get("pairwise", {})
    lines = [
        "# Pairwise vs Pointwise supervision experiment",
        "",
        "## Setup",
        f"- traces: `{trace_glob}` ({len(trace_paths)} traces)",
        f"- capacities: `{capacities}`",
        f"- horizon: `{horizon}`",
        f"- seeds: `{seeds}`",
        f"- pointwise model: `{summary['pointwise_model_family']}`",
        f"- pairwise model: `{summary['pairwise_model_family']}`",
        "",
        "## Downstream aggregate",
        "| style | workloads | total_misses | hit_rate | delta_vs_lru | avg_rank |",
        "|---|---:|---:|---:|---:|---:|",
        f"| pointwise | {p.get('workloads', 0)} | {p.get('total_misses', 0)} | {p.get('hit_rate', 0.0):.4f} | {p.get('delta_misses_vs_lru', 0)} | {p.get('avg_rank', 0.0):.4f} |",
        f"| pairwise | {q.get('workloads', 0)} | {q.get('total_misses', 0)} | {q.get('hit_rate', 0.0):.4f} | {q.get('delta_misses_vs_lru', 0)} | {q.get('avg_rank', 0.0):.4f} |",
        "",
        f"- Pairwise wins/ties/losses vs pointwise: **{wins}/{ties}/{losses}**",
    ]
    if wins > losses:
        lines.append("- Pairwise looks promising enough to prioritize as next branch.")
    elif wins == losses:
        lines.append("- Evidence is mixed; keep pairwise as active experimental branch.")
    else:
        lines.append("- Pairwise does not yet beat pointwise downstream; keep pointwise as default while iterating pairwise design.")

    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled pairwise vs pointwise eviction supervision experiment")
    ap.add_argument("--trace-glob", default="data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json")
    ap.add_argument("--capacities", default="2,3,4,5")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--supervision-style", choices=["pointwise", "pairwise", "both"], default="both")
    ap.add_argument("--output-dir", default="analysis/pairwise_vs_pointwise")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    summary = run_experiment(
        trace_glob=args.trace_glob,
        capacities=capacities,
        horizon=args.horizon,
        max_requests_per_trace=args.max_requests_per_trace,
        seeds=seeds,
        output_dir=args.output_dir,
        supervision_style=args.supervision_style,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
