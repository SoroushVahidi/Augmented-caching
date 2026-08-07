from __future__ import annotations

import argparse
import collections
import csv
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DecisionState:
    decision_id: str
    trace: str
    capacity: int
    t: int
    split: str
    candidate_page_ids: List[str]
    incoming_features: Dict[str, float]
    history_summary: Dict[str, float]
    candidate_features: List[Dict[str, float]]
    y_losses: List[float]
    best_index: int


class JointSoftmaxEvictor:
    def __init__(self, *, lr: float = 0.05, epochs: int = 120, l2: float = 1e-4, seed: int = 7) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.w: np.ndarray | None = None

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        e = np.exp(z)
        return e / np.sum(e)

    def _joint_features(self, cand_mat: np.ndarray, global_vec: np.ndarray) -> np.ndarray:
        return np.concatenate([cand_mat] + [cand_mat * g for g in global_vec], axis=1)

    def fit(self, decisions: Sequence[DecisionState]) -> None:
        if not decisions:
            raise ValueError("No decisions to train joint model")
        d = len(decisions[0].candidate_features[0])
        g = len(decisions[0].incoming_features) + len(decisions[0].history_summary)
        dim = d * (g + 1)
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0.0, 0.01, size=(dim,))

        for _ in range(self.epochs):
            for s in decisions:
                cand = np.asarray([list(x.values()) for x in s.candidate_features], dtype=float)
                gv = np.asarray(list(s.incoming_features.values()) + list(s.history_summary.values()), dtype=float)
                z = self._joint_features(cand, gv)
                scores = z @ w
                probs = self._softmax(scores)
                y = np.zeros_like(probs)
                y[int(s.best_index)] = 1.0
                grad = z.T @ (probs - y)
                grad += self.l2 * w
                w -= self.lr * grad
        self.w = w

    def predict_choice(self, state: DecisionState) -> int:
        if self.w is None:
            raise ValueError("Model not fitted")
        cand = np.asarray([list(x.values()) for x in state.candidate_features], dtype=float)
        gv = np.asarray(list(state.incoming_features.values()) + list(state.history_summary.values()), dtype=float)
        z = self._joint_features(cand, gv)
        scores = z @ self.w
        return int(np.argmax(scores))


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


def _iter_repo_light_traces() -> List[Tuple[str, list, dict]]:
    traces: List[Tuple[str, list, dict]] = []
    for p in ["data/example_unweighted.json", "data/example_atlas_v1.json", "data/example_general_caching.json"]:
        reqs, pages = load_trace(p)
        traces.append((p, reqs, pages))
    stress = {
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
    for name, (reqs, pages) in stress.items():
        traces.append((name, reqs, pages))
    return traces


def _decision_groups(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    g: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        g.setdefault(str(r["decision_id"]), []).append(r)
    return g


def _build_decision_states(rows: Sequence[Dict[str, object]], candidate_cols: List[str]) -> List[DecisionState]:
    states: List[DecisionState] = []
    for did, items in _decision_groups(rows).items():
        ordered = sorted(items, key=lambda r: str(r["candidate_page_id"]))
        losses = [float(r["y_loss"]) for r in ordered]
        best_i = min(range(len(ordered)), key=lambda i: (losses[i], str(ordered[i]["candidate_page_id"])))
        first = ordered[0]
        incoming = {
            "request_bucket": float(first["request_bucket"]),
            "request_confidence": float(first["request_confidence"]),
        }
        hist_summary = {
            "hist_global_unique_ratio_w16": float(first["hist_global_unique_ratio_w16"]),
            "hist_global_repeat_rate_w16": float(first["hist_global_repeat_rate_w16"]),
            "hist_global_hit_ratio_w16": float(first["hist_global_hit_ratio_w16"]),
            "cache_bucket_mean": float(first["cache_bucket_mean"]),
            "cache_bucket_std": float(first["cache_bucket_std"]),
            "cache_confidence_mean": float(first["cache_confidence_mean"]),
            "cache_confidence_std": float(first["cache_confidence_std"]),
        }
        states.append(
            DecisionState(
                decision_id=did,
                trace=str(first["trace"]),
                capacity=int(first["capacity"]),
                t=int(first["t"]),
                split=str(first["split"]),
                candidate_page_ids=[str(r["candidate_page_id"]) for r in ordered],
                incoming_features=incoming,
                history_summary=hist_summary,
                candidate_features=[{c: float(r[c]) for c in candidate_cols} for r in ordered],
                y_losses=losses,
                best_index=best_i,
            )
        )
    return states


def _ranking_from_index(states: Sequence[DecisionState], chosen: Dict[str, int]) -> Dict[str, float]:
    top1 = 0
    regrets: List[float] = []
    for s in states:
        idx = chosen.get(s.decision_id)
        if idx is None:
            continue
        top1 += int(int(idx) == int(s.best_index))
        regrets.append(float(s.y_losses[int(idx)] - s.y_losses[s.best_index]))
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
                total += float(clf.predict_proba(diff)[0][1])
                count += 1
            utils[aid] = total / max(count, 1)
        out[did] = max(utils.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return out


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


def _rate(hist: Sequence[PageId], pid: PageId) -> float:
    if not hist:
        return 0.0
    return float(sum(1 for x in hist if x == pid) / len(hist))


def _replay_pairwise_misses(*, requests: Sequence[Request], capacity: int, feature_columns: List[str], clf: HistGradientBoostingClassifier, history_window: int) -> int:
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
            base.update(_history_extra_features(candidate=c, recent_req_hist=list(recent_req_hist), recent_hit_hist=list(recent_hit_hist), history_window=history_window))
            feat_rows.append(base)

        utilities: Dict[str, float] = {}
        for i, c in enumerate(candidates):
            total = 0.0
            count = 0
            for j, d in enumerate(candidates):
                if i == j:
                    continue
                diff = np.asarray([[float(feat_rows[i][k]) - float(feat_rows[j][k]) for k in feature_columns]], dtype=float)
                total += float(clf.predict_proba(diff)[0][1])
                count += 1
            utilities[str(c)] = total / max(count, 1)
        victim = max(utilities.items(), key=lambda kv: (kv[1], kv[0]))[0]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses


def _replay_joint_misses(*, requests: Sequence[Request], capacity: int, model: JointSoftmaxEvictor, candidate_cols: List[str], history_window: int) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: collections.deque[PageId] = collections.deque(maxlen=history_window)
    recent_hit_hist: collections.deque[PageId] = collections.deque(maxlen=history_window)
    misses = 0

    for t, req in enumerate(requests):
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
        candidate_rows: List[Dict[str, float]] = []
        for c in candidates:
            feats = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=c,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=_rate(recent_req_hist, c),
                recent_hit_rate=_rate(recent_hit_hist, c),
            ).as_dict()
            feats.update(_history_extra_features(candidate=c, recent_req_hist=list(recent_req_hist), recent_hit_hist=list(recent_hit_hist), history_window=history_window))
            candidate_rows.append(feats)

        state = DecisionState(
            decision_id=f"replay|cap={capacity}|t={t}",
            trace="replay",
            capacity=capacity,
            t=t,
            split="test",
            candidate_page_ids=[str(c) for c in candidates],
            incoming_features={"request_bucket": float(req_bucket), "request_confidence": float(req_conf)},
            history_summary={
                "hist_global_unique_ratio_w16": float(candidate_rows[0]["hist_global_unique_ratio_w16"]),
                "hist_global_repeat_rate_w16": float(candidate_rows[0]["hist_global_repeat_rate_w16"]),
                "hist_global_hit_ratio_w16": float(candidate_rows[0]["hist_global_hit_ratio_w16"]),
                "cache_bucket_mean": float(candidate_rows[0]["cache_bucket_mean"]),
                "cache_bucket_std": float(candidate_rows[0]["cache_bucket_std"]),
                "cache_confidence_mean": float(candidate_rows[0]["cache_confidence_mean"]),
                "cache_confidence_std": float(candidate_rows[0]["cache_confidence_std"]),
            },
            candidate_features=[{c: float(fr[c]) for c in candidate_cols} for fr in candidate_rows],
            y_losses=[0.0 for _ in candidates],
            best_index=0,
        )
        victim_idx = model.predict_choice(state)
        victim = candidates[victim_idx]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)
    return misses


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint-state eviction reasoning ablation (lightweight)")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-requests-per-trace", type=int, default=2500)
    ap.add_argument("--out-dir", default="analysis/joint_state_reasoning_light")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    cfg = HistoryAblationConfig(horizons=(args.horizon,), history_window=64)
    trace_items = _iter_repo_light_traces()

    base_rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    loaded_traces: List[Dict[str, object]] = []

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
    hist_pairwise = _fit_pairwise_classifier(hist_splits["train"], hist_cols, seed=args.seed)

    joint_candidate_cols = [
        "candidate_bucket",
        "candidate_confidence",
        "candidate_age_norm",
        "candidate_predictor_score",
        "candidate_lru_score",
        "candidate_is_predictor_victim",
        "candidate_is_lru_victim",
        "score_gap_to_predictor_best",
        "score_gap_to_lru_victim",
        "recent_candidate_request_rate",
        "recent_candidate_hit_rate",
        "hist_w8_candidate_request_rate",
        "hist_w16_candidate_request_rate",
        "hist_w32_candidate_request_rate",
        "hist_w8_candidate_hit_rate",
        "hist_w16_candidate_hit_rate",
        "hist_candidate_last_seen_gap_norm",
        "hist_candidate_last_hit_gap_norm",
        "hist_candidate_interarrival_mean_norm",
        "hist_candidate_burst_max_norm",
        "hist_candidate_recent_trend_w8_minus_w32",
    ]

    decision_states = _build_decision_states(hist_splits["train"] + hist_splits["val"] + hist_splits["test"], joint_candidate_cols)
    split_states: Dict[str, List[DecisionState]] = {"train": [], "val": [], "test": []}
    for s in decision_states:
        split_states[s.split].append(s)

    joint_model = JointSoftmaxEvictor(seed=args.seed)
    joint_model.fit(split_states["train"])

    base_val = _ranking_from_chosen(base_splits["val"], _choose_by_regression(base_splits["val"], base_cols, base_reg))
    base_test = _ranking_from_chosen(base_splits["test"], _choose_by_regression(base_splits["test"], base_cols, base_reg))
    pair_val = _ranking_from_chosen(hist_splits["val"], _choose_by_pairwise(hist_splits["val"], hist_cols, hist_pairwise))
    pair_test = _ranking_from_chosen(hist_splits["test"], _choose_by_pairwise(hist_splits["test"], hist_cols, hist_pairwise))
    joint_val = _ranking_from_index(split_states["val"], {s.decision_id: joint_model.predict_choice(s) for s in split_states["val"]})
    joint_test = _ranking_from_index(split_states["test"], {s.decision_id: joint_model.predict_choice(s) for s in split_states["test"]})

    comparison_rows = [
        {
            "variant": "base_regression",
            "objective": "replay_loss_regression",
            "state_type": "independent_candidate",
            "feature_count": len(base_cols),
            "val_decisions": int(base_val["decision_count"]),
            "val_top1_eviction_match": base_val["top1_eviction_match"],
            "val_mean_regret": base_val["mean_regret_vs_oracle"],
            "test_decisions": int(base_test["decision_count"]),
            "test_top1_eviction_match": base_test["top1_eviction_match"],
            "test_mean_regret": base_test["mean_regret_vs_oracle"],
        },
        {
            "variant": "history_pairwise",
            "objective": "pairwise_classifier",
            "state_type": "independent_candidate",
            "feature_count": len(hist_cols),
            "val_decisions": int(pair_val["decision_count"]),
            "val_top1_eviction_match": pair_val["top1_eviction_match"],
            "val_mean_regret": pair_val["mean_regret_vs_oracle"],
            "test_decisions": int(pair_test["decision_count"]),
            "test_top1_eviction_match": pair_test["top1_eviction_match"],
            "test_mean_regret": pair_test["mean_regret_vs_oracle"],
        },
        {
            "variant": "joint_state_softmax",
            "objective": "direct_victim_softmax_ce",
            "state_type": "joint_decision_state",
            "feature_count": len(joint_candidate_cols),
            "val_decisions": int(joint_val["decision_count"]),
            "val_top1_eviction_match": joint_val["top1_eviction_match"],
            "val_mean_regret": joint_val["mean_regret_vs_oracle"],
            "test_decisions": int(joint_test["decision_count"]),
            "test_top1_eviction_match": joint_test["top1_eviction_match"],
            "test_mean_regret": joint_test["mean_regret_vs_oracle"],
        },
    ]

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
            miss_pair = _replay_pairwise_misses(
                requests=reqs,
                capacity=cap,
                feature_columns=hist_cols,
                clf=hist_pairwise,
                history_window=cfg.history_window,
            )
            miss_joint = _replay_joint_misses(
                requests=reqs,
                capacity=cap,
                model=joint_model,
                candidate_cols=joint_candidate_cols,
                history_window=cfg.history_window,
            )
            replay_rows.append(
                {
                    "trace": trace_name,
                    "capacity": cap,
                    "base_regression_misses": miss_base,
                    "history_pairwise_misses": miss_pair,
                    "joint_state_softmax_misses": miss_joint,
                    "delta_base_minus_joint": float(miss_base - miss_joint),
                    "delta_pairwise_minus_joint": float(miss_pair - miss_joint),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "model_comparison.csv", comparison_rows)
    _write_csv(out_dir / "downstream_replay.csv", replay_rows)

    dataset_rows: List[Dict[str, object]] = []
    for s in decision_states:
        dataset_rows.append(
            {
                "decision_id": s.decision_id,
                "trace": s.trace,
                "split": s.split,
                "capacity": s.capacity,
                "t": s.t,
                "incoming_request_bucket": s.incoming_features["request_bucket"],
                "incoming_request_confidence": s.incoming_features["request_confidence"],
                "history_summary": json.dumps(s.history_summary, sort_keys=True),
                "candidate_page_ids": json.dumps(s.candidate_page_ids),
                "candidate_features": json.dumps(s.candidate_features),
                "y_losses": json.dumps(s.y_losses),
                "best_victim_page_id": s.candidate_page_ids[s.best_index],
            }
        )
    _write_csv(out_dir / "joint_state_dataset.csv", dataset_rows)

    mean_base = mean(float(r["base_regression_misses"]) for r in replay_rows)
    mean_pair = mean(float(r["history_pairwise_misses"]) for r in replay_rows)
    mean_joint = mean(float(r["joint_state_softmax_misses"]) for r in replay_rows)

    summary = {
        "traces": loaded_traces,
        "capacities": capacities,
        "horizon": args.horizon,
        "history_window": cfg.history_window,
        "joint_state_dataset": {
            "path": str(out_dir / "joint_state_dataset.csv"),
            "decision_count": len(decision_states),
            "format": "one_row_per_eviction_decision_with_candidate_set",
        },
        "models": {
            "base": "HistGradientBoostingRegressor on evict_value_v1 candidate rows",
            "history_pairwise": "HistGradientBoostingClassifier pairwise objective on history-aware candidate rows",
            "joint_state": "lightweight shared candidate encoder with set-wise softmax victim prediction",
        },
        "model_comparison": comparison_rows,
        "downstream_replay_means": {
            "base_regression": mean_base,
            "history_pairwise": mean_pair,
            "joint_state_softmax": mean_joint,
        },
        "pivot_assessment": {
            "joint_beats_base": bool(mean_joint < mean_base),
            "joint_beats_history_pairwise": bool(mean_joint < mean_pair),
            "recommendation": "pivot_candidate" if (mean_joint < mean_base and mean_joint <= mean_pair) else "future_work",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Joint-state eviction reasoning ablation (lightweight)")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Fully separate lightweight path; no heavy_r1 reruns and no manuscript-artifact edits.")
    lines.append("- Same compact subset as recent lightweight ablations: 3 repository traces + 3 stress traces.")
    lines.append("- Same capacities and horizon used in that lane (default capacities 2/3/4, horizon=8).")
    lines.append("- Comparison targets: base `evict_value_v1` regression and history-aware pairwise objective variant.")
    lines.append("")
    lines.append("## New decision-state dataset")
    lines.append("- Each row is a single eviction state with incoming item, full current cache residents, compact history summary, and oracle victim label.")
    lines.append(f"- Rows: {len(dataset_rows)} decisions (`joint_state_dataset.csv`).")
    lines.append("")
    lines.append("## Candidate-decision quality (test split)")
    for row in comparison_rows:
        lines.append(
            f"- {row['variant']}: top1={float(row['test_top1_eviction_match']):.4f}, "
            f"mean_regret={float(row['test_mean_regret']):.4f}"
        )
    lines.append("")
    lines.append("## Downstream replay misses (mean over trace×capacity)")
    lines.append(f"- base_regression: {mean_base:.3f}")
    lines.append(f"- history_pairwise: {mean_pair:.3f}")
    lines.append(f"- joint_state_softmax: {mean_joint:.3f}")
    lines.append("")
    lines.append("## Explicit answers")
    promise = "yes" if (mean_joint < mean_base or float(joint_test["top1_eviction_match"]) >= float(base_test["top1_eviction_match"])) else "no"
    lines.append(f"1. Does joint cache-state reasoning show meaningful promise? **{promise}**.")
    if mean_joint < mean_pair:
        lines.append("2. Does it outperform independent scoring enough for a pivot? **Potentially yes in this lightweight run**, but requires larger-scale confirmation before manuscript-safe replacement.")
    elif mean_joint < mean_base:
        lines.append("2. Does it outperform independent scoring enough for a pivot? **Not yet**; it improves over base pointwise but does not clear the best history-aware pairwise baseline.")
    else:
        lines.append("2. Does it outperform independent scoring enough for a pivot? **No** in this lightweight run.")
    lines.append("3. Should it remain future work for now? **Yes** unless/until it consistently beats the best lightweight pairwise baseline across broader traces.")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
