from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.types import PageId, Request

HISTORY_AWARE_EXTRA_COLUMNS: List[str] = [
    "hist_w8_candidate_request_rate",
    "hist_w16_candidate_request_rate",
    "hist_w32_candidate_request_rate",
    "hist_w8_candidate_hit_rate",
    "hist_w16_candidate_hit_rate",
    "hist_candidate_last_seen_gap_norm",
    "hist_candidate_last_hit_gap_norm",
    "hist_candidate_interarrival_mean_norm",
    "hist_candidate_interarrival_std_norm",
    "hist_candidate_burst_max_norm",
    "hist_candidate_recent_trend_w8_minus_w32",
    "hist_global_unique_ratio_w16",
    "hist_global_repeat_rate_w16",
    "hist_global_hit_ratio_w16",
    "hist_transition_prev_to_candidate_rate_w16",
]


@dataclass(frozen=True)
class HistoryAblationConfig:
    horizons: Tuple[int, ...] = (8,)
    history_window: int = 64


def _split_for_decision(decision_id: str) -> str:
    h = int(hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "val"
    return "test"


def _rate(hist: Sequence[PageId], pid: PageId) -> float:
    if not hist:
        return 0.0
    return float(sum(1 for x in hist if x == pid) / len(hist))


def _rate_window(hist: Sequence[PageId], pid: PageId, window: int) -> float:
    if not hist:
        return 0.0
    w = list(hist)[-window:]
    return float(sum(1 for x in w if x == pid) / len(w)) if w else 0.0


def _last_gap_norm(hist: Sequence[PageId], pid: PageId, denom: int) -> float:
    try:
        idx = list(reversed(hist)).index(pid)
        return float((idx + 1) / max(denom, 1))
    except ValueError:
        return float((denom + 1) / max(denom, 1))


def _interarrival_stats(hist: Sequence[PageId], pid: PageId, denom: int) -> Tuple[float, float]:
    positions = [i for i, x in enumerate(hist) if x == pid]
    if len(positions) < 2:
        return 1.0, 0.0
    gaps = [positions[i] - positions[i - 1] for i in range(1, len(positions))]
    mean_gap = float(np.mean(gaps) / max(denom, 1))
    std_gap = float(np.std(gaps) / max(denom, 1))
    return mean_gap, std_gap


def _max_burst_run(hist: Sequence[PageId], pid: PageId, denom: int) -> float:
    best = 0
    cur = 0
    for x in hist:
        if x == pid:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best / max(denom, 1))


def _transition_prev_to_candidate_rate(hist: Sequence[PageId], candidate: PageId, window: int) -> float:
    w = list(hist)[-window:]
    if len(w) < 2:
        return 0.0
    prevs = [a for a, _ in zip(w[:-1], w[1:])]
    total = len(prevs)
    hits = sum(1 for a, b in zip(w[:-1], w[1:]) if b == candidate and a != candidate)
    return float(hits / total) if total else 0.0


def _history_extra_features(
    *,
    candidate: PageId,
    recent_req_hist: Sequence[PageId],
    recent_hit_hist: Sequence[PageId],
    history_window: int,
) -> Dict[str, float]:
    w8_req = _rate_window(recent_req_hist, candidate, 8)
    w16_req = _rate_window(recent_req_hist, candidate, 16)
    w32_req = _rate_window(recent_req_hist, candidate, 32)
    w8_hit = _rate_window(recent_hit_hist, candidate, 8)
    w16_hit = _rate_window(recent_hit_hist, candidate, 16)

    mean_gap, std_gap = _interarrival_stats(recent_req_hist, candidate, history_window)
    w16 = list(recent_req_hist)[-16:]
    uniq_ratio = float(len(set(w16)) / len(w16)) if w16 else 0.0
    repeat_rate = float(1.0 - uniq_ratio) if w16 else 0.0
    hit_w16 = list(recent_hit_hist)[-16:]
    hit_ratio = float(len(hit_w16) / len(w16)) if w16 else 0.0

    return {
        "hist_w8_candidate_request_rate": w8_req,
        "hist_w16_candidate_request_rate": w16_req,
        "hist_w32_candidate_request_rate": w32_req,
        "hist_w8_candidate_hit_rate": w8_hit,
        "hist_w16_candidate_hit_rate": w16_hit,
        "hist_candidate_last_seen_gap_norm": _last_gap_norm(recent_req_hist, candidate, history_window),
        "hist_candidate_last_hit_gap_norm": _last_gap_norm(recent_hit_hist, candidate, history_window),
        "hist_candidate_interarrival_mean_norm": mean_gap,
        "hist_candidate_interarrival_std_norm": std_gap,
        "hist_candidate_burst_max_norm": _max_burst_run(recent_req_hist, candidate, history_window),
        "hist_candidate_recent_trend_w8_minus_w32": w8_req - w32_req,
        "hist_global_unique_ratio_w16": uniq_ratio,
        "hist_global_repeat_rate_w16": repeat_rate,
        "hist_global_hit_ratio_w16": hit_ratio,
        "hist_transition_prev_to_candidate_rate_w16": _transition_prev_to_candidate_rate(recent_req_hist, candidate, 16),
    }


def build_rows(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    cfg: HistoryAblationConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)

    base_rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []

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
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))
        decision_id = f"{trace_name}|cap={capacity}|t={t}"
        split = _split_for_decision(decision_id)

        future = requests[t + 1 :]
        for candidate in candidates:
            base_feats = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=candidate,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=_rate(recent_req_hist, candidate),
                recent_hit_rate=_rate(recent_hit_hist, candidate),
            ).as_dict()
            hist_feats = dict(base_feats)
            hist_feats.update(
                _history_extra_features(
                    candidate=candidate,
                    recent_req_hist=list(recent_req_hist),
                    recent_hit_hist=list(recent_hit_hist),
                    history_window=cfg.history_window,
                )
            )

            for h in cfg.horizons:
                fut_h = future[:h]
                after = [p for p in candidates if p != candidate] + [pid]
                y_loss = float(_simulate_lru_misses(after, fut_h, capacity=capacity))
                common: Dict[str, object] = {
                    "decision_id": decision_id,
                    "trace": trace_name,
                    "capacity": capacity,
                    "t": t,
                    "candidate_page_id": candidate,
                    "horizon": int(h),
                    "split": split,
                    "y_loss": y_loss,
                }
                rb = dict(common)
                rb.update(base_feats)
                base_rows.append(rb)

                rh = dict(common)
                rh.update(hist_feats)
                hist_rows.append(rh)

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return base_rows, hist_rows


def _xy(rows: List[Dict[str, object]], columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in columns] for r in rows], dtype=float)
    y = np.asarray([float(r["y_loss"]) for r in rows], dtype=float)
    return x, y


def ranking_metrics(rows: List[Dict[str, object]], preds: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, pred in zip(rows, preds):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(pred)))

    top1 = 0
    regrets: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x[0]["y_loss"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == best[0]["candidate_page_id"])
        regrets.append(float(chosen[0]["y_loss"]) - float(best[0]["y_loss"]))

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_eviction_match": float(top1 / denom),
        "mean_regret_vs_oracle": float(np.mean(regrets) if regrets else 0.0),
    }


def train_hist_gb(rows_train: List[Dict[str, object]], feature_columns: List[str], seed: int) -> HistGradientBoostingRegressor:
    x_train, y_train = _xy(rows_train, feature_columns)
    est = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=seed)
    est.fit(x_train, y_train)
    return est


def replay_misses(
    *,
    requests: Sequence[Request],
    capacity: int,
    feature_columns: List[str],
    model: HistGradientBoostingRegressor,
    history_window: int,
    history_aware: bool,
) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=history_window)
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

        scored: List[Tuple[str, float]] = []
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
            if history_aware:
                feats.update(
                    _history_extra_features(
                        candidate=c,
                        recent_req_hist=list(recent_req_hist),
                        recent_hit_hist=list(recent_hit_hist),
                        history_window=history_window,
                    )
                )
            x = np.asarray([[float(feats[col]) for col in feature_columns]], dtype=float)
            scored.append((c, float(model.predict(x)[0])))

        victim = min(scored, key=lambda x: (x[1], str(x[0])))[0]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses


def split_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    all_rows = [dict(r) for r in rows]
    out: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in all_rows:
        out[str(row.get("split", "train"))].append(dict(row))
    if out["train"] and out["val"] and out["test"]:
        return out

    out = {"train": [], "val": [], "test": []}
    for i, row in enumerate(all_rows):
        mod = i % 10
        if mod < 7:
            out["train"].append(dict(row))
        elif mod < 9:
            out["val"].append(dict(row))
        else:
            out["test"].append(dict(row))
    if not out["val"] and out["train"]:
        out["val"] = [dict(out["train"][0])]
    if not out["test"] and out["train"]:
        out["test"] = [dict(out["train"][0])]
    return out
