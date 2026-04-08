from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from lafc.learned_gate.features import compute_lru_scores, compute_predictor_scores
from lafc.learned_gate.features_v2 import compute_gate_features_v2
from lafc.types import PageId, Request


@dataclass(frozen=True)
class GateDatasetV2Config:
    horizons: Tuple[int, ...] = (4, 8, 16)
    margin: float = 0.0
    history_window: int = 64


def _simulate_lru_misses(cache_pages: Sequence[PageId], future_reqs: Sequence[Request], capacity: int) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict((p, None) for p in cache_pages)
    misses = 0
    for req in future_reqs:
        pid = req.page_id
        if pid in order:
            order.move_to_end(pid)
            continue
        misses += 1
        if len(order) >= capacity:
            oldest = next(iter(order))
            order.pop(oldest)
        order[pid] = None
    return misses


def _split_by_trace_and_capacity(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        key = f"{row['trace']}|cap={row['capacity']}"
        h = sum(ord(c) for c in key) % 10
        if h <= 5:
            out["train"].append(dict(row))
        elif h <= 7:
            out["val"].append(dict(row))
        else:
            out["test"].append(dict(row))
    return out


def build_gate_examples_v2(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    cfg: GateDatasetV2Config,
) -> List[Dict[str, object]]:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}

    regret_hist: Deque[int] = collections.deque(maxlen=cfg.history_window)
    disagree_hist: Deque[int] = collections.deque(maxlen=cfg.history_window)
    ctx_hist: Deque[int] = collections.deque(maxlen=cfg.history_window)
    ctx_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)

    rows: List[Dict[str, object]] = []

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            continue

        if len(order) < capacity:
            order[pid] = None
            continue

        candidates = list(order.keys())
        if not candidates:
            order[pid] = None
            continue

        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))
        conf_bin = 0 if req_conf <= 0.33 else (1 if req_conf <= 0.66 else 2)
        ctx = (req_bucket, conf_bin)
        ctx_count = ctx_counts[ctx]
        ctx_freq = (sum(ctx_hist) / len(ctx_hist)) if ctx_hist else 0.0

        feats = compute_gate_features_v2(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_regret_rate=(sum(regret_hist) / len(regret_hist)) if regret_hist else 0.0,
            recent_disagree_rate=(sum(disagree_hist) / len(disagree_hist)) if disagree_hist else 0.0,
            context_seen_count=ctx_count,
            recent_context_frequency=ctx_freq,
        )

        p_scores = compute_predictor_scores(candidates, bucket_by_page)
        l_scores = compute_lru_scores(candidates)
        pred_victim = max(candidates, key=lambda x: (p_scores[x], -candidates.index(x)))
        lru_victim = max(candidates, key=lambda x: (l_scores[x], -candidates.index(x)))

        base_cache = list(candidates)
        after_pred = [p for p in base_cache if p != pred_victim] + [pid]
        after_lru = [p for p in base_cache if p != lru_victim] + [pid]
        future = requests[t + 1 :]

        reg_by_h: Dict[int, float] = {}
        cls_by_h: Dict[int, int] = {}
        tri_by_h: Dict[int, int] = {}

        for h in cfg.horizons:
            fut_h = future[:h]
            loss_pred = _simulate_lru_misses(after_pred, fut_h, capacity=capacity)
            loss_lru = _simulate_lru_misses(after_lru, fut_h, capacity=capacity)
            y_reg = float(loss_lru - loss_pred)
            y_cls = int(y_reg > cfg.margin)
            y_tri = 1 if y_reg > cfg.margin else (-1 if y_reg < -cfg.margin else 0)
            reg_by_h[h] = y_reg
            cls_by_h[h] = y_cls
            tri_by_h[h] = y_tri

            row: Dict[str, object] = {
                "trace": trace_name,
                "capacity": capacity,
                "t": t,
                "horizon": h,
                "predictor_victim": pred_victim,
                "lru_victim": lru_victim,
                "loss_pred": loss_pred,
                "loss_lru": loss_lru,
                "y_reg": y_reg,
                "y_cls": y_cls,
                "y_tri": y_tri,
            }
            row.update(feats)
            rows.append(row)

        anchor_h = min(cfg.horizons)
        regret_hist.append(int(reg_by_h[anchor_h] <= 0 and pred_victim != lru_victim))
        disagree_hist.append(int(pred_victim != lru_victim))
        ctx_hist.append(1)
        ctx_counts[ctx] += 1

        order.pop(lru_victim, None)
        order[pid] = None

    return rows


__all__ = [
    "GateDatasetV2Config",
    "build_gate_examples_v2",
    "_split_by_trace_and_capacity",
]
