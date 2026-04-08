from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from lafc.evict_value_features_v1 import compute_candidate_features_v1
from lafc.types import PageId, Request


@dataclass(frozen=True)
class EvictValueDatasetV1Config:
    horizons: Tuple[int, ...] = (4, 8, 16)
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


def build_evict_value_examples_v1(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    cfg: EvictValueDatasetV1Config,
) -> List[Dict[str, object]]:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}

    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)

    rows: List[Dict[str, object]] = []

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        hit = pid in order
        if hit:
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
        future = requests[t + 1 :]
        decision_id = f"{trace_name}|c{capacity}|t{t}"

        for candidate in candidates:
            req_rate = (sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)) if recent_req_hist else 0.0
            hit_rate = (sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)) if recent_hit_hist else 0.0
            feats = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=candidate,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()

            for h in cfg.horizons:
                fut_h = future[:h]
                after = [p for p in candidates if p != candidate] + [pid]
                loss = _simulate_lru_misses(after, fut_h, capacity=capacity)
                row: Dict[str, object] = {
                    "decision_id": decision_id,
                    "trace": trace_name,
                    "capacity": capacity,
                    "t": t,
                    "candidate_page_id": candidate,
                    "horizon": int(h),
                    "y_loss": float(loss),
                    "y_value": float(-loss),
                }
                row.update(feats)
                rows.append(row)

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return rows


__all__ = [
    "EvictValueDatasetV1Config",
    "build_evict_value_examples_v1",
    "_simulate_lru_misses",
    "_split_by_trace_and_capacity",
]
