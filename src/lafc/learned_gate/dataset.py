from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence

from lafc.learned_gate.features import compute_gate_features, compute_lru_scores, compute_predictor_scores
from lafc.types import PageId, Request


@dataclass(frozen=True)
class GateDatasetConfig:
    horizon: int = 4
    regret_window: int = 32


def _future_distances(page_ids: Sequence[PageId]) -> List[Dict[PageId, int]]:
    n = len(page_ids)
    out: List[Dict[PageId, int]] = [dict() for _ in range(n)]
    last: Dict[PageId, int] = {}
    for t in range(n - 1, -1, -1):
        pid = page_ids[t]
        last[pid] = t
        out[t] = dict(last)
    return out


def build_gate_examples(
    requests: Sequence[Request],
    capacity: int,
    cfg: GateDatasetConfig,
    trace_name: str,
) -> List[Dict[str, float | int | str]]:
    if capacity < 1:
        raise ValueError("capacity must be >=1")
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}

    page_ids = [r.page_id for r in requests]
    future = _future_distances(page_ids)

    disagreements: Deque[int] = collections.deque(maxlen=cfg.regret_window)
    regrets: Deque[int] = collections.deque(maxlen=cfg.regret_window)
    rows: List[Dict[str, float | int | str]] = []

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
        recent_disagree_rate = (sum(disagreements) / len(disagreements)) if disagreements else 0.0
        recent_regret_rate = (sum(regrets) / len(regrets)) if regrets else 0.0

        feat = compute_gate_features(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_regret_rate=recent_regret_rate,
            recent_disagree_rate=recent_disagree_rate,
        )

        p_scores = compute_predictor_scores(candidates, bucket_by_page)
        l_scores = compute_lru_scores(candidates)
        pred_victim = max(candidates, key=lambda x: (p_scores[x], -candidates.index(x)))
        lru_victim = max(candidates, key=lambda x: (l_scores[x], -candidates.index(x)))

        next_map = future[t + 1] if (t + 1) < len(future) else {}
        pred_next = next_map.get(pred_victim, math.inf)
        lru_next = next_map.get(lru_victim, math.inf)

        pred_delta = (pred_next - t) if math.isfinite(pred_next) else math.inf
        lru_delta = (lru_next - t) if math.isfinite(lru_next) else math.inf

        pred_penalty = 1 if pred_delta <= cfg.horizon else 0
        lru_penalty = 1 if lru_delta <= cfg.horizon else 0
        label = int(pred_penalty < lru_penalty)

        disagreements.append(int(pred_victim != lru_victim))
        regrets.append(int(label == 0 and pred_victim != lru_victim))

        row: Dict[str, float | int | str] = {
            "trace": trace_name,
            "capacity": capacity,
            "t": t,
            "predictor_victim": pred_victim,
            "lru_victim": lru_victim,
            "pred_next_delta": pred_delta,
            "lru_next_delta": lru_delta,
            "pred_penalty_h": pred_penalty,
            "lru_penalty_h": lru_penalty,
            "y": label,
        }
        row.update(feat.as_dict())
        rows.append(row)

        victim = pred_victim
        order.pop(victim, None)
        order[pid] = None

    return rows


def split_by_trace(rows: Iterable[Dict[str, float | int | str]]) -> Dict[str, List[Dict[str, float | int | str]]]:
    groups: Dict[str, List[Dict[str, float | int | str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        trace = str(row["trace"])
        bucket = sum(ord(c) for c in trace) % 10
        if bucket <= 5:
            groups["train"].append(dict(row))
        elif bucket <= 7:
            groups["val"].append(dict(row))
        else:
            groups["test"].append(dict(row))
    return groups
