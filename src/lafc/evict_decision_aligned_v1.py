from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.types import PageId, Request


@dataclass(frozen=True)
class DecisionAlignedEvictConfig:
    """Configuration for decision-aligned eviction dataset builders."""

    horizon: int = 8
    history_window: int = 64
    include_ties: bool = False


def _compute_candidate_feature_rows(
    *,
    candidates: List[PageId],
    req_bucket: int,
    req_conf: float,
    bucket_by_page: Dict[PageId, int],
    conf_by_page: Dict[PageId, float],
    recent_req_hist: Deque[PageId],
    recent_hit_hist: Deque[PageId],
) -> Dict[PageId, Dict[str, float]]:
    out: Dict[PageId, Dict[str, float]] = {}
    for candidate in candidates:
        req_rate = (sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)) if recent_req_hist else 0.0
        hit_rate = (sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)) if recent_hit_hist else 0.0
        out[candidate] = compute_candidate_features_v1(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            candidate=candidate,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_request_rate=req_rate,
            recent_hit_rate=hit_rate,
        ).as_dict()
    return out


def build_evict_regret_examples_v1(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    cfg: DecisionAlignedEvictConfig,
) -> List[Dict[str, object]]:
    """Build candidate-level examples with regret supervision for eviction decisions."""

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
        decision_id = f"{trace_name}|c{capacity}|t{t}|h{cfg.horizon}"
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))
        features_by_candidate = _compute_candidate_feature_rows(
            candidates=candidates,
            req_bucket=req_bucket,
            req_conf=req_conf,
            bucket_by_page=bucket_by_page,
            conf_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
        )

        future = requests[t + 1 : t + 1 + cfg.horizon]
        losses: Dict[PageId, float] = {}
        for candidate in candidates:
            after = [p for p in candidates if p != candidate] + [pid]
            losses[candidate] = float(_simulate_lru_misses(after, future, capacity=capacity))

        best_loss = min(losses.values()) if losses else 0.0
        for candidate in candidates:
            regret = float(losses[candidate] - best_loss)
            row: Dict[str, object] = {
                "decision_id": decision_id,
                "trace": trace_name,
                "capacity": capacity,
                "t": t,
                "horizon": cfg.horizon,
                "candidate_page_id": candidate,
                "y_loss": float(losses[candidate]),
                "y_regret": regret,
                "y_is_best": float(regret == 0.0),
            }
            row.update(features_by_candidate[candidate])
            rows.append(row)

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return rows


def build_evict_pairwise_examples_v1(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    cfg: DecisionAlignedEvictConfig,
) -> List[Dict[str, object]]:
    """Build pairwise preference examples between eviction candidates."""

    pointwise = build_evict_regret_examples_v1(requests=requests, capacity=capacity, trace_name=trace_name, cfg=cfg)
    by_decision: Dict[str, List[Dict[str, object]]] = {}
    for row in pointwise:
        by_decision.setdefault(str(row["decision_id"]), []).append(row)

    rows: List[Dict[str, object]] = []
    for decision_id, items in by_decision.items():
        items_sorted = sorted(items, key=lambda r: str(r["candidate_page_id"]))
        for i in range(len(items_sorted)):
            for j in range(i + 1, len(items_sorted)):
                ri = items_sorted[i]
                rj = items_sorted[j]
                regret_i = float(ri["y_regret"])
                regret_j = float(rj["y_regret"])
                if regret_i == regret_j and not cfg.include_ties:
                    continue
                label = 1 if regret_i < regret_j else 0

                pair: Dict[str, object] = {
                    "decision_id": decision_id,
                    "trace": ri["trace"],
                    "capacity": ri["capacity"],
                    "t": ri["t"],
                    "horizon": cfg.horizon,
                    "candidate_i_page_id": ri["candidate_page_id"],
                    "candidate_j_page_id": rj["candidate_page_id"],
                    "loss_i": float(ri["y_loss"]),
                    "loss_j": float(rj["y_loss"]),
                    "regret_i": regret_i,
                    "regret_j": regret_j,
                    "label_i_better": label,
                    "is_tie": float(regret_i == regret_j),
                }

                for c in EVICT_VALUE_V1_FEATURE_COLUMNS:
                    fi = float(ri[c])
                    fj = float(rj[c])
                    pair[f"i_{c}"] = fi
                    pair[f"j_{c}"] = fj
                    pair[f"delta_{c}"] = fi - fj
                rows.append(pair)

    return rows


__all__ = [
    "DecisionAlignedEvictConfig",
    "build_evict_regret_examples_v1",
    "build_evict_pairwise_examples_v1",
]
