from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.types import PageId, Request


@dataclass(frozen=True)
class EvictValueV2RolloutConfig:
    """Config for finite-horizon rollout labeling of eviction candidates."""

    horizons: Tuple[int, ...] = (4, 8, 16, 32)
    history_window: int = 64
    reference_policy: str = "lru"
    include_ties: bool = False


def _next_use_distance(cache_page: PageId, future_reqs: Sequence[Request], at_idx: int) -> float:
    for idx in range(at_idx, len(future_reqs)):
        if future_reqs[idx].page_id == cache_page:
            return float(idx)
    return math.inf


def _choose_victim(order: collections.OrderedDict[PageId, None], future_reqs: Sequence[Request], step_idx: int, policy: str) -> PageId:
    candidates = list(order.keys())
    if policy == "lru":
        return candidates[0]
    if policy == "blind_oracle":
        return max(candidates, key=lambda p: (_next_use_distance(p, future_reqs, step_idx), -candidates.index(p)))
    raise ValueError(f"Unsupported reference policy: {policy}")


def simulate_rollout_misses(
    *,
    cache_pages: Sequence[PageId],
    future_reqs: Sequence[Request],
    capacity: int,
    reference_policy: str,
) -> int:
    """Simulate finite-horizon misses with a configurable continuation policy."""

    order: collections.OrderedDict[PageId, None] = collections.OrderedDict((p, None) for p in cache_pages)
    misses = 0
    for step_idx, req in enumerate(future_reqs):
        pid = req.page_id
        if pid in order:
            order.move_to_end(pid)
            continue

        misses += 1
        victim = _choose_victim(order, future_reqs, step_idx, reference_policy)
        order.pop(victim)
        order[pid] = None
    return misses


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


def _rank_from_losses(losses: Dict[PageId, float]) -> Dict[PageId, int]:
    ordered = sorted(losses.items(), key=lambda x: (x[1], str(x[0])))
    return {candidate: idx + 1 for idx, (candidate, _loss) in enumerate(ordered)}


def build_rollout_candidate_rows_v2(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: EvictValueV2RolloutConfig,
) -> List[Dict[str, object]]:
    """Build candidate rows with rollout loss and rollout regret labels."""

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
        feature_rows = _compute_candidate_feature_rows(
            candidates=candidates,
            req_bucket=req_bucket,
            req_conf=req_conf,
            bucket_by_page=bucket_by_page,
            conf_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
        )

        for horizon in cfg.horizons:
            future = requests[t + 1 : t + 1 + horizon]
            losses: Dict[PageId, float] = {}
            for candidate in candidates:
                forced_cache = [p for p in candidates if p != candidate] + [pid]
                losses[candidate] = float(
                    simulate_rollout_misses(
                        cache_pages=forced_cache,
                        future_reqs=future,
                        capacity=capacity,
                        reference_policy=cfg.reference_policy,
                    )
                )

            best_loss = min(losses.values()) if losses else 0.0
            ranks = _rank_from_losses(losses)

            for candidate in candidates:
                regret = float(losses[candidate] - best_loss)
                row: Dict[str, object] = {
                    "trace": trace_name,
                    "family": trace_family,
                    "decision_id": f"{trace_name}|c{capacity}|t{t}|h{horizon}",
                    "request_t": t,
                    "capacity": capacity,
                    "horizon": int(horizon),
                    "reference_policy": cfg.reference_policy,
                    "candidate_page_id": candidate,
                    "rollout_loss_h": float(losses[candidate]),
                    "rollout_regret_h": regret,
                    "candidate_is_rollout_optimal": float(regret == 0.0),
                    "candidate_rank": int(ranks[candidate]),
                    "candidate_count": len(candidates),
                }
                row.update(feature_rows[candidate])
                rows.append(row)

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return rows


def build_pairwise_rows_from_candidate_rows(
    candidate_rows: Iterable[Dict[str, object]],
    include_ties: bool = False,
) -> List[Dict[str, object]]:
    """Convert rollout-labeled candidate rows to pairwise ranking rows."""

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)

    pairwise_rows: List[Dict[str, object]] = []
    for decision_id, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: str(r["candidate_page_id"]))
        for i in range(len(items_sorted)):
            for j in range(i + 1, len(items_sorted)):
                left = items_sorted[i]
                right = items_sorted[j]
                reg_i = float(left["rollout_regret_h"])
                reg_j = float(right["rollout_regret_h"])
                if reg_i == reg_j and not include_ties:
                    continue

                label = 1 if reg_i < reg_j else 0
                pair: Dict[str, object] = {
                    "decision_id": decision_id,
                    "trace": left["trace"],
                    "family": left.get("family", "unknown"),
                    "request_t": left["request_t"],
                    "capacity": left["capacity"],
                    "horizon": left["horizon"],
                    "reference_policy": left["reference_policy"],
                    "candidate_i_page_id": left["candidate_page_id"],
                    "candidate_j_page_id": right["candidate_page_id"],
                    "rollout_loss_i": float(left["rollout_loss_h"]),
                    "rollout_loss_j": float(right["rollout_loss_h"]),
                    "rollout_regret_i": reg_i,
                    "rollout_regret_j": reg_j,
                    "rollout_regret_diff": reg_i - reg_j,
                    "label_i_better": int(label),
                    "is_tie": float(reg_i == reg_j),
                }
                for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
                    fi = float(left[col])
                    fj = float(right[col])
                    pair[f"i_{col}"] = fi
                    pair[f"j_{col}"] = fj
                    pair[f"delta_{col}"] = fi - fj
                pairwise_rows.append(pair)

    return pairwise_rows


__all__ = [
    "EvictValueV2RolloutConfig",
    "build_rollout_candidate_rows_v2",
    "build_pairwise_rows_from_candidate_rows",
    "simulate_rollout_misses",
]
