"""Supervision-objective ablation: shared multi-label dataset construction
and objective-specific models (see docs/supervision_objective_ablation_protocol.md
and configs/supervision_objective_ablation_v1.json for the frozen protocol).

Central design choice (protocol Section 5 / task Section 11): ONE shared
per-decision loop computes the feature vector once per candidate and
attaches every objective's label to the SAME row, so objectives A
(eviction loss), B (next arrival), and C (reuse distance) are trained on
literally the same candidate-decision examples -- not independently
resampled datasets that could differ for reasons unrelated to the
supervision target.

Reuses, unmodified:
- `compute_candidate_features_v1` / `EVICT_VALUE_V1_FEATURE_COLUMNS`
  (src/lafc/evict_value_features_v1.py) -- identical features for all
  four objectives.
- `_next_use_distance` (src/lafc/evict_value_v2_rollout.py) -- the
  existing next-arrival-distance primitive, reused rather than
  reimplemented.
- `_simulate_lru_misses` (src/lafc/evict_value_dataset_v1.py) -- the
  exact eviction-loss kernel already used by the canonical pipeline.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_v2_rollout import _next_use_distance
from lafc.types import PageId, Request


@dataclass(frozen=True)
class ObjectiveAblationConfig:
    horizon: int = 4
    history_window: int = 64


def _forward_reuse_distance(page: PageId, future_reqs: Sequence[Request], at_idx: int) -> float:
    """Standard forward reuse distance: number of DISTINCT other objects
    requested strictly before `page`'s next reoccurrence (not the same as
    next-arrival's raw request-count distance -- deliberately kept
    distinct, per protocol Section 3.C).
    """
    seen: set = set()
    for idx in range(at_idx, len(future_reqs)):
        pid = future_reqs[idx].page_id
        if pid == page:
            return float(len(seen))
        seen.add(pid)
    return math.inf


def build_multi_label_candidate_rows(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
) -> List[Dict[str, object]]:
    """One row per (candidate, decision), carrying every objective's label.

    Structurally mirrors iter_candidate_rows / build_rollout_candidate_rows_v2
    (same candidate enumeration, same feature computation), but computes
    all label views for the SAME candidate set at each decision instead of
    building four separate datasets.
    """
    H = cfg.horizon
    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
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
        decision_id = f"{trace_name}|cap={capacity}|t={t}"
        future = requests[t + 1 :]
        future_h = future[:H]

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

            # A: eviction loss (evict `candidate`, admit `pid`, replay future_h under LRU)
            after = [p for p in candidates if p != candidate] + [pid]
            eviction_loss = float(_simulate_lru_misses(after, future_h, capacity=capacity))

            # B: next-arrival distance (steps until candidate's next reoccurrence)
            raw_next = _next_use_distance(candidate, future, 0)
            next_raw = raw_next + 1.0 if math.isfinite(raw_next) else float(len(future) + 1)
            next_censored = min(next_raw, float(H))

            # C: forward reuse distance (distinct other objects before next reoccurrence)
            raw_reuse = _forward_reuse_distance(candidate, future, 0)
            reuse_raw = raw_reuse if math.isfinite(raw_reuse) else float(len(set(r.page_id for r in future)))
            reuse_censored = min(reuse_raw, float(H))

            row: Dict[str, object] = {
                "trace_name": trace_name,
                "trace_family": trace_family,
                "capacity": int(capacity),
                "horizon": int(H),
                "decision_id": decision_id,
                "decision_t": int(t),
                "candidate_page_id": candidate,
                "eviction_loss_label": eviction_loss,
                "next_arrival_label_raw": float(next_raw),
                "next_arrival_label_censored": float(next_censored),
                "reuse_distance_label_raw": float(reuse_raw),
                "reuse_distance_label_censored": float(reuse_censored),
            }
            row.update(feats)
            rows.append(row)

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return rows


def build_pairwise_rows(
    candidate_rows: Iterable[Dict[str, object]],
    source: str = "next_arrival",
    include_ties: bool = False,
) -> List[Dict[str, object]]:
    """Derive pairwise preference rows from the shared multi-label candidate rows.

    source="next_arrival" (primary, protocol Section 3.D): label i>j iff
    i's censored next-arrival distance is smaller (i reused sooner) --
    independent of the eviction-loss computation entirely, HALP-style.

    source="regret" (secondary diagnostic ONLY, protocol Section 3.D):
    label i>j iff i's eviction_loss_label is smaller -- DERIVED DIRECTLY
    from objective_eviction_loss; never use as a primary comparison arm.
    """
    if source not in ("next_arrival", "regret"):
        raise ValueError(f"Unsupported pairwise source: {source!r}")

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)

    label_col = "next_arrival_label_censored" if source == "next_arrival" else "eviction_loss_label"

    pairwise_rows: List[Dict[str, object]] = []
    for decision_id, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: str(r["candidate_page_id"]))
        for i in range(len(items_sorted)):
            for j in range(i + 1, len(items_sorted)):
                left, right = items_sorted[i], items_sorted[j]
                vi, vj = float(left[label_col]), float(right[label_col])
                if vi == vj and not include_ties:
                    continue
                label = 1 if vi < vj else 0
                pair: Dict[str, object] = {
                    "decision_id": decision_id,
                    "trace_name": left["trace_name"],
                    "trace_family": left.get("trace_family", "unknown"),
                    "decision_t": left["decision_t"],
                    "capacity": left["capacity"],
                    "horizon": left["horizon"],
                    "pairwise_label_source": source,
                    "candidate_i_page_id": left["candidate_page_id"],
                    "candidate_j_page_id": right["candidate_page_id"],
                    "value_i": vi,
                    "value_j": vj,
                    "label_i_preferred": int(label),
                    "is_tie": float(vi == vj),
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
    "ObjectiveAblationConfig",
    "build_multi_label_candidate_rows",
    "build_pairwise_rows",
]
