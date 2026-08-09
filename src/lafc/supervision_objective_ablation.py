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

import bisect
import collections
import math
import random
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _build_occurrence_index(requests: Sequence[Request]) -> Dict[PageId, List[int]]:
    """page_id -> sorted list of ALL positions where it's requested.
    O(n) once per trace; lets the hot loop below answer "does candidate X
    ever occur again after position t, and if so where" in O(log n) via
    bisect, instead of a linear scan of the remaining trace."""
    idx: Dict[PageId, List[int]] = {}
    for i, r in enumerate(requests):
        idx.setdefault(r.page_id, []).append(i)
    return idx


def _build_distinct_suffix_counts(requests: Sequence[Request]) -> List[int]:
    """distinct_suffix[t] = number of distinct page_ids in requests[t:],
    for t in 0..len(requests) (distinct_suffix[len(requests)] == 0). O(n)
    once per trace via a single backward pass."""
    n = len(requests)
    out = [0] * (n + 1)
    seen: set = set()
    for t in range(n - 1, -1, -1):
        seen.add(requests[t].page_id)
        out[t] = len(seen)
    return out


def _next_arrival_and_reuse_distance_fast(
    candidate: PageId,
    t: int,
    n_requests: int,
    occurrence_index: Dict[PageId, List[int]],
    distinct_suffix_counts: List[int],
    requests: Sequence[Request],
) -> Tuple[float, float]:
    """Equivalent of (next_raw, reuse_raw) as computed by
    _next_use_distance / _forward_reuse_distance on future=requests[t+1:],
    but without ever linearly scanning to the end of the trace for a
    candidate that never reoccurs -- the dominant cost on high-cardinality
    traces (e.g. cloudphysics: 41010/50000 unique pages), where most
    evicted candidates never appear again. Verified bit-for-bit equal to
    the original functions in tests/test_supervision_objective_ablation.py
    (test_fast_distance_helpers_match_reference_scan_implementation).
    """
    occ = occurrence_index.get(candidate)
    future_len = n_requests - t - 1
    if not occ:
        # Candidate never occurs at all (shouldn't happen -- it was just
        # evicted from cache, so it occurred at least once -- but handled
        # defensively) or only occurs at/before t.
        return float(future_len + 1), float(distinct_suffix_counts[t + 1])

    pos = bisect.bisect_right(occ, t)
    if pos >= len(occ):
        # No occurrence after t.
        return float(future_len + 1), float(distinct_suffix_counts[t + 1])

    abs_pos = occ[pos]
    next_raw = float(abs_pos - t)
    # Distinct OTHER objects strictly between t and abs_pos -- bounded by
    # the true (usually small) reuse gap, same cost as the original
    # bounded scan in the "reoccurs" case (not the pathological case).
    seen: set = set()
    for idx in range(t + 1, abs_pos):
        seen.add(requests[idx].page_id)
    reuse_raw = float(len(seen))
    return next_raw, reuse_raw


def iter_multi_label_candidate_rows(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
    selected_decision_ids: Optional[set[str]] = None,
) -> Iterable[Dict[str, object]]:
    """Generator form of build_multi_label_candidate_rows.

    Yields one row at a time instead of accumulating a full list, so a
    caller can stream rows to disk in bounded shards (mirroring
    scripts/build_evict_value_dataset_wulver_v1.py's shard-flush pattern)
    without holding an entire multi-family, multi-capacity dataset in
    memory at once. This matters here specifically because each row carries
    5 label columns (vs. 1 in the canonical single-objective pipeline), so
    an unbounded in-memory list is proportionally more expensive.

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

    n_requests = len(requests)
    occurrence_index = _build_occurrence_index(requests)
    distinct_suffix_counts = _build_distinct_suffix_counts(requests)

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
        if selected_decision_ids is not None and decision_id not in selected_decision_ids:
            lru_victim = candidates[0]
            order.pop(lru_victim)
            order[pid] = None
            recent_req_hist.append(pid)
            continue
        # Bounded H-step slice only (not the whole remaining trace): the
        # eviction-loss rollout only ever looks H steps ahead.
        future_h = requests[t + 1 : t + 1 + H]

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

            # B/C: next-arrival + forward reuse distance, via the O(log n)
            # occurrence-index fast path (exact equivalent of
            # _next_use_distance / _forward_reuse_distance on
            # requests[t+1:], see _next_arrival_and_reuse_distance_fast).
            next_raw, reuse_raw = _next_arrival_and_reuse_distance_fast(
                candidate, t, n_requests, occurrence_index, distinct_suffix_counts, requests
            )
            next_censored = min(next_raw, float(H))
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
            yield row

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)


def build_multi_label_candidate_rows(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
    selected_decision_ids: Optional[set[str]] = None,
) -> List[Dict[str, object]]:
    """List-returning convenience wrapper over iter_multi_label_candidate_rows,
    for small traces / tests where materializing the full list is fine."""
    return list(
        iter_multi_label_candidate_rows(
            requests,
            capacity,
            trace_name,
            trace_family,
            cfg,
            selected_decision_ids=selected_decision_ids,
        )
    )


def _max_items_for_pair_budget(budget: int) -> int:
    """Largest k such that C(k,2) <= budget (k >= 2)."""
    k = 2
    while (k + 1) * k // 2 <= budget:
        k += 1
    return k


def build_pairwise_rows(
    candidate_rows: Iterable[Dict[str, object]],
    source: str = "next_arrival",
    include_ties: bool = False,
    max_pairs_per_decision: Optional[int] = None,
    sample_seed: int = 0,
) -> List[Dict[str, object]]:
    """Derive pairwise preference rows from the shared multi-label candidate rows.

    source="next_arrival" (primary, protocol Section 3.D): label i>j iff
    i's censored next-arrival distance is smaller (i reused sooner) --
    independent of the eviction-loss computation entirely, HALP-style.

    source="regret" (secondary diagnostic ONLY, protocol Section 3.D):
    label i>j iff i's eviction_loss_label is smaller -- DERIVED DIRECTLY
    from objective_eviction_loss; never use as a primary comparison arm.

    max_pairs_per_decision (resource-safety addendum, not a semantics
    change): pair count per decision is C(k,2) in the candidate-set size
    k, which is quadratic and can be very large at capacity=128. When set,
    caps the candidate SET used to build pairs at each decision to the
    largest k' with C(k',2) <= max_pairs_per_decision, via a deterministic
    seeded sample (keyed on decision_id + sample_seed) -- so which pairs
    are included is reproducible, and the same budget is applied uniformly
    to every fold/capacity/decision. This bounds TRAINING dataset size; it
    does not alter the pairwise preference definition itself, and has no
    effect on held-out evaluation (a full cache-simulation replay, not a
    pairwise count).
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
        if max_pairs_per_decision is not None:
            k_max = _max_items_for_pair_budget(max_pairs_per_decision)
            if len(items_sorted) > k_max:
                rng = random.Random(f"{decision_id}:{sample_seed}")
                items_sorted = sorted(
                    rng.sample(items_sorted, k_max), key=lambda r: str(r["candidate_page_id"])
                )
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
    "iter_multi_label_candidate_rows",
    "build_multi_label_candidate_rows",
    "build_pairwise_rows",
]
