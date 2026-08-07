"""Distribution-shift ablation: state-generation (behavior policy) ablation
distinct from the label-continuation-policy axis already explored in
src/lafc/evict_value_v2_rollout.py (see
docs/distribution_shift_ablation_protocol.md Section 1 for the prior-art
audit that establishes this is not a duplicate).

Central design: every existing candidate-row builder in this repository
(iter_candidate_rows, build_rollout_candidate_rows_v2, and the
objective-ablation worktree's iter_multi_label_candidate_rows) ends each
eviction decision with `lru_victim = candidates[0]; order.pop(lru_victim)`
-- the cache trajectory that determines which states get visited next is
unconditionally LRU. This module generalizes that one line: the ACTUAL
evicted candidate driving subsequent states can instead be chosen by a
frozen model's own argmin decision (the same rule as
supervision_objective_ablation_policy.ScalarObjectivePolicy in the sibling
worktree), while the GROUND-TRUTH label attached to every row is always
computed independently via the frozen, unmodified _simulate_lru_misses --
never the model's own predicted score (anti-circularity, protocol Section 7).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class DistributionShiftConfig:
    horizon: int = 4
    history_window: int = 64


def _choose_state_generation_victim(
    candidates: List[PageId],
    behavior_model: Optional[EvictValueV1Model],
    feat_rows: Optional[List[Dict[str, float]]],
) -> PageId:
    """The single line this whole module generalizes: which candidate is
    ACTUALLY evicted (driving what states get visited next). None ->
    LRU (candidates[0], matches every existing builder exactly). A frozen
    model -> argmin predicted eviction loss (same rule as
    ScalarObjectivePolicy(direction="min"))."""
    if behavior_model is None:
        return candidates[0]
    assert feat_rows is not None
    preds = behavior_model.predict_loss_batch(feat_rows)
    best_idx = min(range(len(candidates)), key=lambda i: (preds[i], i))
    return candidates[best_idx]


def iter_candidate_rows_with_behavior_policy(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: DistributionShiftConfig,
    behavior_model: Optional[EvictValueV1Model] = None,
    behavior_policy_name: str = "lru",
):
    """Generator yielding one row per (candidate, decision), with the
    ACTUAL cache trajectory driven by `behavior_model` (None = LRU / off
    policy; a frozen model = its own argmin decisions / on policy). Labels
    (eviction_loss_label) are always computed via the frozen
    _simulate_lru_misses, independent of behavior_model (anti-circularity).

    Every row also carries `state_generation_policy` = behavior_policy_name
    for provenance/auditing.
    """
    H = cfg.horizon
    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)

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
        decision_id = f"{trace_name}|cap={capacity}|t={t}|pol={behavior_policy_name}"
        future_h = requests[t + 1 : t + 1 + H]

        feat_rows: List[Dict[str, float]] = []
        for candidate in candidates:
            req_rate = (sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)) if recent_req_hist else 0.0
            hit_rate = (sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)) if recent_hit_hist else 0.0
            feat_rows.append(
                compute_candidate_features_v1(
                    request_bucket=req_bucket, request_confidence=req_conf,
                    candidates=candidates, candidate=candidate,
                    bucket_by_page=bucket_by_page, confidence_by_page=conf_by_page,
                    recent_request_rate=req_rate, recent_hit_rate=hit_rate,
                ).as_dict()
            )

        for candidate, feats in zip(candidates, feat_rows):
            after = [p for p in candidates if p != candidate] + [pid]
            eviction_loss = float(_simulate_lru_misses(after, future_h, capacity=capacity))
            row: Dict[str, object] = {
                "trace_name": trace_name,
                "trace_family": trace_family,
                "capacity": int(capacity),
                "horizon": int(H),
                "decision_id": decision_id,
                "decision_t": int(t),
                "candidate_page_id": candidate,
                "eviction_loss_label": eviction_loss,
                "state_generation_policy": behavior_policy_name,
            }
            row.update(feats)
            yield row

        victim = _choose_state_generation_victim(candidates, behavior_model, feat_rows)
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)


# ---------------------------------------------------------------------
# State-shift metrics
# ---------------------------------------------------------------------

def _wasserstein_1(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except ImportError:
        sa, sb = sorted(a), sorted(b)
        na, nb = len(sa), len(sb)
        if na == 0 or nb == 0:
            return 0.0
        grid = sorted(set(sa) | set(sb))
        total = 0.0
        for lo, hi in zip(grid[:-1], grid[1:]):
            mid = (lo + hi) / 2.0
            fa = sum(1 for x in sa if x <= mid) / na
            fb = sum(1 for x in sb if x <= mid) / nb
            total += abs(fa - fb) * (hi - lo)
        return total


def _standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / len(a)
    vb = sum((x - mb) ** 2 for x in b) / len(b)
    pooled = math.sqrt((va + vb) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (ma - mb) / pooled


@dataclass(frozen=True)
class StateShiftReport:
    per_feature_smd: Dict[str, float]
    per_feature_wasserstein: Dict[str, float]
    aggregate_state_shift_index: float


def compute_state_shift(
    train_rows: Sequence[Dict[str, object]],
    deploy_rows: Sequence[Dict[str, object]],
    feature_columns: Sequence[str] = tuple(EVICT_VALUE_V1_FEATURE_COLUMNS),
) -> StateShiftReport:
    """Compare the TRAINING state distribution to the DEPLOYMENT state
    distribution (states actually visited during held-out replay), per
    protocol Section 4. Frozen formula: aggregate = mean of per-feature
    Wasserstein-1 distances, each min-max normalized by the TRAINING
    distribution's range for that feature."""
    smd: Dict[str, float] = {}
    wass: Dict[str, float] = {}
    normalized: List[float] = []
    for col in feature_columns:
        a = [float(r[col]) for r in train_rows]
        b = [float(r[col]) for r in deploy_rows]
        smd[col] = _standardized_mean_difference(a, b)
        w = _wasserstein_1(a, b)
        wass[col] = w
        rng = (max(a) - min(a)) if a else 0.0
        normalized.append(w / rng if rng > 0 else 0.0)
    agg = float(sum(normalized) / len(normalized)) if normalized else 0.0
    return StateShiftReport(per_feature_smd=smd, per_feature_wasserstein=wass, aggregate_state_shift_index=agg)


# ---------------------------------------------------------------------
# Trajectory divergence diagnostics
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryDivergenceReport:
    first_divergence_index: Optional[int]
    fraction_decisions_diverged: float
    mean_cache_set_jaccard_similarity: float
    longest_identical_prefix_length: int
    distinct_cache_states_visited_reference: int
    distinct_cache_states_visited_other: int


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_trajectory_divergence(
    requests: Sequence[Request],
    capacity: int,
    reference_model: Optional[EvictValueV1Model],
    other_model: Optional[EvictValueV1Model],
) -> TrajectoryDivergenceReport:
    """Replay the SAME trace under two policies (reference: None=LRU;
    other: a frozen model, or None=LRU) side by side, comparing chosen
    victims and resulting cache sets at every eviction decision."""

    def _run(model: Optional[EvictValueV1Model]):
        order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
        bucket_by_page: Dict[PageId, int] = {}
        conf_by_page: Dict[PageId, float] = {}
        recent_req_hist: Deque[PageId] = collections.deque(maxlen=64)
        recent_hit_hist: Deque[PageId] = collections.deque(maxlen=64)
        victims: List[PageId] = []
        cache_sets: List[frozenset] = []
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
            if len(order) < capacity:
                order[pid] = None
                recent_req_hist.append(pid)
                continue
            candidates = list(order.keys())
            req_bucket = int(req.metadata.get("bucket", 0))
            req_conf = float(req.metadata.get("confidence", 0.5))
            feat_rows = []
            if model is not None:
                for c in candidates:
                    req_rate = (sum(1 for x in recent_req_hist if x == c) / len(recent_req_hist)) if recent_req_hist else 0.0
                    hit_rate = (sum(1 for x in recent_hit_hist if x == c) / len(recent_hit_hist)) if recent_hit_hist else 0.0
                    feat_rows.append(
                        compute_candidate_features_v1(
                            request_bucket=req_bucket, request_confidence=req_conf,
                            candidates=candidates, candidate=c,
                            bucket_by_page=bucket_by_page, confidence_by_page=conf_by_page,
                            recent_request_rate=req_rate, recent_hit_rate=hit_rate,
                        ).as_dict()
                    )
            victim = _choose_state_generation_victim(candidates, model, feat_rows if model is not None else None)
            order.pop(victim)
            order[pid] = None
            recent_req_hist.append(pid)
            victims.append(victim)
            cache_sets.append(frozenset(order.keys()))
        return victims, cache_sets

    ref_victims, ref_sets = _run(reference_model)
    oth_victims, oth_sets = _run(other_model)

    n = min(len(ref_victims), len(oth_victims))
    first_div: Optional[int] = None
    diverged = 0
    jaccards: List[float] = []
    longest_prefix = 0
    prefix_broken = False
    for i in range(n):
        same = ref_victims[i] == oth_victims[i]
        if not same:
            diverged += 1
            if first_div is None:
                first_div = i
        if not prefix_broken:
            if ref_sets[i] == oth_sets[i]:
                longest_prefix += 1
            else:
                prefix_broken = True
        jaccards.append(_jaccard(ref_sets[i], oth_sets[i]))

    return TrajectoryDivergenceReport(
        first_divergence_index=first_div,
        fraction_decisions_diverged=(diverged / n) if n else 0.0,
        mean_cache_set_jaccard_similarity=(sum(jaccards) / len(jaccards)) if jaccards else 1.0,
        longest_identical_prefix_length=longest_prefix,
        distinct_cache_states_visited_reference=len(set(ref_sets)),
        distinct_cache_states_visited_other=len(set(oth_sets)),
    )


class DistributionShiftEvalPolicy(BasePolicy):
    """run_policy()-compatible frozen eviction policy for held-out
    evaluation: argmin predicted L_H over ALL current candidates, one
    batched predict_loss_batch() call per decision (avoids the
    single-sample-predict thread-pool overhead observed when scoring
    candidates one at a time)."""

    name: str = "distribution_shift_eval"

    def __init__(self, model_path: str, history_window: int = 64):
        self.model_path = model_path
        self.history_window = history_window

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._model = EvictValueV1Model.load(self.model_path)
        self._order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._recent_req_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._recent_hit_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self.deployment_state_rows: List[Dict[str, float]] = []

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        if request.metadata.get("bucket") is not None:
            self._bucket_by_page[pid] = int(request.metadata["bucket"])
        if request.metadata.get("confidence") is not None:
            self._confidence_by_page[pid] = max(0.0, min(1.0, float(request.metadata["confidence"])))

        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            self._recent_req_hist.append(pid)
            self._recent_hit_hist.append(pid)
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        self._record_miss(1.0)
        evicted = None
        if self._cache.is_full():
            evicted = self._choose_victim(request)
            self._evict(evicted)
            self._order.pop(evicted, None)

        self._add(pid)
        self._order[pid] = None
        self._recent_req_hist.append(pid)
        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted)

    def _choose_victim(self, request: Request) -> PageId:
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))
        feat_rows = []
        for cand in candidates:
            req_rate = (
                sum(1 for x in self._recent_req_hist if x == cand) / len(self._recent_req_hist)
            ) if self._recent_req_hist else 0.0
            hit_rate = (
                sum(1 for x in self._recent_hit_hist if x == cand) / len(self._recent_hit_hist)
            ) if self._recent_hit_hist else 0.0
            feat_rows.append(
                compute_candidate_features_v1(
                    request_bucket=req_bucket, request_confidence=req_conf,
                    candidates=candidates, candidate=cand,
                    bucket_by_page=self._bucket_by_page, confidence_by_page=self._confidence_by_page,
                    recent_request_rate=req_rate, recent_hit_rate=hit_rate,
                ).as_dict()
            )
        for fr in feat_rows:
            fr_tagged = dict(fr)
            fr_tagged["decision_t"] = int(request.t)
            self.deployment_state_rows.append(fr_tagged)
        preds = self._model.predict_loss_batch(feat_rows)
        best_idx = min(range(len(candidates)), key=lambda i: (preds[i], i))
        return candidates[best_idx]


__all__ = [
    "DistributionShiftConfig",
    "iter_candidate_rows_with_behavior_policy",
    "compute_state_shift",
    "StateShiftReport",
    "compute_trajectory_divergence",
    "TrajectoryDivergenceReport",
    "DistributionShiftEvalPolicy",
]
