"""ATLAS v1: experimental confidence-aware LA caching for unweighted paging.

This is an experimental first-version framework policy. It combines:
1) bucket-derived eviction score,
2) confidence-aware trust weight,
3) LRU fallback score.

No theorem/competitive guarantee is claimed for this implementation.
"""

from __future__ import annotations

import collections
import math
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class AtlasDecision:
    """Per-step eviction diagnostics for atlas_v1."""

    t: int
    request_page: PageId
    chosen_eviction: Optional[PageId]
    candidate_buckets: Dict[PageId, Optional[int]]
    candidate_confidences: Dict[PageId, Optional[float]]
    candidate_lambdas: Dict[PageId, float]
    candidate_base_scores: Dict[PageId, float]
    candidate_pred_scores: Dict[PageId, float]
    candidate_combined_scores: Dict[PageId, float]
    decision_mode: str


class AtlasV1Policy(BasePolicy):
    """Experimental confidence-aware policy with LRU fallback.

    Score formula (for cached page i at time t):
        score_t(i) = lambda_{i,t} * pred_score_t(i)
                     + (1 - lambda_{i,t}) * base_score_t(i)

    with lambda_{i,t} = confidence_{i,t} if available, else default_confidence.
    """

    name: str = "atlas_v1"

    def __init__(
        self,
        default_confidence: float = 0.5,
        low_confidence_threshold: float = 0.3,
        error_window: int = 32,
        soon_bucket_threshold: int = 1,
        soon_reuse_window: int = 2,
    ) -> None:
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0, 1]")
        self.default_confidence = float(default_confidence)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.error_window = int(error_window)
        self.soon_bucket_threshold = int(soon_bucket_threshold)
        self.soon_reuse_window = int(soon_reuse_window)

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}

        self._lambda_values: List[float] = []
        self._fallback_dominated: int = 0
        self._low_confidence_count: int = 0
        self._decisions: List[AtlasDecision] = []

        self._last_prediction_at_request: Dict[PageId, int] = {}
        self._recent_mismatch_window: Deque[int] = deque(maxlen=max(1, self.error_window))
        self._recent_mismatch_total: int = 0

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        self._update_error_proxy(pid, request.t)
        self._update_prediction_state(pid, request)

        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        step_diag: Dict[str, object] = {}

        if self._cache.is_full():
            evicted, step_diag = self._choose_victim(request)
            self._evict(evicted)
            self._order.pop(evicted, None)

        self._add(pid)
        self._order[pid] = None

        return CacheEvent(
            t=request.t,
            page_id=pid,
            hit=False,
            cost=1.0,
            evicted=evicted,
            diagnostics=step_diag,
        )

    def diagnostics_summary(self) -> Dict[str, object]:
        avg_lambda = mean(self._lambda_values) if self._lambda_values else 0.0
        total_decisions = len(self._decisions)
        return {
            "average_lambda": avg_lambda,
            "fraction_low_confidence_decisions": (
                self._low_confidence_count / total_decisions if total_decisions else 0.0
            ),
            "fraction_fallback_dominated_decisions": (
                self._fallback_dominated / total_decisions if total_decisions else 0.0
            ),
            "decision_count": total_decisions,
            "recent_error_proxy_rate": (
                self._recent_mismatch_total / len(self._recent_mismatch_window)
                if self._recent_mismatch_window
                else 0.0
            ),
            "error_proxy_window_size": len(self._recent_mismatch_window),
        }

    def decision_log(self) -> List[AtlasDecision]:
        return list(self._decisions)

    def _update_prediction_state(self, pid: PageId, request: Request) -> None:
        bucket = request.metadata.get("bucket")
        if bucket is not None:
            self._bucket_by_page[pid] = int(bucket)
            self._last_prediction_at_request[pid] = request.t

        conf = request.metadata.get("confidence")
        if conf is not None:
            conf_f = float(conf)
            self._confidence_by_page[pid] = min(1.0, max(0.0, conf_f))

    def _update_error_proxy(self, pid: PageId, t: int) -> None:
        """Track recent mismatch between prior 'soon' hints and realized reuse."""
        prev_t = self._last_prediction_at_request.get(pid)
        if prev_t is None:
            return

        observed_gap = t - prev_t
        bucket = self._bucket_by_page.get(pid)
        if bucket is None:
            return

        predicted_soon = bucket <= self.soon_bucket_threshold
        reused_soon = observed_gap <= self.soon_reuse_window
        mismatch = int(predicted_soon != reused_soon)

        if len(self._recent_mismatch_window) == self._recent_mismatch_window.maxlen:
            self._recent_mismatch_total -= self._recent_mismatch_window[0]
        self._recent_mismatch_window.append(mismatch)
        self._recent_mismatch_total += mismatch

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        if not candidates:
            raise RuntimeError("No candidate available for eviction")

        base_scores = self._compute_lru_base_scores(candidates)
        pred_scores = self._compute_bucket_scores(candidates)

        confidences: Dict[PageId, Optional[float]] = {}
        lambdas: Dict[PageId, float] = {}
        scores: Dict[PageId, float] = {}

        for page in candidates:
            conf = self._confidence_by_page.get(page)
            confidences[page] = conf
            # v1 trust rule: confidence if provided; otherwise global default.
            lam = conf if conf is not None else self.default_confidence
            lam = max(0.0, min(1.0, float(lam)))
            lambdas[page] = lam
            self._lambda_values.append(lam)
            if lam <= self.low_confidence_threshold:
                self._low_confidence_count += 1

            # Combined score: trust-weighted prediction + fallback LRU.
            scores[page] = lam * pred_scores[page] + (1.0 - lam) * base_scores[page]

        victim = max(
            candidates,
            key=lambda q: (scores[q], base_scores[q], pred_scores[q], q),
        )

        pred_component = lambdas[victim] * pred_scores[victim]
        base_component = (1.0 - lambdas[victim]) * base_scores[victim]
        decision_mode = (
            "predictor_dominated" if pred_component > base_component else "fallback_dominated"
        )
        if decision_mode == "fallback_dominated":
            self._fallback_dominated += 1

        decision = AtlasDecision(
            t=request.t,
            request_page=request.page_id,
            chosen_eviction=victim,
            candidate_buckets={p: self._bucket_by_page.get(p) for p in candidates},
            candidate_confidences=confidences,
            candidate_lambdas=lambdas,
            candidate_base_scores=base_scores,
            candidate_pred_scores=pred_scores,
            candidate_combined_scores=scores,
            decision_mode=decision_mode,
        )
        self._decisions.append(decision)

        diagnostics = {
            "chosen_eviction": victim,
            "candidate_buckets": decision.candidate_buckets,
            "candidate_confidences": decision.candidate_confidences,
            "candidate_lambdas": decision.candidate_lambdas,
            "candidate_base_scores": decision.candidate_base_scores,
            "candidate_pred_scores": decision.candidate_pred_scores,
            "candidate_combined_scores": decision.candidate_combined_scores,
            "decision_mode": decision.decision_mode,
        }
        return victim, diagnostics

    def _compute_lru_base_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        n = len(candidates)
        if n == 1:
            return {candidates[0]: 1.0}

        out: Dict[PageId, float] = {}
        for idx, page in enumerate(candidates):
            # Oldest page gets highest fallback eviction score (1.0).
            out[page] = (n - 1 - idx) / (n - 1)
        return out

    def _compute_bucket_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        buckets = [self._bucket_by_page.get(page, 0) for page in candidates]
        max_bucket = max(buckets) if buckets else 0
        if max_bucket <= 0:
            return {page: 0.0 for page in candidates}
        return {
            page: self._bucket_by_page.get(page, 0) / max_bucket
            for page in candidates
        }
