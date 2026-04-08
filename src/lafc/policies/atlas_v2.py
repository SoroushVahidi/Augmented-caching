"""ATLAS v2: experimental confidence-aware LA caching with dynamic trust adaptation.

ATLAS v2 extends atlas_v1 by adding:
1) a global online trust multiplier gamma_t in [0,1],
2) a rolling mismatch proxy from delayed online observations,
3) stronger bucket-score normalization for narrow bucket spreads,
4) confidence-sensitive deterministic tie-breaking.

This policy is experimental and does not claim theorem-level guarantees.
"""

from __future__ import annotations

import collections
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class AtlasV2Decision:
    """Per-step eviction diagnostics for atlas_v2."""

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
    gamma_before: float
    mismatch_rate: float
    tie_break_used: bool
    tie_break_mode: Optional[str]


@dataclass
class _PendingPredictorCheck:
    evicted_page: PageId
    evicted_at_t: int
    bucket_hint: int


class AtlasV2Policy(BasePolicy):
    """Experimental confidence-aware policy with dynamic online trust adaptation.

    For cached page i at time t:
        Score_t(i) = lambda_{i,t} * PredScore_t(i)
                     + (1 - lambda_{i,t}) * BaseScore_t(i)

    where
        lambda_{i,t} = gamma_t * confidence_{i,t}            if confidence present
                       gamma_t * default_confidence          otherwise

    with online gamma update:
        E_t = rolling mismatch rate over most recent predictor-dominated
              events resolved online (window size W)
        gamma_{t+1} = clip((1-rho) * gamma_t + rho * (1 - E_t), 0, 1)

    Mismatch proxy (online-safe delayed bookkeeping):
    - Track predictor-dominated eviction decisions.
    - For each such decision, if the evicted page is requested again within
      `mismatch_threshold` requests and its bucket hint was "not soon"
      (`bucket_hint >= soon_bucket_cutoff`), count mismatch=1.
    - Otherwise count mismatch=0 (resolved either by late/no quick reuse, or by
      "soon" hints that are not penalized by this proxy).
    """

    name: str = "atlas_v2"

    def __init__(
        self,
        default_confidence: float = 0.5,
        atlas_window: int = 32,
        atlas_rho: float = 0.3,
        atlas_initial_gamma: float = 0.8,
        atlas_mismatch_threshold: int = 2,
        soon_bucket_cutoff: int = 2,
        tie_epsilon: float = 1e-9,
        tie_confidence_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0, 1]")
        if atlas_window <= 0:
            raise ValueError("atlas_window must be >= 1")
        if not 0.0 <= atlas_rho <= 1.0:
            raise ValueError("atlas_rho must be in [0, 1]")
        if not 0.0 <= atlas_initial_gamma <= 1.0:
            raise ValueError("atlas_initial_gamma must be in [0, 1]")
        if atlas_mismatch_threshold < 1:
            raise ValueError("atlas_mismatch_threshold must be >= 1")

        self.default_confidence = float(default_confidence)
        self.atlas_window = int(atlas_window)
        self.atlas_rho = float(atlas_rho)
        self.atlas_initial_gamma = float(atlas_initial_gamma)
        self.atlas_mismatch_threshold = int(atlas_mismatch_threshold)
        self.soon_bucket_cutoff = int(soon_bucket_cutoff)
        self.tie_epsilon = float(tie_epsilon)
        self.tie_confidence_threshold = float(tie_confidence_threshold)

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}

        self._gamma: float = self.atlas_initial_gamma
        self._gamma_history: List[float] = []
        self._mismatch_rate_history: List[float] = []

        self._lambda_values: List[float] = []
        self._predictor_dominated: int = 0
        self._fallback_dominated: int = 0
        self._tie_break_predictor: int = 0
        self._tie_break_fallback: int = 0

        self._gamma_low_count: int = 0
        self._gamma_medium_count: int = 0
        self._gamma_high_count: int = 0

        self._recent_mismatch_window: Deque[int] = deque(maxlen=self.atlas_window)
        self._recent_mismatch_total: int = 0

        self._pending_predictor_checks: Dict[PageId, _PendingPredictorCheck] = {}
        self._decisions: List[AtlasV2Decision] = []

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        self._resolve_pending_checks_for_time(request.t)
        self._resolve_pending_check_for_page(pid, request.t)
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
        total_decisions = len(self._decisions)
        mismatch_rate = self._current_mismatch_rate()
        return {
            "average_lambda": mean(self._lambda_values) if self._lambda_values else 0.0,
            "gamma_final": self._gamma,
            "rolling_mismatch_rate": mismatch_rate,
            "predictor_dominated_decisions": self._predictor_dominated,
            "fallback_dominated_decisions": self._fallback_dominated,
            "fraction_predictor_dominated": (
                self._predictor_dominated / total_decisions if total_decisions else 0.0
            ),
            "fraction_fallback_dominated": (
                self._fallback_dominated / total_decisions if total_decisions else 0.0
            ),
            "gamma_low_fraction": self._gamma_low_count / total_decisions if total_decisions else 0.0,
            "gamma_medium_fraction": self._gamma_medium_count / total_decisions if total_decisions else 0.0,
            "gamma_high_fraction": self._gamma_high_count / total_decisions if total_decisions else 0.0,
            "tie_break_predictor_count": self._tie_break_predictor,
            "tie_break_fallback_count": self._tie_break_fallback,
            "tie_break_predictor_fraction": self._tie_break_predictor / total_decisions if total_decisions else 0.0,
            "tie_break_fallback_fraction": self._tie_break_fallback / total_decisions if total_decisions else 0.0,
            "decision_count": total_decisions,
            "mismatch_window_size": len(self._recent_mismatch_window),
        }

    def time_series_diagnostics(self) -> Dict[str, object]:
        return {
            "gamma_t": list(self._gamma_history),
            "rolling_mismatch_rate_t": list(self._mismatch_rate_history),
        }

    def decision_log(self) -> List[AtlasV2Decision]:
        return list(self._decisions)

    def _update_prediction_state(self, pid: PageId, request: Request) -> None:
        bucket = request.metadata.get("bucket")
        if bucket is not None:
            self._bucket_by_page[pid] = int(bucket)

        conf = request.metadata.get("confidence")
        if conf is not None:
            self._confidence_by_page[pid] = min(1.0, max(0.0, float(conf)))

    def _resolve_pending_checks_for_time(self, t: int) -> None:
        expired: List[PageId] = []
        for page, check in self._pending_predictor_checks.items():
            if (t - check.evicted_at_t) > self.atlas_mismatch_threshold:
                expired.append(page)

        for page in expired:
            self._register_mismatch_outcome(0)
            self._pending_predictor_checks.pop(page, None)

    def _resolve_pending_check_for_page(self, pid: PageId, t: int) -> None:
        check = self._pending_predictor_checks.get(pid)
        if check is None:
            return

        delta = t - check.evicted_at_t
        predicted_not_soon = check.bucket_hint >= self.soon_bucket_cutoff
        mismatch = int(predicted_not_soon and (delta <= self.atlas_mismatch_threshold))

        self._register_mismatch_outcome(mismatch)
        self._pending_predictor_checks.pop(pid, None)

    def _register_mismatch_outcome(self, mismatch: int) -> None:
        if len(self._recent_mismatch_window) == self._recent_mismatch_window.maxlen:
            self._recent_mismatch_total -= self._recent_mismatch_window[0]
        self._recent_mismatch_window.append(mismatch)
        self._recent_mismatch_total += mismatch

        mismatch_rate = self._current_mismatch_rate()
        target = 1.0 - mismatch_rate
        self._gamma = min(1.0, max(0.0, (1.0 - self.atlas_rho) * self._gamma + self.atlas_rho * target))

    def _current_mismatch_rate(self) -> float:
        if not self._recent_mismatch_window:
            return 0.0
        return self._recent_mismatch_total / len(self._recent_mismatch_window)

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
            page_conf = conf if conf is not None else self.default_confidence
            lam = min(1.0, max(0.0, page_conf * self._gamma))
            lambdas[page] = lam
            self._lambda_values.append(lam)
            scores[page] = lam * pred_scores[page] + (1.0 - lam) * base_scores[page]

        max_score = max(scores[p] for p in candidates)
        tied = [p for p in candidates if (max_score - scores[p]) <= self.tie_epsilon]

        tie_break_used = len(tied) > 1
        tie_break_mode: Optional[str] = None

        if tie_break_used:
            tie_lambda = mean(lambdas[p] for p in tied)
            if tie_lambda >= self.tie_confidence_threshold:
                tie_break_mode = "predictor"
                self._tie_break_predictor += 1
                victim = max(tied, key=lambda q: (pred_scores[q], base_scores[q], q))
            else:
                tie_break_mode = "fallback"
                self._tie_break_fallback += 1
                victim = max(tied, key=lambda q: (base_scores[q], pred_scores[q], q))
        else:
            victim = max(candidates, key=lambda q: (scores[q], base_scores[q], pred_scores[q], q))

        pred_component = lambdas[victim] * pred_scores[victim]
        base_component = (1.0 - lambdas[victim]) * base_scores[victim]
        decision_mode = "predictor_dominated" if pred_component > base_component else "fallback_dominated"

        if decision_mode == "predictor_dominated":
            self._predictor_dominated += 1
            victim_bucket = self._bucket_by_page.get(victim, 0)
            self._pending_predictor_checks[victim] = _PendingPredictorCheck(
                evicted_page=victim,
                evicted_at_t=request.t,
                bucket_hint=victim_bucket,
            )
        else:
            self._fallback_dominated += 1

        if self._gamma < 0.33:
            self._gamma_low_count += 1
        elif self._gamma < 0.66:
            self._gamma_medium_count += 1
        else:
            self._gamma_high_count += 1

        mismatch_rate = self._current_mismatch_rate()
        self._gamma_history.append(self._gamma)
        self._mismatch_rate_history.append(mismatch_rate)

        decision = AtlasV2Decision(
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
            gamma_before=self._gamma,
            mismatch_rate=mismatch_rate,
            tie_break_used=tie_break_used,
            tie_break_mode=tie_break_mode,
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
            "gamma": self._gamma,
            "rolling_mismatch_rate": mismatch_rate,
            "tie_break_used": tie_break_used,
            "tie_break_mode": tie_break_mode,
        }
        return victim, diagnostics

    def _compute_lru_base_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        n = len(candidates)
        if n == 1:
            return {candidates[0]: 1.0}

        out: Dict[PageId, float] = {}
        for idx, page in enumerate(candidates):
            out[page] = (n - 1 - idx) / (n - 1)
        return out

    def _compute_bucket_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        """Compute monotone bucket scores with narrow-spread amplification.

        Let b_i be candidate bucket hint and B=[b_min,b_max]. Then:
            raw_i = (b_i - b_min) / max(1, b_max - b_min)
            rank_i = rank(b_i among unique bucket values) normalized to [0,1]
            spread = (b_max - b_min) / ((b_max - b_min) + 1)
            PredScore_i = spread * raw_i + (1 - spread) * rank_i

        Properties:
        - Monotone in bucket value (larger bucket => larger score).
        - When spread is narrow (e.g. range=1), rank still contributes strongly
          so predictor influence does not vanish.
        """
        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        bucket_map = {page: self._bucket_by_page.get(page, 0) for page in candidates}
        b_values = list(bucket_map.values())
        b_min = min(b_values)
        b_max = max(b_values)
        b_range = b_max - b_min

        unique = sorted(set(b_values))
        if len(unique) == 1:
            rank_map = {unique[0]: 0.5}
        else:
            rank_map = {
                bucket: idx / (len(unique) - 1)
                for idx, bucket in enumerate(unique)
            }

        spread = b_range / (b_range + 1.0)

        out: Dict[PageId, float] = {}
        denom = max(1, b_range)
        for page in candidates:
            b = bucket_map[page]
            raw = (b - b_min) / denom
            rank_score = rank_map[b]
            out[page] = spread * raw + (1.0 - spread) * rank_score
        return out
