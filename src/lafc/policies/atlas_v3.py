"""ATLAS v3: confidence-aware local-trust experimental policy.

ATLAS v3 replaces atlas_v2's single global trust multiplier with a context-local
trust table T[(bucket, confidence_bin)] in [0, 1].

For candidate page p at time t:
    lambda_{p,t} = T[ctx(p,t)] * confidence_{p,t}

Eviction score:
    Score_t(p) = lambda_{p,t} * PredScore_t(p) + (1-lambda_{p,t}) * BaseScore_t(p)

where BaseScore is LRU-normalized in [0,1], and PredScore is an aggressive
rank-based bucket score in [0,1] to preserve separation even when bucket values
are numerically close.

This policy is experimental and does not claim theorem-level guarantees.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class AtlasV3Decision:
    """Per-step eviction diagnostics for atlas_v3."""

    t: int
    request_page: PageId
    chosen_eviction: Optional[PageId]
    chosen_lambda: float
    chosen_context: Optional[str]
    candidate_buckets: Dict[PageId, int]
    candidate_confidences: Dict[PageId, float]
    candidate_contexts: Dict[PageId, str]
    candidate_local_trust: Dict[PageId, float]
    candidate_lambdas: Dict[PageId, float]
    candidate_base_scores: Dict[PageId, float]
    candidate_pred_scores: Dict[PageId, float]
    candidate_combined_scores: Dict[PageId, float]
    decision_mode: str


@dataclass
class _PendingLocalCheck:
    page_id: PageId
    evicted_at_t: int
    context: Tuple[str, str]
    confidence: float
    tolerated_horizon: int


class AtlasV3Policy(BasePolicy):
    """Confidence-calibrated local trust (CCLT) policy, first experimental version."""

    name: str = "atlas_v3"

    def __init__(
        self,
        default_confidence: float = 0.5,
        atlas_initial_local_trust: float = 0.7,
        atlas_confidence_bins: str = "0.33,0.66",
        atlas_eta_pos: float = 0.03,
        atlas_eta_neg: float = 0.12,
        atlas_bucket_regret_mode: str = "linear",
        atlas_tie_epsilon: float = 1e-9,
        atlas_adaptive_tie_coef: float = 0.0,
        atlas_context_mode: str = "bucket_confidence",
        atlas_bucket_group_size: int = 2,
        bucket_horizon: int = 2,
    ) -> None:
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0, 1]")
        if not 0.0 <= atlas_initial_local_trust <= 1.0:
            raise ValueError("atlas_initial_local_trust must be in [0, 1]")
        if atlas_eta_pos < 0.0 or atlas_eta_neg < 0.0:
            raise ValueError("atlas_eta_pos and atlas_eta_neg must be >= 0")
        if atlas_eta_neg < atlas_eta_pos:
            raise ValueError("atlas_eta_neg should be >= atlas_eta_pos")
        if atlas_bucket_regret_mode not in {"linear", "exp2", "sqrt"}:
            raise ValueError("atlas_bucket_regret_mode must be one of: linear, exp2, sqrt")
        if atlas_adaptive_tie_coef < 0.0:
            raise ValueError("atlas_adaptive_tie_coef must be >= 0")
        if atlas_context_mode not in {
            "bucket_confidence",
            "bucket_only",
            "confidence_only",
            "bucket_group_confidence",
        }:
            raise ValueError(
                "atlas_context_mode must be one of: bucket_confidence, bucket_only, confidence_only, bucket_group_confidence"
            )
        if atlas_bucket_group_size < 1:
            raise ValueError("atlas_bucket_group_size must be >= 1")
        if bucket_horizon < 1:
            raise ValueError("bucket_horizon must be >= 1")

        self.default_confidence = float(default_confidence)
        self.atlas_initial_local_trust = float(atlas_initial_local_trust)
        self.confidence_thresholds = self._parse_confidence_bins(atlas_confidence_bins)
        self.atlas_eta_pos = float(atlas_eta_pos)
        self.atlas_eta_neg = float(atlas_eta_neg)
        self.atlas_bucket_regret_mode = atlas_bucket_regret_mode
        self.atlas_tie_epsilon = float(atlas_tie_epsilon)
        self.atlas_adaptive_tie_coef = float(atlas_adaptive_tie_coef)
        self.atlas_context_mode = atlas_context_mode
        self.atlas_bucket_group_size = int(atlas_bucket_group_size)
        self.bucket_horizon = int(bucket_horizon)

    @staticmethod
    def _parse_confidence_bins(spec: str) -> List[float]:
        values: List[float] = []
        for chunk in str(spec).split(","):
            text = chunk.strip()
            if not text:
                continue
            val = float(text)
            if not 0.0 < val < 1.0:
                raise ValueError("atlas_confidence_bins thresholds must be in (0, 1)")
            values.append(val)
        return sorted(set(values))

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._actual_next_by_page: Dict[PageId, float] = {}

        self._local_trust: Dict[Tuple[str, str], float] = {}
        self._pending_local_checks: Dict[PageId, _PendingLocalCheck] = {}
        self._ctx_good: Dict[str, int] = collections.defaultdict(int)
        self._ctx_bad: Dict[str, int] = collections.defaultdict(int)
        self._ctx_history: Dict[str, List[float]] = collections.defaultdict(list)

        self._decisions: List[AtlasV3Decision] = []
        self._lambda_values: List[float] = []
        self._predictor_dominated: int = 0
        self._fallback_dominated: int = 0
        self._tie_region: int = 0

        self._match_lru_count: int = 0
        self._match_blind_oracle_count: int = 0
        self._match_predictive_marker_count: int = 0
        self._eviction_decision_count: int = 0

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
        total = len(self._decisions)
        contexts = sorted(self._local_trust.keys())
        trust_table = {self._ctx_to_key(c): self._local_trust[c] for c in contexts}
        return {
            "average_lambda": mean(self._lambda_values) if self._lambda_values else 0.0,
            "predictor_dominated_decisions": self._predictor_dominated,
            "fallback_dominated_decisions": self._fallback_dominated,
            "tie_region_decisions": self._tie_region,
            "fraction_predictor_dominated": self._predictor_dominated / total if total else 0.0,
            "fraction_fallback_dominated": self._fallback_dominated / total if total else 0.0,
            "fraction_tie_region": self._tie_region / total if total else 0.0,
            "contexts_seen": len(contexts),
            "local_trust_table": trust_table,
            "context_good_counts": dict(self._ctx_good),
            "context_bad_counts": dict(self._ctx_bad),
            "match_rate_lru": self._match_lru_count / self._eviction_decision_count if self._eviction_decision_count else 0.0,
            "match_rate_blind_oracle": self._match_blind_oracle_count / self._eviction_decision_count if self._eviction_decision_count else 0.0,
            "match_rate_predictive_marker": self._match_predictive_marker_count / self._eviction_decision_count if self._eviction_decision_count else 0.0,
            "pending_local_checks": len(self._pending_local_checks),
        }

    def time_series_diagnostics(self) -> Dict[str, object]:
        return {
            "context_trust_evolution": {k: list(v) for k, v in self._ctx_history.items()},
            "contexts_seen_t": [d.chosen_context for d in self._decisions],
            "chosen_lambda_t": [d.chosen_lambda for d in self._decisions],
        }

    def decision_log(self) -> List[AtlasV3Decision]:
        return list(self._decisions)

    def _update_prediction_state(self, pid: PageId, request: Request) -> None:
        bucket = request.metadata.get("bucket")
        if bucket is not None:
            self._bucket_by_page[pid] = int(bucket)
        conf = request.metadata.get("confidence")
        if conf is not None:
            self._confidence_by_page[pid] = min(1.0, max(0.0, float(conf)))
        self._actual_next_by_page[pid] = float(request.actual_next)

    def _confidence_bin(self, confidence: float) -> str:
        for idx, thr in enumerate(self.confidence_thresholds):
            if confidence <= thr:
                return f"bin_{idx}"
        return f"bin_{len(self.confidence_thresholds)}"

    def _context_for_page(self, page: PageId) -> Tuple[str, str]:
        bucket = int(self._bucket_by_page.get(page, 0))
        confidence = float(self._confidence_by_page.get(page, self.default_confidence))
        conf_bin = self._confidence_bin(confidence)
        if self.atlas_context_mode == "bucket_only":
            return (f"bucket={bucket}", "conf=all")
        if self.atlas_context_mode == "confidence_only":
            return ("bucket=all", conf_bin)
        if self.atlas_context_mode == "bucket_group_confidence":
            grp = bucket // self.atlas_bucket_group_size
            return (f"bucket_group={grp}", conf_bin)
        return (f"bucket={bucket}", conf_bin)

    @staticmethod
    def _ctx_to_key(ctx: Tuple[str, str]) -> str:
        return f"{ctx[0]}|{ctx[1]}"

    def _local_trust_for_context(self, ctx: Tuple[str, str]) -> float:
        if ctx not in self._local_trust:
            self._local_trust[ctx] = self.atlas_initial_local_trust
            self._ctx_history[self._ctx_to_key(ctx)].append(self.atlas_initial_local_trust)
        return self._local_trust[ctx]

    def _tolerated_return_horizon(self, bucket: int) -> int:
        b = max(0, int(bucket))
        if self.atlas_bucket_regret_mode == "exp2":
            return max(1, self.bucket_horizon * (2**b))
        if self.atlas_bucket_regret_mode == "sqrt":
            return max(1, self.bucket_horizon * int((b + 1) ** 0.5))
        return max(1, self.bucket_horizon * (b + 1))

    def _resolve_pending_checks_for_time(self, t: int) -> None:
        expired: List[PageId] = []
        for page, check in self._pending_local_checks.items():
            if (t - check.evicted_at_t) > check.tolerated_horizon:
                expired.append(page)

        for page in expired:
            check = self._pending_local_checks.pop(page)
            self._apply_local_trust_update(check.context, check.confidence, is_bad=False)

    def _resolve_pending_check_for_page(self, pid: PageId, t: int) -> None:
        check = self._pending_local_checks.get(pid)
        if check is None:
            return

        delta = t - check.evicted_at_t
        is_bad = delta <= check.tolerated_horizon
        self._pending_local_checks.pop(pid, None)
        self._apply_local_trust_update(check.context, check.confidence, is_bad=is_bad)

    def _apply_local_trust_update(self, ctx: Tuple[str, str], confidence: float, is_bad: bool) -> None:
        prior = self._local_trust_for_context(ctx)
        if is_bad:
            updated = max(0.0, min(1.0, prior - self.atlas_eta_neg * confidence))
            self._ctx_bad[self._ctx_to_key(ctx)] += 1
        else:
            updated = max(0.0, min(1.0, prior + self.atlas_eta_pos * confidence))
            self._ctx_good[self._ctx_to_key(ctx)] += 1
        self._local_trust[ctx] = updated
        self._ctx_history[self._ctx_to_key(ctx)].append(updated)

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        if not candidates:
            raise RuntimeError("No candidate available for eviction")

        base_scores = self._compute_lru_base_scores(candidates)
        pred_scores = self._compute_aggressive_pred_scores(candidates)

        candidate_buckets: Dict[PageId, int] = {}
        candidate_conf: Dict[PageId, float] = {}
        candidate_ctxs: Dict[PageId, str] = {}
        candidate_trust: Dict[PageId, float] = {}
        lambdas: Dict[PageId, float] = {}
        scores: Dict[PageId, float] = {}

        for p in candidates:
            confidence = float(self._confidence_by_page.get(p, self.default_confidence))
            ctx = self._context_for_page(p)
            trust = self._local_trust_for_context(ctx)
            lam = max(0.0, min(1.0, trust * confidence))

            candidate_buckets[p] = int(self._bucket_by_page.get(p, 0))
            candidate_conf[p] = confidence
            candidate_ctxs[p] = self._ctx_to_key(ctx)
            candidate_trust[p] = trust
            lambdas[p] = lam
            scores[p] = lam * pred_scores[p] + (1.0 - lam) * base_scores[p]
            self._lambda_values.append(lam)

        top_score = max(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        score_std = (sum((v - avg_score) ** 2 for v in scores.values()) / len(scores)) ** 0.5
        effective_tie_eps = max(self.atlas_tie_epsilon, self.atlas_adaptive_tie_coef * score_std)
        top_candidates = [p for p in candidates if abs(scores[p] - top_score) <= effective_tie_eps]
        victim = min(top_candidates, key=lambda p: list(self._order.keys()).index(p))

        predictor_choice = self._argmax_tie_lru(pred_scores, candidates)
        fallback_choice = self._argmax_tie_lru(base_scores, candidates)
        blind_oracle_choice = self._blind_oracle_choice(candidates)
        marker_choice = self._argmax_tie_lru({p: float(self._bucket_by_page.get(p, 0)) for p in candidates}, candidates)

        self._eviction_decision_count += 1
        if victim == fallback_choice:
            self._match_lru_count += 1
        if victim == blind_oracle_choice:
            self._match_blind_oracle_count += 1
        if victim == marker_choice:
            self._match_predictive_marker_count += 1

        decision_mode = "tie-region"
        if victim == predictor_choice and victim != fallback_choice:
            decision_mode = "predictor-dominated"
            self._predictor_dominated += 1
        elif victim == fallback_choice and victim != predictor_choice:
            decision_mode = "fallback-dominated"
            self._fallback_dominated += 1
        else:
            self._tie_region += 1

        chosen_ctx: Optional[Tuple[str, str]] = None
        chosen_ctx_key = None
        if victim is not None:
            chosen_ctx = self._context_for_page(victim)
            chosen_ctx_key = self._ctx_to_key(chosen_ctx)

        if decision_mode == "predictor-dominated" and victim is not None and chosen_ctx is not None:
            self._pending_local_checks[victim] = _PendingLocalCheck(
                page_id=victim,
                evicted_at_t=request.t,
                context=chosen_ctx,
                confidence=candidate_conf[victim],
                tolerated_horizon=self._tolerated_return_horizon(candidate_buckets[victim]),
            )

        diag = {
            "decision_mode": decision_mode,
            "chosen_lambda": lambdas[victim],
            "chosen_context": chosen_ctx_key,
            "num_contexts_seen": len(self._local_trust),
            "effective_tie_epsilon": effective_tie_eps,
        }

        self._decisions.append(
            AtlasV3Decision(
                t=request.t,
                request_page=request.page_id,
                chosen_eviction=victim,
                chosen_lambda=lambdas[victim],
                chosen_context=chosen_ctx_key,
                candidate_buckets=candidate_buckets,
                candidate_confidences=candidate_conf,
                candidate_contexts=candidate_ctxs,
                candidate_local_trust=candidate_trust,
                candidate_lambdas=lambdas,
                candidate_base_scores=base_scores,
                candidate_pred_scores=pred_scores,
                candidate_combined_scores=scores,
                decision_mode=decision_mode,
            )
        )

        return victim, diag

    def _compute_lru_base_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        # oldest should receive highest eviction score.
        if len(candidates) == 1:
            return {candidates[0]: 1.0}
        result: Dict[PageId, float] = {}
        denom = len(candidates) - 1
        for idx, p in enumerate(candidates):
            result[p] = 1.0 - (idx / denom)
        return result

    def _compute_aggressive_pred_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        """Rank-based predictor score (bucket-larger => evict-larger), squared for separation."""
        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        bucket_values = {p: int(self._bucket_by_page.get(p, 0)) for p in candidates}
        unique = sorted(set(bucket_values.values()))
        if len(unique) == 1:
            return {p: 0.5 for p in candidates}

        rank = {b: i for i, b in enumerate(unique)}
        denom = len(unique) - 1
        return {p: (rank[bucket_values[p]] / denom) ** 2 for p in candidates}

    def _argmax_tie_lru(self, scores: Dict[PageId, float], candidates: List[PageId]) -> PageId:
        best = max(scores[p] for p in candidates)
        ties = [p for p in candidates if scores[p] == best or abs(scores[p] - best) <= self.atlas_tie_epsilon]
        return min(ties, key=lambda p: candidates.index(p))

    def _blind_oracle_choice(self, candidates: List[PageId]) -> PageId:
        # Farthest true next-use gets evicted; tie-broken by LRU order.
        score = {p: float(self._actual_next_by_page.get(p, float("inf"))) for p in candidates}
        return self._argmax_tie_lru(score, candidates)
