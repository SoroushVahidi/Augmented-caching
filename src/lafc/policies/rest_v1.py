"""ReST v1: Regret-Driven Selective Trust for unweighted paging.

ReST v1 is an experimental *gating* policy (not a score blend): at each eviction,
it either:
- TRUST: follow a predictor-driven eviction rule, or
- ABSTAIN: fall back to LRU.

Context is request-local and interpretable by default:
    ctx_t = (bucket(request_t), confidence_bin(request_t))

Per-context trust state G[ctx] in [0, 1] controls the gate:
    TRUST iff G[ctx] >= trust_threshold; else ABSTAIN.

Delayed online-safe feedback for TRUST decisions uses a fixed horizon H:
- bad outcome: trusted-evicted page returns within H requests
- good outcome: trusted-evicted page does not return within H requests

Update rule on resolution:
- bad:  G[ctx] <- clip01(G[ctx] - eta_neg)
- good: G[ctx] <- clip01(G[ctx] + eta_pos)

No theorem-level guarantees are claimed; this is an experimental architectural pivot
away from calibration-heavy blending.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class RestV1Decision:
    """Per-eviction diagnostics for ReST v1."""

    t: int
    request_page: PageId
    chosen_eviction: Optional[PageId]
    mode: str
    context: str
    trust_score_before: float
    trust_score_after: float
    candidate_buckets: Dict[PageId, int]
    candidate_confidences: Dict[PageId, float]
    predictor_scores: Dict[PageId, float]
    lru_scores: Dict[PageId, float]
    predictor_choice: Optional[PageId]
    lru_choice: Optional[PageId]
    blind_oracle_choice: Optional[PageId]
    predictive_marker_choice: Optional[PageId]


@dataclass
class _PendingTrustCheck:
    page_id: PageId
    evicted_at_t: int
    context: Tuple[str, str]
    horizon: int


class RestV1Policy(BasePolicy):
    """Experimental selective-trust (TRUST vs ABSTAIN-to-LRU) policy."""

    name: str = "rest_v1"

    def __init__(
        self,
        default_confidence: float = 0.5,
        rest_initial_trust: float = 0.5,
        rest_eta_pos: float = 0.05,
        rest_eta_neg: float = 0.10,
        rest_horizon: int = 2,
        rest_confidence_bins: str = "0.33,0.66",
        rest_trust_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0, 1]")
        if not 0.0 <= rest_initial_trust <= 1.0:
            raise ValueError("rest_initial_trust must be in [0, 1]")
        if rest_eta_pos < 0.0 or rest_eta_neg < 0.0:
            raise ValueError("rest_eta_pos/rest_eta_neg must be >= 0")
        if rest_horizon < 1:
            raise ValueError("rest_horizon must be >= 1")
        if not 0.0 <= rest_trust_threshold <= 1.0:
            raise ValueError("rest_trust_threshold must be in [0, 1]")

        self.default_confidence = float(default_confidence)
        self.rest_initial_trust = float(rest_initial_trust)
        self.rest_eta_pos = float(rest_eta_pos)
        self.rest_eta_neg = float(rest_eta_neg)
        self.rest_horizon = int(rest_horizon)
        self.rest_confidence_thresholds = self._parse_confidence_bins(rest_confidence_bins)
        self.rest_trust_threshold = float(rest_trust_threshold)

    @staticmethod
    def _parse_confidence_bins(spec: str) -> List[float]:
        values: List[float] = []
        for chunk in str(spec).split(","):
            text = chunk.strip()
            if not text:
                continue
            val = float(text)
            if not 0.0 < val < 1.0:
                raise ValueError("rest_confidence_bins thresholds must be in (0, 1)")
            values.append(val)
        return sorted(set(values))

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._actual_next_by_page: Dict[PageId, float] = {}

        self._trust_by_ctx: Dict[Tuple[str, str], float] = {}
        self._pending_checks: Dict[PageId, _PendingTrustCheck] = {}

        self._trust_count = 0
        self._abstain_count = 0
        self._ctx_trust_count: Dict[str, int] = collections.defaultdict(int)
        self._ctx_abstain_count: Dict[str, int] = collections.defaultdict(int)
        self._ctx_good: Dict[str, int] = collections.defaultdict(int)
        self._ctx_bad: Dict[str, int] = collections.defaultdict(int)
        self._ctx_history: Dict[str, List[float]] = collections.defaultdict(list)

        self._match_lru_count = 0
        self._match_blind_oracle_count = 0
        self._match_predictive_marker_count = 0
        self._eviction_decision_count = 0

        self._decisions: List[RestV1Decision] = []
        self._modes_t: List[str] = []
        self._contexts_t: List[str] = []
        self._g_t: List[float] = []

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
        total = self._eviction_decision_count
        contexts = sorted(self._trust_by_ctx.keys())
        trust_table = {self._ctx_to_key(c): self._trust_by_ctx[c] for c in contexts}

        per_context: Dict[str, Dict[str, float]] = {}
        for ctx in trust_table:
            t = self._ctx_trust_count.get(ctx, 0)
            a = self._ctx_abstain_count.get(ctx, 0)
            n = t + a
            per_context[ctx] = {
                "trust_count": t,
                "abstain_count": a,
                "trust_rate": (t / n) if n else 0.0,
                "abstain_rate": (a / n) if n else 0.0,
                "good_trust_outcomes": self._ctx_good.get(ctx, 0),
                "bad_trust_outcomes": self._ctx_bad.get(ctx, 0),
            }

        ranked = sorted(
            per_context.items(),
            key=lambda kv: (kv[1]["trust_rate"], kv[1]["trust_count"]),
            reverse=True,
        )

        return {
            "predictor_rule": "atlas_v3_aggressive_bucket_rank_squared",
            "decision_rule": "TRUST if G[ctx] >= threshold else ABSTAIN->LRU",
            "trust_threshold": self.rest_trust_threshold,
            "rest_horizon": self.rest_horizon,
            "trust_decisions": self._trust_count,
            "abstain_decisions": self._abstain_count,
            "trust_coverage": (self._trust_count / total) if total else 0.0,
            "contexts_seen": len(contexts),
            "trust_table": trust_table,
            "per_context": per_context,
            "trusted_contexts_top": [k for k, _ in ranked[:5]],
            "context_good_counts": dict(self._ctx_good),
            "context_bad_counts": dict(self._ctx_bad),
            "match_rate_lru": self._match_lru_count / total if total else 0.0,
            "match_rate_blind_oracle": self._match_blind_oracle_count / total if total else 0.0,
            "match_rate_predictive_marker": self._match_predictive_marker_count / total if total else 0.0,
            "pending_trust_checks": len(self._pending_checks),
        }

    def time_series_diagnostics(self) -> Dict[str, object]:
        return {
            "mode_t": list(self._modes_t),
            "context_t": list(self._contexts_t),
            "g_t": list(self._g_t),
            "context_trust_evolution": {k: list(v) for k, v in self._ctx_history.items()},
            "mean_g": mean(self._g_t) if self._g_t else 0.0,
        }

    def decision_log(self) -> List[RestV1Decision]:
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
        for idx, thr in enumerate(self.rest_confidence_thresholds):
            if confidence <= thr:
                return f"bin_{idx}"
        return f"bin_{len(self.rest_confidence_thresholds)}"

    def _context_for_request(self, request: Request) -> Tuple[str, str]:
        bucket = int(request.metadata.get("bucket", 0))
        conf = float(request.metadata.get("confidence", self.default_confidence))
        conf = min(1.0, max(0.0, conf))
        return (f"bucket={bucket}", self._confidence_bin(conf))

    @staticmethod
    def _ctx_to_key(ctx: Tuple[str, str]) -> str:
        return f"{ctx[0]}|{ctx[1]}"

    def _trust_for_context(self, ctx: Tuple[str, str]) -> float:
        if ctx not in self._trust_by_ctx:
            self._trust_by_ctx[ctx] = self.rest_initial_trust
            self._ctx_history[self._ctx_to_key(ctx)].append(self.rest_initial_trust)
        return self._trust_by_ctx[ctx]

    def _resolve_pending_checks_for_time(self, t: int) -> None:
        expired: List[PageId] = []
        for page, check in self._pending_checks.items():
            if (t - check.evicted_at_t) > check.horizon:
                expired.append(page)

        for page in expired:
            check = self._pending_checks.pop(page)
            self._apply_trust_update(check.context, is_bad=False)

    def _resolve_pending_check_for_page(self, pid: PageId, t: int) -> None:
        check = self._pending_checks.get(pid)
        if check is None:
            return
        delta = t - check.evicted_at_t
        is_bad = delta <= check.horizon
        self._pending_checks.pop(pid, None)
        self._apply_trust_update(check.context, is_bad=is_bad)

    def _apply_trust_update(self, ctx: Tuple[str, str], is_bad: bool) -> None:
        prior = self._trust_for_context(ctx)
        if is_bad:
            updated = max(0.0, min(1.0, prior - self.rest_eta_neg))
            self._ctx_bad[self._ctx_to_key(ctx)] += 1
        else:
            updated = max(0.0, min(1.0, prior + self.rest_eta_pos))
            self._ctx_good[self._ctx_to_key(ctx)] += 1
        self._trust_by_ctx[ctx] = updated
        self._ctx_history[self._ctx_to_key(ctx)].append(updated)

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        if not candidates:
            raise RuntimeError("No candidate available for eviction")

        ctx = self._context_for_request(request)
        ctx_key = self._ctx_to_key(ctx)
        g_before = self._trust_for_context(ctx)

        lru_scores = self._compute_lru_base_scores(candidates)
        predictor_scores = self._compute_aggressive_pred_scores(candidates)

        predictor_choice = self._argmax_tie_lru(predictor_scores, candidates)
        lru_choice = self._argmax_tie_lru(lru_scores, candidates)
        bo_choice = self._blind_oracle_choice(candidates)
        marker_choice = self._argmax_tie_lru({p: float(self._bucket_by_page.get(p, 0)) for p in candidates}, candidates)

        trust_mode = g_before >= self.rest_trust_threshold
        mode = "TRUST" if trust_mode else "ABSTAIN"
        victim = predictor_choice if trust_mode else lru_choice

        self._eviction_decision_count += 1
        if trust_mode:
            self._trust_count += 1
            self._ctx_trust_count[ctx_key] += 1
            self._pending_checks[victim] = _PendingTrustCheck(
                page_id=victim,
                evicted_at_t=request.t,
                context=ctx,
                horizon=self.rest_horizon,
            )
        else:
            self._abstain_count += 1
            self._ctx_abstain_count[ctx_key] += 1

        if victim == lru_choice:
            self._match_lru_count += 1
        if victim == bo_choice:
            self._match_blind_oracle_count += 1
        if victim == marker_choice:
            self._match_predictive_marker_count += 1

        g_after = self._trust_for_context(ctx)
        candidate_buckets = {p: int(self._bucket_by_page.get(p, 0)) for p in candidates}
        candidate_conf = {p: float(self._confidence_by_page.get(p, self.default_confidence)) for p in candidates}

        self._modes_t.append(mode)
        self._contexts_t.append(ctx_key)
        self._g_t.append(g_before)

        self._decisions.append(
            RestV1Decision(
                t=request.t,
                request_page=request.page_id,
                chosen_eviction=victim,
                mode=mode,
                context=ctx_key,
                trust_score_before=g_before,
                trust_score_after=g_after,
                candidate_buckets=candidate_buckets,
                candidate_confidences=candidate_conf,
                predictor_scores=predictor_scores,
                lru_scores=lru_scores,
                predictor_choice=predictor_choice,
                lru_choice=lru_choice,
                blind_oracle_choice=bo_choice,
                predictive_marker_choice=marker_choice,
            )
        )

        diag = {
            "decision_mode": mode,
            "context": ctx_key,
            "g_ctx": g_before,
            "trust_threshold": self.rest_trust_threshold,
            "predictor_choice": predictor_choice,
            "lru_choice": lru_choice,
        }
        return victim, diag

    def _compute_lru_base_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        if len(candidates) == 1:
            return {candidates[0]: 1.0}
        result: Dict[PageId, float] = {}
        denom = len(candidates) - 1
        for idx, p in enumerate(candidates):
            result[p] = 1.0 - (idx / denom)
        return result

    def _compute_aggressive_pred_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
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
        ties = [p for p in candidates if scores[p] == best]
        return min(ties, key=lambda p: candidates.index(p))

    def _blind_oracle_choice(self, candidates: List[PageId]) -> PageId:
        score = {p: float(self._actual_next_by_page.get(p, float("inf"))) for p in candidates}
        return self._argmax_tie_lru(score, candidates)
