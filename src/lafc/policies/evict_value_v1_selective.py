"""Selective/hybrid invocation and top-k candidate pruning for evict_value_v1.

Practical-significance ablation (Reviewer 1, Concern 4;
docs/practical_significance_ablation_protocol.md Sections 4-5). Unlike
``evict_value_v1_optimized.py``, the variants here are quality/cost
tradeoffs, not decision-preserving: they may pick a different victim than
canonical on some decisions, trading a (measured, reported) miss-ratio
delta for reduced learned-scorer invocation.

All selection rules are simple and predeclared (no thresholds tuned on
held-out/test outcomes), and never use future information.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional

from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.evict_value_v1_optimized import score_candidates_vectorized_cached_exact
from lafc.types import CacheEvent, PageId, Request


class _ShadowSieveMixin:
    """Cheap, approximate 'what would SIEVE currently evict' signal.

    Maintains visited bits + a hand pointer over the REAL resident
    candidate set (the set actually produced by this policy's own
    eviction choices), updated on every request regardless of whether the
    learned scorer is invoked that decision. This is an explicit
    approximation of SIEVE's algorithm, not a faithful independently
    simulated SIEVE cache (which would in general hold different pages
    than this policy's real cache) -- documented in the protocol doc,
    Section 4. It costs O(1) amortized bookkeeping per request plus an
    O(k) scan only at eviction decisions, far cheaper than the learned
    scorer's feature+model cost, so it never smuggles in the expensive
    computation it is meant to gate.
    """

    def _shadow_sieve_reset(self) -> None:
        self._shadow_visited: Dict[PageId, bool] = {}
        self._shadow_hand: Optional[PageId] = None

    def _shadow_sieve_on_request(self, pid: PageId, event: CacheEvent) -> None:
        if event.hit:
            self._shadow_visited[pid] = True
        else:
            self._shadow_visited[pid] = False
        if event.evicted is not None:
            self._shadow_visited.pop(event.evicted, None)

    def _shadow_sieve_victim(self, candidates: List[PageId]) -> PageId:
        n = len(candidates)
        index_of = {p: i for i, p in enumerate(candidates)}
        if self._shadow_hand is not None and self._shadow_hand in index_of:
            idx = index_of[self._shadow_hand]
        else:
            idx = 0
        while self._shadow_visited.get(candidates[idx], False):
            self._shadow_visited[candidates[idx]] = False
            idx += 1
            if idx >= n:
                idx = 0
        victim = candidates[idx]
        prev_idx = idx + 1
        self._shadow_hand = candidates[prev_idx] if prev_idx < n else None
        return victim


class EvictValueV1SelectiveDisagreementPolicy(_ShadowSieveMixin, EvictValueV1Policy):
    """Invoke the learned scorer only when the cheap LRU victim and the
    cheap shadow-SIEVE victim disagree; otherwise evict the LRU victim
    directly. `disagreement_lru_sieve` in the frozen protocol."""

    name: str = "evict_value_v1_selective_disagreement_lru_sieve"

    def reset(self, capacity: int, pages) -> None:
        super().reset(capacity, pages)
        self._shadow_sieve_reset()
        self.n_eviction_decisions: int = 0
        self.n_learned_scorer_invocations: int = 0

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        was_hit = self.in_cache(pid)
        event = super().on_request(request)
        self._shadow_sieve_on_request(pid, event)
        assert was_hit == event.hit
        return event

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        self.n_eviction_decisions += 1
        lru_victim = candidates[0]
        sieve_victim = self._shadow_sieve_victim(candidates)
        if lru_victim == sieve_victim:
            return lru_victim, {
                "mode": "SELECTIVE_FALLBACK_LRU",
                "invoked_learned_scorer": False,
                "candidate_count": len(candidates),
                "rule": "disagreement_lru_sieve",
            }
        self.n_learned_scorer_invocations += 1
        pred_losses = score_candidates_vectorized_cached_exact(self, request, candidates)
        idx_of = {p: i for i, p in enumerate(candidates)}
        victim = min(candidates, key=lambda p: (pred_losses[p], idx_of[p]))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "invoked_learned_scorer": True,
            "predicted_loss": pred_losses[victim],
            "candidate_count": len(candidates),
            "rule": "disagreement_lru_sieve",
        }

    def invocation_rate(self) -> float:
        return (self.n_learned_scorer_invocations / self.n_eviction_decisions) if self.n_eviction_decisions else 0.0


class EvictValueV1SelectivePeriodicPolicy(EvictValueV1Policy):
    """Invoke the learned scorer on 1 of every K evictions (fixed,
    predeclared); otherwise evict the LRU victim. `periodic` in the frozen
    protocol."""

    name: str = "evict_value_v1_selective_periodic"

    def __init__(self, *args, period_k: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.period_k = int(period_k)
        if self.period_k < 1:
            raise ValueError("period_k must be >= 1")

    def reset(self, capacity: int, pages) -> None:
        super().reset(capacity, pages)
        self.n_eviction_decisions: int = 0
        self.n_learned_scorer_invocations: int = 0

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        decision_index = self.n_eviction_decisions
        self.n_eviction_decisions += 1
        invoke = (decision_index % self.period_k) == 0
        if not invoke:
            return candidates[0], {
                "mode": "SELECTIVE_FALLBACK_LRU",
                "invoked_learned_scorer": False,
                "candidate_count": len(candidates),
                "rule": "periodic",
                "period_k": self.period_k,
            }
        self.n_learned_scorer_invocations += 1
        pred_losses = score_candidates_vectorized_cached_exact(self, request, candidates)
        idx_of = {p: i for i, p in enumerate(candidates)}
        victim = min(candidates, key=lambda p: (pred_losses[p], idx_of[p]))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "invoked_learned_scorer": True,
            "predicted_loss": pred_losses[victim],
            "candidate_count": len(candidates),
            "rule": "periodic",
            "period_k": self.period_k,
        }

    def invocation_rate(self) -> float:
        return (self.n_learned_scorer_invocations / self.n_eviction_decisions) if self.n_eviction_decisions else 0.0


class EvictValueV1TopKPolicy(EvictValueV1Policy):
    """Cheap LRU prefilter selects the k oldest resident candidates; the
    learned scorer evaluates only those k. Reports whether the canonical
    (full-resident-set) victim would have been inside the pruned set, as
    the quality/retention signal."""

    name: str = "evict_value_v1_topk"

    def __init__(self, *args, k: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be >= 1")

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        pruned = candidates[: self.k] if self.k < len(candidates) else candidates
        pred_losses = score_candidates_vectorized_cached_exact(self, request, pruned)
        idx_of = {p: i for i, p in enumerate(pruned)}
        victim = min(pruned, key=lambda p: (pred_losses[p], idx_of[p]))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE_TOPK",
            "predicted_loss": pred_losses[victim],
            "candidate_count": len(candidates),
            "pruned_candidate_count": len(pruned),
            "k": self.k,
        }


def canonical_victim_would_be_pruned(canonical_victim: PageId, candidates: List[PageId], k: int) -> bool:
    """True if the canonical (unpruned) victim falls outside the k-oldest
    prefilter -- used by the runner to compute top-k victim-retention rate
    without needing to instantiate a policy."""
    pruned = set(candidates[:k]) if k < len(candidates) else set(candidates)
    return canonical_victim not in pruned
