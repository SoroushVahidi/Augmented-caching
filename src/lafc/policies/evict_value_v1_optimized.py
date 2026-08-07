"""Exact-decision-preserving optimized variants of EvictValueV1Policy.

Practical-significance ablation (Reviewer 1, Concern 4;
docs/practical_significance_ablation_protocol.md). Every class in this
module subclasses ``EvictValueV1Policy`` and overrides only
``_choose_victim``: ``reset``/``on_request``/``_choose_scorer``/
``diagnostics_summary`` are inherited unchanged. Nothing in
``lafc.policies.evict_value_v1`` or ``lafc.evict_value_features_v1`` is
modified -- those modules are read-only imports here, deliberately, because
other Reviewer-1 concerns' jobs depend on them while this ablation runs.

Two real, provable redundant-recomputation costs in the canonical
``_choose_victim`` are what these variants remove (see the protocol doc
Section 2 for the full audit):

1. ``compute_predictor_scores``/``compute_lru_scores`` and the cache-level
   aggregate stats (bucket mean/std/min/max, confidence mean/std, unique
   bucket count) do not depend on which candidate is being scored, yet the
   canonical per-candidate loop recomputes them from scratch for every one
   of the ``k`` candidates: O(k^2 log k) instead of O(k log k).
2. ``recent_candidate_request_rate``/``recent_candidate_hit_rate`` are
   computed via an O(history_window) linear scan per candidate instead of a
   single O(history_window) Counter pass shared across all candidates.
3. The model is invoked once per candidate (``predict_loss_one``) instead
   of once per decision (``predict_loss_batch``, which already exists on
   ``EvictValueV1Model`` but is unused by the canonical policy).

``candidates.index(x)`` (used by canonical for tie-breaking) is also O(k)
per call; the hoisted variants replace it with a precomputed ``idx_of``
dict, which returns bit-identical index values (candidates are always a
duplicate-free list, drawn from an ``OrderedDict``'s keys) and therefore
cannot change any decision.
"""

from __future__ import annotations

import collections
from typing import Dict, List

from lafc.evict_value_features_v1 import _std
from lafc.learned_gate.features import compute_lru_scores, compute_predictor_scores
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.types import PageId, Request


def score_candidates_vectorized_cached_exact(
    policy: EvictValueV1Policy, request: Request, candidates: List[PageId]
) -> Dict[PageId, float]:
    """Shared scoring core used by ``EvictValueV1VectorizedCachedExactPolicy``
    and by the selective/top-k variants in ``evict_value_v1_selective.py``
    (both need "score exactly this candidate set the fast way" as a
    building block, not just "score the full resident set").
    """
    req_bucket = int(request.metadata.get("bucket", 0))
    req_conf = float(request.metadata.get("confidence", 0.5))

    p_scores = compute_predictor_scores(candidates, policy._bucket_by_page)
    l_scores = compute_lru_scores(candidates)
    idx_of = {p: i for i, p in enumerate(candidates)}
    pred_victim = max(candidates, key=lambda x: (p_scores[x], -idx_of[x]))
    lru_victim = max(candidates, key=lambda x: (l_scores[x], -idx_of[x]))

    buckets = [float(policy._bucket_by_page.get(p, 0)) for p in candidates]
    confs = [float(policy._confidence_by_page.get(p, 0.5)) for p in candidates]
    cache_bucket_mean = sum(buckets) / len(buckets)
    cache_bucket_std = _std(buckets)
    cache_bucket_min = min(buckets)
    cache_bucket_max = max(buckets)
    cache_unique_bucket_count = float(len(set(buckets)))
    cache_confidence_mean = sum(confs) / len(confs)
    cache_confidence_std = _std(confs)
    predictor_lru_disagree = float(pred_victim != lru_victim)
    denom = max(len(candidates) - 1, 1)

    req_counts = collections.Counter(policy._recent_req_hist)
    hit_counts = collections.Counter(policy._recent_hit_hist)
    n_req_hist = len(policy._recent_req_hist)
    n_hit_hist = len(policy._recent_hit_hist)

    rows: List[Dict[str, float]] = []
    for cand in candidates:
        recency_rank = float(idx_of[cand])
        req_rate = (req_counts[cand] / n_req_hist) if n_req_hist else 0.0
        hit_rate = (hit_counts[cand] / n_hit_hist) if n_hit_hist else 0.0
        rows.append(
            {
                "request_bucket": float(req_bucket),
                "request_confidence": float(req_conf),
                "candidate_bucket": float(policy._bucket_by_page.get(cand, 0)),
                "candidate_confidence": float(policy._confidence_by_page.get(cand, 0.5)),
                "candidate_recency_rank": recency_rank,
                "candidate_age_norm": recency_rank / float(denom),
                "candidate_predictor_score": float(p_scores[cand]),
                "candidate_lru_score": float(l_scores[cand]),
                "candidate_is_predictor_victim": float(cand == pred_victim),
                "candidate_is_lru_victim": float(cand == lru_victim),
                "score_gap_to_predictor_best": float(p_scores[cand] - p_scores[pred_victim]),
                "score_gap_to_lru_victim": float(l_scores[cand] - l_scores[lru_victim]),
                "bucket_gap_to_predictor_best": float(
                    policy._bucket_by_page.get(cand, 0) - policy._bucket_by_page.get(pred_victim, 0)
                ),
                "bucket_gap_to_lru_victim": float(
                    policy._bucket_by_page.get(cand, 0) - policy._bucket_by_page.get(lru_victim, 0)
                ),
                "confidence_gap_to_predictor_best": float(
                    policy._confidence_by_page.get(cand, 0.5) - policy._confidence_by_page.get(pred_victim, 0.5)
                ),
                "confidence_gap_to_lru_victim": float(
                    policy._confidence_by_page.get(cand, 0.5) - policy._confidence_by_page.get(lru_victim, 0.5)
                ),
                "cache_bucket_mean": cache_bucket_mean,
                "cache_bucket_std": cache_bucket_std,
                "cache_bucket_min": cache_bucket_min,
                "cache_bucket_max": cache_bucket_max,
                "cache_unique_bucket_count": cache_unique_bucket_count,
                "cache_confidence_mean": cache_confidence_mean,
                "cache_confidence_std": cache_confidence_std,
                "predictor_lru_disagree": predictor_lru_disagree,
                "recent_candidate_request_rate": float(req_rate),
                "recent_candidate_hit_rate": float(hit_rate),
            }
        )

    losses = _predict_batch(policy._scorer, rows)
    return dict(zip(candidates, losses))


def _predict_batch(scorer: object, rows: List[Dict[str, float]]) -> List[float]:
    """Batched prediction across all candidate rows in one call.

    Falls back to per-row ``predict_loss_one`` for the lightweight text
    surrogate (already O(1) per call; no artifact model to batch against).
    For the artifact-backed scorer, delegates to
    ``EvictValueV1Model.predict_loss_batch``, documented as producing
    identical per-row results to ``predict_loss_one`` -- one
    ``estimator.predict()`` call instead of N.
    """
    model = getattr(scorer, "model", None)
    if model is not None and hasattr(model, "predict_loss_batch"):
        return list(model.predict_loss_batch(rows))
    return [scorer.predict_loss_one(row) for row in rows]  # type: ignore[attr-defined]


class EvictValueV1CachedExactPolicy(EvictValueV1Policy):
    """Hoists per-decision invariants out of the per-candidate loop.

    Still calls ``predict_loss_one`` once per candidate (isolates the
    hoisting effect from the batching effect below).
    """

    name: str = "evict_value_v1_cached_exact"

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))

        p_scores = compute_predictor_scores(candidates, self._bucket_by_page)
        l_scores = compute_lru_scores(candidates)
        idx_of = {p: i for i, p in enumerate(candidates)}
        pred_victim = max(candidates, key=lambda x: (p_scores[x], -idx_of[x]))
        lru_victim = max(candidates, key=lambda x: (l_scores[x], -idx_of[x]))

        buckets = [float(self._bucket_by_page.get(p, 0)) for p in candidates]
        confs = [float(self._confidence_by_page.get(p, 0.5)) for p in candidates]
        cache_bucket_mean = sum(buckets) / len(buckets)
        cache_bucket_std = _std(buckets)
        cache_bucket_min = min(buckets)
        cache_bucket_max = max(buckets)
        cache_unique_bucket_count = float(len(set(buckets)))
        cache_confidence_mean = sum(confs) / len(confs)
        cache_confidence_std = _std(confs)
        predictor_lru_disagree = float(pred_victim != lru_victim)
        denom = max(len(candidates) - 1, 1)

        req_counts = collections.Counter(self._recent_req_hist)
        hit_counts = collections.Counter(self._recent_hit_hist)
        n_req_hist = len(self._recent_req_hist)
        n_hit_hist = len(self._recent_hit_hist)

        pred_losses: Dict[PageId, float] = {}
        for cand in candidates:
            recency_rank = float(idx_of[cand])
            req_rate = (req_counts[cand] / n_req_hist) if n_req_hist else 0.0
            hit_rate = (hit_counts[cand] / n_hit_hist) if n_hit_hist else 0.0
            row = {
                "request_bucket": float(req_bucket),
                "request_confidence": float(req_conf),
                "candidate_bucket": float(self._bucket_by_page.get(cand, 0)),
                "candidate_confidence": float(self._confidence_by_page.get(cand, 0.5)),
                "candidate_recency_rank": recency_rank,
                "candidate_age_norm": recency_rank / float(denom),
                "candidate_predictor_score": float(p_scores[cand]),
                "candidate_lru_score": float(l_scores[cand]),
                "candidate_is_predictor_victim": float(cand == pred_victim),
                "candidate_is_lru_victim": float(cand == lru_victim),
                "score_gap_to_predictor_best": float(p_scores[cand] - p_scores[pred_victim]),
                "score_gap_to_lru_victim": float(l_scores[cand] - l_scores[lru_victim]),
                "bucket_gap_to_predictor_best": float(
                    self._bucket_by_page.get(cand, 0) - self._bucket_by_page.get(pred_victim, 0)
                ),
                "bucket_gap_to_lru_victim": float(
                    self._bucket_by_page.get(cand, 0) - self._bucket_by_page.get(lru_victim, 0)
                ),
                "confidence_gap_to_predictor_best": float(
                    self._confidence_by_page.get(cand, 0.5) - self._confidence_by_page.get(pred_victim, 0.5)
                ),
                "confidence_gap_to_lru_victim": float(
                    self._confidence_by_page.get(cand, 0.5) - self._confidence_by_page.get(lru_victim, 0.5)
                ),
                "cache_bucket_mean": cache_bucket_mean,
                "cache_bucket_std": cache_bucket_std,
                "cache_bucket_min": cache_bucket_min,
                "cache_bucket_max": cache_bucket_max,
                "cache_unique_bucket_count": cache_unique_bucket_count,
                "cache_confidence_mean": cache_confidence_mean,
                "cache_confidence_std": cache_confidence_std,
                "predictor_lru_disagree": predictor_lru_disagree,
                "recent_candidate_request_rate": float(req_rate),
                "recent_candidate_hit_rate": float(hit_rate),
            }
            pred_losses[cand] = self._scorer.predict_loss_one(row)

        victim = min(candidates, key=lambda p: (pred_losses[p], idx_of[p]))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "predicted_loss": pred_losses[victim],
            "scorer_mode": self._scorer_mode_active,
            "model": self._scorer.diagnostics().get("model_name", self._scorer.name),
            "candidate_count": len(candidates),
            "optimization_variant": self.name,
        }


class EvictValueV1VectorizedExactPolicy(EvictValueV1Policy):
    """Canonical (non-hoisted) per-candidate feature construction, but one
    batched model call across all k candidates instead of k single-row
    calls. Isolates the batching effect from the hoisting effect above."""

    name: str = "evict_value_v1_vectorized_exact"

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))

        rows = [
            self._build_candidate_features(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=cand,
            )
            for cand in candidates
        ]
        losses = _predict_batch(self._scorer, rows)
        pred_losses: Dict[PageId, float] = dict(zip(candidates, losses))

        victim = min(candidates, key=lambda p: (pred_losses[p], candidates.index(p)))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "predicted_loss": pred_losses[victim],
            "scorer_mode": self._scorer_mode_active,
            "model": self._scorer.diagnostics().get("model_name", self._scorer.name),
            "candidate_count": len(candidates),
            "optimization_variant": self.name,
        }


class EvictValueV1VectorizedCachedExactPolicy(EvictValueV1Policy):
    """Combines cached_exact's invariant hoisting with vectorized_exact's
    batched model call -- the fastest exact variant."""

    name: str = "evict_value_v1_vectorized_cached_exact"

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        idx_of = {p: i for i, p in enumerate(candidates)}
        pred_losses = score_candidates_vectorized_cached_exact(self, request, candidates)
        victim = min(candidates, key=lambda p: (pred_losses[p], idx_of[p]))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "predicted_loss": pred_losses[victim],
            "scorer_mode": self._scorer_mode_active,
            "model": self._scorer.diagnostics().get("model_name", self._scorer.name),
            "candidate_count": len(candidates),
            "optimization_variant": self.name,
        }


OPTIMIZATION_VARIANT_CLASSES = {
    "canonical": EvictValueV1Policy,
    "cached_exact": EvictValueV1CachedExactPolicy,
    "vectorized_exact": EvictValueV1VectorizedExactPolicy,
    "vectorized_cached_exact": EvictValueV1VectorizedCachedExactPolicy,
}
