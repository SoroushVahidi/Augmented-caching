from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from lafc.types import PageId

ML_GATE_FEATURE_COLUMNS: List[str] = [
    "request_bucket",
    "request_confidence",
    "predictor_bucket",
    "predictor_confidence",
    "predictor_score",
    "lru_score",
    "score_gap_pred_minus_lru",
    "predictor_lru_disagree",
    "predictor_recency_rank",
    "lru_recency_rank",
    "bucket_gap_pred_minus_lru",
    "confidence_gap_pred_minus_lru",
    "cache_unique_bucket_count",
    "cache_bucket_mean",
    "cache_bucket_std",
    "recent_regret_rate",
    "recent_disagree_rate",
]


@dataclass(frozen=True)
class GateFeatureContext:
    request_bucket: int
    request_confidence: float
    predictor_bucket: int
    predictor_confidence: float
    predictor_score: float
    lru_score: float
    score_gap_pred_minus_lru: float
    predictor_lru_disagree: int
    predictor_recency_rank: int
    lru_recency_rank: int
    bucket_gap_pred_minus_lru: int
    confidence_gap_pred_minus_lru: float
    cache_unique_bucket_count: int
    cache_bucket_mean: float
    cache_bucket_std: float
    recent_regret_rate: float
    recent_disagree_rate: float

    def as_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in ML_GATE_FEATURE_COLUMNS}


def compute_lru_scores(candidates: List[PageId]) -> Dict[PageId, float]:
    if len(candidates) == 1:
        return {candidates[0]: 1.0}
    denom = len(candidates) - 1
    return {p: 1.0 - (idx / denom) for idx, p in enumerate(candidates)}


def compute_predictor_scores(candidates: List[PageId], bucket_by_page: Dict[PageId, int]) -> Dict[PageId, float]:
    if len(candidates) == 1:
        return {candidates[0]: 1.0}
    bucket_values = {p: int(bucket_by_page.get(p, 0)) for p in candidates}
    uniq = sorted(set(bucket_values.values()))
    if len(uniq) == 1:
        return {p: 0.5 for p in candidates}
    rank = {b: i for i, b in enumerate(uniq)}
    denom = len(uniq) - 1
    return {p: (rank[bucket_values[p]] / denom) ** 2 for p in candidates}


def _argmax_tie_lru(scores: Dict[PageId, float], candidates: List[PageId], tie_eps: float = 1e-9) -> PageId:
    best = max(scores[p] for p in candidates)
    ties = [p for p in candidates if abs(scores[p] - best) <= tie_eps]
    return min(ties, key=lambda p: candidates.index(p))


def predictor_choice(candidates: List[PageId], bucket_by_page: Dict[PageId, int]) -> PageId:
    return _argmax_tie_lru(compute_predictor_scores(candidates, bucket_by_page), candidates)


def lru_choice(candidates: List[PageId]) -> PageId:
    return _argmax_tie_lru(compute_lru_scores(candidates), candidates)


def compute_gate_features(
    *,
    request_bucket: int,
    request_confidence: float,
    candidates: List[PageId],
    bucket_by_page: Dict[PageId, int],
    confidence_by_page: Dict[PageId, float],
    recent_regret_rate: float,
    recent_disagree_rate: float,
) -> GateFeatureContext:
    p_scores = compute_predictor_scores(candidates, bucket_by_page)
    l_scores = compute_lru_scores(candidates)
    p_choice = _argmax_tie_lru(p_scores, candidates)
    l_choice = _argmax_tie_lru(l_scores, candidates)

    recency_rank = {p: idx for idx, p in enumerate(candidates)}
    buckets = [int(bucket_by_page.get(p, 0)) for p in candidates]
    mean_bucket = sum(buckets) / len(buckets)
    var_bucket = sum((b - mean_bucket) ** 2 for b in buckets) / len(buckets)

    return GateFeatureContext(
        request_bucket=int(request_bucket),
        request_confidence=float(request_confidence),
        predictor_bucket=int(bucket_by_page.get(p_choice, 0)),
        predictor_confidence=float(confidence_by_page.get(p_choice, 0.5)),
        predictor_score=float(p_scores[p_choice]),
        lru_score=float(l_scores[l_choice]),
        score_gap_pred_minus_lru=float(p_scores[p_choice] - l_scores[l_choice]),
        predictor_lru_disagree=int(p_choice != l_choice),
        predictor_recency_rank=int(recency_rank[p_choice]),
        lru_recency_rank=int(recency_rank[l_choice]),
        bucket_gap_pred_minus_lru=int(bucket_by_page.get(p_choice, 0) - bucket_by_page.get(l_choice, 0)),
        confidence_gap_pred_minus_lru=float(confidence_by_page.get(p_choice, 0.5) - confidence_by_page.get(l_choice, 0.5)),
        cache_unique_bucket_count=len(set(buckets)),
        cache_bucket_mean=float(mean_bucket),
        cache_bucket_std=float(var_bucket ** 0.5),
        recent_regret_rate=float(recent_regret_rate),
        recent_disagree_rate=float(recent_disagree_rate),
    )
