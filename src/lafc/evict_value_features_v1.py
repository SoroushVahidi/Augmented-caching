from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from lafc.learned_gate.features import compute_lru_scores, compute_predictor_scores
from lafc.types import PageId

EVICT_VALUE_V1_FEATURE_COLUMNS: List[str] = [
    "request_bucket",
    "request_confidence",
    "candidate_bucket",
    "candidate_confidence",
    "candidate_recency_rank",
    "candidate_age_norm",
    "candidate_predictor_score",
    "candidate_lru_score",
    "candidate_is_predictor_victim",
    "candidate_is_lru_victim",
    "score_gap_to_predictor_best",
    "score_gap_to_lru_victim",
    "bucket_gap_to_predictor_best",
    "bucket_gap_to_lru_victim",
    "confidence_gap_to_predictor_best",
    "confidence_gap_to_lru_victim",
    "cache_bucket_mean",
    "cache_bucket_std",
    "cache_bucket_min",
    "cache_bucket_max",
    "cache_unique_bucket_count",
    "cache_confidence_mean",
    "cache_confidence_std",
    "predictor_lru_disagree",
    "recent_candidate_request_rate",
    "recent_candidate_hit_rate",
]


def _std(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


@dataclass(frozen=True)
class CandidateFeatureContext:
    values: Dict[str, float]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)


def compute_candidate_features_v1(
    *,
    request_bucket: int,
    request_confidence: float,
    candidates: List[PageId],
    candidate: PageId,
    bucket_by_page: Dict[PageId, int],
    confidence_by_page: Dict[PageId, float],
    recent_request_rate: float,
    recent_hit_rate: float,
) -> CandidateFeatureContext:
    p_scores = compute_predictor_scores(candidates, bucket_by_page)
    l_scores = compute_lru_scores(candidates)
    pred_victim = max(candidates, key=lambda x: (p_scores[x], -candidates.index(x)))
    lru_victim = max(candidates, key=lambda x: (l_scores[x], -candidates.index(x)))

    buckets = [float(bucket_by_page.get(p, 0)) for p in candidates]
    confs = [float(confidence_by_page.get(p, 0.5)) for p in candidates]
    recency_rank = float(candidates.index(candidate))
    denom = max(len(candidates) - 1, 1)

    vals = {
        "request_bucket": float(request_bucket),
        "request_confidence": float(request_confidence),
        "candidate_bucket": float(bucket_by_page.get(candidate, 0)),
        "candidate_confidence": float(confidence_by_page.get(candidate, 0.5)),
        "candidate_recency_rank": recency_rank,
        "candidate_age_norm": recency_rank / float(denom),
        "candidate_predictor_score": float(p_scores[candidate]),
        "candidate_lru_score": float(l_scores[candidate]),
        "candidate_is_predictor_victim": float(candidate == pred_victim),
        "candidate_is_lru_victim": float(candidate == lru_victim),
        "score_gap_to_predictor_best": float(p_scores[candidate] - p_scores[pred_victim]),
        "score_gap_to_lru_victim": float(l_scores[candidate] - l_scores[lru_victim]),
        "bucket_gap_to_predictor_best": float(bucket_by_page.get(candidate, 0) - bucket_by_page.get(pred_victim, 0)),
        "bucket_gap_to_lru_victim": float(bucket_by_page.get(candidate, 0) - bucket_by_page.get(lru_victim, 0)),
        "confidence_gap_to_predictor_best": float(confidence_by_page.get(candidate, 0.5) - confidence_by_page.get(pred_victim, 0.5)),
        "confidence_gap_to_lru_victim": float(confidence_by_page.get(candidate, 0.5) - confidence_by_page.get(lru_victim, 0.5)),
        "cache_bucket_mean": sum(buckets) / len(buckets),
        "cache_bucket_std": _std(buckets),
        "cache_bucket_min": min(buckets),
        "cache_bucket_max": max(buckets),
        "cache_unique_bucket_count": float(len(set(buckets))),
        "cache_confidence_mean": sum(confs) / len(confs),
        "cache_confidence_std": _std(confs),
        "predictor_lru_disagree": float(pred_victim != lru_victim),
        "recent_candidate_request_rate": float(recent_request_rate),
        "recent_candidate_hit_rate": float(recent_hit_rate),
    }
    return CandidateFeatureContext(values=vals)
