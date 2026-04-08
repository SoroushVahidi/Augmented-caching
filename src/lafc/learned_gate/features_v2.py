from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from lafc.learned_gate.features import compute_gate_features
from lafc.types import PageId

ML_GATE_V2_FEATURE_COLUMNS: List[str] = [
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
    "context_id_hash",
    "context_seen_count",
    "recent_context_frequency",
]


@dataclass
class OnlineFeatureStats:
    regret_hist: Deque[int]
    disagree_hist: Deque[int]
    context_hist: Deque[int]
    context_counts: Dict[Tuple[int, int], int]


def _confidence_bin(conf: float) -> int:
    if conf <= 0.33:
        return 0
    if conf <= 0.66:
        return 1
    return 2


def compute_gate_features_v2(
    *,
    request_bucket: int,
    request_confidence: float,
    candidates: List[PageId],
    bucket_by_page: Dict[PageId, int],
    confidence_by_page: Dict[PageId, float],
    recent_regret_rate: float,
    recent_disagree_rate: float,
    context_seen_count: int,
    recent_context_frequency: float,
) -> Dict[str, float]:
    base = compute_gate_features(
        request_bucket=request_bucket,
        request_confidence=request_confidence,
        candidates=candidates,
        bucket_by_page=bucket_by_page,
        confidence_by_page=confidence_by_page,
        recent_regret_rate=recent_regret_rate,
        recent_disagree_rate=recent_disagree_rate,
    ).as_dict()
    conf_bin = _confidence_bin(request_confidence)
    ctx_hash = ((int(request_bucket) + 17) * 31 + conf_bin) % 97
    base["context_id_hash"] = float(ctx_hash)
    base["context_seen_count"] = float(context_seen_count)
    base["recent_context_frequency"] = float(recent_context_frequency)
    return base
