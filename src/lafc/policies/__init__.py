"""Policies sub-package."""

from __future__ import annotations

from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.follow_the_prediction import FollowThePredictionPolicy
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.policies.la_weighted_paging_randomized import LAWeightedPagingRandomized
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.policies.weighted_lru import WeightedLRUPolicy

__all__ = [
    "LRUPolicy",
    "WeightedLRUPolicy",
    "AdviceTrustingPolicy",
    "BlindOraclePolicy",
    "FollowThePredictionPolicy",
    "LAWeightedPagingDeterministic",
    "LAWeightedPagingRandomized",
    "MarkerPolicy",
    "PredictiveMarkerPolicy",
    "TrustAndDoubtPolicy",
]
