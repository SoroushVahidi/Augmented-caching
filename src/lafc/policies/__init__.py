"""Policies sub-package."""

from __future__ import annotations

from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.blind_oracle_randomized_combiner import BlindOracleRandomizedCombiner
from lafc.policies.equitable import EquitablePolicy
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.policies.la_weighted_paging_det_faithful import LAWeightedPagingDeterministicFaithful
from lafc.policies.la_weighted_paging_randomized import LAWeightedPagingRandomized
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
from lafc.policies.atlas_cga_v2 import AtlasCGAV2Policy
from lafc.policies.offline_belady import OfflineBeladyPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.policies.weighted_lru import WeightedLRUPolicy

__all__ = [
    "LRUPolicy",
    "WeightedLRUPolicy",
    "AdviceTrustingPolicy",
    "LAWeightedPagingDeterministic",
    "LAWeightedPagingDeterministicFaithful",
    "LAWeightedPagingRandomized",
    "MarkerPolicy",
    "BlindOraclePolicy",
    "PredictiveMarkerPolicy",
    "AtlasV1Policy",
    "AtlasV2Policy",
    "AtlasV3Policy",
    "AtlasCGAV1Policy",
    "AtlasCGAV2Policy",
    # Baseline 4 (Wei 2020)
    "BlindOracleLRUCombiner",
    "OfflineBeladyPolicy",
    "TrustAndDoubtPolicy",
    # Scaffolds (not yet implemented)
    "EquitablePolicy",
    "BlindOracleRandomizedCombiner",
]
