from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_pairwise_model_v1 import EvictValuePairwiseV1Model
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass
class _ScorerChoice:
    scorer: "_PairwiseScorer"
    requested_mode: str
    active_mode: str
    artifact_found: bool


class _PairwiseScorer:
    name: str = "unknown"

    def predict_a_beats_b(self, a_features: Dict[str, float], b_features: Dict[str, float]) -> float:
        raise NotImplementedError


class _ArtifactPairwiseScorer(_PairwiseScorer):
    name = "artifact_model"

    def __init__(self, model_path: str) -> None:
        self.model = EvictValuePairwiseV1Model.load(model_path)

    def predict_a_beats_b(self, a_features: Dict[str, float], b_features: Dict[str, float]) -> float:
        return self.model.predict_a_beats_b_proba(a_features, b_features)


class _LightweightPairwiseScorer(_PairwiseScorer):
    name = "lightweight_delta_linear"

    DEFAULT_WEIGHTS = {
        "candidate_predictor_score": 0.45,
        "candidate_lru_score": 0.15,
        "candidate_age_norm": 0.20,
        "recent_candidate_request_rate": -0.25,
        "recent_candidate_hit_rate": -0.12,
        "candidate_confidence": -0.08,
    }

    def predict_a_beats_b(self, a_features: Dict[str, float], b_features: Dict[str, float]) -> float:
        score = 0.0
        for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
            delta = float(a_features.get(col, 0.0)) - float(b_features.get(col, 0.0))
            score += self.DEFAULT_WEIGHTS.get(col, 0.0) * delta
        return float(1.0 / (1.0 + pow(2.718281828, -score)))


class EvictValuePairwiseV1Policy(BasePolicy):
    name: str = "evict_value_pairwise_v1"

    def __init__(
        self,
        model_path: str = "models/evict_value_pairwise_v1_best.pkl",
        history_window: int = 64,
        scorer_mode: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.history_window = history_window
        mode = str(scorer_mode).strip().lower()
        if mode not in {"auto", "artifact", "lightweight"}:
            raise ValueError("scorer_mode must be one of: auto, artifact, lightweight")
        self.scorer_mode = mode

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._recent_req_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._recent_hit_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        choice = self._choose_scorer()
        self._scorer = choice.scorer
        self._scorer_mode_active = choice.active_mode

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        if request.metadata.get("bucket") is not None:
            self._bucket_by_page[pid] = int(request.metadata["bucket"])
        if request.metadata.get("confidence") is not None:
            self._confidence_by_page[pid] = max(0.0, min(1.0, float(request.metadata["confidence"])))

        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            self._recent_req_hist.append(pid)
            self._recent_hit_hist.append(pid)
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        if self._cache.is_full():
            evicted = self._choose_victim(request)
            self._evict(evicted)
            self._order.pop(evicted, None)

        self._add(pid)
        self._order[pid] = None
        self._recent_req_hist.append(pid)
        return CacheEvent(
            t=request.t,
            page_id=pid,
            hit=False,
            cost=1.0,
            evicted=evicted,
            diagnostics={"mode": "PAIRWISE_WIN_AGG", "scorer_mode": self._scorer_mode_active},
        )

    def _choose_victim(self, request: Request) -> PageId:
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))

        feat_by_candidate: Dict[PageId, Dict[str, float]] = {}
        for candidate in candidates:
            req_rate = (sum(1 for x in self._recent_req_hist if x == candidate) / len(self._recent_req_hist)) if self._recent_req_hist else 0.0
            hit_rate = (sum(1 for x in self._recent_hit_hist if x == candidate) / len(self._recent_hit_hist)) if self._recent_hit_hist else 0.0
            feat_by_candidate[candidate] = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=candidate,
                bucket_by_page=self._bucket_by_page,
                confidence_by_page=self._confidence_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()

        wins = {c: 0.0 for c in candidates}
        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                p_a = self._scorer.predict_a_beats_b(feat_by_candidate[a], feat_by_candidate[b])
                wins[a] += p_a
                wins[b] += 1.0 - p_a

        return max(candidates, key=lambda c: (wins[c], str(c)))

    def _choose_scorer(self) -> _ScorerChoice:
        artifact_exists = Path(self.model_path).exists()
        if self.scorer_mode == "artifact":
            if not artifact_exists:
                raise FileNotFoundError(f"pairwise artifact mode requested but model file not found: {self.model_path}")
            return _ScorerChoice(_ArtifactPairwiseScorer(self.model_path), self.scorer_mode, "artifact", True)
        if self.scorer_mode == "lightweight":
            return _ScorerChoice(_LightweightPairwiseScorer(), self.scorer_mode, "lightweight", artifact_exists)
        if artifact_exists:
            return _ScorerChoice(_ArtifactPairwiseScorer(self.model_path), self.scorer_mode, "artifact", True)
        return _ScorerChoice(_LightweightPairwiseScorer(), self.scorer_mode, "lightweight", False)
