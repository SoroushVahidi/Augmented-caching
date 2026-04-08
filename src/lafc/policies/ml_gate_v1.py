"""Experimental learned gate v1: choose TRUST(predictor) vs ABSTAIN(LRU)."""

from __future__ import annotations

import collections
from statistics import mean
from typing import Deque, Dict, List, Optional

from lafc.learned_gate.features import compute_gate_features, compute_lru_scores, compute_predictor_scores
from lafc.learned_gate.model import LearnedGateModel
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class MLGateV1Policy(BasePolicy):
    name: str = "ml_gate_v1"

    def __init__(
        self,
        model_path: str = "models/ml_gate_v1.pkl",
        gate_threshold: float = 0.5,
        regret_window: int = 32,
    ) -> None:
        if regret_window < 1:
            raise ValueError("regret_window must be >= 1")
        self.model_path = model_path
        self.gate_threshold = gate_threshold
        self.regret_window = regret_window

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._model = LearnedGateModel.load(self.model_path)
        self._model.threshold = self.gate_threshold

        self._disagree_hist: Deque[int] = collections.deque(maxlen=self.regret_window)
        self._regret_hist: Deque[int] = collections.deque(maxlen=self.regret_window)

        self._trust_count = 0
        self._abstain_count = 0
        self._gate_probs: List[float] = []

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        if request.metadata.get("bucket") is not None:
            self._bucket_by_page[pid] = int(request.metadata["bucket"])
        if request.metadata.get("confidence") is not None:
            self._confidence_by_page[pid] = max(0.0, min(1.0, float(request.metadata["confidence"])))

        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        diag: Dict[str, object] = {}

        if self._cache.is_full():
            evicted, diag = self._choose_victim(request)
            self._evict(evicted)
            self._order.pop(evicted, None)

        self._add(pid)
        self._order[pid] = None

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted, diagnostics=diag)

    def diagnostics_summary(self) -> Dict[str, object]:
        total = self._trust_count + self._abstain_count
        return {
            "trust_decisions": self._trust_count,
            "abstain_decisions": self._abstain_count,
            "trust_coverage": (self._trust_count / total) if total else 0.0,
            "mean_gate_probability": mean(self._gate_probs) if self._gate_probs else 0.0,
            "recent_disagree_rate": (sum(self._disagree_hist) / len(self._disagree_hist)) if self._disagree_hist else 0.0,
            "recent_regret_rate": (sum(self._regret_hist) / len(self._regret_hist)) if self._regret_hist else 0.0,
            "model_path": self.model_path,
        }

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        p_scores = compute_predictor_scores(candidates, self._bucket_by_page)
        l_scores = compute_lru_scores(candidates)

        pred_victim = max(candidates, key=lambda x: (p_scores[x], -candidates.index(x)))
        lru_victim = max(candidates, key=lambda x: (l_scores[x], -candidates.index(x)))

        feat = compute_gate_features(
            request_bucket=int(request.metadata.get("bucket", 0)),
            request_confidence=float(request.metadata.get("confidence", 0.5)),
            candidates=candidates,
            bucket_by_page=self._bucket_by_page,
            confidence_by_page=self._confidence_by_page,
            recent_regret_rate=(sum(self._regret_hist) / len(self._regret_hist)) if self._regret_hist else 0.0,
            recent_disagree_rate=(sum(self._disagree_hist) / len(self._disagree_hist)) if self._disagree_hist else 0.0,
        )
        feat_row = feat.as_dict()
        prob = self._model.predict_proba_one(feat_row)
        trust = prob >= self.gate_threshold
        victim = pred_victim if trust else lru_victim

        self._gate_probs.append(prob)
        self._disagree_hist.append(int(pred_victim != lru_victim))
        self._regret_hist.append(int((not trust) and (pred_victim != lru_victim)))

        if trust:
            self._trust_count += 1
        else:
            self._abstain_count += 1

        diag = {
            "mode": "TRUST" if trust else "ABSTAIN",
            "gate_probability": prob,
            "predictor_victim": pred_victim,
            "lru_victim": lru_victim,
        }
        return victim, diag
