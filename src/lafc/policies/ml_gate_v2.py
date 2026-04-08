"""Experimental learned gate v2 with stronger labels/model families."""

from __future__ import annotations

import collections
from typing import Deque, Dict, List, Optional, Tuple

from lafc.learned_gate.features import compute_lru_scores, compute_predictor_scores
from lafc.learned_gate.features_v2 import compute_gate_features_v2
from lafc.learned_gate.model_v2 import LearnedGateV2Model
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class MLGateV2Policy(BasePolicy):
    name: str = "ml_gate_v2"

    def __init__(self, model_path: str = "models/ml_gate_v2_random_forest.pkl", history_window: int = 64) -> None:
        self.model_path = model_path
        self.history_window = history_window

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._model = LearnedGateV2Model.load(self.model_path)

        self._regret_hist: Deque[int] = collections.deque(maxlen=self.history_window)
        self._disagree_hist: Deque[int] = collections.deque(maxlen=self.history_window)
        self._ctx_hist: Deque[int] = collections.deque(maxlen=self.history_window)
        self._ctx_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)

        self._trust = 0
        self._abstain = 0

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

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        p_scores = compute_predictor_scores(candidates, self._bucket_by_page)
        l_scores = compute_lru_scores(candidates)
        pred_victim = max(candidates, key=lambda x: (p_scores[x], -candidates.index(x)))
        lru_victim = max(candidates, key=lambda x: (l_scores[x], -candidates.index(x)))

        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))
        conf_bin = 0 if req_conf <= 0.33 else (1 if req_conf <= 0.66 else 2)
        ctx = (req_bucket, conf_bin)

        feat = compute_gate_features_v2(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            bucket_by_page=self._bucket_by_page,
            confidence_by_page=self._confidence_by_page,
            recent_regret_rate=(sum(self._regret_hist) / len(self._regret_hist)) if self._regret_hist else 0.0,
            recent_disagree_rate=(sum(self._disagree_hist) / len(self._disagree_hist)) if self._disagree_hist else 0.0,
            context_seen_count=self._ctx_counts[ctx],
            recent_context_frequency=(sum(self._ctx_hist) / len(self._ctx_hist)) if self._ctx_hist else 0.0,
        )
        prob = self._model.predict_proba_one(feat)
        trust = prob >= self._model.threshold
        victim = pred_victim if trust else lru_victim

        self._trust += int(trust)
        self._abstain += int(not trust)
        self._disagree_hist.append(int(pred_victim != lru_victim))
        self._regret_hist.append(int((not trust) and (pred_victim != lru_victim)))
        self._ctx_counts[ctx] += 1
        self._ctx_hist.append(1)

        return victim, {
            "mode": "TRUST" if trust else "ABSTAIN",
            "gate_probability": prob,
            "predictor_victim": pred_victim,
            "lru_victim": lru_victim,
            "model": self._model.model_name,
        }

    def diagnostics_summary(self) -> Dict[str, object]:
        total = self._trust + self._abstain
        return {
            "trust_decisions": self._trust,
            "abstain_decisions": self._abstain,
            "trust_coverage": (self._trust / total) if total else 0.0,
            "model_path": self.model_path,
            "model_name": self._model.model_name,
        }
