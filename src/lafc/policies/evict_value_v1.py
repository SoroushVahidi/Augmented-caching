"""Direct candidate eviction-value predictor (experimental v1)."""

from __future__ import annotations

import collections
from typing import Deque, Dict, Optional

from lafc.evict_value_features_v1 import compute_candidate_features_v1
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class EvictValueV1Policy(BasePolicy):
    name: str = "evict_value_v1"

    def __init__(self, model_path: str = "models/evict_value_v1_hist_gb.pkl", history_window: int = 64) -> None:
        self.model_path = model_path
        self.history_window = history_window

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._recent_req_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._recent_hit_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._model = EvictValueV1Model.load(self.model_path)
        self._evictions = 0

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
        diag: Dict[str, object] = {}

        if self._cache.is_full():
            evicted, diag = self._choose_victim(request)
            self._evict(evicted)
            self._order.pop(evicted, None)
            self._evictions += 1

        self._add(pid)
        self._order[pid] = None
        self._recent_req_hist.append(pid)
        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted, diagnostics=diag)

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))

        pred_losses: Dict[PageId, float] = {}
        for cand in candidates:
            req_rate = (sum(1 for x in self._recent_req_hist if x == cand) / len(self._recent_req_hist)) if self._recent_req_hist else 0.0
            hit_rate = (sum(1 for x in self._recent_hit_hist if x == cand) / len(self._recent_hit_hist)) if self._recent_hit_hist else 0.0
            feat = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=cand,
                bucket_by_page=self._bucket_by_page,
                confidence_by_page=self._confidence_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()
            pred_losses[cand] = self._model.predict_loss_one(feat)

        victim = min(candidates, key=lambda p: (pred_losses[p], candidates.index(p)))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "predicted_loss": pred_losses[victim],
            "model": self._model.model_name,
            "candidate_count": len(candidates),
        }

    def diagnostics_summary(self) -> Dict[str, object]:
        return {
            "model_path": self.model_path,
            "model_name": self._model.model_name,
            "evictions": self._evictions,
            "history_window": self.history_window,
        }
