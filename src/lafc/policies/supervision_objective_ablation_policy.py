"""Held-out inference policy wrappers for the supervision-objective
ablation (docs/supervision_objective_ablation_protocol.md).

Both wrappers score every in-cache candidate once per eviction decision
using the SAME feature computation as training
(compute_candidate_features_v1, reused unmodified) and the SAME
architecture as the canonical evict_value_v1 policy
(src/lafc/policies/evict_value_v1.py): frozen, pretrained model, no online
adaptation during test, deterministic tie-breaking by cache order. The
only thing that varies between objectives is which trained model is
loaded and the eviction direction (argmin vs argmax / lowest-reward), not
the candidate-generation or feature machinery.
"""

from __future__ import annotations

import collections
from typing import Deque, Dict, Optional

import numpy as np

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.halp_model import HALPModel
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class ScalarObjectivePolicy(BasePolicy):
    """Eviction policy driven by a frozen scalar regressor trained on one
    of objective_eviction_loss / objective_next_arrival / objective_reuse_distance.

    direction="min": evict the candidate with the LOWEST predicted label
    (objective_eviction_loss: evicting it is predicted to cost least).
    direction="max": evict the candidate with the HIGHEST predicted label
    (objective_next_arrival / objective_reuse_distance: evict what's
    predicted to return latest / be least reused-soon).
    """

    name: str = "supervision_objective_ablation_scalar"

    def __init__(self, model_path: str, direction: str, history_window: int = 64):
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
        self.model_path = model_path
        self.direction = direction
        self.history_window = history_window

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._model = EvictValueV1Model.load(self.model_path)
        self._order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._recent_req_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._recent_hit_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
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

    def _choose_victim(self, request: Request):
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))
        feat_rows = []
        for cand in candidates:
            req_rate = (
                sum(1 for x in self._recent_req_hist if x == cand) / len(self._recent_req_hist)
            ) if self._recent_req_hist else 0.0
            hit_rate = (
                sum(1 for x in self._recent_hit_hist if x == cand) / len(self._recent_hit_hist)
            ) if self._recent_hit_hist else 0.0
            feat_rows.append(
                compute_candidate_features_v1(
                    request_bucket=req_bucket, request_confidence=req_conf,
                    candidates=candidates, candidate=cand,
                    bucket_by_page=self._bucket_by_page, confidence_by_page=self._confidence_by_page,
                    recent_request_rate=req_rate, recent_hit_rate=hit_rate,
                ).as_dict()
            )
        # Batched: one estimator.predict() call for all candidates instead
        # of one call per candidate -- same per-row result, far less
        # per-call overhead (see supervision_objective_ablation_policy
        # module history / EvictValueV1Model.predict_loss_batch).
        batch_preds = self._model.predict_loss_batch(feat_rows)
        preds: Dict[PageId, float] = dict(zip(candidates, batch_preds))

        pick = min if self.direction == "min" else max
        victim = pick(candidates, key=lambda p: (preds[p], candidates.index(p)))
        return victim, {
            "mode": f"SCALAR_{self.direction.upper()}",
            "predicted_value": preds[victim],
            "candidate_count": len(candidates),
        }

    def diagnostics_summary(self) -> Dict[str, object]:
        return {"model_path": self.model_path, "direction": self.direction, "evictions": self._evictions}


class PairwiseObjectivePolicy(BasePolicy):
    """Eviction policy driven by a frozen shared-weight pairwise preference
    model (HALPModel). Scores every in-cache candidate's reward R(x) and
    evicts the LOWEST-reward candidate (same convention as
    src/lafc/policies/halp.py's `_choose_victim`-equivalent block: lower
    reward = less preferred to keep)."""

    name: str = "supervision_objective_ablation_pairwise"

    def __init__(self, model_path: str, history_window: int = 64):
        self.model_path = model_path
        self.history_window = history_window

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._model = HALPModel.load(self.model_path)
        self._order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._recent_req_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._recent_hit_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
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

    def _choose_victim(self, request: Request):
        candidates = list(self._order.keys())
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))
        feats = []
        for cand in candidates:
            req_rate = (
                sum(1 for x in self._recent_req_hist if x == cand) / len(self._recent_req_hist)
            ) if self._recent_req_hist else 0.0
            hit_rate = (
                sum(1 for x in self._recent_hit_hist if x == cand) / len(self._recent_hit_hist)
            ) if self._recent_hit_hist else 0.0
            feat_row = compute_candidate_features_v1(
                request_bucket=req_bucket, request_confidence=req_conf,
                candidates=candidates, candidate=cand,
                bucket_by_page=self._bucket_by_page, confidence_by_page=self._confidence_by_page,
                recent_request_rate=req_rate, recent_hit_rate=hit_rate,
            ).as_dict()
            feats.append([float(feat_row[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS])

        rewards = self._model.predict_rewards(np.asarray(feats, dtype=float))
        best_idx = 0
        lowest = rewards[0]
        for idx in range(1, len(candidates)):
            if rewards[idx] < lowest:
                lowest = rewards[idx]
                best_idx = idx
        victim = candidates[best_idx]
        return victim, {
            "mode": "PAIRWISE_LOWEST_REWARD",
            "predicted_reward": float(lowest),
            "candidate_count": len(candidates),
        }

    def diagnostics_summary(self) -> Dict[str, object]:
        return {"model_path": self.model_path, "evictions": self._evictions}


__all__ = ["ScalarObjectivePolicy", "PairwiseObjectivePolicy"]
