"""Direct candidate eviction-value predictor (experimental v1).

Supports two scorer paths:
- artifact-backed scorer (pickle/joblib model artifact), and
- lightweight text-only surrogate scorer for local/debug comparison.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

from lafc.evict_value_features_v1 import compute_candidate_features_v1
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass
class _ScorerChoice:
    scorer: "_EvictValueScorer"
    requested_mode: str
    active_mode: str
    artifact_found: bool


class _EvictValueScorer:
    name: str = "unknown"
    is_surrogate: bool = False

    def predict_loss_one(self, row: Dict[str, float]) -> float:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, object]:
        return {
            "scorer_name": self.name,
            "is_surrogate": self.is_surrogate,
        }


class _ArtifactBackedScorer(_EvictValueScorer):
    name = "artifact_model"
    is_surrogate = False

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)
        self.model = EvictValueV1Model.load(model_path)

    def predict_loss_one(self, row: Dict[str, float]) -> float:
        return self.model.predict_loss_one(row)

    def diagnostics(self) -> Dict[str, object]:
        return {
            **super().diagnostics(),
            "model_path": self.model_path,
            "model_name": self.model.model_name,
        }


class _LinearTextSurrogateScorer(_EvictValueScorer):
    name = "lightweight_text_surrogate"
    is_surrogate = True

    # Deterministic text-only coefficients.
    DEFAULT_INTERCEPT = 0.25
    DEFAULT_WEIGHTS: Dict[str, float] = {
        # Favor evicting candidates with larger expected-reuse distance proxies.
        "candidate_predictor_score": 0.55,
        "candidate_lru_score": 0.15,
        "candidate_age_norm": 0.12,
        # Penalize evicting candidates that predictor/LRU currently favor keeping.
        "candidate_is_predictor_victim": -0.20,
        "candidate_is_lru_victim": -0.08,
        # Contextual signals.
        "request_bucket": -0.05,
        "request_confidence": -0.18,
        "candidate_confidence": -0.12,
        "candidate_recency_rank": 0.08,
        "recent_candidate_request_rate": -0.20,
        "recent_candidate_hit_rate": -0.10,
        # Gap features for relative preference.
        "score_gap_to_predictor_best": 0.08,
        "score_gap_to_lru_victim": 0.05,
    }

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = str(config_path) if config_path else None
        self.intercept = float(self.DEFAULT_INTERCEPT)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if self.config_path:
            self._load_config(self.config_path)

    def _load_config(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "intercept" in payload:
            self.intercept = float(payload["intercept"])
        for key, value in dict(payload.get("weights", {})).items():
            self.weights[str(key)] = float(value)

    def predict_loss_one(self, row: Dict[str, float]) -> float:
        score = self.intercept
        for feature, weight in self.weights.items():
            score += weight * float(row.get(feature, 0.0))
        return float(score)

    def diagnostics(self) -> Dict[str, object]:
        return {
            **super().diagnostics(),
            "config_path": self.config_path,
            "intercept": self.intercept,
            "weights_count": len(self.weights),
            "model_name": "evict_value_v1_text_surrogate_linear",
        }


class EvictValueV1Policy(BasePolicy):
    name: str = "evict_value_v1"

    def __init__(
        self,
        model_path: str = "models/evict_value_v1_hist_gb.pkl",
        history_window: int = 64,
        scorer_mode: str = "auto",
        lightweight_config_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.history_window = history_window
        mode = str(scorer_mode).strip().lower()
        if mode not in {"auto", "artifact", "lightweight"}:
            raise ValueError("scorer_mode must be one of: auto, artifact, lightweight")
        self.scorer_mode = mode
        self.lightweight_config_path = lightweight_config_path

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._recent_req_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        self._recent_hit_hist: Deque[PageId] = collections.deque(maxlen=self.history_window)
        choice = self._choose_scorer()
        self._scorer = choice.scorer
        self._scorer_mode_requested = choice.requested_mode
        self._scorer_mode_active = choice.active_mode
        self._artifact_found = choice.artifact_found
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
            feat = self._build_candidate_features(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=cand,
            )
            pred_losses[cand] = self._scorer.predict_loss_one(feat)

        victim = min(candidates, key=lambda p: (pred_losses[p], candidates.index(p)))
        return victim, {
            "mode": "DIRECT_EVICT_VALUE",
            "predicted_loss": pred_losses[victim],
            "scorer_mode": self._scorer_mode_active,
            "model": self._scorer.diagnostics().get("model_name", self._scorer.name),
            "candidate_count": len(candidates),
        }

    def _choose_scorer(self) -> _ScorerChoice:
        artifact_exists = Path(self.model_path).exists()
        if self.scorer_mode == "artifact":
            if not artifact_exists:
                raise FileNotFoundError(
                    f"evict_value_v1 artifact mode requested but model file not found: {self.model_path}"
                )
            return _ScorerChoice(
                scorer=_ArtifactBackedScorer(self.model_path),
                requested_mode=self.scorer_mode,
                active_mode="artifact",
                artifact_found=True,
            )
        if self.scorer_mode == "lightweight":
            return _ScorerChoice(
                scorer=_LinearTextSurrogateScorer(self.lightweight_config_path),
                requested_mode=self.scorer_mode,
                active_mode="lightweight",
                artifact_found=artifact_exists,
            )

        # auto mode
        if artifact_exists:
            return _ScorerChoice(
                scorer=_ArtifactBackedScorer(self.model_path),
                requested_mode=self.scorer_mode,
                active_mode="artifact",
                artifact_found=True,
            )
        return _ScorerChoice(
            scorer=_LinearTextSurrogateScorer(self.lightweight_config_path),
            requested_mode=self.scorer_mode,
            active_mode="lightweight",
            artifact_found=False,
        )

    def _build_candidate_features(
        self,
        *,
        request_bucket: int,
        request_confidence: float,
        candidates: list[PageId],
        candidate: PageId,
    ) -> Dict[str, float]:
        req_rate = (
            sum(1 for x in self._recent_req_hist if x == candidate) / len(self._recent_req_hist)
        ) if self._recent_req_hist else 0.0
        hit_rate = (
            sum(1 for x in self._recent_hit_hist if x == candidate) / len(self._recent_hit_hist)
        ) if self._recent_hit_hist else 0.0
        return compute_candidate_features_v1(
            request_bucket=request_bucket,
            request_confidence=request_confidence,
            candidates=candidates,
            candidate=candidate,
            bucket_by_page=self._bucket_by_page,
            confidence_by_page=self._confidence_by_page,
            recent_request_rate=req_rate,
            recent_hit_rate=hit_rate,
        ).as_dict()

    def diagnostics_summary(self) -> Dict[str, object]:
        scorer_diag = self._scorer.diagnostics()
        return {
            "model_path": self.model_path,
            "model_name": scorer_diag.get("model_name", "unknown"),
            "evictions": self._evictions,
            "history_window": self.history_window,
            "scorer_mode_requested": self._scorer_mode_requested,
            "scorer_mode": self._scorer_mode_active,
            "artifact_found": self._artifact_found,
            "lightweight_config_path": self.lightweight_config_path,
            "scorer": scorer_diag,
        }
