"""RobustFtP-D (MARKER fallback) experimental baseline.

Reference
---------
Chłędowski, Polak, Szabucki, Żołna.
"Robust Learning-Augmented Caching: An Experimental Study."
ICML 2021, PMLR 139.

Paper mapping (important)
-------------------------
The ICML'21 paper studies **RobustFtP** (Robust Follow-the-Prediction),
which combines:
  (1) a robust classical policy (MARKER in their main reported variants), and
  (2) a consistent policy that blindly follows policy predictions.

The paper also discusses deterministic vs randomized black-box combiners and
reports RobustFtPD / RobustFtPR variants. This implementation is the
**deterministic** variant with **MARKER** fallback, chosen because it is the
clearest robust-switching practical baseline in the paper's experiments.

Repository-interface adaptation
-------------------------------
RobustFtP uses *policy predictions* (predictor suggests cache content / eviction
behavior), not reuse-distance predictions. In this repository, policy predictions
are represented as ``request.metadata['predicted_cache']`` (same interface used
by TRUST&DOUBT).

The predictor-following expert is implemented as:
- if request is a hit, keep cache,
- on miss with full cache, evict any page in ``cache \ predicted_cache``;
  tie-break deterministically by lexicographically smallest page id,
- if ``cache \ predicted_cache`` is empty (imperfect/underspecified advice),
  fall back to lexicographically smallest cached page.

This follows the paper's spirit but is still an interpreted approximation of
"follow policy predictions" because the exact low-level tie rules are not
fully specified in the experimental section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.policies.marker import MarkerPolicy
from lafc.types import CacheEvent, Page, PageId, Request


class FollowPredictedCachePolicy(BasePolicy):
    """Blindly follow policy predictions provided via metadata['predicted_cache']."""

    name: str = "follow_predicted_cache"

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        if self.in_cache(pid):
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        self._record_miss(1.0)
        evicted: Optional[PageId] = None

        if self._cache.is_full():
            predicted_cache = self._predicted_cache(request)
            cache_now = set(self._cache.current_cache())
            candidates = sorted(cache_now - predicted_cache)
            if not candidates:
                candidates = sorted(cache_now)
            evicted = candidates[0]
            self._evict(evicted)

        self._add(pid)
        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted)

    def _predicted_cache(self, request: Request) -> Set[PageId]:
        pc = request.metadata.get("predicted_cache")
        if pc is None:
            raise ValueError(
                "RobustFtP requires metadata['predicted_cache']; "
                "use --derive-predicted-caches or provide predicted_caches in the trace"
            )
        return set(str(x) for x in pc)


@dataclass
class RobustFtPStepLog:
    t: int
    page_id: PageId
    chosen_expert: str
    switched: bool
    robust_misses_before: int
    predictor_misses_before: int
    robust_misses_after: int
    predictor_misses_after: int
    hit: bool
    evicted: Optional[PageId]


class RobustFtPDeterministicMarkerCombiner(BasePolicy):
    """Deterministic robust-switching combiner for RobustFtP with MARKER fallback."""

    name: str = "robust_ftp_d_marker"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)

        self._shadow_robust = MarkerPolicy()
        self._shadow_robust.reset(capacity, pages)

        self._shadow_predictor = FollowPredictedCachePolicy()
        self._shadow_predictor.reset(capacity, pages)

        self._active = "predictor"
        self._step_log: List[RobustFtPStepLog] = []
        self._switch_points: List[int] = []
        self._followed_counts = {"predictor": 0, "robust": 0}

    def _select_expert(self, robust_misses: int, predictor_misses: int) -> str:
        # Deterministic tie-break in favor of predictor-following expert.
        return "predictor" if predictor_misses <= robust_misses else "robust"

    def on_request(self, request: Request) -> CacheEvent:
        robust_before = self._shadow_robust._misses
        predictor_before = self._shadow_predictor._misses

        chosen = self._select_expert(robust_before, predictor_before)
        switched = bool(self._step_log) and chosen != self._active
        if switched:
            self._switch_points.append(request.t)
        self._active = chosen

        if chosen == "predictor":
            chosen_event = self._shadow_predictor.on_request(request)
            self._shadow_robust.on_request(request)
        else:
            chosen_event = self._shadow_robust.on_request(request)
            self._shadow_predictor.on_request(request)

        if chosen_event.hit:
            self._record_hit()
        else:
            self._record_miss(1.0)

        self._followed_counts[chosen] += 1

        self._step_log.append(
            RobustFtPStepLog(
                t=request.t,
                page_id=request.page_id,
                chosen_expert=chosen,
                switched=switched,
                robust_misses_before=robust_before,
                predictor_misses_before=predictor_before,
                robust_misses_after=self._shadow_robust._misses,
                predictor_misses_after=self._shadow_predictor._misses,
                hit=chosen_event.hit,
                evicted=chosen_event.evicted,
            )
        )

        return CacheEvent(
            t=request.t,
            page_id=request.page_id,
            hit=chosen_event.hit,
            cost=0.0 if chosen_event.hit else 1.0,
            evicted=chosen_event.evicted,
            diagnostics={
                "chosen_expert": chosen,
                "switched": switched,
                "robust_misses_before": robust_before,
                "predictor_misses_before": predictor_before,
                "robust_misses_after": self._shadow_robust._misses,
                "predictor_misses_after": self._shadow_predictor._misses,
            },
        )

    def step_log(self) -> List[RobustFtPStepLog]:
        return list(self._step_log)

    def diagnostics_summary(self) -> Dict[str, float]:
        total = max(1, len(self._step_log))
        return {
            "followed_predictor_steps": float(self._followed_counts["predictor"]),
            "followed_robust_steps": float(self._followed_counts["robust"]),
            "switch_count": float(len(self._switch_points)),
            "switch_fraction": float(len(self._switch_points)) / float(total),
            "shadow_predictor_total_misses": float(self._shadow_predictor._misses),
            "shadow_robust_total_misses": float(self._shadow_robust._misses),
        }

    def switch_points(self) -> List[int]:
        return list(self._switch_points)
