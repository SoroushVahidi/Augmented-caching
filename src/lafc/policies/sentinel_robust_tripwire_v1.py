"""Sentinel Robust Tripwire v1: robust-first, learned-second prototype.

Design intent (from docs/next_primary_method_ideas.md)
-------------------------------------------------------
- Keep a strong robust line as the default controller.
- Allow predictor-following overrides only when online risk is low.
- Enforce a trust budget and an early-return burst tripwire.
- If tripwire fires, force robust-only mode for a short guard interval.

This is a minimal, interpretable prototype intended for lightweight evaluation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.policies.robust_ftp_marker_combiner import (
    FollowPredictedCachePolicy,
    RobustFtPDeterministicMarkerCombiner,
)
from lafc.simulator.cache_state import CacheState
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass
class SentinelTripwireStep:
    t: int
    page_id: PageId
    chosen_line: str
    forced_robust: bool
    used_predictor_override: bool
    risk_score: float
    suspicious_count_window: int
    budget_before: int
    budget_after: int
    guard_remaining_after: int
    robust_hit: bool
    predictor_hit: bool
    chosen_hit: bool
    chosen_evicted: Optional[PageId]


class SentinelRobustTripwireV1Policy(BasePolicy):
    """Robust-first policy with conservative predictor override permissions."""

    name: str = "sentinel_robust_tripwire_v1"

    def __init__(
        self,
        *,
        warmup_steps: int = 8,
        risk_threshold: float = 0.20,
        budget_init: int = 2,
        budget_max: int = 2,
        budget_recover: int = 1,
        early_return_window: int = 2,
        trigger_window: int = 16,
        guard_trigger_threshold: int = 2,
        guard_duration: int = 8,
        history_window: int = 32,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in [0,1]")
        if budget_init < 0 or budget_max < 0 or budget_recover < 0:
            raise ValueError("budget params must be >= 0")
        if budget_init > budget_max:
            raise ValueError("budget_init must be <= budget_max")
        if early_return_window < 1 or trigger_window < 1 or guard_duration < 1 or history_window < 1:
            raise ValueError("window/duration params must be >= 1")
        if guard_trigger_threshold < 1:
            raise ValueError("guard_trigger_threshold must be >= 1")

        self.warmup_steps = int(warmup_steps)
        self.risk_threshold = float(risk_threshold)
        self.budget_init = int(budget_init)
        self.budget_max = int(budget_max)
        self.budget_recover = int(budget_recover)
        self.early_return_window = int(early_return_window)
        self.trigger_window = int(trigger_window)
        self.guard_trigger_threshold = int(guard_trigger_threshold)
        self.guard_duration = int(guard_duration)
        self.history_window = int(history_window)

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)

        self._shadow_robust = RobustFtPDeterministicMarkerCombiner()
        self._shadow_robust.reset(capacity, pages)

        self._shadow_predictor = FollowPredictedCachePolicy()
        self._shadow_predictor.reset(capacity, pages)

        self._step = 0
        self._budget = self.budget_init
        self._guard_remaining = 0

        self._predictor_evicted_at: Dict[PageId, int] = {}
        self._suspicious_times: Deque[int] = deque()

        self._disagree_hist: Deque[int] = deque(maxlen=self.history_window)
        self._suspicious_hist: Deque[int] = deque(maxlen=self.history_window)

        self._step_log: List[SentinelTripwireStep] = []
        self._line_counts = {"robust": 0, "predictor": 0}
        self._guard_triggers = 0
        self._forced_robust_steps = 0

    def on_request(self, request: Request) -> CacheEvent:
        self._step += 1
        budget_before = self._budget

        early_return, suspicious_count = self._detect_early_return(request.t, request.page_id)
        self._suspicious_hist.append(1 if early_return else 0)

        disagreement_rate = (sum(self._disagree_hist) / len(self._disagree_hist)) if self._disagree_hist else 0.0
        suspicious_rate = (sum(self._suspicious_hist) / len(self._suspicious_hist)) if self._suspicious_hist else 0.0
        budget_pressure = 1.0 - (float(self._budget) / float(max(1, self.budget_max)))
        risk_score = self._risk_score(disagreement_rate, suspicious_rate, budget_pressure)

        robust_event = self._shadow_robust.on_request(request)
        predictor_event = self._shadow_predictor.on_request(request)

        disagree_now = int(robust_event.evicted != predictor_event.evicted)
        self._disagree_hist.append(disagree_now)

        forced_robust = self._guard_remaining > 0
        if forced_robust:
            self._guard_remaining = max(0, self._guard_remaining - 1)
            self._forced_robust_steps += 1

        # Refinement: only spend predictor overrides on disagreement steps where
        # robust and predictor shadows propose different evictions.
        allow_predictor = (
            (not forced_robust)
            and (self._step > self.warmup_steps)
            and (self._budget > 0)
            and (disagree_now == 1)
            and (suspicious_count < self.guard_trigger_threshold)
            and (risk_score <= self.risk_threshold)
        )

        chosen = "predictor" if allow_predictor else "robust"
        chosen_event = predictor_event if allow_predictor else robust_event

        if allow_predictor:
            self._budget = max(0, self._budget - 1)
            if predictor_event.evicted is not None:
                self._predictor_evicted_at[predictor_event.evicted] = int(request.t)
        elif not early_return:
            self._budget = min(self.budget_max, self._budget + self.budget_recover)

        if (not forced_robust) and suspicious_count >= self.guard_trigger_threshold:
            self._guard_remaining = self.guard_duration
            self._guard_triggers += 1

        if chosen_event.hit:
            self._record_hit()
        else:
            self._record_miss(1.0)

        self._line_counts[chosen] += 1
        self._sync_visible_cache(chosen)

        self._step_log.append(
            SentinelTripwireStep(
                t=request.t,
                page_id=request.page_id,
                chosen_line=chosen,
                forced_robust=forced_robust,
                used_predictor_override=allow_predictor,
                risk_score=risk_score,
                suspicious_count_window=suspicious_count,
                budget_before=budget_before,
                budget_after=self._budget,
                guard_remaining_after=self._guard_remaining,
                robust_hit=robust_event.hit,
                predictor_hit=predictor_event.hit,
                chosen_hit=chosen_event.hit,
                chosen_evicted=chosen_event.evicted,
            )
        )

        return CacheEvent(
            t=request.t,
            page_id=request.page_id,
            hit=chosen_event.hit,
            cost=0.0 if chosen_event.hit else 1.0,
            evicted=chosen_event.evicted,
            diagnostics={
                "chosen_line": chosen,
                "risk_score": risk_score,
                "forced_robust": forced_robust,
                "used_predictor_override": allow_predictor,
                "suspicious_count_window": suspicious_count,
                "budget_before": budget_before,
                "budget_after": self._budget,
                "guard_remaining_after": self._guard_remaining,
            },
        )

    @staticmethod
    def _risk_score(disagreement_rate: float, suspicious_rate: float, budget_pressure: float) -> float:
        # Interpretable fixed-weight logistic head (learned-like, conservative by design).
        z = -2.0 + 2.5 * disagreement_rate + 4.0 * suspicious_rate + 1.0 * budget_pressure
        return 1.0 / (1.0 + math.exp(-z))

    def _detect_early_return(self, t: int, req_pid: PageId) -> tuple[bool, int]:
        early_return = False
        evicted_at = self._predictor_evicted_at.get(req_pid)
        if evicted_at is not None:
            if int(t) - int(evicted_at) <= self.early_return_window:
                early_return = True
                self._suspicious_times.append(int(t))
            self._predictor_evicted_at.pop(req_pid, None)

        min_t = int(t) - self.trigger_window + 1
        while self._suspicious_times and self._suspicious_times[0] < min_t:
            self._suspicious_times.popleft()

        return early_return, len(self._suspicious_times)

    def _sync_visible_cache(self, chosen: str) -> None:
        target = self._shadow_predictor.current_cache() if chosen == "predictor" else self._shadow_robust.current_cache()
        new_cache = CacheState(self._cache.capacity, self._pages)
        for pid in sorted(target):
            new_cache.add(pid)
        self._cache = new_cache

    def diagnostics_summary(self) -> Dict[str, float]:
        total = max(1, len(self._step_log))
        return {
            "robust_steps": float(self._line_counts["robust"]),
            "predictor_steps": float(self._line_counts["predictor"]),
            "predictor_coverage": float(self._line_counts["predictor"]) / float(total),
            "guard_triggers": float(self._guard_triggers),
            "forced_robust_steps": float(self._forced_robust_steps),
            "final_budget": float(self._budget),
            "risk_threshold": self.risk_threshold,
            "warmup_steps": float(self.warmup_steps),
            "guard_duration": float(self.guard_duration),
            "guard_trigger_threshold": float(self.guard_trigger_threshold),
        }

    def step_log(self) -> List[SentinelTripwireStep]:
        return list(self._step_log)
