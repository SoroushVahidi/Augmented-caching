"""Generic guard-style robust wrapper for learning-augmented caching policies.

This module provides a lightweight, black-box robustification wrapper:
- run a base (possibly learned/prediction-heavy) policy,
- monitor suspicious online behavior,
- temporarily switch to a robust fallback policy,
- return to base mode after a bounded guard interval.

Design note
-----------
This implementation is inspired by recent guard-style robustification ideas in
learning-augmented caching, but is a repository-compatible approximation rather
than a theorem-faithful implementation of any one paper.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.simulator.cache_state import CacheState
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass
class GuardStepRecord:
    t: int
    page_id: PageId
    mode_before: str
    mode_after: str
    hit: bool
    evicted: Optional[PageId]
    base_hit: bool
    fallback_hit: bool
    early_return_detected: bool
    suspicious_count_window: int
    guard_triggered: bool
    trigger_reason: Optional[str]


class GuardWrapperPolicy(BasePolicy):
    """Generic guard wrapper around a base and fallback policy.

    Detection rule
    --------------
    Early-return detector over base-policy evictions:
      - if a page evicted by the base policy is requested again within
        `early_return_window` requests, mark a suspicious event.
      - maintain suspicious events in a sliding request-time window of size
        `trigger_window`.
      - if suspicious count in window >= `trigger_threshold`, trigger guard mode.

    Guard mode
    ----------
    On trigger, switch to fallback policy for `guard_duration` requests.
    Base and fallback shadows are both advanced online so the wrapper can
    return immediately after guard mode ends.
    """

    name: str = "guard_wrapper"

    def __init__(
        self,
        *,
        base_policy: BasePolicy,
        fallback_policy: BasePolicy,
        early_return_window: int = 2,
        trigger_threshold: int = 2,
        trigger_window: int = 16,
        guard_duration: int = 8,
        wrapper_name: str = "guard_wrapper",
    ) -> None:
        if early_return_window < 1:
            raise ValueError("early_return_window must be >= 1")
        if trigger_threshold < 1:
            raise ValueError("trigger_threshold must be >= 1")
        if trigger_window < 1:
            raise ValueError("trigger_window must be >= 1")
        if guard_duration < 1:
            raise ValueError("guard_duration must be >= 1")

        self._base = base_policy
        self._fallback = fallback_policy
        self.early_return_window = int(early_return_window)
        self.trigger_threshold = int(trigger_threshold)
        self.trigger_window = int(trigger_window)
        self.guard_duration = int(guard_duration)
        self.name = wrapper_name

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._base.reset(capacity, pages)
        self._fallback.reset(capacity, pages)

        self._guard_remaining = 0
        self._evicted_at_base: Dict[PageId, int] = {}
        self._suspicious_times: Deque[int] = deque()

        self._guard_triggers = 0
        self._guard_trigger_times: List[int] = []
        self._guard_trigger_reasons: List[str] = []
        self._guard_time_steps = 0
        self._base_steps = 0
        self._fallback_steps = 0
        self._base_evictions = 0
        self._fallback_evictions = 0
        self._early_return_events = 0

        self._step_log: List[GuardStepRecord] = []

    def on_request(self, request: Request) -> CacheEvent:
        mode_before = "fallback" if self._guard_remaining > 0 else "base"

        # Advance both shadows to keep hot-swapping cheap and consistent.
        base_event = self._base.on_request(request)
        fallback_event = self._fallback.on_request(request)

        chosen_event = fallback_event if mode_before == "fallback" else base_event
        if mode_before == "fallback":
            self._fallback_steps += 1
            self._guard_time_steps += 1
            if chosen_event.evicted is not None:
                self._fallback_evictions += 1
            self._guard_remaining = max(0, self._guard_remaining - 1)
        else:
            self._base_steps += 1
            if chosen_event.evicted is not None:
                self._base_evictions += 1

        if chosen_event.hit:
            self._record_hit()
        else:
            self._record_miss(1.0)

        # Keep visible cache aligned with active shadow.
        self._sync_visible_cache(mode_before)

        # Update detector using base-policy behavior only.
        early_return, suspicious_count = self._update_detector(request, base_event, mode_before)

        guard_triggered = False
        reason: Optional[str] = None
        if mode_before == "base" and self._guard_remaining == 0:
            if suspicious_count >= self.trigger_threshold:
                self._guard_remaining = self.guard_duration
                self._guard_triggers += 1
                self._guard_trigger_times.append(request.t)
                reason = "base_early_return_burst"
                self._guard_trigger_reasons.append(reason)
                guard_triggered = True

        mode_after = "fallback" if self._guard_remaining > 0 else "base"

        self._step_log.append(
            GuardStepRecord(
                t=request.t,
                page_id=request.page_id,
                mode_before=mode_before,
                mode_after=mode_after,
                hit=chosen_event.hit,
                evicted=chosen_event.evicted,
                base_hit=base_event.hit,
                fallback_hit=fallback_event.hit,
                early_return_detected=early_return,
                suspicious_count_window=suspicious_count,
                guard_triggered=guard_triggered,
                trigger_reason=reason,
            )
        )

        return CacheEvent(
            t=request.t,
            page_id=request.page_id,
            hit=chosen_event.hit,
            cost=0.0 if chosen_event.hit else 1.0,
            evicted=chosen_event.evicted,
            diagnostics={
                "mode_before": mode_before,
                "mode_after": mode_after,
                "guard_triggered": guard_triggered,
                "trigger_reason": reason,
                "suspicious_count_window": suspicious_count,
                "early_return_detected": early_return,
            },
        )

    def _sync_visible_cache(self, mode: str) -> None:
        target = self._base.current_cache() if mode == "base" else self._fallback.current_cache()
        new_cache = CacheState(self._cache.capacity, self._pages)
        for pid in sorted(target):
            new_cache.add(pid)
        self._cache = new_cache

    def _update_detector(self, request: Request, base_event: CacheEvent, mode_before: str) -> tuple[bool, int]:
        # Track base evictions when base mode is actually active.
        if mode_before == "base" and base_event.evicted is not None:
            self._evicted_at_base[base_event.evicted] = int(request.t)

        early_return = False
        req_pid = request.page_id
        if req_pid in self._evicted_at_base:
            age = int(request.t) - int(self._evicted_at_base[req_pid])
            if age <= self.early_return_window:
                early_return = True
                self._early_return_events += 1
                self._suspicious_times.append(int(request.t))
            self._evicted_at_base.pop(req_pid, None)

        min_t = int(request.t) - self.trigger_window + 1
        while self._suspicious_times and self._suspicious_times[0] < min_t:
            self._suspicious_times.popleft()

        return early_return, len(self._suspicious_times)

    def diagnostics_summary(self) -> Dict[str, Any]:
        return {
            "guard_triggers": self._guard_triggers,
            "guard_trigger_times": list(self._guard_trigger_times),
            "guard_trigger_reasons": list(self._guard_trigger_reasons),
            "guard_time_steps": self._guard_time_steps,
            "base_time_steps": self._base_steps,
            "fallback_time_steps": self._fallback_steps,
            "base_evictions": self._base_evictions,
            "fallback_evictions": self._fallback_evictions,
            "early_return_events": self._early_return_events,
            "early_return_window": self.early_return_window,
            "trigger_threshold": self.trigger_threshold,
            "trigger_window": self.trigger_window,
            "guard_duration": self.guard_duration,
            "base_policy_name": self._base.name,
            "fallback_policy_name": self._fallback.name,
        }

    def step_log(self) -> List[GuardStepRecord]:
        return list(self._step_log)


class EvictValueV1GuardedPolicy(GuardWrapperPolicy):
    """Concrete guarded variant for evict_value_v1 with robust fallback."""

    name: str = "evict_value_v1_guarded"

    def __init__(
        self,
        *,
        model_path: str = "models/evict_value_v1_hist_gb.pkl",
        history_window: int = 64,
        fallback_policy: str = "lru",
        early_return_window: int = 2,
        trigger_threshold: int = 2,
        trigger_window: int = 16,
        guard_duration: int = 8,
    ) -> None:
        base = EvictValueV1Policy(model_path=model_path, history_window=history_window)
        if fallback_policy == "lru":
            fallback = LRUPolicy()
        elif fallback_policy == "marker":
            fallback = MarkerPolicy()
        else:
            raise ValueError(f"Unsupported fallback_policy='{fallback_policy}'")

        super().__init__(
            base_policy=base,
            fallback_policy=fallback,
            early_return_window=early_return_window,
            trigger_threshold=trigger_threshold,
            trigger_window=trigger_window,
            guard_duration=guard_duration,
            wrapper_name="evict_value_v1_guarded",
        )
        self._fallback_choice = fallback_policy

    def diagnostics_summary(self) -> Dict[str, Any]:
        payload = super().diagnostics_summary()
        payload["fallback_choice"] = self._fallback_choice
        return payload
