"""Shared infrastructure for the reviewer-facing fairness protocol
(docs/reviewer_fairness_protocol.md, configs/reviewer_fairness_protocol.json).

Central idea: `run_policy()` already returns a full, in-order,
one-event-per-request `SimulationResult.events` list (verified directly
against `src/lafc/runner/run_policy.py`: `for req in requests: events.
append(policy.on_request(req))`, strictly 1:1). That means the "Controlled
Test-Window Fairness" metric -- misses over an identical held-out suffix,
after every policy has processed an identical history/warm-up prefix under
its own legitimate online behavior -- can be reconstructed **losslessly
from a single full-stream execution**, by slicing `events` at the score
boundary and recounting. No policy needs a special "windowed" code path,
and no policy needs to be rerun merely to obtain the windowed metric if a
trustworthy full-stream per-request event log is available.

This module provides that slicing primitive plus the common result-row
schema (docs/reviewer_fairness_protocol.md, "Common result schema") shared
by every fairness-runner script, so field names/semantics can't drift
between policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from lafc.types import CacheEvent

# Frozen protocol constants (see docs/reviewer_fairness_protocol.md,
# "Common primary protocol", for the audit justifying this exact split).
# Independently adopted -- before this audit existed -- by three of the
# four new external-baseline runners' own --validation-fraction defaults
# (LRB, 3L-Cache) and HALP's own training_trigger default (10,000 of a
# 50,000-request trace = 20%), which is why it is the frozen choice here
# rather than an arbitrary one.
CANONICAL_TRACE_REQUESTS = 50000
HISTORY_START = 0
HISTORY_END = 10000
SCORE_START = 10000
SCORE_END = 50000
PROTOCOL_VERSION = "reviewer_fairness_v1"


@dataclass
class WindowScore:
    history_requests: int
    scored_requests: int
    score_start: int
    score_end: int
    hits: int
    misses: int
    miss_ratio: float


def score_window(
    events: List[CacheEvent], score_start: int, score_end: Optional[int] = None
) -> WindowScore:
    """Recompute hits/misses over `events[score_start:score_end]` only.

    Does not re-simulate anything: `events` must already reflect a full,
    in-order run over the *entire* request stream (history included), so
    that cache/model state at `score_start` reflects genuinely having
    processed the history under the policy's own legitimate behavior.
    Raises ValueError on an out-of-range window rather than silently
    clamping -- a caller passing a bad boundary should find out
    immediately, not get a quietly-wrong row in the fairness certificate.
    """
    n = len(events)
    if score_end is None:
        score_end = n
    if not (0 <= score_start <= score_end <= n):
        raise ValueError(
            f"score_window: invalid window [{score_start}, {score_end}) "
            f"for {n} events"
        )
    window = events[score_start:score_end]
    misses = sum(1 for e in window if not e.hit)
    hits = len(window) - misses
    scored = len(window)
    return WindowScore(
        history_requests=score_start,
        scored_requests=scored,
        score_start=score_start,
        score_end=score_end,
        hits=hits,
        misses=misses,
        miss_ratio=(misses / scored) if scored else float("nan"),
    )


# Common result-row schema (docs/reviewer_fairness_protocol.md section
# "Common result schema"). Every fairness-runner script writes rows with
# exactly these fields (plus policy-specific extras appended after them);
# a row missing any of these is invalid and must not be compared.
COMMON_SCHEMA_FIELDS = [
    "experiment_protocol_version",
    "policy",
    "policy_variant",
    "implementation_source",
    "implementation_commit",
    "trace",
    "trace_sha256",
    "capacity",
    "capacity_semantics",
    "object_size_semantics",
    "history_start",
    "history_end",
    "score_start",
    "score_end",
    "history_requests",
    "scored_requests",
    "hits",
    "misses",
    "miss_ratio",
    "cache_warmup",
    "model_training_mode",
    "model_training_data",
    "model_frozen_during_test",
    "online_adaptation_during_test",
    "hyperparameter_source",
    "random_seed",
    "future_information",
    "runtime_seconds",
    "status",
    "failure_reason",
]


def validate_common_row(row: Dict[str, object]) -> None:
    """Raise ValueError if `row` is missing any required common-schema
    field. Called before every write so a malformed row never reaches a
    fairness-certificate comparison silently.
    """
    missing = [f for f in COMMON_SCHEMA_FIELDS if f not in row]
    if missing:
        raise ValueError(f"Fairness result row missing required fields: {missing}")
