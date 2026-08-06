"""Feature computation for the LRB (Learning Relaxed Belady) policy.

Reference
---------
Song, Berger, Li, Lloyd. "Learning Relaxed Belady for Content Distribution
Network Caching." NSDI 2020, Section 4.3.1.
Official implementation: https://github.com/sunnyszy/lrb (BSD-2-Clause),
commit ``9e8b4423383c01c4528deb447f152f0437a37c3a`` (pinned 2026-08-06),
``include/webcachesim/caches/lrb.h`` (``MetaExtra``, ``TrainingData::emplace_back``)
and ``src/caches/lrb.cpp`` (``rank()``).

Feature row layout (``n_feature`` slots total, matches the official code's
CSR column indices exactly)::

    [0]                 age = sample_timestamp - meta.past_timestamp
    [1 .. D]            past request deltas, most-recent-first, D <= max_n_past_distances
    [max_n_past_timestamps]                 object size (constant 1.0 -- see
                                             docs/lrb_method_spec.md, "unit-size
                                             specialization")
    [max_n_past_timestamps + 1]              n_within (# of the above deltas whose
                                             running sum stays under memory_window)
    [max_n_past_timestamps + 2 .. + 1 + E]  E exponentially-decayed counters (EDCs)

Unfilled delta slots are left as NaN (LightGBM treats NaN as "missing",
matching the official code's use of a sparse CSR row that simply omits
unset columns -- see paper Section 4.3.2, "GBM ... can handle missing
values efficiently").

This repository's trace format carries no CDN-style categorical "extra
fields" (object type, etc.), so ``n_extra_fields`` is fixed at 0 here,
unlike the official code which supports up to 4. This is a documented
adaptation, not a silent approximation (see docs/lrb_method_spec.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

N_EDC_FEATURE = 10
BASE_EDC_WINDOW = 10
DEFAULT_MAX_N_PAST_TIMESTAMPS = 32


def edc_windows(
    base_edc_window: int = BASE_EDC_WINDOW, n_edc_feature: int = N_EDC_FEATURE
) -> List[int]:
    """EDC window sizes: ``2**(base_edc_window + i)`` for i in [0, n_edc_feature)."""
    return [2 ** (base_edc_window + i) for i in range(n_edc_feature)]


def hash_edc_table(memory_window: int, base_edc_window: int = BASE_EDC_WINDOW) -> List[float]:
    """Precomputed decay table ``hash_edc[i] = 0.5**i``.

    ``max_hash_edc_idx = floor(memory_window / 2**base_edc_window) - 1`` (clamped
    to >= 0), matching ``LRBCache::set_hash_edc`` in the official code exactly.
    """
    max_idx = max(0, (memory_window // (2**base_edc_window)) - 1)
    return [0.5**i for i in range(max_idx + 1)]


def _edc_distance_index(distance: float, window: int, hash_edc: Sequence[float]) -> int:
    idx = int(distance // window)
    return min(idx, len(hash_edc) - 1)


def n_feature_count(max_n_past_timestamps: int, n_extra_fields: int = 0, n_edc_feature: int = N_EDC_FEATURE) -> int:
    """Total feature-row width, matches the official code's ``n_feature`` field."""
    return max_n_past_timestamps + n_extra_fields + 2 + n_edc_feature


@dataclass
class ObjectMeta:
    """Per-object online metadata.

    Mirrors ``Meta``/``MetaExtra`` in the official ``lrb.h``: ``past_distances``
    is kept most-recent-first (a plain capped Python list here, rather than the
    official code's circular-buffer array -- same semantics, simpler data
    structure; a documented adaptation, not a behavior change). ``edc`` is
    ``None`` until the object has been requested a second time, mirroring
    ``MetaExtra* _extra == nullptr`` for "one-hit-wonder" objects.
    """

    key: str
    past_timestamp: int
    size: float = 1.0
    past_distances: List[int] = field(default_factory=list)
    edc: Optional[List[float]] = None
    sample_times: List[int] = field(default_factory=list)

    def record_request(
        self,
        timestamp: int,
        *,
        max_n_past_distances: int,
        windows: Sequence[int],
        hash_edc: Sequence[float],
    ) -> None:
        """Update deltas/EDC/timestamp for a new observed request at ``timestamp``.

        Must be called AFTER any pending ``sample_times`` have been matured
        against the *old* state (mirrors the official code's explicit ordering
        comment in ``LRBCache::lookup``: "make this update after update
        training, otherwise the last timestamp will change").
        """
        distance = timestamp - self.past_timestamp
        if distance <= 0:
            raise ValueError(
                f"Non-increasing request timestamp for page '{self.key}': "
                f"{timestamp} <= {self.past_timestamp}"
            )
        if self.edc is None:
            # First-ever "second visit": mirrors MetaExtra's constructor, which
            # seeds edc[i] = hash_edc[idx] + 1 (i.e. as if the prior EDC value
            # were 1, not 0) -- an exact, faithful detail of the official code.
            self.past_distances = [distance]
            self.edc = [
                hash_edc[_edc_distance_index(distance, windows[i], hash_edc)] + 1.0
                for i in range(len(windows))
            ]
        else:
            self.past_distances.insert(0, distance)
            if len(self.past_distances) > max_n_past_distances:
                self.past_distances.pop()
            self.edc = [
                self.edc[i] * hash_edc[_edc_distance_index(distance, windows[i], hash_edc)] + 1.0
                for i in range(len(windows))
            ]
        self.past_timestamp = timestamp


def compute_n_within(past_distances: Sequence[int], memory_window: int) -> int:
    """Count of leading (most-recent-first) deltas whose running sum stays < memory_window.

    Matches ``LRBCache::rank``/``TrainingData::emplace_back``'s inline loop exactly
    (the official code's ``else break;`` short-circuit is commented out, i.e. the
    loop deliberately keeps iterating past the window boundary to also emit all
    remaining delta features -- only the *count* stops accumulating).
    """
    running = 0
    n_within = 0
    for d in past_distances:
        running += d
        if running < memory_window:
            n_within += 1
    return n_within


def compute_lrb_feature_row(
    meta: ObjectMeta,
    sample_timestamp: int,
    *,
    memory_window: int,
    max_n_past_timestamps: int,
    windows: Sequence[int],
    hash_edc: Sequence[float],
) -> List[float]:
    """Build the dense feature row for ``meta`` as observed at ``sample_timestamp``.

    ``sample_timestamp`` may be in the past relative to ``meta``'s *current*
    ``past_timestamp`` (delayed-label maturation replays the state as of the
    original sampling time) or equal to it (fresh eviction-candidate scoring).
    """
    max_n_past_distances = max_n_past_timestamps - 1
    n_edc = len(windows)
    n = n_feature_count(max_n_past_timestamps, n_extra_fields=0, n_edc_feature=n_edc)
    row: List[float] = [float("nan")] * n

    age = sample_timestamp - meta.past_timestamp
    row[0] = float(age)

    distances = meta.past_distances[:max_n_past_distances]
    for j, d in enumerate(distances):
        row[1 + j] = float(d)

    row[max_n_past_timestamps] = float(meta.size)
    row[max_n_past_timestamps + 1] = float(compute_n_within(distances, memory_window))

    edc_offset = max_n_past_timestamps + 2
    for k in range(n_edc):
        idx = _edc_distance_index(age, windows[k], hash_edc)
        decay = hash_edc[idx]
        if meta.edc is not None:
            row[edc_offset + k] = meta.edc[k] * decay
        else:
            row[edc_offset + k] = decay

    return row


def label_from_future_interval(future_interval: float) -> float:
    """LRB's regression target: ``log1p(time-to-next-request)`` (paper Section 4.3.3)."""
    return math.log1p(future_interval)
