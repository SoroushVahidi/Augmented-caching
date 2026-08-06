"""Feature computation for the 3L-Cache policy.

Reference
---------
Zhou, Niu, Xiong, Fang, Wang. "3L-Cache: Low Overhead and Precise
Learning-based Eviction Policy for Caches." FAST 2025, Sections 4.1/4.2.2.
Official implementation: https://github.com/optiq-lab/3L-Cache (GPL-3.0),
commit ``134cd159b635cdab75419a4281bed1a330fef31f`` (pinned 2026-08-06),
``3LCache/TLCache.h`` (``Meta``/``MetaExtra``) and ``3LCache/TLCache.cpp``
(``TrainingData::emplace_back``, ``prediction``).

Feature row layout (6 slots total, matches the official code's CSR column
indices exactly, ``n_feature = max_n_past_timestamps(4) + 2``)::

    [0]        age = sample_timestamp - meta.past_timestamp
    [1 .. 3]   up to 3 most-recent inter-arrival deltas, most-recent-first
    [4]        object size (constant 1.0 under this repository's unit-size
               specialization -- see docs/three_l_cache_method_spec.md)
    [5]        frequency: total observed-request count for this object
               since its metadata was (re)created, capped at 65535

Unlike LRB's EDC counters, 3L-Cache uses a plain running frequency count
(``Meta._freq``) as its popularity feature -- simpler, and reset only when
an object's metadata is destroyed (i.e. it exits the sliding window).
Unfilled delta slots are left as NaN (LightGBM "missing"), matching the
official sparse-CSR row that simply omits unset columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

MAX_N_PAST_TIMESTAMPS = 4
MAX_N_PAST_DISTANCES = 3
N_FEATURE = MAX_N_PAST_TIMESTAMPS + 2  # age+deltas(3), size, frequency = 6
MAX_FREQ = 65535


@dataclass
class ObjectMeta:
    """Per-object online metadata. Mirrors ``Meta``/``MetaExtra`` in the
    official ``TLCache.h``: ``past_distances`` is kept most-recent-first (a
    capped Python list, not the official code's small ring buffer -- same
    semantics, simpler structure, a documented adaptation with no behavior
    change, exactly as established for the LRB port).
    """

    key: str
    past_timestamp: int
    size: float = 1.0
    freq: int = 1
    past_distances: List[int] = field(default_factory=list)
    sample_time: int = 0  # 0 means "no pending unlabeled sample"

    def record_request(self, timestamp: int) -> None:
        distance = timestamp - self.past_timestamp
        if distance <= 0:
            raise ValueError(
                f"Non-increasing request timestamp for page '{self.key}': "
                f"{timestamp} <= {self.past_timestamp}"
            )
        self.past_distances.insert(0, distance)
        if len(self.past_distances) > MAX_N_PAST_DISTANCES:
            self.past_distances.pop()
        self.past_timestamp = timestamp
        if self.freq < MAX_FREQ:
            self.freq += 1


def compute_three_l_cache_feature_row(meta: ObjectMeta, sample_timestamp: int) -> List[float]:
    """Build the dense 6-feature row for ``meta`` as observed at ``sample_timestamp``."""
    row: List[float] = [float("nan")] * N_FEATURE
    row[0] = float(sample_timestamp - meta.past_timestamp)
    for j, d in enumerate(meta.past_distances[:MAX_N_PAST_DISTANCES]):
        row[1 + j] = float(d)
    row[MAX_N_PAST_TIMESTAMPS] = float(meta.size)
    row[MAX_N_PAST_TIMESTAMPS + 1] = float(meta.freq)
    return row


def label_from_future_interval(future_interval: float) -> float:
    """3L-Cache's regression target: ``log1p(time-to-next-request)`` (paper Section 4.2.2)."""
    return math.log1p(future_interval)


def score_to_reuse_time(log_score: float) -> float:
    """Inverse of the training-time ``log1p`` transform used at prediction time.

    The official code inverts with plain ``exp()``, not ``expm1()``
    (``TLCache.cpp:451,461``) -- a real asymmetry (systematically off by a
    constant +1 in linear space relative to a model perfectly calibrated to
    its own ``log1p`` training target). Reproduced exactly, not "corrected",
    per docs/three_l_cache_method_spec.md.
    """
    return math.exp(log_score)
