"""Feature computation for the HALP baseline.

We use standard recency, frequency, and inter-arrival deltas, matching
3L-Cache features, to cleanly isolate the architectural/paradigm
difference (pairwise ranking vs pointwise interval-prediction).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from dataclasses import dataclass

PageId = str


@dataclass
class ObjectMeta:
    """Track request history per object."""
    key: PageId
    past_timestamp: int
    freq: int = 1

    def __init__(self, key: PageId, past_timestamp: int):
        self.key = key
        self.past_timestamp = past_timestamp
        self.past_distances: List[int] = []
        self.freq = 1

    def record_request(self, timestamp: int) -> None:
        """Update the timestamps of requests for this object."""
        if timestamp < self.past_timestamp:
            raise ValueError(f"Timestamp decreased from {self.past_timestamp} to {timestamp}")
        distance = timestamp - self.past_timestamp
        self.past_distances.insert(0, distance)
        if len(self.past_distances) > 3:
            self.past_distances.pop()
        self.past_timestamp = timestamp
        self.freq += 1


def compute_halp_feature_row(meta: ObjectMeta, sample_timestamp: int) -> List[float]:
    """Compute the 5-feature vector: Age, Frequency, Delta 1, Delta 2, Delta 3."""
    row = [float("nan")] * 5
    row[0] = float(sample_timestamp - meta.past_timestamp) # Age (recency)
    row[1] = float(meta.freq) # Frequency
    for j, d in enumerate(meta.past_distances[:3]):
        row[2 + j] = float(d) # Delta 1, 2, 3
    return row
