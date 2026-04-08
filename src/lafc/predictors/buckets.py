"""Bucket-based prediction helpers for atlas_v1 experiments."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from lafc.types import Request


MAX_BUCKET = 3


def distance_to_bucket(distance: float, bucket_horizon: int = 2) -> int:
    """Map a next-arrival distance into one of 4 eviction buckets.

    Bucket semantics (larger => more evictable):
    - 0: very soon (distance <= 1)
    - 1: somewhat soon (2 .. bucket_horizon)
    - 2: medium (bucket_horizon+1 .. 2*bucket_horizon)
    - 3: far or unknown (distance > 2*bucket_horizon or inf)
    """
    if math.isinf(distance):
        return MAX_BUCKET
    if distance <= 1:
        return 0
    if distance <= bucket_horizon:
        return 1
    if distance <= 2 * bucket_horizon:
        return 2
    return MAX_BUCKET


def attach_perfect_buckets(requests: List[Request], bucket_horizon: int = 2) -> List[Request]:
    """Return copied requests with perfect bucket metadata from actual_next."""
    out: List[Request] = []
    for req in requests:
        dist = req.actual_next - req.t if not math.isinf(req.actual_next) else math.inf
        md = dict(req.metadata)
        md["bucket"] = distance_to_bucket(dist, bucket_horizon=bucket_horizon)
        out.append(
            Request(
                t=req.t,
                page_id=req.page_id,
                predicted_next=req.predicted_next,
                actual_next=req.actual_next,
                metadata=md,
            )
        )
    return out


def maybe_corrupt_buckets(
    requests: List[Request],
    noise_prob: float,
    seed: int = 0,
    max_bucket: int = MAX_BUCKET,
) -> List[Request]:
    """Return copied requests with probabilistically corrupted bucket metadata."""
    if noise_prob <= 0.0:
        return requests

    rng = random.Random(seed)
    out: List[Request] = []
    for req in requests:
        md: Dict[str, Any] = dict(req.metadata)
        bucket = md.get("bucket")
        if bucket is not None and rng.random() < noise_prob:
            choices = [b for b in range(max_bucket + 1) if b != int(bucket)]
            md["bucket"] = rng.choice(choices)
        out.append(
            Request(
                t=req.t,
                page_id=req.page_id,
                predicted_next=req.predicted_next,
                actual_next=req.actual_next,
                metadata=md,
            )
        )
    return out


def extract_trace_prediction_records(requests: List[Request]) -> List[Dict[str, Optional[float]]]:
    """Extract lightweight bucket/confidence metadata for diagnostics."""
    records: List[Dict[str, Optional[float]]] = []
    for req in requests:
        conf = req.metadata.get("confidence")
        records.append(
            {
                "bucket": req.metadata.get("bucket"),
                "confidence": None if conf is None else float(conf),
            }
        )
    return records
