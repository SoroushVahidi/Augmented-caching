from __future__ import annotations

import gzip
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import CanonicalTraceRecord

BRIGHTKITE_FILENAME = "loc-brightkite_totalCheckins.txt.gz"


def _open_brightkite(raw_path: Path):
    if raw_path.suffix == ".gz":
        return gzip.open(raw_path, "rt", encoding="utf-8")
    return raw_path.open("r", encoding="utf-8")


def parse_brightkite(raw_path: Path, limit: Optional[int] = None) -> List[CanonicalTraceRecord]:
    """Parse BrightKite check-ins.

    Default mapping: cacheable item_id := venue_id (4th column).
    Expected columns: user_id, timestamp, lat, lon, venue_id.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"BrightKite raw file not found at {raw_path}. "
            "Run scripts/datasets/download_brightkite.py first."
        )

    records: List[CanonicalTraceRecord] = []
    with _open_brightkite(raw_path) as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            user_id, ts, lat, lon, venue_id = parts[:5]
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                iso_ts = dt.isoformat() + "Z"
            except ValueError:
                iso_ts = ts

            records.append(
                CanonicalTraceRecord(
                    request_index=len(records),
                    item_id=str(venue_id),
                    source_dataset="brightkite",
                    timestamp=iso_ts,
                    cost=1.0,
                    metadata={
                        "user_id": user_id,
                        "latitude": lat,
                        "longitude": lon,
                        "raw_timestamp": ts,
                    },
                )
            )
    if not records:
        raise ValueError("Parsed zero BrightKite records; verify raw format.")
    return records
