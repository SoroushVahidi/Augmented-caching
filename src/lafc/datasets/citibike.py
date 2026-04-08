from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .base import CanonicalTraceRecord


def _row_get(row: Dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return default


def parse_citibike(raw_csv: Path, limit: Optional[int] = None) -> List[CanonicalTraceRecord]:
    """Parse CitiBike CSV.

    Default mapping: item_id := start_station_id (station demand as cacheable item).
    """
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"CitiBike raw file not found at {raw_csv}. "
            "Run scripts/datasets/download_citibike.py first or place CSV manually."
        )

    records: List[CanonicalTraceRecord] = []
    with raw_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if limit is not None and len(records) >= limit:
                break
            start_station_id = _row_get(row, ["start_station_id", "start station id", "start_station_id "])
            if not start_station_id:
                continue
            started_at = _row_get(row, ["started_at", "starttime", "starttime "])
            end_station_id = _row_get(row, ["end_station_id", "end station id"])
            bike_id = _row_get(row, ["bikeid", "bike_id"])
            duration = _row_get(row, ["tripduration", "trip_duration_seconds"])

            records.append(
                CanonicalTraceRecord(
                    request_index=len(records),
                    item_id=str(start_station_id),
                    source_dataset="citibike",
                    timestamp=started_at or None,
                    cost=1.0,
                    metadata={
                        "end_station_id": end_station_id,
                        "bike_id": bike_id,
                        "tripduration": duration,
                    },
                )
            )

    if not records:
        raise ValueError("Parsed zero CitiBike records; verify CSV headers/format.")
    return records
