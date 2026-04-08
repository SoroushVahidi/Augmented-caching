from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from .base import CanonicalTraceRecord


def parse_wiki2018(raw_path: Path, limit: Optional[int] = None) -> List[CanonicalTraceRecord]:
    """Parse wiki2018-style request traces from CSV/TSV.

    Required columns (case-sensitive): object_id
    Optional columns: timestamp, size, cost
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"wiki2018 raw file not found: {raw_path}. "
            "See data/raw/wiki2018/INSTRUCTIONS.md for manual acquisition instructions."
        )

    delimiter = "\t" if raw_path.suffix.lower() in {".tsv", ".txt"} else ","

    records: List[CanonicalTraceRecord] = []
    with raw_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if "object_id" not in (reader.fieldnames or []):
            raise ValueError("wiki2018 raw input must include an 'object_id' column")

        for row in reader:
            if limit is not None and len(records) >= limit:
                break
            object_id = row.get("object_id", "").strip()
            if not object_id:
                continue
            size = row.get("size", "").strip()
            cost = float(row["cost"]) if row.get("cost") else 1.0
            records.append(
                CanonicalTraceRecord(
                    request_index=len(records),
                    item_id=object_id,
                    source_dataset="wiki2018",
                    timestamp=row.get("timestamp") or None,
                    size=int(size) if size else None,
                    cost=cost,
                    metadata={k: v for k, v in row.items() if k not in {"object_id", "timestamp", "size", "cost"}},
                )
            )
    if not records:
        raise ValueError("Parsed zero wiki2018 records; check raw input file")
    return records
