from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CanonicalTraceRecord:
    """Canonical record for dataset-derived caching traces."""

    request_index: int
    item_id: str
    source_dataset: str
    split: str = "full"
    timestamp: Optional[str] = None
    size: Optional[int] = None
    cost: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


REQUIRED_FIELDS = {"request_index", "item_id", "source_dataset", "cost"}


def validate_records(records: Iterable[CanonicalTraceRecord]) -> None:
    last_index = -1
    for rec in records:
        d = asdict(rec)
        missing = REQUIRED_FIELDS - set(d.keys())
        if missing:
            raise ValueError(f"Canonical record is missing fields: {sorted(missing)}")
        if rec.request_index <= last_index:
            raise ValueError("request_index values must be strictly increasing")
        if not rec.item_id:
            raise ValueError("item_id must be non-empty")
        if rec.cost <= 0:
            raise ValueError("cost must be > 0")
        last_index = rec.request_index


def write_records(records: List[CanonicalTraceRecord], output_path: Path, fmt: str = "jsonl") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_records(records)

    if fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        return

    if fmt == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "request_index",
                    "item_id",
                    "source_dataset",
                    "split",
                    "timestamp",
                    "size",
                    "cost",
                    "metadata",
                ],
            )
            writer.writeheader()
            for r in records:
                data = asdict(r)
                data["metadata"] = json.dumps(data["metadata"], sort_keys=True) if data["metadata"] is not None else ""
                writer.writerow(data)
        return

    raise ValueError(f"Unsupported format '{fmt}'. Expected jsonl or csv.")


def write_request_sequence(records: List[CanonicalTraceRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(f"{rec.item_id}\n")
