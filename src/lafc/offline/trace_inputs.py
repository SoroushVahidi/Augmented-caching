"""Trace helpers for offline general caching baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

from lafc.simulator.request_trace import load_trace
from lafc.types import Page, PageId, Request


def load_trace_with_sizes(path: str) -> Tuple[list[Request], Dict[PageId, Page], Dict[PageId, float]]:
    """Load requests/pages and require per-page sizes for general caching.

    Supported size encodings:
    - JSON: top-level "sizes": {"page_id": size}
    - CSV:  "size" column (consistent per page_id)
    """
    requests, pages = load_trace(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        raw_sizes = data.get("sizes")
        if not isinstance(raw_sizes, dict):
            raise ValueError(
                "General caching requires a top-level 'sizes' map in JSON traces."
            )
        page_sizes = {str(k): float(v) for k, v in raw_sizes.items()}
    elif suffix == ".csv":
        page_sizes: Dict[PageId, float] = {}
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "size" not in (reader.fieldnames or []):
                raise ValueError("General caching CSV traces must include a 'size' column.")
            for row in reader:
                pid = str(row["page_id"])
                size_val = float(row["size"])
                if pid in page_sizes and abs(page_sizes[pid] - size_val) > 1e-12:
                    raise ValueError(
                        f"CSV size mismatch for page '{pid}': {page_sizes[pid]} vs {size_val}"
                    )
                page_sizes[pid] = size_val
    else:
        raise ValueError(f"Unsupported trace format '{suffix}'. Use .json or .csv")

    missing = sorted({r.page_id for r in requests if r.page_id not in page_sizes})
    if missing:
        raise ValueError(
            "Missing sizes for requested pages: "
            f"{missing}. Add entries in JSON 'sizes' or CSV 'size' column."
        )

    return requests, pages, page_sizes
