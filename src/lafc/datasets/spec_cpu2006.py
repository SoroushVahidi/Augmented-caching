from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import CanonicalTraceRecord


def _parse_line(line: str) -> Optional[Dict[str, str]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2:
            return {"address": parts[0], "op": parts[1], "size": parts[2] if len(parts) > 2 else "64"}

    parts = s.split()
    if len(parts) >= 1:
        address = parts[0]
        op = parts[1] if len(parts) >= 2 else "R"
        size = parts[2] if len(parts) >= 3 else "64"
        return {"address": address, "op": op, "size": size}

    return None


def parse_spec_from_manifest(manifest_path: Path, limit: Optional[int] = None) -> List[CanonicalTraceRecord]:
    """Parse local SPEC CPU2006-derived traces listed in a manifest JSON.

    Manifest format:
    {
      "traces": [
        {"name": "401.bzip2", "path": "401.bzip2.trace"},
        ...
      ]
    }
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"SPEC manifest not found: {manifest_path}. "
            "Create data/raw/spec_cpu2006/manifest.json per docs/datasets.md."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    traces = manifest.get("traces", [])
    if not traces:
        raise ValueError("SPEC manifest contains no traces entries")

    records: List[CanonicalTraceRecord] = []
    base_dir = manifest_path.parent
    for entry in traces:
        trace_name = entry["name"]
        trace_path = base_dir / entry["path"]
        if not trace_path.exists():
            raise FileNotFoundError(f"SPEC trace path does not exist: {trace_path}")

        with trace_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if limit is not None and len(records) >= limit:
                    return records
                parsed = _parse_line(line)
                if not parsed:
                    continue
                records.append(
                    CanonicalTraceRecord(
                        request_index=len(records),
                        item_id=parsed["address"],
                        source_dataset="spec_cpu2006",
                        cost=1.0,
                        size=int(parsed["size"], 0) if parsed["size"] else 64,
                        metadata={"op": parsed["op"], "trace": trace_name},
                    )
                )

    if not records:
        raise ValueError("Parsed zero SPEC records; check input trace format.")
    return records
