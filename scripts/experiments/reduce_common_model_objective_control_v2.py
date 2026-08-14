"""Fail-closed reducer for common_model_objective_control_v2 unit outputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
CAPACITIES = [32, 64, 128]
OBJECTIVES = [
    "objective_eviction_loss",
    "objective_next_arrival",
    "objective_reuse_distance",
    "objective_pairwise",
]


def atomic_json(p: Path, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False, encoding="utf8") as f:
        json.dump(x, f, indent=2, sort_keys=True)
        f.write("\n")
        q = Path(f.name)
    os.replace(q, p)


def read_unit(root: Path, family: str, capacity: int) -> Dict[str, object]:
    path = root / "units" / f"{family}_cap{capacity}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing unit summary: {path}")
    data = json.loads(path.read_text())
    if data.get("status") != "COMPLETE":
        raise ValueError(f"unit not complete: {path}")
    metadata = data.get("metadata", {})
    if metadata.get("family") != family or int(metadata.get("capacity")) != capacity:
        raise ValueError(f"metadata family/capacity mismatch: {path}")
    rows = data.get("rows", [])
    if len(rows) != len(OBJECTIVES):
        raise ValueError(f"expected 4 rows in {path}, found {len(rows)}")
    seen_obj = {str(r.get("objective")) for r in rows}
    if seen_obj != set(OBJECTIVES):
        raise ValueError(f"objective coverage mismatch in {path}: {sorted(seen_obj)}")
    for row in rows:
        if row.get("held_out_family") != family or int(row.get("capacity")) != capacity:
            raise ValueError(f"row key mismatch in {path}: {row}")
        if not row.get("trace_sha256"):
            raise ValueError(f"missing trace hash in {path}: {row}")
    return data


def reduce(root: Path) -> None:
    rows: List[Dict[str, object]] = []
    units: Dict[str, Dict[str, object]] = {}
    source_heads = set()
    protocols = set()
    trace_hashes: Dict[Tuple[str, int], str] = {}
    keys = set()

    for family in FAMILIES:
        for capacity in CAPACITIES:
            data = read_unit(root, family, capacity)
            metadata = data["metadata"]
            source_heads.add(str(metadata.get("source_head")))
            protocols.add(str(metadata.get("protocol_id")))
            units[f"{family}_cap{capacity}"] = {
                "status": "COMPLETE",
                "trace_sha256": metadata.get("trace_sha256"),
                "source_head": metadata.get("source_head"),
            }
            for row in data["rows"]:
                key = (row["held_out_family"], int(row["capacity"]), row["objective"])
                if key in keys:
                    raise ValueError(f"duplicate primary key: {key}")
                keys.add(key)
                trace_hashes[(str(row["held_out_family"]), int(row["capacity"]))] = str(row["trace_sha256"])
                rows.append(dict(row))

    expected_keys = {(family, capacity, obj) for family in FAMILIES for capacity in CAPACITIES for obj in OBJECTIVES}
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        raise ValueError(f"primary-key coverage mismatch missing={missing} extra={extra}")
    if len(rows) != 84:
        raise ValueError(f"expected 84 rows, found {len(rows)}")
    if len(source_heads) != 1:
        raise ValueError(f"source commit mismatch across units: {sorted(source_heads)}")
    if protocols != {"common_model_objective_control_v2"}:
        raise ValueError(f"protocol mismatch: {sorted(protocols)}")

    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    atomic_json(
        root / "completion_manifest.json",
        {
            "status": "COMPLETE",
            "expected_units": 21,
            "completed_units": 21,
            "expected_rows": 84,
            "rows": 84,
            "source_head": next(iter(source_heads)),
            "units": units,
        },
    )
    atomic_json(
        root / "integrity_audit.json",
        {
            "status": "PASS",
            "rows": 84,
            "expected_rows": 84,
            "unique_keys": len(keys),
            "families": FAMILIES,
            "capacities": CAPACITIES,
            "objectives": OBJECTIVES,
            "trace_hashes": {f"{family}_cap{capacity}": h for (family, capacity), h in trace_hashes.items()},
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/common_model_objective_control_wulver_v2"))
    args = ap.parse_args()
    reduce(args.root)


if __name__ == "__main__":
    main()
