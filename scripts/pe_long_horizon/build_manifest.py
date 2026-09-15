"""Build the canonical PE long-horizon production manifest.

Deterministically enumerates:
  - 20 production TASKS (one per family x capacity; each task's single
    generator invocation computes horizons {32,64,128} together, matching
    the generator's real reuse boundary -- see docs/pe_long_horizon_design_notes.md)
  - 60 logical CELLS (one per family x capacity x horizon), used for
    scientific coverage bookkeeping/validation, each cross-referenced to the
    task that produces it.

Re-running this script must produce byte-identical output (pure function of
the constants below) -- this is what "deterministic array-index -> manifest
record mapping" means for Task 5's sbatch.

Does not touch Wulver, does not generate data. Local planning artifact only.
"""
from __future__ import annotations

import json
from pathlib import Path

FAMILIES = ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
READER_FACING_NAME = {
    "cloudphysics": "alibaba-block",
    "metacdn": "metacdn",
    "metakv": "metakv",
    "twemcache": "twemcache",
    "wiki2018": "wiki2018",
}
CAPACITIES = [32, 64, 128, 256]
PRODUCTION_HORIZONS = [32, 64, 128]  # H16 is NOT a production cell -- canonical control only
EXPECTED_WULVER_SOURCE_SHA = "3a04a25094de1e8c57a603e2662b321bcf1d9e88"
EXPECTED_WULVER_SOURCE_BRANCH = "main"
RUN_ROOT = "data/derived/pe_long_horizon_production_v1"
TRACE_PATH_TEMPLATE = "data/processed/{family}/trace.jsonl"


def build_manifest() -> dict:
    tasks = []
    cells = []
    task_index = 0
    for family in FAMILIES:
        for capacity in CAPACITIES:
            task_id = f"{family}__cap{capacity}"
            out_dir = f"{RUN_ROOT}/raw/{task_id}"
            task = {
                "array_index": task_index,
                "task_id": task_id,
                "family": family,
                "reader_facing_family": READER_FACING_NAME[family],
                "capacity": capacity,
                "horizons": list(PRODUCTION_HORIZONS),
                "trace_path": TRACE_PATH_TEMPLATE.format(family=family),
                "out_dir": out_dir,
                "expected_source_sha": EXPECTED_WULVER_SOURCE_SHA,
                "expected_source_branch": EXPECTED_WULVER_SOURCE_BRANCH,
                "status": "NOT_RUN",
            }
            tasks.append(task)
            for horizon in PRODUCTION_HORIZONS:
                cells.append(
                    {
                        "cell_id": f"{family}__cap{capacity}__H{horizon}",
                        "family": family,
                        "reader_facing_family": READER_FACING_NAME[family],
                        "capacity": capacity,
                        "horizon": horizon,
                        "trace_path": TRACE_PATH_TEMPLATE.format(family=family),
                        "produced_by_task_id": task_id,
                        "produced_by_array_index": task_index,
                        "intended_output_dir": out_dir,
                        "expected_source_sha": EXPECTED_WULVER_SOURCE_SHA,
                        "status": "NOT_RUN",
                    }
                )
            task_index += 1

    assert len(tasks) == 20, f"expected 20 production tasks, got {len(tasks)}"
    assert len(cells) == 60, f"expected 60 logical cells, got {len(cells)}"
    for f in FAMILIES:
        assert f not in ("brightkite", "citibike")
    for c in cells:
        assert c["horizon"] != 16, "H16 must not appear as a production cell (canonical control only)"

    return {
        "format": "pe_long_horizon_production_manifest_v1",
        "run_root": RUN_ROOT,
        "families": FAMILIES,
        "capacities": CAPACITIES,
        "production_horizons": PRODUCTION_HORIZONS,
        "excluded_families": ["brightkite", "citibike"],
        "h16_reuse_source": "canonical validated H16 artifacts (not regenerated)",
        "expected_source_sha": EXPECTED_WULVER_SOURCE_SHA,
        "expected_source_branch": EXPECTED_WULVER_SOURCE_BRANCH,
        "num_production_tasks": len(tasks),
        "num_logical_cells": len(cells),
        "tasks": tasks,
        "cells": cells,
    }


def main() -> None:
    manifest = build_manifest()
    out_path = Path(__file__).resolve().parents[2] / "configs" / "pe_long_horizon_production" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"tasks={manifest['num_production_tasks']} cells={manifest['num_logical_cells']}")


if __name__ == "__main__":
    main()
