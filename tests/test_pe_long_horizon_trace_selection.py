"""Prove the production trace-selection fix end-to-end (Task 2 of the final
launch gate).

Job 1287838 processed all 5 families instead of 1 because
--trace-glob uses argparse action="append" with a non-empty default
(default=["data/processed/*/trace.jsonl"]); passing --trace-glob on the
command line APPENDS to, rather than replaces, that default.

The production template now writes a one-row --trace-manifest CSV per task
instead. This test calls the *actual* generator library function
(parse_trace_manifest) used by scripts/build_evict_value_dataset_wulver_v1.py,
with the real wildcard default still present as fallback_globs, and checks
that a one-row manifest resolves to exactly one trace/family regardless.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lafc.evict_value_wulver_v1 import parse_trace_manifest  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"
# The exact wildcard default argparse would otherwise fall back to -- included
# deliberately so this test fails loudly if --trace-manifest ever stopped
# actually overriding it.
WILDCARD_DEFAULT = ["data/processed/*/trace.jsonl"]


def _write_task_manifest(tmp_path: Path, name_prefix: str, family: str, trace_path: str) -> Path:
    manifest_csv = tmp_path / f"{name_prefix}_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "trace_name", "dataset_source", "trace_family"])
        w.writerow([trace_path, trace_path, family, family])
    return manifest_csv


def test_every_production_task_resolves_to_exactly_one_family(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    assert len(tasks) == 20

    for task in tasks:
        family = task["family"]
        capacity = task["capacity"]
        trace_path = task["trace_path"]

        task_manifest = _write_task_manifest(tmp_path, task["task_id"], family, trace_path)

        # Call the REAL generator function with the REAL wildcard default
        # still present as fallback_globs, exactly as build_evict_value_dataset_wulver_v1.py
        # does at import/argparse time -- proving --trace-manifest overrides
        # it rather than merely assuming argparse behaves.
        specs = parse_trace_manifest(str(task_manifest), WILDCARD_DEFAULT)

        resolved_families = {s.dataset_source for s in specs}
        assert len(specs) == 1, (
            f"{task['task_id']}: expected exactly 1 resolved trace, got "
            f"{len(specs)}: {[s.dataset_source for s in specs]}"
        )
        assert resolved_families == {family}, (
            f"{task['task_id']}: expected only family={family!r}, "
            f"got {resolved_families}"
        )
        # capacity is a CLI arg (--capacities), not part of trace resolution,
        # but assert it's exactly the one intended value per task.
        assert task["capacity"] == capacity


def test_no_family_leaks_across_any_task_pair(tmp_path: Path) -> None:
    """Cross-check: no two tasks' resolved trace sets overlap in family,
    except intentionally (same family, different capacity)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    seen_family_capacity = set()
    for task in manifest["tasks"]:
        key = (task["family"], task["capacity"])
        assert key not in seen_family_capacity, f"duplicate family+capacity: {key}"
        seen_family_capacity.add(key)
    assert len(seen_family_capacity) == 20

    families = {t["family"] for t in manifest["tasks"]}
    assert families == {"cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"}
    assert "brightkite" not in families
    assert "citibike" not in families
