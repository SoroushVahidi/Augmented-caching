"""Repair only stale atomic-staging paths in finalized campaign metadata.

This utility changes JSON metadata path strings only. It never changes
scientific values, aggregates, models, traces, or configuration. Every stale
path must map to one existing canonical artifact; missing or ambiguous targets
fail closed instead of being guessed. A second pass is therefore idempotent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


STALE_PATH = re.compile(r"^(?P<prefix>.*?/units/)\.(?P<unit>[^/]+)\.tmp-[^/]+/(?P<name>[^/]+)$")
FINALIZED_UNIT_FILES = {"summary.json", "provenance.json", "report.md"}


def finalized_output_paths(outputs: dict[str, Any], final_dir: Path) -> dict[str, str]:
    """Build output references from the finalized unit directory."""
    return {key: str(final_dir / Path(value).name) for key, value in outputs.items()}


def canonicalize_stale_path(value: str) -> str | None:
    """Return the unambiguous finalized path for one stale staging path."""
    match = STALE_PATH.match(value)
    if not match:
        return None
    if match.group("name") not in FINALIZED_UNIT_FILES:
        raise ValueError(f"unsupported or ambiguous finalized artifact name: {value}")
    canonical = Path(match.group("prefix")) / match.group("unit") / match.group("name")
    if not canonical.is_file():
        raise FileNotFoundError(f"canonical artifact is missing for {value}: {canonical}")
    return str(canonical)


def repair_payload(payload: Any) -> tuple[Any, int]:
    """Replace only stale path strings, refusing missing/ambiguous targets."""
    changed = 0

    def visit(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            replacement = canonicalize_stale_path(value)
            if replacement is not None:
                changed += 1
                return replacement
        return value

    return visit(payload), changed


def repair_json_file(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    repaired, changed = repair_payload(payload)
    if changed:
        path.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return changed


def repair_campaign(campaign_dir: Path) -> int:
    return sum(repair_json_file(path) for path in sorted(campaign_dir.rglob("*.json")))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path, nargs="+")
    args = parser.parse_args()
    for campaign in args.campaign:
        print(f"{campaign}: repaired {repair_campaign(campaign)} references")
