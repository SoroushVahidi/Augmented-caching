from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "cells" in payload:
            return list(payload["cells"])
        if isinstance(payload, list):
            return list(payload)
        raise SystemExit(f"{path}: expected a JSON list or an object with a 'cells' list")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["policy"]), str(row["family"]), int(row["capacity"]))


def validate_plan(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    allowed = {_cell_key(row) for row in manifest["new_baseline_cells"]}
    tier1 = set(manifest["tier1_reuse_policies"])
    families = set(manifest["families"])
    capacities = {int(c) for c in manifest["capacities"]}
    seen: set[tuple[str, str, int]] = set()

    learned_gate = manifest["learned_policy_gate"]
    allowed_model_sha = learned_gate.get("allowed_model_sha256")
    contaminated_sha = learned_gate.get("contaminated_model_sha256")

    for idx, row in enumerate(rows, start=1):
        try:
            key = _cell_key(row)
        except Exception as exc:
            failures.append(f"row {idx}: missing or invalid policy/family/capacity fields: {exc}")
            continue

        policy, family, capacity = key
        if key in seen:
            failures.append(f"row {idx}: duplicate cell {key}")
        seen.add(key)

        if family not in families:
            failures.append(f"row {idx}: family {family!r} is not frozen in manifest")
        if capacity not in capacities:
            failures.append(f"row {idx}: capacity {capacity} is not frozen in manifest")
        if policy in tier1:
            failures.append(f"row {idx}: {policy} is Tier-1 reuse and must not be recomputed")
        if policy == "evict_value_v1":
            failures.append(f"row {idx}: learned replay gate is {learned_gate['status']}; no new learned cells are allowed")
            model_sha = row.get("model_sha256")
            if model_sha == contaminated_sha:
                failures.append(f"row {idx}: contaminated heavy_r1 model SHA reused")
            if allowed_model_sha is not None and model_sha != allowed_model_sha:
                failures.append(f"row {idx}: model SHA mismatch")
            if row.get("source") == "pilot_20260913":
                failures.append(f"row {idx}: frozen pilot may be imported as existing evidence, not rerun as new")
        elif key not in allowed:
            failures.append(f"row {idx}: cell {key} is not in frozen new_baseline_cells")

    missing = allowed - seen
    if missing:
        failures.append(f"missing frozen cells: {sorted(missing)}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-closed PE Tier-2 launch guard.")
    ap.add_argument("--manifest", type=Path, default=Path("configs/pe_tier2_campaign_manifest_20260915.json"))
    ap.add_argument("--plan", type=Path, required=True, help="CSV or JSON launch plan containing policy,family,capacity rows.")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = _load_rows(args.plan)
    failures = validate_plan(manifest, rows)
    if failures:
        print("LAUNCH_GUARD_FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    print("LAUNCH_GUARD_PASS")


if __name__ == "__main__":
    main()
