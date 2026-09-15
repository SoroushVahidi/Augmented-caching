from __future__ import annotations

import json
from pathlib import Path

from scripts.validation.pe_tier2_launch_guard import validate_plan


def _manifest() -> dict:
    path = Path("configs/pe_tier2_campaign_manifest_20260915.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_launch_guard_accepts_frozen_lfu_cells():
    manifest = _manifest()
    rows = list(manifest["new_baseline_cells"])
    assert validate_plan(manifest, rows) == []


def test_launch_guard_rejects_tier1_recompute():
    manifest = _manifest()
    rows = list(manifest["new_baseline_cells"]) + [
        {"policy": "lru", "family": "twemcache", "capacity": 32}
    ]
    failures = validate_plan(manifest, rows)
    assert any("Tier-1 reuse" in failure for failure in failures)


def test_launch_guard_rejects_contaminated_learned_cell():
    manifest = _manifest()
    rows = list(manifest["new_baseline_cells"]) + [
        {
            "policy": "evict_value_v1",
            "family": "twemcache",
            "capacity": 32,
            "model_sha256": manifest["learned_policy_gate"]["contaminated_model_sha256"],
            "source": "pilot_20260913",
        }
    ]
    failures = validate_plan(manifest, rows)
    assert any("learned replay gate" in failure for failure in failures)
    assert any("contaminated heavy_r1" in failure for failure in failures)
    assert any("pilot may be imported" in failure for failure in failures)
