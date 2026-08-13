from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments.run_exact_target_oracle_replication import _validate_summary, _read_config
from scripts.maintenance.repair_campaign_metadata_paths import finalized_output_paths


def test_replication_config_is_canonical():
    cfg = _read_config(Path("configs/exact_target_oracle_replication_v1.json"))
    assert cfg["horizon"] == 4
    assert cfg["history"] == [0, 10000]
    assert cfg["score"] == [10000, 50000]
    assert len(cfg["families"]) * len(cfg["capacities"]) == 21


def test_unit_summary_validation_rejects_partial_or_wrong_protocol():
    summary = {
        "status": "COMPLETE",
        "protocol": {"horizon": 4, "history_start": 0, "score_start": 10000, "score_end": 50000, "capacity": 64},
        "trace": {"family": "brightkite", "sha256": "trace"},
        "policies": {
            "lru": {"misses": 13225, "miss_ratio": 0.3},
            "exact_finite_horizon_eviction_loss_oracle": {"misses": 19079, "miss_ratio": 0.4},
        },
    }
    _validate_summary(summary, "brightkite", 64, "trace")
    summary["protocol"]["horizon"] = 8
    with pytest.raises(ValueError, match="H=4"):
        _validate_summary(summary, "brightkite", 64, "trace")


def test_finalized_unit_output_paths_do_not_retain_staging_directory(tmp_path):
    outputs = {"summary_json": str(tmp_path / ".unit.tmp-1" / "summary.json"), "report_md": str(tmp_path / ".unit.tmp-1" / "report.md")}
    finalized = finalized_output_paths(outputs, tmp_path / "unit")
    assert finalized == {"summary_json": str(tmp_path / "unit" / "summary.json"), "report_md": str(tmp_path / "unit" / "report.md")}
