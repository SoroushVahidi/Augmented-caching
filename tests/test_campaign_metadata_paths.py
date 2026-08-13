from pathlib import Path

import pytest

from scripts.maintenance.repair_campaign_metadata_paths import (
    canonicalize_stale_path,
    finalized_output_paths,
    repair_payload,
)


def test_repair_maps_stale_path_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    final = Path("analysis/campaign/units/unit")
    final.mkdir(parents=True)
    (final / "summary.json").write_text("{}\n", encoding="utf-8")
    payload = {"outputs": {"summary_json": "analysis/campaign/units/.unit.tmp-123/summary.json"}, "value": 7}

    repaired, count = repair_payload(payload)
    assert count == 1
    assert repaired["outputs"]["summary_json"] == "analysis/campaign/units/unit/summary.json"
    again, second_count = repair_payload(repaired)
    assert second_count == 0
    assert again == repaired


def test_repair_rejects_missing_canonical_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        canonicalize_stale_path("analysis/campaign/units/.unit.tmp-123/summary.json")


def test_repair_rejects_unsupported_ambiguous_artifact_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    final = Path("analysis/campaign/units/unit")
    final.mkdir(parents=True)
    (final / "unknown.bin").write_bytes(b"artifact")
    with pytest.raises(ValueError, match="unsupported or ambiguous"):
        canonicalize_stale_path("analysis/campaign/units/.unit.tmp-123/unknown.bin")


def test_finalized_output_paths_never_retain_staging_directory(tmp_path):
    outputs = {"summary_json": ".unit.tmp-123/summary.json"}

    finalized = finalized_output_paths(outputs, tmp_path / "unit")

    assert finalized["summary_json"] == str(tmp_path / "unit" / "summary.json")
    assert ".tmp-" not in finalized["summary_json"]
