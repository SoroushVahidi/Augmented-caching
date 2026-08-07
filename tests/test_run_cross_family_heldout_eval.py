"""Gate tests for scripts/experiments/run_cross_family_heldout_eval.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path("scripts/experiments").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = _SCRIPTS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _SCRIPTS_DIR)
    yield
    if inserted and _SCRIPTS_DIR in sys.path:
        sys.path.remove(_SCRIPTS_DIR)


def _import_module():
    import run_cross_family_heldout_eval as m
    return m


def test_missing_registry_blocked(tmp_path):
    m = _import_module()
    with pytest.raises(m.GateBlocked, match="No model registry"):
        m.check_gate(tmp_path / "nope.json")


def test_unfrozen_registry_blocked(tmp_path):
    m = _import_module()
    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(json.dumps({"MODEL_SELECTION_FROZEN": False, "missing_folds": ["citibike: not ready"]}))
    with pytest.raises(m.GateBlocked, match="MODEL_SELECTION_FROZEN=False"):
        m.check_gate(reg_path)


def test_scoped_registry_blocked(tmp_path):
    m = _import_module()
    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(json.dumps({
        "MODEL_SELECTION_FROZEN": True, "is_full_campaign_scope": False, "scope_families": ["brightkite"],
    }))
    with pytest.raises(m.GateBlocked, match="not the full 7-family campaign scope"):
        m.check_gate(reg_path)


def test_tampered_model_hash_blocked(tmp_path):
    m = _import_module()
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"original bytes")
    records = [
        {"held_out_family": f, "model_artifact_path": str(model_path), "model_artifact_sha256": "not_the_real_hash"}
        for f in m.FAMILIES
    ]
    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(json.dumps({
        "MODEL_SELECTION_FROZEN": True, "is_full_campaign_scope": True, "records": records,
    }))
    with pytest.raises(m.GateBlocked, match="Model hash mismatch"):
        m.check_gate(reg_path)


def test_fully_valid_registry_passes(tmp_path):
    m = _import_module()
    from lafc.experiments.external_baseline_common import sha256_of_file
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"real model bytes")
    real_hash = sha256_of_file(model_path)
    records = [
        {"held_out_family": f, "model_artifact_path": str(model_path), "model_artifact_sha256": real_hash,
         "training_families": ["a", "b", "c", "d", "e"]}
        for f in m.FAMILIES
    ]
    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(json.dumps({
        "MODEL_SELECTION_FROZEN": True, "is_full_campaign_scope": True, "records": records,
        "registry_sha256": "abc123",
    }))
    registry = m.check_gate(reg_path)
    assert len(registry["records"]) == 7
