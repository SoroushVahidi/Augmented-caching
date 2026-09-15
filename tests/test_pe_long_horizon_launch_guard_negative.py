"""Prove the extended launch guard (Task 6 of the final launch gate) actually
CATCHES bad states, not just passes trivially on the good one.

Loads the real launch_guard.py module and calls its check_* functions
directly against deliberately-corrupted synthetic manifests -- no
subprocess, no Wulver, no canonical-code changes.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "pe_long_horizon"


def _load_guard():
    spec = importlib.util.spec_from_file_location("launch_guard", SCRIPTS / "launch_guard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _good_manifest() -> dict:
    return json.loads((REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json").read_text())


def test_guard_passes_on_the_real_unmodified_manifest():
    guard = _load_guard()
    manifest = _good_manifest()
    assert guard.check_manifest_structure(manifest) == []
    assert guard.check_trace_selection(manifest) == []


def test_guard_catches_brightkite_injection():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["tasks"][0]["family"] = "brightkite"
    errors = guard.check_manifest_structure(manifest)
    assert any("forbidden family" in e for e in errors)


def test_guard_catches_h16_as_production_horizon():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["tasks"][0]["horizons"] = [16, 32, 64]
    errors = guard.check_manifest_structure(manifest)
    assert any("forbidden production horizon" in e for e in errors)


def test_guard_catches_duplicate_output_directory():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["tasks"][1]["out_dir"] = manifest["tasks"][0]["out_dir"]
    errors = guard.check_manifest_structure(manifest)
    assert any("duplicate output directory" in e for e in errors)


def test_guard_catches_wrong_task_count():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["tasks"] = manifest["tasks"][:19]
    errors = guard.check_manifest_structure(manifest)
    assert any("expected 20 physical tasks" in e for e in errors)


def test_guard_catches_generator_hash_mismatch():
    guard = _load_guard()
    params = {
        "EXPECTED_GENERATOR_LIB_SHA256": "0" * 64,
        "EXPECTED_GENERATOR_DRIVER_SHA256": "0" * 64,
    }
    errors = guard.check_generator_hashes(params)
    assert any("mismatch" in e for e in errors)


def test_guard_catches_probe_output_reuse():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["tasks"][0]["out_dir"] = (
        "data/derived/evict_value_v1_long_horizon_probe_cap256_cloudphysics_20260915"
    )
    errors = guard.check_probe_output_not_reused(manifest)
    assert errors and "timeout probe" in errors[0]
