"""Storage-guard regression tests (Task 11 of the storage-recovery audit).

Proves the launch guard rejects a HOME-based run_root -- the exact defect
that let job 1288536 launch and fail campaign-wide on a disk-quota error
-- and enforces the quota-audit freshness/threshold requirement.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import time
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


def test_current_manifest_run_root_is_not_under_home():
    guard = _load_guard()
    manifest = _good_manifest()
    assert guard.check_run_root_location(manifest) == []
    assert manifest["run_root"].startswith("/mmfs1/scratch/")


def test_guard_rejects_home_run_root():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["run_root"] = "/home/sv96/lafc-work/Augmented-caching/data/derived/pe_long_horizon_production_v1"
    errors = guard.check_run_root_location(manifest)
    assert errors, "expected the guard to reject a HOME-based run_root"
    assert "1288536" in errors[0]


def test_guard_rejects_mmfs1_home_run_root():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["run_root"] = "/mmfs1/home/sv96/lafc-work/Augmented-caching/data/derived/pe_long_horizon_production_v1"
    errors = guard.check_run_root_location(manifest)
    assert errors


def test_guard_rejects_unapproved_prefix():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["run_root"] = "/tmp/whatever"
    errors = guard.check_run_root_location(manifest)
    assert errors


def test_guard_accepts_project_prefix():
    guard = _load_guard()
    manifest = copy.deepcopy(_good_manifest())
    manifest["run_root"] = "/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1"
    assert guard.check_run_root_location(manifest) == []


def test_guard_requires_quota_audit_artifact(tmp_path, monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "QUOTA_AUDIT_PATH", tmp_path / "does_not_exist.json")
    errors = guard.check_quota_audit({})
    assert errors and "no quota-audit artifact" in errors[0]


def test_guard_rejects_stale_quota_audit(tmp_path, monkeypatch):
    guard = _load_guard()
    stale_path = tmp_path / "quota_audit.json"
    stale_path.write_text(json.dumps({
        "captured_at_epoch": time.time() - 48 * 3600,
        "scratch_available_bytes": 999_999_999_999,
        "quota_domain": "SCRATCH",
    }))
    monkeypatch.setattr(guard, "QUOTA_AUDIT_PATH", stale_path)
    errors = guard.check_quota_audit({})
    assert any("old" in e for e in errors)


def test_guard_rejects_insufficient_scratch_available(tmp_path, monkeypatch):
    guard = _load_guard()
    small_path = tmp_path / "quota_audit.json"
    small_path.write_text(json.dumps({
        "captured_at_epoch": time.time(),
        "scratch_available_bytes": 1_000_000,  # far below 200GB
        "quota_domain": "SCRATCH",
    }))
    monkeypatch.setattr(guard, "QUOTA_AUDIT_PATH", small_path)
    errors = guard.check_quota_audit({})
    assert any("below the required minimum" in e for e in errors)


def test_guard_rejects_quota_audit_reporting_home_domain(tmp_path, monkeypatch):
    guard = _load_guard()
    bad_path = tmp_path / "quota_audit.json"
    bad_path.write_text(json.dumps({
        "captured_at_epoch": time.time(),
        "scratch_available_bytes": 999_999_999_999,
        "quota_domain": "HOME",
    }))
    monkeypatch.setattr(guard, "QUOTA_AUDIT_PATH", bad_path)
    errors = guard.check_quota_audit({})
    assert any("forbidden domain" in e for e in errors)


def test_real_frozen_quota_audit_passes():
    guard = _load_guard()
    params = guard.load_params(guard.PARAMS_PATH)
    assert guard.check_quota_audit(params) == []


def test_full_guard_passes_end_to_end():
    guard = _load_guard()
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(SCRIPTS / "launch_guard.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
