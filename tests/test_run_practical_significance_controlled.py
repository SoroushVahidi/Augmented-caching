"""Gate-logic tests for scripts/experiments/run_practical_significance_controlled.py,
using mocked process lists / resource readings (no real `ps`/`/proc` calls,
no real subprocess launches -- see task instruction to test the gate logic
against mocked process lists)."""

from __future__ import annotations

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
    import run_practical_significance_controlled as m
    return m


def _patch_idle_machine(monkeypatch, m, tmp_path):
    monkeypatch.setattr(m, "_ps_lines", lambda: ["  1234  0.1 some_unrelated_process"])
    monkeypatch.setattr(m, "_load_average", lambda: (0.5, 0.4, 0.3))
    monkeypatch.setattr(m, "_free_ram_gb", lambda: 40.0)
    monkeypatch.setattr(m, "_free_disk_gb", lambda path=".": 100.0)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "20\n"})())
    cfg = tmp_path / "practical_significance_ablation_v1.json"
    cfg.write_text("{}")
    monkeypatch.setattr(m, "CONFIG_PATH", cfg)


def test_idle_machine_ready(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    gate = m.check_gate()
    assert gate["ready"] is True
    assert gate["reasons"] == []


def test_active_c1_process_blocks(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_ps_lines", lambda: [
        "3314009 99.9 .venv_fairness/bin/python -u scripts/train_evict_value_wulver_v1.py --manifest x",
    ])
    gate = m.check_gate()
    assert gate["ready"] is False
    assert "concern_1_cross_family_training" in gate["blocking_processes"]
    assert any("concern_1_cross_family_training" in r for r in gate["reasons"])


def test_legacy_lrb_blocks_too(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_ps_lines", lambda: [
        "113981 100 .venv_kbs_heavy_r1/bin/python3 scripts/experiments/run_lrb_external_baseline.py --capacities 32,64,128",
    ])
    gate = m.check_gate()
    assert gate["ready"] is False
    assert "legacy_lrb_archival" in gate["blocking_processes"]


def test_own_process_never_self_blocks(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_ps_lines", lambda: [
        "999999 1.0 python run_practical_significance_controlled.py",
    ])
    gate = m.check_gate()
    assert gate["ready"] is True


def test_high_load_average_blocks(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_load_average", lambda: (13.38, 11.69, 8.1))
    gate = m.check_gate()
    assert gate["ready"] is False
    assert any("load average" in r for r in gate["reasons"])


def test_insufficient_ram_blocks(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_free_ram_gb", lambda: 2.0)
    gate = m.check_gate()
    assert gate["ready"] is False
    assert any("RAM" in r for r in gate["reasons"])


def test_insufficient_disk_blocks(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_free_disk_gb", lambda path=".": 1.0)
    gate = m.check_gate()
    assert gate["ready"] is False
    assert any("disk" in r for r in gate["reasons"])


def test_missing_protocol_config_blocks(monkeypatch, tmp_path):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "CONFIG_PATH", tmp_path / "does_not_exist.json")
    gate = m.check_gate()
    assert gate["ready"] is False
    assert any("protocol config not found" in r for r in gate["reasons"])


def test_launch_refused_when_gate_blocked(monkeypatch, tmp_path, capsys):
    m = _import_module()
    _patch_idle_machine(monkeypatch, m, tmp_path)
    monkeypatch.setattr(m, "_ps_lines", lambda: [
        "113981 100 .venv_kbs_heavy_r1/bin/python3 scripts/experiments/run_lrb_external_baseline.py",
    ])
    called = []
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: called.append(a) or type("R", (), {"stdout": "20\n"})())
    monkeypatch.setattr(sys, "argv", ["run_practical_significance_controlled.py", "--launch"])
    with pytest.raises(SystemExit) as exc_info:
        m.main()
    assert exc_info.value.code == 1
    # subprocess.run was only called for the nproc lookup, never for the actual campaign command.
    assert not any(str(m.UNDERLYING_RUNNER) in str(c) for c in called)
