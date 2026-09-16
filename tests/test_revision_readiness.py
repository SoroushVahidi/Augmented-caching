"""Fixture tests for scripts/validation/revision_readiness.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path("scripts/validation").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = _SCRIPTS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _SCRIPTS_DIR)
    yield
    if inserted and _SCRIPTS_DIR in sys.path:
        sys.path.remove(_SCRIPTS_DIR)


def _import_module():
    import revision_readiness as m
    return m


def test_next_action_waits_when_c1_c2_running_even_if_c3_resumable():
    m = _import_module()
    c1r = {"training": "RUNNING", "registry": "BLOCKED", "eval": "BLOCKED"}
    c2r = {"training": "RUNNING", "registry": "BLOCKED", "eval": "BLOCKED", "audits": "NOT_RUN"}
    c3r = {"campaign": "PARTIAL", "resume": "READY", "completion_audit": "BLOCKED"}
    c4r = {"smoke": "COMPLETE", "controlled_timing": "BLOCKED_BY_ACTIVE_JOBS"}
    assert m.next_action(c1r, c2r, c3r, c4r) == "WAIT_FOR_RUNNING_JOBS"


def test_next_action_c3_resume_surfaces_once_c1_c2_training_done():
    m = _import_module()
    c1r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "BLOCKED"}
    c2r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "BLOCKED", "audits": "COMPLETE"}
    c3r = {"campaign": "PARTIAL", "resume": "READY", "completion_audit": "BLOCKED"}
    c4r = {"smoke": "COMPLETE", "controlled_timing": "BLOCKED_BY_ACTIVE_JOBS"}
    assert m.next_action(c1r, c2r, c3r, c4r) == "CONCERN_3_READY_TO_RESUME"


def test_next_action_registry_freeze_takes_priority_even_while_others_run():
    m = _import_module()
    c1r = {"training": "COMPLETE", "registry": "READY_TO_FREEZE", "eval": "BLOCKED"}
    c2r = {"training": "RUNNING", "registry": "BLOCKED", "eval": "BLOCKED", "audits": "NOT_RUN"}
    c3r = {"campaign": "PARTIAL", "resume": "READY", "completion_audit": "BLOCKED"}
    c4r = {"smoke": "COMPLETE", "controlled_timing": "BLOCKED_BY_ACTIVE_JOBS"}
    assert m.next_action(c1r, c2r, c3r, c4r) == "CONCERN_1_READY_FOR_NEXT_STAGE"


def test_next_action_timing_ready_when_everything_else_settled():
    m = _import_module()
    c1r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "COMPLETE"}
    c2r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "COMPLETE", "audits": "COMPLETE"}
    c3r = {"campaign": "COMPLETE", "resume": "N/A", "completion_audit": "READY"}
    c4r = {"smoke": "COMPLETE", "controlled_timing": "READY"}
    assert m.next_action(c1r, c2r, c3r, c4r) == "CONCERN_4_TIMING_READY"


def test_next_action_run_c2_final_audits_when_eval_complete_audits_required():
    m = _import_module()
    c1r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "BLOCKED"}
    c2r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED",
           "audits": "NOT_RUN"}
    c3r = {"campaign": "PARTIAL", "resume": "READY", "completion_audit": "BLOCKED"}
    c4r = {"smoke": "COMPLETE", "controlled_timing": "BLOCKED_BY_ACTIVE_JOBS"}
    assert m.next_action(c1r, c2r, c3r, c4r) == "RUN_C2_FINAL_AUDITS"


def test_next_action_review_audit_failure_when_blocked():
    m = _import_module()
    c1r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "BLOCKED"}
    c2r = {"training": "COMPLETE", "registry": "FROZEN", "eval": "BLOCKED_BY_AUDIT", "audits": "PARTIAL"}
    c3r = {"campaign": "PARTIAL", "resume": "READY", "completion_audit": "BLOCKED"}
    c4r = {"smoke": "COMPLETE", "controlled_timing": "BLOCKED_BY_ACTIVE_JOBS"}
    assert m.next_action(c1r, c2r, c3r, c4r) == "REVIEW_C2_AUDIT_FAILURE"


def test_concern_1_readiness_blocked_while_training(tmp_path):
    m = _import_module()
    c1 = {"models_done": 2, "models_total": 7, "registry_frozen": False,
          "eval_rows_total": 0, "eval_rows_primary_expected": 21}
    r = m.concern_1_readiness(c1)
    assert r == {"training": "RUNNING", "registry": "BLOCKED", "eval": "BLOCKED"}


def test_concern_1_readiness_ready_to_freeze_when_all_models_done(tmp_path):
    m = _import_module()
    c1 = {"models_done": 7, "models_total": 7, "registry_frozen": False,
          "eval_rows_total": 0, "eval_rows_primary_expected": 21}
    r = m.concern_1_readiness(c1)
    assert r["registry"] == "READY_TO_FREEZE"
    assert r["training"] == "COMPLETE"


def test_concern_2_readiness_training_running_eval_blocked(tmp_path):
    m = _import_module()
    c2 = {"models_done": 24, "models_total": 28, "registry_frozen": False,
          "eval_rows": 0, "eval_rows_expected": 84, "eval_state": "NOT_STARTED"}
    r = m.concern_2_readiness(c2, ablation_root=None)
    assert r["training"] == "RUNNING"
    assert r["registry"] == "BLOCKED"
    assert r["eval"] == "NOT_STARTED"


def test_concern_2_readiness_ready_to_start_when_frozen_no_eval_state(tmp_path):
    m = _import_module()
    c2 = {"models_done": 28, "models_total": 28, "registry_frozen": True,
          "eval_rows": 84, "eval_rows_expected": 84, "eval_state": "NOT_STARTED"}
    r = m.concern_2_readiness(c2, ablation_root=None)
    assert "acceptance_note" not in r
    assert r["eval"] == "READY_TO_START"


def test_concern_2_readiness_running_pre_gate_has_acceptance_note():
    m = _import_module()
    c2 = {"models_done": 28, "models_total": 28, "registry_frozen": True,
          "eval_rows": 40, "eval_rows_expected": 84, "eval_state": "CURRENT_RUN_RUNNING_PRE_GATE",
          "same_example_final_pass": False, "fairness_final_pass": False}
    r = m.concern_2_readiness(c2, ablation_root=None)
    assert r["eval"] == "CURRENT_RUN_RUNNING_PRE_GATE"
    assert "acceptance_note" in r


def test_concern_2_readiness_eval_complete_audits_required():
    m = _import_module()
    c2 = {"models_done": 28, "models_total": 28, "registry_frozen": True,
          "eval_rows": 84, "eval_rows_expected": 84,
          "eval_state": "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED",
          "same_example_final_pass": False, "fairness_final_pass": False}
    r = m.concern_2_readiness(c2, ablation_root=None)
    assert r["eval"] == "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED"
    assert "acceptance_note" in r


def test_concern_2_readiness_blocked_by_audit():
    m = _import_module()
    c2 = {"models_done": 28, "models_total": 28, "registry_frozen": True,
          "eval_rows": 84, "eval_rows_expected": 84,
          "eval_state": "BLOCKED_BY_AUDIT",
          "same_example_final_pass": True, "fairness_final_pass": False}
    r = m.concern_2_readiness(c2, ablation_root=None)
    assert r["eval"] == "BLOCKED_BY_AUDIT"
    assert "acceptance_note" in r


def test_audit_final_state_reads_final_flag(tmp_path):
    m = _import_module()
    p = tmp_path / "audit.json"
    p.write_text(json.dumps({"FINAL": False}))
    assert m._audit_final_state(p) == "PARTIAL"
    p.write_text(json.dumps({"FINAL": True}))
    assert m._audit_final_state(p) == "COMPLETE"
    assert m._audit_final_state(tmp_path / "nope.json") == "NOT_RUN"


def test_concern_3_readiness_already_running_when_session_active():
    m = _import_module()
    c3 = {"primary_rows": 18, "primary_rows_expected": 42}
    r = m.concern_3_readiness(c3, active_tmux=["distribution_shift_2h_continue"])
    assert r["resume"] == "ALREADY_RUNNING"


def test_concern_3_readiness_ready_when_no_session_active():
    m = _import_module()
    c3 = {"primary_rows": 18, "primary_rows_expected": 42}
    r = m.concern_3_readiness(c3, active_tmux=["evict_cross_family_resume"])
    assert r["resume"] == "READY"


def test_concern_4_readiness_reflects_gate():
    m = _import_module()
    c4 = {"smoke_artifacts_expected": 9, "smoke_artifacts_count": 9, "smoke_artifacts_missing": [],
          "timing_gate": "READY", "controlled_campaign_started": False}
    r = m.concern_4_readiness(c4)
    assert r == {"smoke": "COMPLETE", "controlled_timing": "READY"}
