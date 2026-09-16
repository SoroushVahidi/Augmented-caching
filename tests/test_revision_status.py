"""Fixture tests for scripts/validation/revision_status.py."""

from __future__ import annotations

import csv
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
    import revision_status as m
    return m


def test_parse_worktree_porcelain():
    m = _import_module()
    porcelain = (
        "worktree /home/soroush/Augmented-caching\n"
        "HEAD abc123\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /home/soroush/Augmented-caching-fairness\n"
        "HEAD def456\n"
        "branch refs/heads/feat/reviewer-fairness-protocol\n"
    )

    class FakeResult:
        stdout = porcelain

    import subprocess as real_subprocess
    orig_run = real_subprocess.run

    def fake_run(cmd, **kwargs):
        if "worktree" in cmd:
            return FakeResult()
        return orig_run(cmd, **kwargs)

    import unittest.mock
    with unittest.mock.patch.object(m.subprocess, "run", side_effect=fake_run):
        worktrees = m.find_worktrees(Path("/fake"))
    assert worktrees["main"] == Path("/home/soroush/Augmented-caching")
    assert worktrees["feat/reviewer-fairness-protocol"] == Path("/home/soroush/Augmented-caching-fairness")


def _write_csv(path: Path, n_data_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b"])
        for i in range(n_data_rows):
            w.writerow([i, i])


def test_concern_1_status_counts_datasets_and_models(tmp_path):
    m = _import_module()
    data_root = tmp_path / "data" / "derived" / "evict_value_v1_cross_family_v1"
    for f in ["brightkite", "citibike"]:
        (data_root / f).mkdir(parents=True)
        (data_root / f / "manifest.json").write_text("{}")
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "evict_value_v1_cross_family_v1_brightkite.pkl").write_bytes(b"x")

    status = m.concern_1_status(tmp_path)
    assert status["datasets_done"] == 2
    assert status["models_done"] == 1
    assert status["registry_frozen"] is False
    assert status["status"] == "RUNNING"


def test_concern_1_status_ready_for_next_stage_when_all_models_done_unfrozen(tmp_path):
    m = _import_module()
    models_root = tmp_path / "models"
    models_root.mkdir()
    for f in m.FAMILIES:
        (models_root / f"evict_value_v1_cross_family_v1_{f}.pkl").write_bytes(b"x")
    status = m.concern_1_status(tmp_path)
    assert status["models_done"] == 7
    assert status["status"] == "TRAINING COMPLETE — READY FOR NEXT STAGE"


def test_concern_2_status_counts_across_4_objectives(tmp_path):
    m = _import_module()
    models_root = tmp_path / "models" / "supervision_objective_ablation_v1"
    for o in m.OBJECTIVES:
        (models_root / o).mkdir(parents=True)
    (models_root / "objective_eviction_loss" / "brightkite.pkl").write_bytes(b"x")
    (models_root / "objective_pairwise" / "brightkite.pkl").write_bytes(b"x")
    status = m.concern_2_status(tmp_path)
    assert status["models_done"] == 2
    assert status["models_total"] == 28


def test_concern_3_status_reads_campaign_state(tmp_path):
    m = _import_module()
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "policy_comparison.csv", 18)
    (out_dir / "campaign_state.json").write_text(json.dumps({"completed_folds": ["brightkite", "citibike", "cloudphysics"]}))
    status = m.concern_3_status(tmp_path)
    assert status["primary_rows"] == 18
    assert status["folds_complete"] == 3
    assert status["status"] == "STOPPED_CLEANLY_PARTIAL"


def test_concern_3_status_complete_when_42_rows(tmp_path):
    m = _import_module()
    out_dir = tmp_path / "analysis" / "distribution_shift_ablation_v1"
    out_dir.mkdir(parents=True)
    _write_csv(out_dir / "policy_comparison.csv", 42)
    status = m.concern_3_status(tmp_path)
    assert status["status"] == "COMPLETE"


def test_count_csv_rows_missing_file_is_zero(tmp_path):
    m = _import_module()
    assert m._count_csv_rows(tmp_path / "nope.csv") == 0


# ---------------------------------------------------------------------------
# Concern 2 explicit evaluation state machine (Reviewer Concern 2 fix)
# ---------------------------------------------------------------------------

FAM = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
OBJS = ["objective_eviction_loss", "objective_next_arrival", "objective_reuse_distance", "objective_pairwise"]


def _make_c2_root(tmp_path: Path, *, n_models=28, n_eval_rows=0, audit_payloads=None) -> Path:
    m = _import_module()
    data_root = tmp_path / "data" / "derived" / "supervision_objective_ablation_v1"
    models_root = tmp_path / "models" / "supervision_objective_ablation_v1"
    out_dir = tmp_path / "analysis" / "supervision_objective_ablation_v1"
    for fam in FAM:
        (data_root / fam).mkdir(parents=True)
        (data_root / fam / "manifest.json").write_text("{}")
    for obj in OBJS:
        (models_root / obj).mkdir(parents=True)
        for fam in FAM:
            if n_models >= OBJS.index(obj) * len(FAM) + FAM.index(fam) + 1:
                (models_root / obj / f"{fam}.pkl").write_bytes(b"x")

    registry = {
        "MODEL_SELECTION_FROZEN": n_models == 28,
        "protocol_id": "supervision_objective_ablation_v1",
        "expected_model_count": 28,
        "actual_model_count": n_models,
        "records": [
            {"objective": obj, "held_out_family": fam}
            for obj in OBJS for fam in FAM
        ][:n_models],
    }
    (out_dir).mkdir(parents=True)
    (out_dir / "model_registry.json").write_text(json.dumps(registry))

    if n_eval_rows:
        import csv as csv_mod
        with (out_dir / "policy_comparison.csv").open("w", newline="") as fh:
            w = csv_mod.DictWriter(fh, fieldnames=["objective", "held_out_family", "capacity", "status", "miss_ratio"])
            w.writeheader()
            for i in range(n_eval_rows):
                obj = OBJS[i % len(OBJS)]
                fam = FAM[(i // len(OBJS)) % len(FAM)]
                cap = [32, 64, 128][i % 3]
                w.writerow({"objective": obj, "held_out_family": fam, "capacity": cap,
                            "status": "ok", "miss_ratio": 0.5})
    else:
        (out_dir / "policy_comparison.csv").write_text("objective,held_out_family,capacity,status,miss_ratio\n")

    for name, payload in (audit_payloads or {}).items():
        (out_dir / name).write_text(json.dumps(payload))

    return tmp_path


def _patch_evaluator_active(m, active: bool):
    import unittest.mock
    return unittest.mock.patch.object(m, "_is_process_running", return_value=active)


def test_c2_state_running_pre_gate_40_of_84_active(tmp_path):
    m = _import_module()
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=40)
    with _patch_evaluator_active(m, True):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "CURRENT_RUN_RUNNING_PRE_GATE"
    assert s["status"] == "CURRENT_RUN_RUNNING_PRE_GATE"


def test_c2_state_partial_stopped_pre_gate_40_of_84_absent(tmp_path):
    m = _import_module()
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=40)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "CURRENT_RUN_PARTIAL_STOPPED_PRE_GATE"
    assert s["status"] == "CURRENT_RUN_PARTIAL_STOPPED_PRE_GATE"


def test_c2_state_eval_complete_audits_required_84_of_84(tmp_path):
    m = _import_module()
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=84)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED"
    assert s["status"] == "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED"


def test_c2_state_complete_and_audited_84_with_both_final_pass(tmp_path):
    m = _import_module()
    audits = {
        "same_example_audit.json": {"FINAL": True, "overall": "PASS"},
        "fairness_audit.json": {"FINAL": True, "overall": "PASS"},
    }
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=84, audit_payloads=audits)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "COMPLETE_AND_AUDITED"
    assert s["status"] == "COMPLETE_AND_AUDITED"


def test_c2_state_blocked_by_audit_84_with_fairness_fail(tmp_path):
    m = _import_module()
    audits = {
        "same_example_audit.json": {"FINAL": True, "overall": "PASS"},
        "fairness_audit.json": {"FINAL": True, "overall": "FAIL"},
    }
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=84, audit_payloads=audits)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "BLOCKED_BY_AUDIT"
    assert s["status"] == "BLOCKED_BY_AUDIT"


def test_c2_state_invalid_overflow_85_rows(tmp_path):
    m = _import_module()
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=85)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "INVALID_OVERFLOW"
    assert s["status"] == "FAILED"


def test_c2_state_invalid_duplicate_keys(tmp_path):
    m = _import_module()
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=0)
    import csv as csv_mod
    out = root / "analysis" / "supervision_objective_ablation_v1" / "policy_comparison.csv"
    # 83 unique rows + one exact duplicate of the first = 84 rows total.
    with out.open("w", newline="") as fh:
        w = csv_mod.writer(fh)
        w.writerow(["objective", "held_out_family", "capacity", "status", "miss_ratio"])
        rows = []
        for i in range(83):
            obj = OBJS[i % len(OBJS)]
            fam = FAM[(i // len(OBJS)) % len(FAM)]
            cap = [32, 64, 128][i % 3]
            rows.append([obj, fam, cap, "ok", 0.5])
        rows.insert(1, list(rows[0]))  # duplicate of row 1
        w.writerows(rows)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_rows"] == 84
    assert s["eval_duplicate_keys"] >= 1
    assert s["eval_state"] == "INVALID_DUPLICATE_KEYS"
    assert s["status"] == "FAILED"


def test_c2_state_training_complete_audits_not_run_no_eval(tmp_path):
    m = _import_module()
    root = _make_c2_root(tmp_path, n_models=28, n_eval_rows=0)
    with _patch_evaluator_active(m, False):
        s = m.concern_2_status(root)
    assert s["eval_state"] == "NOT_STARTED"
    assert s["status"] == "TRAINING_COMPLETE_AUDITS_NOT_RUN"


def test_c4_authoritative_smoke_artifact_list_is_9():
    m = _import_module()
    assert len(m.C4_SMOKE_ARTIFACTS) == 9
    assert m.C4_SMOKE_ARTIFACTS == [
        "exact_optimization_equivalence.json",
        "profiler_breakdown.csv",
        "selective_invocation.csv",
        "topk_tradeoff.csv",
        "model_complexity_tradeoff.csv",
        "break_even_miss_cost.csv",
        "miss_cost_sweep.csv",
        "weighted_cost.csv",
        "pareto_frontier.csv",
    ]
