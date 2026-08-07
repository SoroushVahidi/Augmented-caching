"""Fixture tests for scripts/experiments/audit_supervision_objective_examples.py."""

from __future__ import annotations

import csv
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
    import audit_supervision_objective_examples as m
    return m


TRAIN5 = ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]


def _write_fold(tmp_path: Path, family="brightkite", validation="citibike", training=TRAIN5) -> None:
    folds_dir = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    fold = {"fold_id": f"cross_family_v1_{family}", "test_family": family,
            "validation_family": validation, "training_families": training}
    (folds_dir / f"{family}.json").write_text(json.dumps(fold))


def _write_manifest(tmp_path: Path, family: str, input_families, per_capacity) -> None:
    d = tmp_path / "data" / "derived" / "supervision_objective_ablation_v1" / family
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "trace_stats": [{"trace_family": f, "split": "train"} for f in input_families],
        "label_stats": {"per_capacity": per_capacity},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))


def _write_shards(tmp_path: Path, family: str, prefix: str, decision_ids_scalar, decision_ids_pairwise) -> None:
    scalar_dir = tmp_path / "data" / "derived" / "supervision_objective_ablation_v1" / family / "scalar"
    pairwise_dir = tmp_path / "data" / "derived" / "supervision_objective_ablation_v1" / family / "pairwise"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    with (scalar_dir / f"{prefix}.part0000.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["decision_id", "example_id"])
        w.writeheader()
        for did in decision_ids_scalar:
            w.writerow({"decision_id": did, "example_id": f"{did}|page1"})

    with (pairwise_dir / f"{prefix}.part0000.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["decision_id", "pair_id"])
        w.writeheader()
        for did in decision_ids_pairwise:
            w.writerow({"decision_id": did, "pair_id": f"{did}|page1|page2"})


def test_not_built_fold_reports_not_built(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    report = m.audit_fold("brightkite", max_shards=3)
    assert report["status"] == "NOT_BUILT"


def test_held_out_family_leak_detected(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    _write_manifest(tmp_path, "brightkite", input_families=["brightkite"] + TRAIN5 + ["citibike"], per_capacity={})
    report = m.audit_fold("brightkite", max_shards=3)
    assert "FAIL" in report["checks"]["held_out_family_isolation"]
    assert report["status"] == "FAIL"


def test_clean_fold_with_matching_pairwise_passes(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    per_cap = {"cap32": {"scalar_rows": 100, "pairwise_rows": 50, "decisions": 10,
                          "next_arrival_censored_count": 5, "reuse_distance_censored_count": 3}}
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5 + ["citibike"], per_capacity=per_cap)
    _write_shards(tmp_path, "brightkite", "brightkite_50k__cap32",
                  decision_ids_scalar=["d1", "d2", "d3"], decision_ids_pairwise=["d1", "d2"])
    report = m.audit_fold("brightkite", max_shards=3)
    assert report["status"] == "PASS"
    subset = report["per_capacity_subsets"]["cap32"]
    assert subset["common_candidate_universe"] == 100
    assert subset["objective_next_arrival_finite_subset"] == 95
    assert subset["objective_reuse_distance_finite_subset"] == 97


def test_pairwise_decision_id_not_in_scalar_detected(tmp_path, monkeypatch):
    m = _import_module()
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path)
    per_cap = {"cap32": {"scalar_rows": 10, "pairwise_rows": 5, "decisions": 2,
                          "next_arrival_censored_count": 0, "reuse_distance_censored_count": 0}}
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5 + ["citibike"], per_capacity=per_cap)
    # pairwise shard references a decision_id ("d99") absent from its
    # matching scalar shard -- must be caught, not silently ignored.
    _write_shards(tmp_path, "brightkite", "brightkite_50k__cap32",
                  decision_ids_scalar=["d1", "d2"], decision_ids_pairwise=["d1", "d99"])
    report = m.audit_fold("brightkite", max_shards=3)
    assert report["status"] == "FAIL"
    assert "FAIL" in report["checks"]["pairwise_candidate_group_consistency"]
    assert report["pairwise_spot_check"]["mismatches"]


def test_main_blocked_without_partial_audit_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for fam in ["brightkite", "citibike"]:
        _write_fold(tmp_path, family=fam, validation="x", training=["a", "b", "c", "d", "e"])
    # Neither fold has a manifest -> NOT_BUILT -> partial.
    m = _import_module()
    monkeypatch.setattr(sys, "argv", ["audit_supervision_objective_examples.py", "--families", "brightkite,citibike"])
    with pytest.raises(SystemExit) as exc_info:
        m.main()
    assert exc_info.value.code == 1
    out_path = tmp_path / "analysis" / "supervision_objective_ablation_v1" / "same_example_audit.json"
    assert not out_path.exists()


def test_main_partial_audit_writes_final_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_fold(tmp_path, family="brightkite")
    per_cap = {"cap32": {"scalar_rows": 10, "pairwise_rows": 0, "decisions": 2,
                          "next_arrival_censored_count": 0, "reuse_distance_censored_count": 0}}
    _write_manifest(tmp_path, "brightkite", input_families=TRAIN5 + ["citibike"], per_capacity=per_cap)

    m = _import_module()
    monkeypatch.setattr(sys, "argv", [
        "audit_supervision_objective_examples.py", "--families", "brightkite", "--partial-audit",
    ])
    m.main()
    out_path = tmp_path / "analysis" / "supervision_objective_ablation_v1" / "same_example_audit.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["FINAL"] is False
