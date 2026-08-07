"""Wrong-fold / fail-closed protection tests for
scripts/experiments/run_evict_value_v1_cross_family_eval.py.

These import the runner script as a module (it lives outside src/, so this
mirrors the pattern already used in tests/test_reviewer_fairness_common.py
for run_reviewer_fairness.py / generate_fairness_certificate.py).
"""

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


def _import_runner():
    import run_evict_value_v1_cross_family_eval as m
    return m


def test_ineligible_heavy_r1_artifact_rejected(tmp_path):
    m = _import_runner()
    fold = {"test_family": "brightkite", "model_output_path": "models/evict_value_wulver_v1_best.pkl"}
    with pytest.raises(m.WrongFoldError, match="ineligible artifact"):
        m._verify_fold_and_model(fold, tmp_path)


def test_ineligible_rejected_fair_v1_artifact_rejected(tmp_path):
    m = _import_runner()
    fold = {"test_family": "citibike", "model_output_path": "models/evict_value_v1_fair_v1.pkl"}
    with pytest.raises(m.WrongFoldError, match="ineligible artifact"):
        m._verify_fold_and_model(fold, tmp_path)


def test_wrong_fold_naming_mismatch_rejected(tmp_path):
    m = _import_runner()
    # brightkite's fold pointing at citibike's model path -- must never be
    # silently accepted, regardless of whether that file happens to exist.
    fold = {
        "test_family": "brightkite",
        "model_output_path": "models/evict_value_v1_cross_family_v1_citibike.pkl",
    }
    with pytest.raises(m.WrongFoldError, match="expected naming convention"):
        m._verify_fold_and_model(fold, tmp_path)


def test_missing_model_raises_not_silently_skipped(tmp_path, monkeypatch):
    m = _import_runner()
    monkeypatch.chdir(tmp_path)
    fold = {
        "test_family": "wiki2018",
        "model_output_path": "models/evict_value_v1_cross_family_v1_wiki2018.pkl",
        "dataset_output_root": str(tmp_path / "nonexistent"),
    }
    with pytest.raises(FileNotFoundError, match="Explicit artifact mode"):
        m._verify_fold_and_model(fold, tmp_path)


def test_test_trace_hash_mismatch_rejected(tmp_path, monkeypatch):
    m = _import_runner()
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "models" / "evict_value_v1_cross_family_v1_metakv.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"not a real model, just needs to exist")

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"item_id": "a"}\n')

    fold = {
        "test_family": "metakv",
        "model_output_path": "models/evict_value_v1_cross_family_v1_metakv.pkl",
        "test_trace_path": "trace.jsonl",
        "test_trace_sha256": "0" * 64,  # deliberately wrong
        "dataset_output_root": str(tmp_path / "nonexistent"),
    }
    with pytest.raises(m.WrongFoldError, match="test trace hash mismatch"):
        m._verify_fold_and_model(fold, tmp_path)


def test_all_seven_frozen_fold_manifests_exist_and_exclude_own_family():
    import json

    folds_dir = Path("configs/fair_cross_family_v1/folds")
    families = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
    for fam in families:
        fold = json.loads((folds_dir / f"{fam}.json").read_text())
        assert fold["test_family"] == fam
        assert fam not in fold["training_families"]
        assert fold["validation_family"] != fam
        assert fold["validation_family"] not in fold["training_families"]
        assert len(fold["training_families"]) == 5

        train_manifest_text = (folds_dir / f"{fam}_train_manifest.csv").read_text()
        assert fam not in train_manifest_text.split("\n")[0]  # sanity: header has no family col named after fam
        # The held-out family name must not appear as a trace_family value
        # in its own fold's training manifest.
        import csv as _csv
        import io as _io
        rows = list(_csv.DictReader(_io.StringIO(train_manifest_text)))
        assert all(r["trace_family"] != fam for r in rows)
