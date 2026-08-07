"""Tests for scripts/experiments/run_distribution_shift_ablation.py.

Follows the established pattern (tests/test_evict_value_v1_cross_family_eval.py):
import the runner script as a module and unit-test its functions directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EXPERIMENTS_DIR = str(Path("scripts/experiments").resolve())


@pytest.fixture(autouse=True)
def _scripts_on_path():
    inserted = _EXPERIMENTS_DIR not in sys.path
    if inserted:
        sys.path.insert(0, _EXPERIMENTS_DIR)
    yield
    if inserted and _EXPERIMENTS_DIR in sys.path:
        sys.path.remove(_EXPERIMENTS_DIR)


def _import_runner():
    import run_distribution_shift_ablation as m
    return m


# ---------------------------------------------------------------------
# Reservoir-sampling row cap (resource-safety: bounds memory regardless
# of trace length / capacity, preventing the uncapped-in-memory-list OOM
# failure mode observed in the canonical single-objective pipeline).
# ---------------------------------------------------------------------

def test_reservoir_sample_bounds_size_and_is_deterministic():
    m = _import_runner()
    stream1 = iter({"i": i} for i in range(100000))
    r1 = m._reservoir_sample_stream(stream1, max_rows=1000, seed=0)
    assert len(r1) == 1000
    stream2 = iter({"i": i} for i in range(100000))
    r2 = m._reservoir_sample_stream(stream2, max_rows=1000, seed=0)
    assert r1 == r2


def test_reservoir_sample_differs_across_seeds():
    m = _import_runner()
    stream_a = iter({"i": i} for i in range(100000))
    stream_b = iter({"i": i} for i in range(100000))
    ra = m._reservoir_sample_stream(stream_a, max_rows=1000, seed=0)
    rb = m._reservoir_sample_stream(stream_b, max_rows=1000, seed=1)
    assert ra != rb


def test_reservoir_sample_keeps_everything_when_stream_smaller_than_cap():
    m = _import_runner()
    stream = iter({"i": i} for i in range(50))
    r = m._reservoir_sample_stream(stream, max_rows=1000, seed=0)
    assert len(r) == 50


# ---------------------------------------------------------------------
# Time-budget controller
# ---------------------------------------------------------------------

def test_time_budget_allows_first_unit_unconditionally():
    m = _import_runner()
    budget = m.TimeBudget(max_wall_hours=9.0)
    assert budget.can_start_new_unit() is True


def test_time_budget_stops_when_remaining_below_average_cost():
    m = _import_runner()
    budget = m.TimeBudget(max_wall_hours=0.001)  # ~3.6s total budget
    budget.record_unit(10.0)  # a single unit already cost 10s, more than the whole budget
    assert budget.can_start_new_unit() is False


def test_time_budget_avg_unit_cost_tracks_recorded_units():
    m = _import_runner()
    budget = m.TimeBudget(max_wall_hours=9.0)
    budget.record_unit(10.0)
    budget.record_unit(20.0)
    assert budget.avg_unit_cost() == pytest.approx(15.0)


# ---------------------------------------------------------------------
# Fail-closed: protocol id / trace hash
# ---------------------------------------------------------------------

def test_main_rejects_wrong_protocol_id(tmp_path, monkeypatch, capsys):
    m = _import_runner()
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps({"protocol_id": "something_else"}))
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(bad_config)])
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        m.main()


def test_run_fold_rejects_trace_hash_mismatch(tmp_path, monkeypatch):
    m = _import_runner()
    monkeypatch.chdir(tmp_path)
    folds_dir = tmp_path / "configs" / "fair_cross_family_v1" / "folds"
    folds_dir.mkdir(parents=True)
    fold = {
        "fold_id": "cross_family_v1_brightkite", "test_family": "brightkite",
        "test_trace_path": "trace.jsonl", "test_trace_sha256": "0" * 64,
        "test_trace_name": "brightkite_50k",
        "training_families": ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"],
        "validation_family": "citibike",
        "train_manifest": str(folds_dir / "brightkite_train_manifest.csv"),
    }
    (folds_dir / "brightkite.json").write_text(json.dumps(fold))
    split_map = {"citibike": "val", "cloudphysics": "train", "metacdn": "train",
                 "metakv": "train", "twemcache": "train", "wiki2018": "train"}
    (folds_dir / "brightkite_family_split_map.json").write_text(json.dumps(split_map))
    with open(folds_dir / "brightkite_train_manifest.csv", "w", newline="") as fh:
        import csv
        w = csv.writer(fh)
        w.writerow(["path", "trace_name", "trace_family"])
    (tmp_path / "trace.jsonl").write_text('{"item_id": "a"}\n')

    with pytest.raises(ValueError, match="trace hash mismatch"):
        m._run_fold("brightkite", [32], tmp_path / "models", 0, None, None, None, tmp_path / "state.json")


# ---------------------------------------------------------------------
# Fold isolation: held-out family absent from own train manifest
# ---------------------------------------------------------------------

def test_all_seven_folds_train_manifest_excludes_own_family():
    m = _import_runner()
    for family in m.FAMILIES:
        recs = m._load_train_manifest(family)
        assert all(r["trace_family"] != family for r in recs)
        split_map = m._load_split_map(family)
        assert family not in split_map


# ---------------------------------------------------------------------
# Section 23 sanity experiment: DAgger states differ from off-policy
# states, and get independent ground-truth labels.
# ---------------------------------------------------------------------

def test_dagger_iteration1_states_differ_from_off_policy_and_get_independent_labels():
    from sklearn.linear_model import LinearRegression
    import numpy as np

    from lafc.distribution_shift_ablation import DistributionShiftConfig, iter_candidate_rows_with_behavior_policy
    from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
    from lafc.evict_value_model_v1 import EvictValueV1Model
    from lafc.simulator.request_trace import build_requests_from_lists

    # Construct a trace where LRU and a "prefer evicting high recency
    # rank" model trajectory are structurally likely to diverge: cyclic
    # reuse pattern with an occasional novel item forcing an eviction
    # among a heterogeneous-age candidate set.
    page_ids = (["a", "b", "c", "d", "e"] * 4) + ["z1"] + (["a", "b", "c", "d", "e"] * 4) + ["z2"] + (["f", "g", "h", "i", "j"] * 4)
    reqs, _ = build_requests_from_lists(page_ids=page_ids)
    cfg = DistributionShiftConfig(horizon=4)

    d0 = list(iter_candidate_rows_with_behavior_policy(reqs, 5, "t", "fam", cfg, behavior_model=None, behavior_policy_name="lru"))
    assert d0

    # A simple frozen model with real coefficients (not all-zero) so its
    # argmin decision can plausibly differ from LRU's candidates[0].
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(EVICT_VALUE_V1_FEATURE_COLUMNS)))
    y = -X[:, EVICT_VALUE_V1_FEATURE_COLUMNS.index("candidate_recency_rank")]
    est = LinearRegression().fit(X, y)
    model = EvictValueV1Model(model_name="toy", estimator=est, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS))

    d1 = list(iter_candidate_rows_with_behavior_policy(reqs, 5, "t", "fam", cfg, behavior_model=model, behavior_policy_name="learned"))
    assert d1

    d0_ids = {(r["decision_id"].split("|pol=")[0], r["candidate_page_id"]) for r in d0}
    d1_ids = {(r["decision_id"].split("|pol=")[0], r["candidate_page_id"]) for r in d1}
    # Mechanically verify D0 != D1 (the states visited genuinely differ
    # once trajectories diverge) -- not asserting a specific difference,
    # just that state generation is not vacuously identical.
    assert d0_ids != d1_ids or len(d0) != len(d1)

    # Independent labeling: D1's eviction_loss_label values must not be a
    # copy of the model's own prediction on that row (the forbidden
    # circularity) -- true labels are non-negative integers (miss counts,
    # capped by the horizon), the model's continuous regression output is
    # not; if labeling were circular every row would match its prediction
    # exactly, which does not hold here (checked in aggregate, since a
    # handful of individual near-zero coincidences are expected and fine).
    feat_cols = list(EVICT_VALUE_V1_FEATURE_COLUMNS)
    preds = [model.predict_loss_one({c: row[c] for c in feat_cols}) for row in d1]
    labels = [row["eviction_loss_label"] for row in d1]
    assert all(float(lbl).is_integer() and 0 <= lbl <= cfg.horizon for lbl in labels)
    n_matching = sum(1 for lbl, p in zip(labels, preds) if abs(lbl - p) < 1e-12)
    assert n_matching < len(labels)  # not every row trivially matches its own prediction
