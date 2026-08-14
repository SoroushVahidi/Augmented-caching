from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    build_candidate_rows_for_full_cache_state,
    iter_multi_label_candidate_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load("common_v2", "scripts/experiments/run_common_model_objective_control_v2.py")
reducer = _load("common_v2_reducer", "scripts/experiments/reduce_common_model_objective_control_v2.py")


def _old_selected_rows_from_requests(requests, cap, n, trace_name, trace_family, cfg):
    ids = []
    for row in iter_multi_label_candidate_rows(requests, cap, trace_name, trace_family, cfg):
        if row["decision_id"] not in ids:
            ids.append(row["decision_id"])
        if len(ids) >= n:
            break
    return list(
        iter_multi_label_candidate_rows(
            requests,
            cap,
            trace_name,
            trace_family,
            cfg,
            selected_decision_ids=set(ids),
        )
    )


def test_pairwise_orientation_uses_label_not_candidate_id():
    pair = {"label_i_preferred": 1}
    for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
        pair[f"i_{col}"] = 0.0
        pair[f"j_{col}"] = 10.0
    A, B = v2.orient_pairwise_rows_for_eviction_score([pair])
    assert np.all(A == 10.0)
    assert np.all(B == 0.0)

    swapped = {"label_i_preferred": 0}
    for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
        swapped[f"i_{col}"] = 10.0
        swapped[f"j_{col}"] = 0.0
    A2, B2 = v2.orient_pairwise_rows_for_eviction_score([swapped])
    assert np.array_equal(A, A2)
    assert np.array_equal(B, B2)


def test_corrected_pairwise_learns_eviction_score_direction_on_synthetic_pairs():
    pairs = []
    for val in np.linspace(0.0, 1.0, 60):
        p = {"label_i_preferred": 1}
        for idx, col in enumerate(EVICT_VALUE_V1_FEATURE_COLUMNS):
            p[f"i_{col}"] = float(val)
            p[f"j_{col}"] = float(val + 0.5 + idx * 0.01)
        pairs.append(p)
    A, B = v2.orient_pairwise_rows_for_eviction_score(pairs)
    model = v2.CommonScorer(hidden=8, lr=0.02, epochs=200, l2=0.0, seed=0)
    model.fit(A, pairs=(A, B), mode="pairwise")
    assert float(np.mean(model.score(A) > model.score(B))) == 1.0


def test_scorer_cached_once_preserves_victim_and_tie_break():
    rows = [{"candidate_page_id": str(i), **{c: float(i) for c in EVICT_VALUE_V1_FEATURE_COLUMNS}} for i in range(5)]
    calls = {"old": 0, "new": 0}

    def scorer_old(rs):
        calls["old"] += 1
        return {str(r["candidate_page_id"]): float(int(r["candidate_page_id"]) % 3) for r in rs}

    ids = [str(r["candidate_page_id"]) for r in rows]
    old_victim = min(ids, key=lambda p: (scorer_old(rows)[p], ids.index(p)))

    class Dummy:
        def score(self, X):
            calls["new"] += 1
            return np.asarray([float(i % 3) for i in range(len(X))])

    scores = v2.score_rows_once(rows, Dummy())
    new_victim = v2.choose_from_scores(rows, scores, "min")
    assert calls["old"] == len(rows)
    assert calls["new"] == 1
    assert old_victim == new_victim


def test_selected_rows_one_pass_matches_old_two_pass_exactly():
    requests, _ = build_requests_from_lists(page_ids=["a", "b", "c", "a", "d", "b", "e", "a", "f", "b", "g"])
    cfg = ObjectiveAblationConfig(horizon=4)
    old = _old_selected_rows_from_requests(requests, 2, 3, "toy", "fam", cfg)
    new = v2.selected_rows_from_requests(requests, 2, 3, "toy", "fam", cfg)
    assert json.dumps(new, sort_keys=True, default=str) == json.dumps(old, sort_keys=True, default=str)
    assert v2.selected_decision_ids(new) == v2.selected_decision_ids(old)


def test_feature_only_rows_match_full_rows_for_feature_columns():
    requests, _ = build_requests_from_lists(page_ids=["a", "b", "c", "a", "d"])
    cfg = ObjectiveAblationConfig(horizon=4)
    order = ["a", "b"]
    req = requests[2]
    kwargs = dict(
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cache_order=order,
        bucket_by_page={},
        confidence_by_page={},
        recent_req_hist=["a", "b"],
        recent_hit_hist=[],
    )
    full = build_candidate_rows_for_full_cache_state(requests=requests, request_index=2, cfg=cfg, **kwargs)
    feat = v2.build_candidate_feature_rows_for_full_cache_state(request=req, request_index=2, **kwargs)
    assert [r["candidate_page_id"] for r in feat] == [r["candidate_page_id"] for r in full]
    for frow, row in zip(feat, full):
        for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
            assert frow[col] == row[col]
        assert "eviction_loss_label" not in frow
        assert "next_arrival_label_censored" not in frow


def test_config_protocol_invariants():
    cfg = json.loads((ROOT / "configs/common_model_objective_control_v2.json").read_text())
    assert cfg["families"] == v2.FAMILIES
    assert cfg["capacities"] == v2.CAPACITIES
    assert list(cfg["objectives"]) == v2.OBJECTIVES
    assert cfg["architecture"]["hidden_units"] == 8
    assert cfg["seed"] == 0
    for held in cfg["families"]:
        fold = json.loads((ROOT / "configs/fair_cross_family_v1/folds" / f"{held}.json").read_text())
        assert held not in fold["training_families"]
        assert held != fold["validation_family"]


def test_resume_idempotence_skips_existing_summary(tmp_path):
    out = tmp_path / "unit"
    out.mkdir()
    summary = out / "summary.json"
    summary.write_text(json.dumps({"status": "COMPLETE", "rows": []}))
    args = type("Args", (), {"family": "brightkite", "capacity": 32, "out": out, "force": False})
    assert v2.run_unit(args, {"families": ["brightkite"], "capacities": [32]}) == summary


def test_reducer_requires_all_units(tmp_path):
    with pytest.raises(FileNotFoundError):
        reducer.reduce(tmp_path)


def test_reducer_accepts_complete_synthetic_campaign(tmp_path):
    for family in reducer.FAMILIES:
        for capacity in reducer.CAPACITIES:
            unit = tmp_path / "units" / f"{family}_cap{capacity}"
            unit.mkdir(parents=True)
            rows = [
                {
                    "objective": objective,
                    "held_out_family": family,
                    "capacity": capacity,
                    "misses": 1,
                    "miss_ratio": 0.1,
                    "delta_vs_lru": None,
                    "validation_mean_regret": 0.0,
                    "model_sha256": "m",
                    "trace_sha256": f"trace-{family}",
                    "seed": 0,
                    "diagnostics_count": 0,
                    "victim_sequence_sha256": "v",
                }
                for objective in reducer.OBJECTIVES
            ]
            data = {
                "status": "COMPLETE",
                "rows": rows,
                "metadata": {
                    "protocol_id": "common_model_objective_control_v2",
                    "source_head": "abc",
                    "family": family,
                    "capacity": capacity,
                    "trace_sha256": f"trace-{family}",
                },
            }
            (unit / "summary.json").write_text(json.dumps(data))
    reducer.reduce(tmp_path)
    assert json.loads((tmp_path / "integrity_audit.json").read_text())["status"] == "PASS"
    assert json.loads((tmp_path / "completion_manifest.json").read_text())["rows"] == 84


def test_slurm_mapping_unique_21_units():
    families = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
    capacities = [32, 64, 128]
    units = []
    for task in range(21):
        units.append((families[task // 3], capacities[task % 3]))
    assert len(units) == 21
    assert len(set(units)) == 21
    assert set(units) == {(f, c) for f in families for c in capacities}
