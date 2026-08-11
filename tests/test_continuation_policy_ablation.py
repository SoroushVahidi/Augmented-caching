from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from lafc.continuation_policy_ablation import (
    ContinuationAblationConfig,
    ContinuationState,
    FrozenPi1Provenance,
    build_decision_aligned_continuation_rows,
    load_frozen_pi1_from_registry,
    simulate_pi1_continuation_misses,
    train_pi2_from_c2_labels,
)
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from scripts.experiments.run_continuation_policy_causal_ablation_smoke import _misses_for_policy


class LruLikeModel:
    def predict_loss_batch(self, rows):
        return [float(r["candidate_recency_rank"]) for r in rows]


class MostRecentModel:
    def predict_loss_batch(self, rows):
        return [-float(r["candidate_recency_rank"]) for r in rows]


class TieModel:
    def predict_loss_batch(self, rows):
        return [0.0 for _ in rows]


class RecordingModel:
    def __init__(self):
        self.calls = []

    def predict_loss_batch(self, rows):
        self.calls.append([dict(r) for r in rows])
        return [0.0 for _ in rows]


def _prov(**kwargs) -> FrozenPi1Provenance:
    base = {
        "held_out_family": "heldout",
        "validation_family": "val",
        "training_families": ("train_a", "train_b"),
        "model_path": "models/pi1.pkl",
        "model_sha256": "a" * 64,
        "registry_path": "registry.json",
        "registry_sha256": "b" * 64,
    }
    base.update(kwargs)
    return FrozenPi1Provenance(**base)


def _rows(page_ids, model, capacity=3, horizon=4):
    reqs, _pages = build_requests_from_lists(page_ids=page_ids)
    return build_decision_aligned_continuation_rows(
        requests=reqs,
        capacity=capacity,
        trace_name="toy",
        trace_family="synthetic",
        cfg=ContinuationAblationConfig(horizon=horizon),
        pi1_model=model,
        pi1_provenance=_prov(),
    )


def test_lru_continuation_and_lru_equivalent_pi1_labels_match():
    rows = _rows(["a", "b", "c", "d", "a", "e", "b", "f", "c", "g"], LruLikeModel(), capacity=3)
    assert rows
    assert all(float(r["c1_label"]) == float(r["c2_label"]) for r in rows)
    assert all(float(r["label_delta"]) == 0.0 for r in rows)


def test_pi1_continuation_can_change_labels_when_pi1_differs_from_lru():
    rows = _rows(
        ["a", "b", "c", "d", "a", "b", "e", "c", "f", "d", "a", "g"],
        MostRecentModel(),
        capacity=3,
        horizon=4,
    )
    assert rows
    assert any(float(r["c1_label"]) != float(r["c2_label"]) for r in rows)


def test_forced_candidate_action_is_applied_before_continuation():
    reqs, _pages = build_requests_from_lists(page_ids=["c", "a"])
    state = ContinuationState(
        order=collections.OrderedDict([("a", None), ("b", None)]),
        bucket_by_page={},
        confidence_by_page={},
        recent_req_hist=collections.deque(["a", "b"], maxlen=64),
        recent_hit_hist=collections.deque([], maxlen=64),
    )
    cfg = ContinuationAblationConfig(horizon=1)

    evict_a_loss = simulate_pi1_continuation_misses(
        pre_decision_state=state,
        forced_candidate="a",
        incoming_request=reqs[0],
        future_reqs=reqs[1:],
        capacity=2,
        model=LruLikeModel(),
        cfg=cfg,
    )
    evict_b_loss = simulate_pi1_continuation_misses(
        pre_decision_state=state,
        forced_candidate="b",
        incoming_request=reqs[0],
        future_reqs=reqs[1:],
        capacity=2,
        model=LruLikeModel(),
        cfg=cfg,
    )

    assert evict_a_loss == 1
    assert evict_b_loss == 0


def test_future_pi1_decisions_are_recomputed_from_updated_state():
    reqs, _pages = build_requests_from_lists(page_ids=["c", "b", "d"])
    model = RecordingModel()
    state = ContinuationState(
        order=collections.OrderedDict([("a", None), ("b", None)]),
        bucket_by_page={},
        confidence_by_page={},
        recent_req_hist=collections.deque(["a", "b"], maxlen=64),
        recent_hit_hist=collections.deque([], maxlen=64),
    )

    simulate_pi1_continuation_misses(
        pre_decision_state=state,
        forced_candidate="a",
        incoming_request=reqs[0],
        future_reqs=reqs[1:],
        capacity=2,
        model=model,
        cfg=ContinuationAblationConfig(horizon=2),
    )

    assert len(model.calls) == 1
    call = model.calls[0]
    by_rank = {int(r["candidate_recency_rank"]): r for r in call}
    assert by_rank[0]["candidate_is_lru_victim"] == 1.0
    assert by_rank[1]["recent_candidate_hit_rate"] > 0.0


def test_same_example_decision_alignment():
    rows = _rows(["a", "b", "c", "d", "a", "e", "b", "f", "c", "g"], MostRecentModel(), capacity=3)
    pairs = {(r["decision_id"], r["candidate_id"]) for r in rows}
    assert len(pairs) == len(rows)
    assert all({"c1_label", "c2_label", "label_delta", "pi1_hash"}.issubset(r.keys()) for r in rows)


def test_c0_lru_smoke_metric_matches_plain_lru_replay():
    reqs, pages = build_requests_from_lists(page_ids=["a", "b", "a", "c", "b", "a"])

    direct = run_policy(LRUPolicy(), reqs, pages, capacity=2)
    smoke_metric = _misses_for_policy(LRUPolicy(), reqs, pages, capacity=2)

    assert direct.total_misses == 5
    assert smoke_metric["misses"] == float(direct.total_misses)
    assert smoke_metric["requests"] == float(len(reqs))
    assert smoke_metric["miss_ratio"] == float(direct.total_misses / len(reqs))


def test_deterministic_tie_behavior_matches_lru():
    rows = _rows(["a", "b", "c", "d", "a", "e", "b", "f", "c", "g"], TieModel(), capacity=3)
    assert rows
    assert all(float(r["c1_label"]) == float(r["c2_label"]) for r in rows)


def test_frozen_pi1_hash_provenance_required():
    reqs, _pages = build_requests_from_lists(page_ids=["a", "b", "c", "d", "a", "e", "b"])
    with pytest.raises(ValueError, match="model_sha256"):
        build_decision_aligned_continuation_rows(
            requests=reqs,
            capacity=3,
            trace_name="toy",
            trace_family="synthetic",
            cfg=ContinuationAblationConfig(horizon=2),
            pi1_model=LruLikeModel(),
            pi1_provenance=_prov(model_sha256=""),
        )


def test_held_out_model_leakage_gate(tmp_path: Path):
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    (folds_dir / "heldout.json").write_text(
        json.dumps(
            {
                "fold_id": "fold-heldout",
                "training_families": ["heldout", "train_b"],
                "validation_family": "val",
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "MODEL_SELECTION_FROZEN": True,
        "records": [
            {
                "objective": "objective_eviction_loss",
                "held_out_family": "heldout",
                "fold_id": "fold-heldout",
                "training_families": ["heldout", "train_b"],
                "validation_family": "val",
                "model_artifact_path": str(tmp_path / "models" / "objective_eviction_loss" / "heldout.pkl"),
                "model_artifact_sha256": "0" * 64,
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="held-out leakage"):
        load_frozen_pi1_from_registry(
            registry_path=registry_path,
            held_out_family="heldout",
            folds_dir=folds_dir,
        )


def test_pi2_training_uses_c2_labels_without_pi1_model_mutation():
    rows = _rows(
        ["a", "b", "c", "d", "a", "b", "e", "c", "f", "d", "a", "g", "h", "a", "b", "i"],
        MostRecentModel(),
        capacity=3,
        horizon=4,
    )
    assert len(rows) >= 6
    train_rows = rows[: max(4, len(rows) // 2)]
    val_rows = rows[max(4, len(rows) // 2) :]
    before_hash = _prov().model_sha256
    pi2 = train_pi2_from_c2_labels(
        train_rows=train_rows,
        val_rows=val_rows,
        seed=0,
        pi1_provenance=_prov(),
    )
    assert pi2.feature_columns == list(EVICT_VALUE_V1_FEATURE_COLUMNS)
    assert _prov().model_sha256 == before_hash
