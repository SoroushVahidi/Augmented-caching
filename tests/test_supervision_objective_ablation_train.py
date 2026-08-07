"""Tests for src/lafc/supervision_objective_ablation_train.py."""

from __future__ import annotations

import numpy as np

from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import ObjectiveAblationConfig, build_multi_label_candidate_rows, build_pairwise_rows
from lafc.supervision_objective_ablation_train import FEATURES, train_pairwise_objective, train_scalar_objective


def _real_rows():
    # A repeating pattern with enough decisions to split train/val/test.
    page_ids = (["a", "b", "c", "d"] * 40) + ["x", "a", "b", "c", "d", "x", "a", "b"]
    reqs, _ = build_requests_from_lists(page_ids=page_ids)
    cfg = ObjectiveAblationConfig(horizon=4)
    return build_multi_label_candidate_rows(reqs, capacity=3, trace_name="t", trace_family="fam", cfg=cfg)


def _split(rows):
    decisions = sorted({r["decision_id"] for r in rows})
    n = len(decisions)
    train_ids = set(decisions[: n // 2])
    val_ids = set(decisions[n // 2 : n * 3 // 4])
    test_ids = set(decisions[n * 3 // 4 :])
    train = [r for r in rows if r["decision_id"] in train_ids]
    val = [r for r in rows if r["decision_id"] in val_ids]
    test = [r for r in rows if r["decision_id"] in test_ids]
    return train, val, test


def test_train_scalar_objective_eviction_loss_min_direction():
    rows = _real_rows()
    train, val, test = _split(rows)
    assert train and val and test
    result = train_scalar_objective(
        objective="objective_eviction_loss",
        label_column="eviction_loss_label",
        direction="min",
        train_rows=train,
        val_rows=val,
        test_rows=test,
        seed=0,
    )
    assert result.best_model_name in {"ridge", "random_forest", "hist_gb"}
    assert result.best_model.feature_columns == FEATURES
    assert len(result.comparison_rows) == 3


def test_train_scalar_objective_next_arrival_max_direction():
    rows = _real_rows()
    train, val, test = _split(rows)
    result = train_scalar_objective(
        objective="objective_next_arrival",
        label_column="next_arrival_label_censored",
        direction="max",
        train_rows=train,
        val_rows=val,
        test_rows=test,
        seed=0,
    )
    assert result.direction == "max"
    for row in result.comparison_rows:
        assert row["val_mean_regret"] >= 0.0
        assert row["test_mean_regret"] >= 0.0


def test_ranking_direction_affects_regret_sign_convention_consistently():
    # Synthetic candidate set where predictions equal true labels exactly ->
    # regret must be exactly zero regardless of min/max direction.
    from lafc.supervision_objective_ablation_train import _ranking_metrics

    rows = [
        {"decision_id": "d0", "candidate_page_id": "p1", "y": 1.0},
        {"decision_id": "d0", "candidate_page_id": "p2", "y": 5.0},
        {"decision_id": "d0", "candidate_page_id": "p3", "y": 3.0},
    ]
    preds = np.asarray([1.0, 5.0, 3.0])
    for direction in ("min", "max"):
        m = _ranking_metrics(rows, preds, "y", direction)
        assert m["mean_regret_vs_oracle"] == 0.0
        assert m["top1_eviction_match"] == 1.0


def test_train_pairwise_objective_ranks_held_out_pair_correctly():
    # Build a real trace with a clear preference structure, train the
    # pairwise model on next-arrival-ordering pairs, and confirm it scores
    # the sooner-reused candidate higher than the later-reused one on a
    # held-out decision (basic sanity: the model learned *something*
    # consistent with the training signal, not just noise).
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"] * 8
    reqs, _ = build_requests_from_lists(page_ids=page_ids)
    cfg = ObjectiveAblationConfig(horizon=6)
    rows = build_multi_label_candidate_rows(reqs, capacity=2, trace_name="t", trace_family="fam", cfg=cfg)
    pairs = build_pairwise_rows(rows, source="next_arrival")
    assert pairs

    n = len(pairs)
    train_pairs = pairs[: n * 3 // 4]
    test_pairs = pairs[n * 3 // 4 :]
    assert train_pairs and test_pairs

    result = train_pairwise_objective(objective="objective_pairwise", train_pairs=train_pairs, seed=0)
    assert result.n_train_pairs == len(train_pairs)

    correct = 0
    for p in test_pairs:
        i_feats = np.asarray([[float(p[f"i_{c}"]) for c in FEATURES]])
        j_feats = np.asarray([[float(p[f"j_{c}"]) for c in FEATURES]])
        r_i = result.model.predict_rewards(i_feats)[0]
        r_j = result.model.predict_rewards(j_feats)[0]
        predicted_i_preferred = int(r_i > r_j)
        if predicted_i_preferred == int(p["label_i_preferred"]):
            correct += 1
    # Not a tight bound (small synthetic trace, few epochs) -- just confirms
    # the trained model is meaningfully better than a coin flip.
    assert correct / len(test_pairs) >= 0.6


def test_train_scalar_objective_deterministic_given_seed():
    rows = _real_rows()
    train, val, test = _split(rows)
    r1 = train_scalar_objective(
        objective="objective_eviction_loss",
        label_column="eviction_loss_label",
        direction="min",
        train_rows=train,
        val_rows=val,
        test_rows=test,
        seed=0,
    )
    r2 = train_scalar_objective(
        objective="objective_eviction_loss",
        label_column="eviction_loss_label",
        direction="min",
        train_rows=train,
        val_rows=val,
        test_rows=test,
        seed=0,
    )
    assert r1.best_model_name == r2.best_model_name
    assert r1.comparison_rows == r2.comparison_rows
