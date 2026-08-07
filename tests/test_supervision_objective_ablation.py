"""Tests for src/lafc/supervision_objective_ablation.py -- the shared
multi-label dataset construction behind the four supervision objectives
(docs/supervision_objective_ablation_protocol.md).
"""

from __future__ import annotations

import math

import pytest

from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    build_multi_label_candidate_rows,
    build_pairwise_rows,
)


def _rows_for(page_ids, capacity, horizon=6):
    reqs, _ = build_requests_from_lists(page_ids=page_ids)
    cfg = ObjectiveAblationConfig(horizon=horizon)
    return build_multi_label_candidate_rows(reqs, capacity, "t", "fam", cfg)


def test_eviction_loss_label_matches_hand_derivation():
    # Verified against the actual module: capacity=2, q1 fills first, q2
    # second; decision at t=2 (miss on "newpid"); future = [q1, q2, z1, q1,
    # z2, q1], H=6. q1 is reused repeatedly, q2 only once.
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    rows = _rows_for(page_ids, capacity=2, horizon=6)
    decision_rows = [r for r in rows if r["decision_t"] == 2]
    assert len(decision_rows) == 2
    by_cand = {r["candidate_page_id"]: r for r in decision_rows}
    # Evicting q1 (repeatedly reused) costs more than evicting q2 (used once).
    assert by_cand["q1"]["eviction_loss_label"] == 5.0
    assert by_cand["q2"]["eviction_loss_label"] == 4.0


def test_next_arrival_and_reuse_distance_labels_match_hand_derivation():
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    rows = _rows_for(page_ids, capacity=2, horizon=6)
    decision_rows = [r for r in rows if r["decision_t"] == 2]
    by_cand = {r["candidate_page_id"]: r for r in decision_rows}

    # q1 reoccurs immediately (distance 1); q2 reoccurs one step later
    # (distance 2) -- the closest two distinct candidates can be, since
    # only one page is requested per step (structural fact used in the
    # corrected manuscript motivating example).
    assert by_cand["q1"]["next_arrival_label_raw"] == 1.0
    assert by_cand["q2"]["next_arrival_label_raw"] == 2.0
    assert by_cand["q1"]["next_arrival_label_censored"] == 1.0
    assert by_cand["q2"]["next_arrival_label_censored"] == 2.0

    # reuse distance: distinct OTHER objects before next reoccurrence.
    # q1's next occurrence is the very next request -> 0 distinct others.
    # q2's next occurrence is preceded by exactly one distinct other (q1).
    assert by_cand["q1"]["reuse_distance_label_raw"] == 0.0
    assert by_cand["q2"]["reuse_distance_label_raw"] == 1.0


def test_motivating_case_next_arrival_nearly_indistinguishable_eviction_loss_differs():
    # The corrected manuscript motivating example, verified end to end
    # through the actual objective-ablation dataset builder (not just the
    # raw simulate_lru_misses call used to originally derive the numbers).
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    rows = _rows_for(page_ids, capacity=2, horizon=6)
    decision_rows = [r for r in rows if r["decision_t"] == 2]
    by_cand = {r["candidate_page_id"]: r for r in decision_rows}

    next_gap = abs(by_cand["q1"]["next_arrival_label_censored"] - by_cand["q2"]["next_arrival_label_censored"])
    loss_gap = abs(by_cand["q1"]["eviction_loss_label"] - by_cand["q2"]["eviction_loss_label"])

    # Next-arrival distances are as close as two distinct candidates can
    # be (gap of exactly 1 -- an exact tie is structurally impossible).
    assert next_gap == 1.0
    # Eviction loss still separates them with the correct direction:
    # evicting the repeatedly-reused candidate (q1) costs strictly more.
    assert by_cand["q1"]["eviction_loss_label"] > by_cand["q2"]["eviction_loss_label"]
    assert loss_gap >= 1.0


def test_censoring_caps_at_horizon():
    # q1 never reoccurs within the horizon window at all.
    page_ids = ["q1", "q2", "newpid"] + ["filler"] * 10
    rows = _rows_for(page_ids, capacity=2, horizon=4)
    decision_rows = [r for r in rows if r["decision_t"] == 2]
    by_cand = {r["candidate_page_id"]: r for r in decision_rows}
    # raw distance exceeds the horizon (q1/q2 never seen again within it)
    assert by_cand["q1"]["next_arrival_label_raw"] > 4.0
    assert by_cand["q2"]["next_arrival_label_raw"] > 4.0
    # censored labels are capped exactly at H
    assert by_cand["q1"]["next_arrival_label_censored"] == 4.0
    assert by_cand["q2"]["next_arrival_label_censored"] == 4.0
    assert by_cand["q1"]["reuse_distance_label_censored"] <= 4.0
    assert by_cand["q2"]["reuse_distance_label_censored"] <= 4.0


def test_features_do_not_include_any_label_column():
    # Structural leakage guard: the feature columns used at inference must
    # never include a label value or a raw hint about future information.
    from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS

    label_columns = {
        "eviction_loss_label", "next_arrival_label_raw", "next_arrival_label_censored",
        "reuse_distance_label_raw", "reuse_distance_label_censored",
    }
    assert label_columns.isdisjoint(set(EVICT_VALUE_V1_FEATURE_COLUMNS))


def test_deterministic_repeated_construction():
    page_ids = ["a", "b", "c", "a", "d", "b", "c", "e", "a", "b"]
    rows_1 = _rows_for(page_ids, capacity=2, horizon=4)
    rows_2 = _rows_for(page_ids, capacity=2, horizon=4)
    assert rows_1 == rows_2


def test_pairwise_next_arrival_source_independent_of_eviction_loss():
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    rows = _rows_for(page_ids, capacity=2, horizon=6)
    pairs = build_pairwise_rows(rows, source="next_arrival")
    decision_pairs = [p for p in pairs if p["decision_t"] == 2]
    assert len(decision_pairs) == 1
    pair = decision_pairs[0]
    # q1 has a smaller (sooner) next-arrival distance than q2 -> q1 preferred
    # to be KEPT, i.e. label_i_preferred reflects the smaller-distance
    # candidate as "i_preferred" per the pairwise_label_source semantics.
    assert pair["pairwise_label_source"] == "next_arrival"
    i_is_q1 = pair["candidate_i_page_id"] == "q1"
    expected_label = 1 if i_is_q1 else 0
    assert pair["label_i_preferred"] == expected_label


def test_pairwise_regret_derived_is_explicitly_labeled_and_separate():
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    rows = _rows_for(page_ids, capacity=2, horizon=6)
    pairs_next = build_pairwise_rows(rows, source="next_arrival")
    pairs_regret = build_pairwise_rows(rows, source="regret")
    assert pairs_next[0]["pairwise_label_source"] == "next_arrival"
    assert pairs_regret[0]["pairwise_label_source"] == "regret"
    # Computed from entirely different label columns (next_arrival_label_censored
    # vs eviction_loss_label) -- distinct numeric values in this example,
    # confirming they are not silently the same computation under two names.
    assert pairs_next[0]["value_i"] != pairs_regret[0]["value_i"]


def test_build_pairwise_rows_rejects_unknown_source():
    with pytest.raises(ValueError):
        build_pairwise_rows([], source="bogus")
