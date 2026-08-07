"""Tests for src/lafc/supervision_objective_ablation.py -- the shared
multi-label dataset construction behind the four supervision objectives
(docs/supervision_objective_ablation_protocol.md).
"""

from __future__ import annotations

import math

import pytest

from lafc.evict_value_v2_rollout import _next_use_distance
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    _build_distinct_suffix_counts,
    _build_occurrence_index,
    _forward_reuse_distance,
    _next_arrival_and_reuse_distance_fast,
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


def test_fast_distance_helpers_match_reference_scan_implementation():
    # Differential test: the O(log n) occurrence-index fast path used by
    # the hot loop must produce EXACTLY the same (next_raw, reuse_raw)
    # values as the original linear-scan reference functions
    # (_next_use_distance / _forward_reuse_distance), for every
    # (candidate, decision time) pair, including candidates that never
    # reoccur (the pathological case that motivated the optimization) and
    # a trace with duplicate/repeated page ids at varying gaps.
    import math
    import random as _random

    rng = _random.Random(7)
    vocab = [f"p{i}" for i in range(15)]
    page_ids = [rng.choice(vocab) for _ in range(200)]
    # Add some pages that appear exactly once (never reoccur).
    page_ids += [f"unique{i}" for i in range(5)]
    reqs, _ = build_requests_from_lists(page_ids=page_ids)

    occurrence_index = _build_occurrence_index(reqs)
    distinct_suffix_counts = _build_distinct_suffix_counts(reqs)
    n = len(reqs)

    candidates = set(page_ids)
    for t in range(0, n - 1, 7):  # sample decision points
        future = reqs[t + 1 :]
        for candidate in candidates:
            ref_next = _next_use_distance(candidate, future, 0)
            ref_next_raw = ref_next + 1.0 if math.isfinite(ref_next) else float(len(future) + 1)
            ref_reuse = _forward_reuse_distance(candidate, future, 0)
            ref_reuse_raw = ref_reuse if math.isfinite(ref_reuse) else float(len(set(r.page_id for r in future)))

            fast_next_raw, fast_reuse_raw = _next_arrival_and_reuse_distance_fast(
                candidate, t, n, occurrence_index, distinct_suffix_counts, reqs
            )
            assert fast_next_raw == ref_next_raw, (t, candidate, fast_next_raw, ref_next_raw)
            assert fast_reuse_raw == ref_reuse_raw, (t, candidate, fast_reuse_raw, ref_reuse_raw)


def test_pairwise_max_pairs_per_decision_caps_and_is_deterministic():
    import random as _random

    page_ids = ["a", "b", "c", "d", "e", "f"]
    rng = _random.Random(1)
    for _ in range(300):
        page_ids.append(rng.choice(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]))
    rows = _rows_for(page_ids, capacity=6, horizon=4)

    uncapped = build_pairwise_rows(rows, source="next_arrival")
    capped = build_pairwise_rows(rows, source="next_arrival", max_pairs_per_decision=6, sample_seed=0)
    capped_again = build_pairwise_rows(rows, source="next_arrival", max_pairs_per_decision=6, sample_seed=0)

    counts: dict = {}
    for p in capped:
        counts[p["decision_id"]] = counts.get(p["decision_id"], 0) + 1
    assert counts and max(counts.values()) <= 6
    assert len(capped) < len(uncapped)
    assert capped == capped_again  # deterministic given the same seed
