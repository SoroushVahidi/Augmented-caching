from __future__ import annotations

import math

from lafc.reuse_tail_horizon import (
    CellAccumulator,
    iter_resident_candidate_observations,
    nearest_rank_quantile,
    summarize_accumulator,
)
from lafc.simulator.request_trace import build_requests_from_lists


def _observations(page_ids, *, capacity, score_start, score_end):
    requests, _pages = build_requests_from_lists(page_ids=list(page_ids))
    return list(
        iter_resident_candidate_observations(
            requests,
            family="toyfam",
            trace_name="toytrace",
            capacity=capacity,
            score_start=score_start,
            score_end=score_end,
        )
    )


def test_reuse_tail_records_within_beyond_and_boundary_request_counts():
    # Decision at t=2 sees resident candidates a,b. a is requested at t=3
    # (T=1), b is requested at t=5 (T=3). T counts request positions, not
    # distinct intervening objects.
    obs = _observations(["a", "b", "x", "a", "q", "b"], capacity=2, score_start=2, score_end=3)
    by_candidate = {row.candidate_page_id: row for row in obs}

    assert set(by_candidate) == {"a", "b"}
    assert by_candidate["a"].next_reuse_request_index == 3
    assert by_candidate["a"].t == 1.0
    assert by_candidate["b"].next_reuse_request_index == 5
    assert by_candidate["b"].t == 3.0

    acc = CellAccumulator("toyfam", "toytrace", 2, horizons=[1, 3])
    acc.decision_points = 1
    for row in obs:
        acc.record(row)

    h1 = summarize_accumulator(acc, 1)
    assert h1["resident_candidate_observations"] == 2
    assert h1["t_le_h_count"] == 1
    assert h1["t_gt_h_count_including_never"] == 1
    assert h1["p_t_gt_h_including_never"] == 0.5

    h3 = summarize_accumulator(acc, 3)
    assert h3["t_le_h_count"] == 2
    assert h3["t_gt_h_count_including_never"] == 0
    assert h3["p_t_gt_h_including_never"] == 0.0


def test_reuse_tail_records_never_reused_as_infinity_and_exceeding_h():
    obs = _observations(["a", "b", "x", "a", "q"], capacity=2, score_start=2, score_end=3)
    by_candidate = {row.candidate_page_id: row for row in obs}

    assert by_candidate["a"].t == 1.0
    assert by_candidate["b"].next_reuse_request_index is None
    assert math.isinf(by_candidate["b"].t)
    assert by_candidate["b"].never_reused

    acc = CellAccumulator("toyfam", "toytrace", 2, horizons=[4])
    acc.decision_points = 1
    for row in obs:
        acc.record(row)
    summary = summarize_accumulator(acc, 4)

    assert summary["finite_reuse_count"] == 1
    assert summary["never_reused_count"] == 1
    assert summary["never_reused_fraction"] == 0.5
    assert summary["t_gt_h_count_including_never"] == 1
    assert summary["t_gt_h_count_eventually_reused"] == 0
    assert summary["p_t_gt_h_eventually_reused"] == 0.0


def test_reuse_tail_multiple_capacities_use_full_cache_miss_population():
    cap1 = _observations(["a", "b", "a"], capacity=1, score_start=1, score_end=2)
    assert len(cap1) == 1
    assert cap1[0].candidate_page_id == "a"
    assert cap1[0].decision_index == 1
    assert cap1[0].t == 1.0

    cap2 = _observations(["a", "b", "c", "a", "b"], capacity=2, score_start=2, score_end=3)
    assert len(cap2) == 2
    assert {row.candidate_page_id for row in cap2} == {"a", "b"}


def test_score_window_filters_decisions_without_changing_lru_state():
    # The decision at t=2 is outside the score window but still advances LRU
    # state. The scored decision at t=4 therefore sees candidates b,c.
    obs = _observations(["a", "b", "c", "b", "d", "c"], capacity=2, score_start=4, score_end=5)
    assert {row.candidate_page_id for row in obs} == {"b", "c"}
    assert all(row.decision_index == 4 for row in obs)


def test_nearest_rank_quantile_is_explicit_for_finite_t_values():
    assert nearest_rank_quantile([], 0.5) is None
    assert nearest_rank_quantile([1, 3, 10, 20], 0.5) == 3.0
    assert nearest_rank_quantile([1, 3, 10, 20], 0.75) == 10.0
    assert nearest_rank_quantile([1, 3, 10, 20], 0.99) == 20.0
