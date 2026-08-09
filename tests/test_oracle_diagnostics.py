from __future__ import annotations

import pytest

from lafc.oracle_diagnostics import (
    EXACT_ORACLE_REQUIRES_CLARIFICATION,
    EXACT_ORACLE_WELL_DEFINED,
    get_exact_oracle_objective_specs,
    replay_exact_target_policy,
    replay_score_driven_policy,
    summarize_decision_diagnostics,
)
from lafc.policies.offline_belady import OfflineBeladyPolicy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import ObjectiveAblationConfig


def test_exact_eviction_loss_oracle_chooses_lowest_true_loss_candidate():
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)
    summary = replay_exact_target_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=6),
        objective="eviction_loss",
    )

    first = summary.decisions[0]
    assert first.exact_candidate == "q2"
    assert first.candidate_values == {"q1": 5.0, "q2": 4.0}
    assert first.target_regret == 0.0
    assert len(summary.hit_sequence) == len(requests)
    assert summary.total_misses == sum(1 for hit in summary.hit_sequence if not hit)


def test_exact_eviction_loss_oracle_horizon_change_can_change_choice():
    page_ids = ["a", "b", "c", "c", "c", "c", "c", "a"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)
    summary_h4 = replay_exact_target_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=4),
        objective="eviction_loss",
    )
    summary_h8 = replay_exact_target_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=8),
        objective="eviction_loss",
    )

    assert summary_h4.decisions[0].candidate_values == {"a": 0.0, "b": 0.0}
    assert summary_h4.decisions[0].exact_candidate == "a"
    assert summary_h8.decisions[0].candidate_values == {"a": 1.0, "b": 0.0}
    assert summary_h8.decisions[0].exact_candidate == "b"


def test_score_driven_policy_logs_regret_when_it_disagrees_with_exact_target():
    page_ids = ["q1", "q2", "newpid", "q1", "q2", "z1", "q1", "z2", "q1"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)

    def always_pick_lexicographically_smallest(rows):
        candidates = sorted(str(row["candidate_page_id"]) for row in rows)
        return {candidate: float(idx) for idx, candidate in enumerate(candidates)}

    summary = replay_score_driven_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=6),
        objective="eviction_loss",
        scorer=always_pick_lexicographically_smallest,
        policy_name="mock_wrong_scalar",
    )

    first = summary.decisions[0]
    metrics = summarize_decision_diagnostics(summary.decisions)
    assert first.exact_candidate == "q2"
    assert first.chosen_candidate == "q1"
    assert first.agrees_with_exact is False
    assert first.target_regret == 1.0
    assert metrics["agreement_rate"] < 1.0
    assert metrics["non_optimal_fraction"] > 0.0


def test_exact_eviction_loss_oracle_and_belady_can_differ():
    page_ids = ["a", "b", "c", "c", "c", "d", "a"]
    requests, pages = build_requests_from_lists(page_ids=page_ids)
    exact = replay_exact_target_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=4),
        objective="eviction_loss",
    )

    belady = OfflineBeladyPolicy()
    belady.reset(2, pages)
    belady_victim = None
    for req in requests[:3]:
        event = belady.on_request(req)
        if req.t == 2:
            belady_victim = event.evicted

    assert exact.decisions[0].candidate_values == {"a": 2.0, "b": 2.0}
    assert exact.decisions[0].exact_candidate == "a"
    assert belady_victim == "b"


def test_exact_target_tie_breaks_lexicographically():
    page_ids = ["a", "b", "c", "c", "c", "d", "a"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)
    summary = replay_exact_target_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=4),
        objective="eviction_loss",
    )

    first = summary.decisions[0]
    assert first.optimal_candidates == ("a", "b")
    assert first.exact_candidate == "a"


def test_score_driven_policy_rejects_infeasible_candidate_choice():
    page_ids = ["a", "b", "c", "a", "b"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)

    with pytest.raises(ValueError, match="exactly one score per candidate"):
        replay_score_driven_policy(
            requests=requests,
            capacity=2,
            trace_name="toy",
            trace_family="fam",
            cfg=ObjectiveAblationConfig(horizon=4),
            objective="eviction_loss",
            scorer=lambda rows: {"ghost": 0.0},
            policy_name="invalid_mock",
        )


def test_score_driven_policy_tie_break_matches_scalar_policy_cache_order_for_min():
    page_ids = ["b", "a", "c", "a", "b"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)

    summary = replay_score_driven_policy(
        requests=requests,
        capacity=2,
        trace_name="toy",
        trace_family="fam",
        cfg=ObjectiveAblationConfig(horizon=4),
        objective="eviction_loss",
        scorer=lambda rows: {str(row["candidate_page_id"]): 0.0 for row in rows},
        policy_name="mock_tied_scalar",
    )

    first = summary.decisions[0]
    assert [str(pid) for pid in first.candidate_values] == ["b", "a"]
    assert first.chosen_candidate == "b"


def test_exact_oracle_objective_classification_records_pairwise_as_requires_clarification():
    specs = get_exact_oracle_objective_specs()

    assert specs["eviction_loss"].status == EXACT_ORACLE_WELL_DEFINED
    assert specs["next_arrival"].status == EXACT_ORACLE_WELL_DEFINED
    assert specs["reuse_distance"].status == EXACT_ORACLE_WELL_DEFINED
    assert specs["objective_pairwise"].status == EXACT_ORACLE_REQUIRES_CLARIFICATION
