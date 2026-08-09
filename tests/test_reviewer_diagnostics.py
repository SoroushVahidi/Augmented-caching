from __future__ import annotations

from lafc.reviewer_diagnostics import (
    build_nested_fraction_subsets,
    compute_preference_cycle_metrics,
    filter_rows_by_decision_subset,
)
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    build_multi_label_candidate_rows,
    build_pairwise_rows,
)


def _candidate_rows(page_ids, capacity=3, horizon=4):
    reqs, _ = build_requests_from_lists(page_ids=page_ids)
    cfg = ObjectiveAblationConfig(horizon=horizon)
    return build_multi_label_candidate_rows(reqs, capacity, "toy_trace", "toy_family", cfg)


def test_nested_fraction_subsets_are_deterministic_and_nested():
    decision_ids = [f"d{i}" for i in range(20)]
    forward = build_nested_fraction_subsets(decision_ids, [0.1, 0.25, 1.0], seed=7)
    reverse = build_nested_fraction_subsets(reversed(decision_ids), [0.1, 0.25, 1.0], seed=7)

    assert forward == reverse
    assert set(forward[0.1]).issubset(set(forward[0.25]))
    assert set(forward[0.25]).issubset(set(forward[1.0]))
    assert len(forward[1.0]) == len(decision_ids)


def test_same_example_subset_can_drive_scalar_and_regret_pairwise_views():
    rows = _candidate_rows(
        ["a", "b", "c", "d", "a", "e", "b", "f", "c", "g", "a", "h", "b", "i", "c", "j"],
        capacity=3,
        horizon=4,
    )
    all_regret_pairs = build_pairwise_rows(rows, source="regret")
    eligible_decision_ids = sorted({str(row["decision_id"]) for row in all_regret_pairs})
    subsets = build_nested_fraction_subsets(eligible_decision_ids, [0.5, 1.0], seed=3)

    scalar_subset = filter_rows_by_decision_subset(rows, subsets[0.5])
    pairwise_subset = build_pairwise_rows(scalar_subset, source="regret")

    assert scalar_subset
    assert pairwise_subset
    assert {str(row["decision_id"]) for row in scalar_subset} == set(subsets[0.5])
    assert {str(row["decision_id"]) for row in pairwise_subset}.issubset(set(subsets[0.5]))
    assert all(row["pairwise_label_source"] == "regret" for row in pairwise_subset)


def test_cycle_metrics_detect_simple_three_cycle():
    metrics = compute_preference_cycle_metrics(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C"), ("C", "A")],
    )

    assert metrics.has_cycle is True
    assert metrics.cycle_triplet_count == 1
    assert metrics.strongly_connected_component_sizes == (3,)


def test_cycle_metrics_report_scalar_total_preorder_as_acyclic():
    rewards = {"A": 0.3, "B": 0.2, "C": 0.2, "D": -0.5}
    preferred_edges = []
    for left, left_reward in rewards.items():
        for right, right_reward in rewards.items():
            if left != right and left_reward > right_reward:
                preferred_edges.append((left, right))

    metrics = compute_preference_cycle_metrics(list(rewards), preferred_edges)

    assert metrics.has_cycle is False
    assert metrics.cycle_triplet_count == 0
    assert metrics.strongly_connected_component_sizes == ()
