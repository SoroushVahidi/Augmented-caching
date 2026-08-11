from __future__ import annotations

from pathlib import Path

import pytest

from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    _build_distinct_suffix_counts,
    _build_occurrence_index,
    build_candidate_rows_for_full_cache_state,
)
from lafc.target_degeneracy import (
    deterministic_exact_tiebreak,
    eviction_loss_values,
    exact_tie_metrics,
    resolve_tied_set_at_long_horizon,
)
from scripts.experiments.analyze_eviction_loss_target_degeneracy import analyze


def _toy_rows(horizon: int, candidate_subset=None):
    page_ids = ["a", "b", "c", "c", "c", "c", "c", "a"]
    requests, _pages = build_requests_from_lists(page_ids=page_ids)
    return build_candidate_rows_for_full_cache_state(
        requests=requests,
        request_index=2,
        capacity=2,
        trace_name="toy",
        trace_family="test",
        cfg=ObjectiveAblationConfig(horizon=horizon),
        cache_order=["a", "b"],
        bucket_by_page={},
        confidence_by_page={},
        recent_req_hist=["a", "b"],
        recent_hit_hist=[],
        occurrence_index=_build_occurrence_index(requests),
        distinct_suffix_counts=_build_distinct_suffix_counts(requests),
        candidate_subset=candidate_subset,
        include_features=False,
    )


def test_multiple_candidates_tied_at_h4():
    values = eviction_loss_values(_toy_rows(horizon=4))
    metrics = exact_tie_metrics(values)

    assert values == {"a": 0.0, "b": 0.0}
    assert metrics.optimal_set_size == 2
    assert metrics.ordinary_margin == 0.0
    assert metrics.strict_distinct_margin is None


def test_same_candidates_separated_at_h8():
    values = eviction_loss_values(_toy_rows(horizon=8))

    assert values == {"a": 1.0, "b": 0.0}


def test_strict_margin_ignores_duplicate_minima():
    metrics = exact_tie_metrics({"a": 0.0, "b": 0.0, "c": 2.0})

    assert metrics.ordinary_margin == 0.0
    assert metrics.strict_distinct_margin == 2.0
    assert metrics.distinct_value_count == 2


def test_deterministic_tie_breaker_regret_computed_inside_tied_set():
    res = resolve_tied_set_at_long_horizon(
        h_long=8,
        h_tied_candidates=("a", "b"),
        long_values={"a": 1.0, "b": 0.0},
        deterministic_choice=deterministic_exact_tiebreak(("a", "b")),
    )

    assert res.deterministic_choice == "a"
    assert res.deterministic_is_long_best is False
    assert res.deterministic_long_regret == 1.0
    assert res.long_spread == 1.0


def test_longer_horizon_evaluation_preserves_initial_candidate_feasibility():
    rows = _toy_rows(horizon=8, candidate_subset={"a"})

    assert [row["candidate_page_id"] for row in rows] == ["a"]
    assert eviction_loss_values(rows) == {"a": 1.0}
    with pytest.raises(ValueError, match="non-cache candidates"):
        _toy_rows(horizon=8, candidate_subset={"ghost"})


def test_diagnostic_is_deterministic(tmp_path: Path):
    requests, pages = build_requests_from_lists(page_ids=["a", "b", "c", "c", "c", "c", "c", "a", "b"])
    kwargs = {
        "requests": requests,
        "pages": pages,
        "trace_name": "toy",
        "trace_family": "test",
        "capacity": 2,
        "horizon": 4,
        "long_horizons": [8],
        "score_start": 0,
        "score_end": len(requests),
        "learned_choose": None,
        "overwrite": False,
        "trace_path": None,
        "trace_sha256": "synthetic",
        "fold": None,
        "learned_model": {"status": "NOT_AVAILABLE", "reason": "test"},
    }

    s1 = analyze(out_dir=tmp_path / "r1", **kwargs)
    s2 = analyze(out_dir=tmp_path / "r2", **kwargs)

    assert s1["event_summary"] == s2["event_summary"]
    assert s1["longer_horizon_resolution"] == s2["longer_horizon_resolution"]
    assert (tmp_path / "r1" / "event_metrics.csv").read_text() == (
        tmp_path / "r2" / "event_metrics.csv"
    ).read_text()
