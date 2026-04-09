from __future__ import annotations

import csv
import json

from lafc.offline import (
    GeneralCachingLPApproxSolver,
    load_trace_with_sizes,
    run_offline_solver,
    save_offline_results,
)
from lafc.simulator.request_trace import build_requests_from_lists


def test_general_solver_prefers_high_value_interval_with_bypass():
    # Capacity=2, A and B both size 2. Keeping A saves 10, keeping B would save 1.
    # The LP+rounding should keep A and bypass B at t=1.
    requests, pages = build_requests_from_lists(
        ["A", "B", "A"],
        {"A": 10.0, "B": 1.0},
    )
    sizes = {"A": 2.0, "B": 2.0}

    result = run_offline_solver(
        GeneralCachingLPApproxSolver(),
        requests=requests,
        pages=pages,
        capacity=2.0,
        page_sizes=sizes,
        allow_bypass=True,
    )

    assert result.total_misses == 2
    assert result.total_cost == 11.0
    # t=1 (request B) should be a miss+bypass in this instance.
    assert result.decisions[1].bypassed


def test_general_solver_respects_capacity_always():
    requests, pages = build_requests_from_lists(
        ["A", "B", "C", "A", "B", "C"],
        {"A": 8.0, "B": 4.0, "C": 3.0},
    )
    sizes = {"A": 3.0, "B": 2.0, "C": 2.0}

    result = run_offline_solver(
        GeneralCachingLPApproxSolver(),
        requests=requests,
        pages=pages,
        capacity=4.0,
        page_sizes=sizes,
        allow_bypass=True,
    )

    assert all((d.cache_occupancy or 0.0) <= 4.0 + 1e-9 for d in result.decisions)
    assert result.diagnostics["lp"]["status"] in {"Optimal", "no_intervals"}


def test_general_solver_outputs_written(tmp_path):
    requests, pages = build_requests_from_lists(["A", "B", "A"], {"A": 5.0, "B": 1.0})
    sizes = {"A": 2.0, "B": 1.0}

    result = run_offline_solver(
        GeneralCachingLPApproxSolver(),
        requests=requests,
        pages=pages,
        capacity=2.0,
        page_sizes=sizes,
        allow_bypass=True,
    )
    save_offline_results(result, str(tmp_path))

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "per_step_decisions.csv").exists()
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "report.md").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["solver_name"] == "offline_general_caching_lp_round"
    assert "total_cost" in summary

    with (tmp_path / "per_step_decisions.csv").open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 4


def test_load_trace_with_sizes_requires_sizes(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"requests": ["A", "B"], "weights": {"A": 1, "B": 1}}), encoding="utf-8")

    try:
        load_trace_with_sizes(str(bad))
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "sizes" in str(exc)
