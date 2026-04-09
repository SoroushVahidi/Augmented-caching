from __future__ import annotations

import csv
import json

import pytest

from lafc.offline import BeladyUniformPagingSolver, run_offline_solver, save_offline_results
from lafc.simulator.request_trace import build_requests_from_lists


def test_belady_uniform_classic_trace_and_victims():
    # Trace: A B C A B C with k=2 has OPT misses=4.
    requests, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "C"])
    result = run_offline_solver(
        BeladyUniformPagingSolver(),
        requests=requests,
        pages=pages,
        capacity=2,
    )

    assert result.total_misses == 4
    # First full-cache eviction (t=2): evict B because next(B)=4 > next(A)=3.
    assert result.decisions[2].evicted == "B"
    # Next full-cache eviction (t=4): evict A (never used again).
    assert result.decisions[4].evicted == "A"
    assert result.decisions[4].evicted_never_used_again


def test_belady_tie_breaking_is_deterministic():
    # At t=2 with cache {A,B}, both A and B are never used again.
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    result = run_offline_solver(
        BeladyUniformPagingSolver(),
        requests=requests,
        pages=pages,
        capacity=2,
    )
    assert result.decisions[2].evicted == "A"
    assert result.decisions[2].tie_size == 2
    assert result.diagnostics["ties_on_eviction"] == 1


def test_belady_rejects_non_uniform_weights_in_strict_mode():
    requests, pages = build_requests_from_lists(["A", "B", "A"], {"A": 1.0, "B": 2.0})
    with pytest.raises(ValueError, match="equal retrieval costs"):
        run_offline_solver(
            BeladyUniformPagingSolver(),
            requests=requests,
            pages=pages,
            capacity=1,
            validation_mode="strict",
        )


def test_belady_coerce_mode_allows_mixed_weights():
    requests, pages = build_requests_from_lists(["A", "B", "A"], {"A": 1.0, "B": 2.0})
    result = run_offline_solver(
        BeladyUniformPagingSolver(),
        requests=requests,
        pages=pages,
        capacity=1,
        validation_mode="coerce",
    )
    assert result.total_misses == 3
    assert result.diagnostics["validation"]["mode"] == "coerce"
    assert not result.diagnostics["validation"]["is_uniform"]


def test_offline_output_files_are_written(tmp_path):
    requests, pages = build_requests_from_lists(["A", "B", "A", "C"])
    result = run_offline_solver(
        BeladyUniformPagingSolver(),
        requests=requests,
        pages=pages,
        capacity=2,
    )

    save_offline_results(result, str(tmp_path))

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "per_step_decisions.csv").exists()
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "report.md").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["solver_name"] == "offline_belady_uniform"

    with (tmp_path / "per_step_decisions.csv").open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == len(requests) + 1
