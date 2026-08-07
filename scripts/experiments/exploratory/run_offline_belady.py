#!/usr/bin/env python3
"""Run the exact offline Belady baseline for uniform paging traces."""

from __future__ import annotations

import argparse

from lafc.offline import BeladyUniformPagingSolver, run_offline_solver, save_offline_results
from lafc.simulator.request_trace import load_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact offline Belady baseline (uniform paging)."
    )
    parser.add_argument("--trace", required=True, help="Path to JSON/CSV trace file.")
    parser.add_argument("--capacity", required=True, type=int, help="Cache capacity in pages.")
    parser.add_argument(
        "--output-dir",
        default="output/offline_belady",
        help="Directory for summary/CSV/diagnostics outputs.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["strict", "coerce"],
        default="strict",
        help="strict=require uniform weights; coerce=allow mixed weights as unit-cost paging.",
    )
    args = parser.parse_args()

    requests, pages = load_trace(args.trace)
    solver = BeladyUniformPagingSolver()
    result = run_offline_solver(
        solver,
        requests=requests,
        pages=pages,
        capacity=args.capacity,
        validation_mode=args.validation_mode,
    )
    save_offline_results(result, args.output_dir)

    print(
        f"Completed {result.solver_name}: misses={result.total_misses}, "
        f"hits={result.total_hits}, hit_rate={result.hit_rate:.6f}"
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
