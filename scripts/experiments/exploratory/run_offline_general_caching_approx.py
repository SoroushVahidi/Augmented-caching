#!/usr/bin/env python3
"""Run offline LP+rounding approximation baseline for general caching."""

from __future__ import annotations

import argparse

from lafc.offline import (
    GeneralCachingLPApproxSolver,
    load_trace_with_sizes,
    run_offline_solver,
    save_offline_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline general caching baseline (LP relaxation + deterministic rounding)."
        )
    )
    parser.add_argument("--trace", required=True, help="Path to JSON/CSV trace file.")
    parser.add_argument(
        "--capacity",
        required=True,
        type=float,
        help="Cache capacity (same units as page sizes).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/offline_general_caching_approx",
        help="Output directory for summary/CSV/diagnostics/report.",
    )
    args = parser.parse_args()

    requests, pages, page_sizes = load_trace_with_sizes(args.trace)
    solver = GeneralCachingLPApproxSolver()

    result = run_offline_solver(
        solver,
        requests=requests,
        pages=pages,
        capacity=args.capacity,
        page_sizes=page_sizes,
        allow_bypass=True,
    )
    save_offline_results(result, args.output_dir)

    print(
        f"Completed {result.solver_name}: total_cost={result.total_cost:.6f}, "
        f"misses={result.total_misses}, hits={result.total_hits}"
    )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
