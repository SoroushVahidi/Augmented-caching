"""Verify per-capacity KBS policy-comparison chunk CSVs before they are merged.

Each chunk is produced by scripts/run_policy_comparison_wulver_v1.py and has
columns: trace_name, trace_family, path, capacity, policy, misses, hit_rate.
This script checks structural and numeric sanity of one or more chunk files
without merging them — use build_kbs_policy_trend_artifacts.py for that.

Run from repository root:

    python scripts/paper/verify_kbs_policy_chunks.py \\
        --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \\
                 analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \\
        --expected-capacities 32,64 \\
        --expected-policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXPECTED_COLUMNS = [
    "trace_name",
    "trace_family",
    "path",
    "capacity",
    "policy",
    "misses",
    "hit_rate",
]

# Diagnostic-only oracle (knows the future); never part of the final 8-policy
# manuscript roster. The fair variant is "blind_oracle_lru_combiner".
DIAGNOSTIC_ONLY_POLICIES = {"blind_oracle"}

DEFAULT_EXPECTED_TRACES = {
    "brightkite_50k",
    "citibike_202401_50k",
    "wiki2018_pageviews_en_50k",
    "twemcache_cluster26_sample100_50k",
    "metakv_kvcache_202206_head_50k",
    "metacdn_cdn_202303_head_50k",
    "cloudphysics_alibaba_block_head_50k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more chunk CSV paths to verify (e.g. cap32_with_sieve_fifo.csv cap64_with_sieve_fifo.csv).",
    )
    parser.add_argument(
        "--expected-capacities",
        required=True,
        help="Comma-separated list of capacities that must be present across --inputs, e.g. 32,64.",
    )
    parser.add_argument(
        "--expected-policies",
        required=True,
        help="Comma-separated list of policy names that must be present for every (trace, capacity) pair.",
    )
    parser.add_argument(
        "--expected-traces",
        default="",
        help=(
            "Comma-separated list of expected trace_name values. "
            "Defaults to the 7-trace wulver_v1 manifest if omitted."
        ),
    )
    parser.add_argument(
        "--allow-diagnostic-blind-oracle",
        action="store_true",
        help="Allow the diagnostic-only 'blind_oracle' policy to be present (default: rejected).",
    )
    return parser.parse_args()


def load_chunk(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: unexpected columns {reader.fieldnames!r}, expected {EXPECTED_COLUMNS!r}"
            )
        return list(reader)


def verify(
    inputs: list[Path],
    expected_capacities: set[str],
    expected_policies: set[str],
    expected_traces: set[str],
    allow_diagnostic_blind_oracle: bool,
) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    all_rows: list[tuple[Path, dict[str, str]]] = []

    for path in inputs:
        if not path.exists():
            errors.append(f"missing input file: {path}")
            continue
        try:
            rows = load_chunk(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not rows:
            errors.append(f"{path}: no data rows")
            continue
        all_rows.extend((path, row) for row in rows)

    if errors:
        return False, errors, warnings

    seen_capacities: set[str] = set()
    seen_policies: set[str] = set()
    seen_traces: set[str] = set()
    dup_check: dict[tuple[str, str, str], list[Path]] = {}

    for path, row in all_rows:
        trace_name = row["trace_name"]
        capacity = row["capacity"]
        policy = row["policy"]
        seen_capacities.add(capacity)
        seen_policies.add(policy)
        seen_traces.add(trace_name)

        key = (trace_name, capacity, policy)
        dup_check.setdefault(key, []).append(path)

        if not allow_diagnostic_blind_oracle and policy in DIAGNOSTIC_ONLY_POLICIES:
            errors.append(
                f"{path}: diagnostic-only policy '{policy}' found in row "
                f"(trace_name={trace_name}, capacity={capacity}); exclude or pass "
                "--allow-diagnostic-blind-oracle if this is intentional"
            )

        try:
            misses = int(row["misses"])
        except ValueError:
            errors.append(f"{path}: non-integer misses value {row['misses']!r} for {key}")
        else:
            if misses < 0:
                errors.append(f"{path}: negative misses {misses} for {key}")

        if row["hit_rate"] != "":
            try:
                hit_rate = float(row["hit_rate"])
            except ValueError:
                errors.append(f"{path}: non-numeric hit_rate {row['hit_rate']!r} for {key}")
            else:
                if not (0.0 <= hit_rate <= 1.0):
                    errors.append(f"{path}: hit_rate {hit_rate} out of [0, 1] for {key}")

    # Duplicate (trace, capacity, policy) rows, whether within one file or
    # across the provided inputs.
    for key, paths in dup_check.items():
        if len(paths) > 1:
            errors.append(
                f"duplicate row for trace_name={key[0]}, capacity={key[1]}, policy={key[2]} "
                f"across {[str(p) for p in paths]}"
            )

    missing_capacities = expected_capacities - seen_capacities
    if missing_capacities:
        errors.append(f"missing expected capacities: {sorted(missing_capacities)}")
    extra_capacities = seen_capacities - expected_capacities
    if extra_capacities:
        warnings.append(f"capacities present but not requested: {sorted(extra_capacities)}")

    missing_policies = expected_policies - seen_policies
    if missing_policies:
        errors.append(f"missing expected policies overall: {sorted(missing_policies)}")
    extra_policies = seen_policies - expected_policies - DIAGNOSTIC_ONLY_POLICIES
    if extra_policies:
        warnings.append(f"policies present but not in expected roster: {sorted(extra_policies)}")

    if expected_traces:
        missing_traces = expected_traces - seen_traces
        if missing_traces:
            errors.append(f"missing expected traces: {sorted(missing_traces)}")
        extra_traces = seen_traces - expected_traces
        if extra_traces:
            warnings.append(f"traces present but not in expected manifest: {sorted(extra_traces)}")

    # Per (trace, capacity): every expected policy must appear exactly once.
    by_trace_capacity: dict[tuple[str, str], set[str]] = {}
    for _, row in all_rows:
        key = (row["trace_name"], row["capacity"])
        by_trace_capacity.setdefault(key, set()).add(row["policy"])

    for (trace_name, capacity), policies in sorted(by_trace_capacity.items()):
        if capacity not in expected_capacities:
            continue
        missing = expected_policies - policies
        if missing:
            errors.append(
                f"trace_name={trace_name}, capacity={capacity}: missing policies {sorted(missing)}"
            )

    return not errors, errors, warnings


def main() -> int:
    args = parse_args()
    inputs = [Path(p) for p in args.inputs]
    expected_capacities = {c.strip() for c in args.expected_capacities.split(",") if c.strip()}
    expected_policies = {p.strip() for p in args.expected_policies.split(",") if p.strip()}
    if args.expected_traces.strip():
        expected_traces = {t.strip() for t in args.expected_traces.split(",") if t.strip()}
    else:
        expected_traces = set(DEFAULT_EXPECTED_TRACES)

    ok, errors, warnings = verify(
        inputs=inputs,
        expected_capacities=expected_capacities,
        expected_policies=expected_policies,
        expected_traces=expected_traces,
        allow_diagnostic_blind_oracle=args.allow_diagnostic_blind_oracle,
    )

    print(f"Verified {len(inputs)} input file(s):")
    for path in inputs:
        print(f"  - {path}")
    print()

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("PASSED: chunks are structurally and numerically sound for merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
