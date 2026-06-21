"""Build DRAFT cap32->cap64->cap128(->cap256) policy trend artifacts.

Reads one or more *verified* per-capacity chunk CSVs (see
verify_kbs_policy_chunks.py) and emits draft, explicitly-labeled trend
artifacts. Outputs are never treated as the final canonical manuscript
table — that requires every capacity in --capacities to be present in
--inputs, and even then this script only writes to the "available
capacities" filenames below, never to the canonical table3 file.

Outputs:
    analysis/kbs_policy_trend_available_capacities.csv
    tables/manuscript/table3_policy_miss_ratio_available_capacities.csv
    reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md

Run from repository root:

    python scripts/paper/build_kbs_policy_trend_artifacts.py \\
        --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \\
                 analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \\
        --capacities 32,64 \\
        --draft-label "DRAFT / AVAILABLE CAPACITIES ONLY"
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_CSV = REPO_ROOT / "analysis" / "kbs_policy_trend_available_capacities.csv"
OUT_TABLE = REPO_ROOT / "tables" / "manuscript" / "table3_policy_miss_ratio_available_capacities.csv"
OUT_MD = REPO_ROOT / "reports" / "manuscript_artifacts" / "kbs_policy_trend_available_capacities.md"

REFERENCE_POLICY = "lru"
PROPOSED_POLICY = "evict_value_v1"
COMPARISON_BASELINES = ["lru", "sieve", "fifo_reinsertion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Verified chunk CSV paths.")
    parser.add_argument(
        "--capacities",
        required=True,
        help="Comma-separated list of capacities this run is *requesting* trend coverage for, e.g. 32,64,128,256.",
    )
    parser.add_argument(
        "--draft-label",
        default="DRAFT / AVAILABLE CAPACITIES ONLY",
        help="Label stamped on every output artifact.",
    )
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main() -> int:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    requested_capacities = [c.strip() for c in args.capacities.split(",") if c.strip()]
    draft_label = args.draft_label

    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"input chunk not found: {path}")

    rows = load_rows(input_paths)
    available_capacities = sorted({row["capacity"] for row in rows}, key=int)
    missing_capacities = [c for c in requested_capacities if c not in available_capacities]
    is_final = not missing_capacities

    # mean misses by (capacity, policy), aggregated across traces
    misses_by_cap_policy: dict[tuple[str, str], list[float]] = defaultdict(list)
    # misses by (capacity, policy, trace_family) for per-family ranking
    misses_by_cap_policy_family: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for row in rows:
        capacity = row["capacity"]
        policy = row["policy"]
        family = row["trace_family"]
        misses = float(row["misses"])
        misses_by_cap_policy[(capacity, policy)].append(misses)
        misses_by_cap_policy_family[(capacity, policy, family)].append(misses)

    policies = sorted({row["policy"] for row in rows})
    families = sorted({row["trace_family"] for row in rows})

    # --- analysis/kbs_policy_trend_available_capacities.csv ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "status_label",
                "capacity",
                "policy",
                "mean_misses",
                "rel_gap_vs_lru_pct",
            ]
        )
        for capacity in available_capacities:
            ref_key = (capacity, REFERENCE_POLICY)
            ref_mean = (
                statistics.mean(misses_by_cap_policy[ref_key])
                if ref_key in misses_by_cap_policy
                else None
            )
            for policy in policies:
                key = (capacity, policy)
                if key not in misses_by_cap_policy:
                    continue
                mean_misses = statistics.mean(misses_by_cap_policy[key])
                rel_gap = (
                    100.0 * (mean_misses - ref_mean) / ref_mean
                    if ref_mean is not None and ref_mean != 0
                    else ""
                )
                writer.writerow(
                    [
                        draft_label,
                        capacity,
                        policy,
                        f"{mean_misses:.4f}",
                        f"{rel_gap:.4f}" if rel_gap != "" else "",
                    ]
                )

    # --- tables/manuscript/table3_policy_miss_ratio_available_capacities.csv ---
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["status_label", "policy"] + [f"mean_misses_cap{c}" for c in available_capacities]
        writer.writerow(header)
        for policy in policies:
            row_out = [draft_label, policy]
            for capacity in available_capacities:
                key = (capacity, policy)
                if key in misses_by_cap_policy:
                    row_out.append(f"{statistics.mean(misses_by_cap_policy[key]):.4f}")
                else:
                    row_out.append("")
            writer.writerow(row_out)

    # --- per-trace-family ranking (best=1 by mean misses) per capacity ---
    family_rankings: dict[str, dict[str, list[tuple[str, float, int]]]] = {}
    for capacity in available_capacities:
        family_rankings[capacity] = {}
        for family in families:
            entries = []
            for policy in policies:
                key = (capacity, policy, family)
                if key in misses_by_cap_policy_family:
                    entries.append((policy, statistics.mean(misses_by_cap_policy_family[key])))
            entries.sort(key=lambda item: item[1])
            ranked = [(policy, mean_misses, rank + 1) for rank, (policy, mean_misses) in enumerate(entries)]
            family_rankings[capacity][family] = ranked

    # --- evict_value_v1 gap vs lru/sieve/fifo_reinsertion per capacity ---
    proposed_gap: dict[str, dict[str, float]] = {}
    for capacity in available_capacities:
        proposed_key = (capacity, PROPOSED_POLICY)
        if proposed_key not in misses_by_cap_policy:
            continue
        proposed_mean = statistics.mean(misses_by_cap_policy[proposed_key])
        proposed_gap[capacity] = {}
        for baseline in COMPARISON_BASELINES:
            baseline_key = (capacity, baseline)
            if baseline_key not in misses_by_cap_policy:
                continue
            baseline_mean = statistics.mean(misses_by_cap_policy[baseline_key])
            if baseline_mean != 0:
                proposed_gap[capacity][baseline] = 100.0 * (proposed_mean - baseline_mean) / baseline_mean

    # --- cap32->cap64->cap128(->cap256) trend for evict_value_v1 mean misses ---
    trend_caps = [c for c in ["32", "64", "128", "256"] if c in available_capacities]
    trend_series = {
        policy: [
            statistics.mean(misses_by_cap_policy[(c, policy)])
            if (c, policy) in misses_by_cap_policy
            else None
            for c in trend_caps
        ]
        for policy in policies
    }

    # --- markdown report ---
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# KBS policy trend — {draft_label}")
    lines.append("")
    lines.append(
        "**This is NOT the final canonical manuscript table.** "
        f"Requested capacities: {', '.join(requested_capacities)}. "
        f"Available capacities in inputs: {', '.join(available_capacities)}."
    )
    if missing_capacities:
        lines.append(f"Missing capacities (not yet run): {', '.join(missing_capacities)}.")
        lines.append("Do not cite this artifact as the final cap32-cap256 trend until all capacities are present.")
    else:
        lines.append(
            "All requested capacities are present in the inputs, but this script still only "
            "writes to the *_available_capacities filenames — promoting to a canonical table "
            "is a separate, explicit step."
        )
    lines.append("")
    lines.append(f"Input chunks: {', '.join(str(p) for p in input_paths)}")
    lines.append("")

    lines.append("## Per-capacity mean misses by policy")
    lines.append("")
    header_cols = "| policy | " + " | ".join(f"cap{c}" for c in available_capacities) + " |"
    sep_cols = "|---|" + "---|" * len(available_capacities)
    lines.append(header_cols)
    lines.append(sep_cols)
    for policy in policies:
        cells = []
        for capacity in available_capacities:
            key = (capacity, policy)
            cells.append(f"{statistics.mean(misses_by_cap_policy[key]):.1f}" if key in misses_by_cap_policy else "-")
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-capacity relative gap vs LRU (%)")
    lines.append("")
    lines.append(header_cols)
    lines.append(sep_cols)
    for policy in policies:
        cells = []
        for capacity in available_capacities:
            ref_key = (capacity, REFERENCE_POLICY)
            key = (capacity, policy)
            if key in misses_by_cap_policy and ref_key in misses_by_cap_policy:
                ref_mean = statistics.mean(misses_by_cap_policy[ref_key])
                mean_misses = statistics.mean(misses_by_cap_policy[key])
                cells.append(f"{100.0 * (mean_misses - ref_mean) / ref_mean:+.2f}" if ref_mean else "-")
            else:
                cells.append("-")
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(f"## {PROPOSED_POLICY} gap vs LRU / SIEVE / FIFO-Reinsertion (%)")
    lines.append("")
    lines.append("| capacity | vs lru | vs sieve | vs fifo_reinsertion |")
    lines.append("|---|---|---|---|")
    for capacity in available_capacities:
        gaps = proposed_gap.get(capacity, {})
        cells = [f"{gaps[b]:+.2f}" if b in gaps else "-" for b in COMPARISON_BASELINES]
        lines.append(f"| {capacity} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-trace-family ranking (1 = lowest mean misses) by capacity")
    lines.append("")
    for capacity in available_capacities:
        lines.append(f"### Capacity {capacity}")
        lines.append("")
        lines.append("| trace_family | ranked policies (best -> worst) |")
        lines.append("|---|---|")
        for family in families:
            ranked = family_rankings[capacity][family]
            ranked_str = ", ".join(f"{p}({m:.0f})" for p, m, _rank in ranked)
            lines.append(f"| {family} | {ranked_str} |")
        lines.append("")

    lines.append(f"## Trend across available capacities ({' -> '.join(trend_caps) if trend_caps else 'none'})")
    lines.append("")
    if len(trend_caps) >= 2:
        lines.append("| policy | " + " | ".join(f"cap{c}" for c in trend_caps) + " | direction |")
        lines.append("|---|" + "---|" * len(trend_caps) + "---|")
        for policy in policies:
            series = trend_series[policy]
            cells = [f"{v:.1f}" if v is not None else "-" for v in series]
            present = [v for v in series if v is not None]
            if len(present) >= 2:
                direction = "decreasing (improving)" if present[-1] < present[0] else (
                    "increasing (worsening)" if present[-1] > present[0] else "flat"
                )
            else:
                direction = "insufficient data"
            lines.append(f"| {policy} | " + " | ".join(cells) + f" | {direction} |")
    else:
        lines.append("Fewer than two capacities available — trend cannot be computed yet.")
    lines.append("")

    lines.append("---")
    lines.append(f"_{draft_label}. Regenerate after each new capacity chunk completes and is verified._")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_MD}")
    print()
    print(f"Status: {draft_label}")
    print(f"Requested capacities: {requested_capacities}")
    print(f"Available capacities: {available_capacities}")
    if missing_capacities:
        print(f"Missing capacities (draft only, not final): {missing_capacities}")
    else:
        print("All requested capacities present in inputs (still written as *_available_capacities, not canonical).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
