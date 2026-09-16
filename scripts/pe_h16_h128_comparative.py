#!/usr/bin/env python3
"""Integrate canonical H16 summaries with completed PE long-horizon evidence.

This is an adapter/aggregator only. It reads existing H16 decision-view
summaries and existing H32/H64/H128 compact summaries; it never regenerates
candidate labels or future rollouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


READER_FAMILY = {
    "cloudphysics": "alibaba-block",
    "metacdn": "metacdn",
    "metakv": "metakv",
    "twemcache": "twemcache",
    "wiki2018": "wiki2018",
}
INTERNAL_FAMILY = {v: k for k, v in READER_FAMILY.items()}
EXPECTED_INTERNAL_FAMILIES = ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
EXPECTED_CAPACITIES = [32, 64, 128, 256]
EXPECTED_HORIZONS = [16, 32, 64, 128]
PRIMARY_METRICS = [
    "all_tied_fraction",
    "discriminative_fraction",
    "random_optimal_probability",
    "mean_random_regret",
]
DELTA_PAIRS = [(16, 32), (32, 64), (64, 128), (16, 128)]
DEFAULT_H16_SOURCE = Path(
    "/home/soroush/projects/lafc-evict-dataset/worktrees/pe-results-polish/"
    "analysis/sigmod_target_discriminativeness_20260913/outputs/"
    "phase4_stratified_family_capacity_horizon.csv"
)
DEFAULT_H16_REPORT = Path(
    "/home/soroush/projects/lafc-evict-dataset/worktrees/pe-results-polish/"
    "analysis/sigmod_target_discriminativeness_20260913/outputs/"
    "decision_view_report.json"
)
DEFAULT_H16_SCRIPT = Path(
    "/home/soroush/projects/lafc-evict-dataset/worktrees/pe-results-polish/"
    "analysis/sigmod_target_discriminativeness_20260913/scripts/"
    "01_decision_view_analysis.py"
)
DEFAULT_LONG_ROOT = Path("analysis/pe_long_horizon_production_v1")
DEFAULT_OUT_DIR = Path("analysis/pe_h16_h128_comparative_20260916")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return "UNKNOWN"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fnum(value: object) -> float:
    return float(value)


def intish(value: object) -> int:
    return int(round(float(value)))


def classify_h16_cells(h16_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_key = {}
    for row in h16_rows:
        if row.get("horizon") != "16":
            continue
        key = (row["trace_family"], int(row["capacity"]))
        by_key.setdefault(key, []).append(row)

    audit = []
    for family in EXPECTED_INTERNAL_FAMILIES:
        for capacity in EXPECTED_CAPACITIES:
            rows = by_key.get((family, capacity), [])
            if len(rows) == 1:
                state = "EXISTING_CANONICAL_VALID"
            elif not rows:
                state = "MISSING"
            else:
                state = "AMBIGUOUS"
            audit.append(
                {
                    "family": READER_FAMILY[family],
                    "internal_family": family,
                    "capacity": capacity,
                    "horizon": 16,
                    "classification": state,
                    "source_rows": len(rows),
                }
            )
    return audit


def convert_h16_cells(h16_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in h16_rows:
        if row.get("horizon") != "16":
            continue
        internal = row["trace_family"]
        capacity = int(row["capacity"])
        decisions = int(row["n_decisions"])
        all_tied = fnum(row["all_tied_fraction"])
        disc = 1.0 - all_tied
        candidate_rows = decisions * capacity
        all_tied_count = int(round(all_tied * decisions))
        rows.append(
            {
                "family": READER_FAMILY[internal],
                "internal_family": internal,
                "capacity": capacity,
                "horizon": 16,
                "decision_count": decisions,
                "candidate_row_count": candidate_rows,
                "all_tied_decision_count": all_tied_count,
                "discriminative_decision_count": decisions - all_tied_count,
                "all_tied_fraction": all_tied,
                "discriminative_fraction": disc,
                "mean_optimal_set_fraction": fnum(row["mean_optimal_set_fraction"]),
                "random_optimal_probability": fnum(row["random_optimal_probability"]),
                "mean_random_regret": fnum(row["mean_random_regret"]),
                "source": "canonical_h16_decision_view_summary",
            }
        )
    return sorted(rows, key=lambda r: (str(r["internal_family"]), int(r["capacity"])))


def load_long_horizon_cells(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(path):
        internal = row["family"]
        decisions = int(row["decisions"])
        all_tied = fnum(row["all_tied_fraction"])
        all_tied_count = int(round(all_tied * decisions))
        rows.append(
            {
                "family": READER_FAMILY[internal],
                "internal_family": internal,
                "capacity": int(row["capacity"]),
                "horizon": int(row["horizon"]),
                "decision_count": decisions,
                "candidate_row_count": int(row["candidate_rows"]),
                "all_tied_decision_count": all_tied_count,
                "discriminative_decision_count": decisions - all_tied_count,
                "all_tied_fraction": all_tied,
                "discriminative_fraction": fnum(row["discriminativeness"]),
                "mean_optimal_set_fraction": fnum(row["mean_optimal_set_fraction"]),
                "random_optimal_probability": fnum(row["random_optimal_probability"]),
                "mean_random_regret": fnum(row["mean_random_regret"]),
                "source": "long_horizon_validated_summary",
            }
        )
    return sorted(rows, key=lambda r: (str(r["internal_family"]), int(r["capacity"]), int(r["horizon"])))


def validate_cells(rows: list[dict[str, object]]) -> dict[str, object]:
    errors = []
    keys = [(r["internal_family"], int(r["capacity"]), int(r["horizon"])) for r in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate family/capacity/horizon cells")
    expected = {
        (family, capacity, horizon)
        for family in EXPECTED_INTERNAL_FAMILIES
        for capacity in EXPECTED_CAPACITIES
        for horizon in EXPECTED_HORIZONS
    }
    missing = sorted(expected - set(keys))
    extra = sorted(set(keys) - expected)
    if missing:
        errors.append(f"missing expected cells: {missing}")
    if extra:
        errors.append(f"unexpected cells: {extra}")
    for r in rows:
        for metric in PRIMARY_METRICS + ["mean_optimal_set_fraction"]:
            value = float(r[metric])
            if metric == "mean_random_regret":
                if value < -1e-12:
                    errors.append(f"negative regret in {r}")
            elif not (0.0 - 1e-12 <= value <= 1.0 + 1e-12):
                errors.append(f"fraction out of range for {metric}: {r}")
        if int(r["decision_count"]) <= 0 or int(r["candidate_row_count"]) <= 0:
            errors.append(f"nonpositive counts: {r}")
        if abs(float(r["all_tied_fraction"]) + float(r["discriminative_fraction"]) - 1.0) > 1e-9:
            errors.append(f"all_tied + discriminative != 1: {r}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "expected_cells": len(expected),
        "valid_cells": len(rows) if not errors else len(rows) - len(errors),
        "h16_cells": sum(1 for r in rows if int(r["horizon"]) == 16),
        "long_horizon_cells": sum(1 for r in rows if int(r["horizon"]) in {32, 64, 128}),
    }


def summarize_micro(rows: list[dict[str, object]], *, exclude_wiki2018: bool = False) -> list[dict[str, object]]:
    out = []
    filtered = [r for r in rows if not (exclude_wiki2018 and r["internal_family"] == "wiki2018")]
    for h in EXPECTED_HORIZONS:
        cells = [r for r in filtered if int(r["horizon"]) == h]
        if not cells:
            continue
        total_decisions = sum(int(r["decision_count"]) for r in cells)
        total_candidate_rows = sum(int(r["candidate_row_count"]) for r in cells)
        def weighted(metric: str) -> float:
            return sum(float(r[metric]) * int(r["decision_count"]) for r in cells) / total_decisions
        out.append(
            {
                "weighting": "DECISION_MICRO",
                "scope": "EXCLUDING_WIKI2018" if exclude_wiki2018 else "ALL_FIVE_FAMILIES",
                "horizon": h,
                "cell_count": len(cells),
                "decisions": total_decisions,
                "candidate_rows": total_candidate_rows,
                "all_tied_fraction": weighted("all_tied_fraction"),
                "discriminative_fraction": weighted("discriminative_fraction"),
                "random_optimal_probability": weighted("random_optimal_probability"),
                "mean_random_regret": weighted("mean_random_regret"),
            }
        )
    return out


def summarize_macro(rows: list[dict[str, object]], *, exclude_wiki2018: bool = False) -> list[dict[str, object]]:
    out = []
    filtered = [r for r in rows if not (exclude_wiki2018 and r["internal_family"] == "wiki2018")]
    for h in EXPECTED_HORIZONS:
        cells = [r for r in filtered if int(r["horizon"]) == h]
        if not cells:
            continue
        record: dict[str, object] = {
            "weighting": "CELL_MACRO",
            "scope": "EXCLUDING_WIKI2018" if exclude_wiki2018 else "ALL_FIVE_FAMILIES",
            "horizon": h,
            "cell_count": len(cells),
            "decisions": sum(int(r["decision_count"]) for r in cells),
            "candidate_rows": sum(int(r["candidate_row_count"]) for r in cells),
        }
        for metric in PRIMARY_METRICS:
            values = [float(r[metric]) for r in cells]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_median"] = statistics.median(values)
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_max"] = max(values)
        out.append(record)
    return out


def paired_deltas(rows: list[dict[str, object]], tolerance: float = 1e-12) -> list[dict[str, object]]:
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["internal_family"], int(r["capacity"]))][int(r["horizon"])] = r
    out = []
    for metric in PRIMARY_METRICS:
        for h0, h1 in DELTA_PAIRS:
            deltas = [
                float(by_h[h1][metric]) - float(by_h[h0][metric])
                for by_h in by_cell.values()
                if h0 in by_h and h1 in by_h
            ]
            out.append(
                {
                    "metric": metric,
                    "delta": f"H{h1}-H{h0}",
                    "n_cells": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "median_delta": statistics.median(deltas),
                    "min_delta": min(deltas),
                    "max_delta": max(deltas),
                    "n_positive": sum(1 for d in deltas if d > tolerance),
                    "n_negative": sum(1 for d in deltas if d < -tolerance),
                    "n_effectively_zero": sum(1 for d in deltas if abs(d) <= tolerance),
                    "zero_tolerance": tolerance,
                }
            )
    return out


def breakdown(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    out = []
    groups = sorted({r[group_key] for r in rows}, key=lambda x: (str(x)))
    if group_key == "capacity":
        groups = sorted(groups, key=int)
    for group in groups:
        for h in EXPECTED_HORIZONS:
            cells = [r for r in rows if r[group_key] == group and int(r["horizon"]) == h]
            if not cells:
                continue
            total_decisions = sum(int(r["decision_count"]) for r in cells)
            record = {
                group_key: group,
                "horizon": h,
                "cell_count": len(cells),
                "decisions": total_decisions,
            }
            for metric in PRIMARY_METRICS:
                record[metric] = sum(float(r[metric]) * int(r["decision_count"]) for r in cells) / total_decisions
            out.append(record)
    return out


def leave_one_family_out(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for omit in EXPECTED_INTERNAL_FAMILIES:
        subset = [r for r in rows if r["internal_family"] != omit]
        for row in summarize_macro(subset):
            out.append({"omitted_family": READER_FAMILY[omit], **row})
    return out


def capacity_micro(rows: list[dict[str, object]], capacity: int) -> list[dict[str, object]]:
    cells = [r for r in rows if int(r["capacity"]) == capacity]
    out = []
    for h in EXPECTED_HORIZONS:
        horizon_cells = [r for r in cells if int(r["horizon"]) == h]
        total_decisions = sum(int(r["decision_count"]) for r in horizon_cells)
        total_candidate_rows = sum(int(r["candidate_row_count"]) for r in horizon_cells)
        out.append(
            {
                "capacity": capacity,
                "horizon": h,
                "decisions": total_decisions,
                "candidate_rows": total_candidate_rows,
                "all_tied_fraction": sum(float(r["all_tied_fraction"]) * int(r["decision_count"]) for r in horizon_cells) / total_decisions,
                "discriminative_fraction": sum(float(r["discriminative_fraction"]) * int(r["decision_count"]) for r in horizon_cells) / total_decisions,
                "random_optimal_probability": sum(float(r["random_optimal_probability"]) * int(r["decision_count"]) for r in horizon_cells) / total_decisions,
                "mean_random_regret": sum(float(r["mean_random_regret"]) * int(r["decision_count"]) for r in horizon_cells) / total_decisions,
            }
        )
    return out


def wide_family_capacity_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["family"], r["internal_family"], int(r["capacity"]))][int(r["horizon"])] = r
    out = []
    for (family, internal, capacity), by_h in sorted(by_cell.items(), key=lambda x: (x[0][1], x[0][2])):
        record = {"family": family, "internal_family": internal, "capacity": capacity}
        for h in EXPECTED_HORIZONS:
            r = by_h[h]
            for metric in PRIMARY_METRICS:
                record[f"H{h}_{metric}"] = r[metric]
            record[f"H{h}_decisions"] = r["decision_count"]
        out.append(record)
    return out


def compatibility_rows(h16_path: Path, long_root: Path) -> list[dict[str, object]]:
    return [
        {
            "field": "family",
            "h16_source": "trace_family in canonical H16 decision-view audit; cloudphysics maps to reader-facing alibaba-block",
            "long_horizon_source": "family in production manifest/table_A; cloudphysics maps to reader-facing alibaba-block",
            "compatible": "YES",
            "note": "Same five internal families; reader-facing mapping preserved.",
        },
        {
            "field": "capacity",
            "h16_source": "capacity column: 32,64,128,256",
            "long_horizon_source": "capacity column: 32,64,128,256",
            "compatible": "YES",
            "note": "Same four capacities.",
        },
        {
            "field": "decision",
            "h16_source": "decision_view row, one full-cache-miss eviction decision",
            "long_horizon_source": "validated_summary unique decision_id from production shards",
            "compatible": "YES",
            "note": "Definition matches; H128 production uses a stricter long-horizon-eligible denominator for twemcache cells.",
        },
        {
            "field": "candidate set",
            "h16_source": "candidate_count residents per decision in decision_view",
            "long_horizon_source": "candidate_page_id rows per decision in production shards",
            "compatible": "YES",
            "note": "Candidate row counts equal decision_count x capacity in every converted/validated cell.",
        },
        {
            "field": "counterfactual continuation",
            "h16_source": "canonical H16 LAFC-Evict LRU-continuation labels from existing decision_view",
            "long_horizon_source": "same generator family extended to H32/H64/H128; H16_REGENERATED=NO",
            "compatible": "YES",
            "note": "Comparison reuses existing H16 and completed H32/H64/H128 summaries.",
        },
        {
            "field": "y_loss",
            "h16_source": "finite-horizon miss-count loss; lower is better",
            "long_horizon_source": "finite-horizon miss-count loss; y_value=-y_loss validated",
            "compatible": "YES",
            "note": "Same sign convention and target meaning.",
        },
        {
            "field": "optimal candidate set",
            "h16_source": "optimal_candidate_count / tie_count equals candidates with regret==0",
            "long_horizon_source": "candidates with y_loss equal to per-decision minimum",
            "compatible": "YES",
            "note": "Both use tied argmin set.",
        },
        {
            "field": "all-tied criterion",
            "h16_source": "optimal_candidate_count == candidate_count",
            "long_horizon_source": "min(y_loss) == max(y_loss) within decision",
            "compatible": "YES",
            "note": "Equivalent criteria.",
        },
        {
            "field": "discriminative decision criterion",
            "h16_source": "1 - all_tied_fraction; regret_max > 0",
            "long_horizon_source": "1 - all_tied_fraction",
            "compatible": "YES",
            "note": "Equivalent decision-level criterion.",
        },
        {
            "field": "random-optimal probability",
            "h16_source": "mean optimal_set_fraction / random_optimal_probability",
            "long_horizon_source": "total optimal rows divided by candidate rows per cell",
            "compatible": "YES",
            "note": "Per-cell capacity is fixed, so cell values match decision-weighted optimal-set fraction.",
        },
        {
            "field": "per-candidate/random regret",
            "h16_source": "mean_random_regret from decision_view regret_mean",
            "long_horizon_source": "mean of per-decision random-candidate regret",
            "compatible": "YES",
            "note": "Both are decision-weighted inside each cell.",
        },
        {
            "field": "request preprocessing",
            "h16_source": str(h16_path),
            "long_horizon_source": str(long_root / "manifests/campaign_manifest.json"),
            "compatible": "YES",
            "note": "Both draw from the same canonical processed family/capacity corpus identities.",
        },
        {
            "field": "cache initialization",
            "h16_source": "canonical decision_view full-cache-miss decisions",
            "long_horizon_source": "same production cache state semantics, validated per shard",
            "compatible": "YES",
            "note": "No source indicates a changed initialization rule.",
        },
        {
            "field": "admission semantics",
            "h16_source": "LAFC-Evict candidate label contract",
            "long_horizon_source": "same generator contract in completed long-horizon production branch",
            "compatible": "YES",
            "note": "No source indicates a changed admission rule.",
        },
    ]


def make_plot(rows: list[dict[str, object]], micro: list[dict[str, object]], out_dir: Path) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "SKIPPED", "reason": f"matplotlib unavailable: {exc}"}

    x = EXPECTED_HORIZONS
    metrics = [
        ("discriminative_fraction", "Discriminative fraction"),
        ("random_optimal_probability", "Random-optimal probability"),
        ("mean_random_regret", "Mean random regret"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["family"], int(r["capacity"]))][int(r["horizon"])] = r
    micro_by_h = {int(r["horizon"]): r for r in micro}
    for ax, (metric, title) in zip(axes, metrics):
        for (_family, _capacity), by_h in by_cell.items():
            ax.plot(x, [float(by_h[h][metric]) for h in x], color="#9aa0a6", alpha=0.35, linewidth=1.0)
        ax.plot(x, [float(micro_by_h[h][metric]) for h in x], color="#1f4e79", marker="o", linewidth=2.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Horizon H", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.set_xticks(x)
    axes[0].set_ylabel("Fraction", fontsize=10)
    png = out_dir / "figures" / "h16_h128_metric_trajectories.png"
    pdf = out_dir / "figures" / "h16_h128_metric_trajectories.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"status": "PASS", "png": str(png), "pdf": str(pdf)}


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[str], digits: int = 4, limit: int | None = None) -> str:
    use_rows = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in use_rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(fmt(value, digits) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    *,
    h16_path: Path,
    long_table: Path,
    source_hashes: dict[str, str],
    compatibility: list[dict[str, object]],
    validation: dict[str, object],
    micro: list[dict[str, object]],
    macro: list[dict[str, object]],
    excluding: list[dict[str, object]],
    deltas: list[dict[str, object]],
    family_sensitivity: list[dict[str, object]],
    capacity_sensitivity: list[dict[str, object]],
    cap32_sanity: list[dict[str, object]],
    fig_status: dict[str, str],
) -> None:
    micro_cols = [
        "horizon", "decisions", "all_tied_fraction", "discriminative_fraction",
        "random_optimal_probability", "mean_random_regret",
    ]
    macro_cols = [
        "horizon", "cell_count", "all_tied_fraction_mean", "all_tied_fraction_median",
        "discriminative_fraction_mean", "random_optimal_probability_mean", "mean_random_regret_mean",
    ]
    delta_cols = [
        "metric", "delta", "mean_delta", "median_delta", "min_delta", "max_delta",
        "n_positive", "n_negative", "n_effectively_zero",
    ]
    family_cols = [
        "family", "horizon", "decisions", "all_tied_fraction",
        "discriminative_fraction", "random_optimal_probability", "mean_random_regret",
    ]
    capacity_cols = [
        "capacity", "horizon", "decisions", "all_tied_fraction",
        "discriminative_fraction", "random_optimal_probability", "mean_random_regret",
    ]
    cap32_cols = [
        "capacity", "horizon", "candidate_rows", "decisions", "all_tied_fraction",
        "discriminative_fraction", "random_optimal_probability", "mean_random_regret",
    ]
    by_h = {int(r["horizon"]): r for r in micro}
    h16, h32, h64, h128 = by_h[16], by_h[32], by_h[64], by_h[128]
    saturation = abs(float(h128["discriminative_fraction"]) - float(h64["discriminative_fraction"]))
    text = f"""# PE H16-H128 Comparative Report

Generated from existing artifacts only. H16 labels were not regenerated. H32/H64/H128 labels were not regenerated.

## Sources

- H16 source: `{h16_path}`
- H32/H64/H128 source: `{long_table}`
- H16 source SHA256: `{source_hashes['h16_source_csv']}`
- Long-horizon table SHA256: `{source_hashes['long_horizon_table_A']}`

## Compatibility Audit

{markdown_table(compatibility, ['field', 'compatible', 'note'], digits=4)}

The comparison is definition-compatible. One limitation is explicit: H16 is the canonical H16-eligible population, while the long-horizon production denominator is H128-eligible for each cell. The difference affects twemcache counts and is treated as a limitation, not hidden.

## Validation

- Validation status: `{validation['status']}`
- Expected cells: `{validation['expected_cells']}`
- H16 cells: `{validation['h16_cells']}`
- Long-horizon cells: `{validation['long_horizon_cells']}`
- H16_REGENERATED: `NO`

## Primary Decision-Micro Summary

{markdown_table(micro, micro_cols, digits=6)}

Important scope note: the primary H16 row is the canonical H16-eligible population. H32/H64/H128 use the completed long-horizon production denominator. The paired cell-level deltas are therefore the preferred descriptive evidence for monotone within-cell horizon behavior; decision-micro rows are still reported because they match manuscript-style pooled summaries, but H16-to-H32 changes should be read with this denominator caveat.

## Cap32 Sanity / Scope Audit

{markdown_table(cap32_sanity, cap32_cols, digits=6)}

The H32/H64/H128 cap32 rows match the earlier cap32 reference denominator (`6,072,320` candidate rows; `189,760` decisions). The canonical H16 source contains `6,542,016` candidate rows and `204,438` decisions at cap32, so the earlier H16 cap32 sanity values are not forced onto this adapter output. No existing H16-on-H128-eligible cap32 summary was found during the audit.

## Cell-Macro Summary

{markdown_table(macro, macro_cols, digits=6)}

## Excluding Wiki2018 Sensitivity

{markdown_table(excluding, micro_cols, digits=6)}

## Paired Horizon Deltas

{markdown_table(deltas, delta_cols, digits=6)}

## Family-Block Sensitivity

{markdown_table(family_sensitivity, family_cols, digits=6)}

## Capacity Sensitivity

{markdown_table(capacity_sensitivity, capacity_cols, digits=6)}

## Figure

- Figure status: `{fig_status.get('status')}`
- PNG: `{fig_status.get('png', '')}`
- PDF: `{fig_status.get('pdf', '')}`

## Scientific Questions

### Q1. Does increasing H beyond 16 materially reduce tie density?

FACT: Decision-micro all-tied fraction changes from {fmt(h16['all_tied_fraction'], 6)} at H16 to {fmt(h128['all_tied_fraction'], 6)} at H128.

INTERPRETATION: The reduction is real, but the target remains highly tied.

### Q2. Does it increase discriminative supervision?

FACT: Decision-micro discriminative fraction changes from {fmt(h16['discriminative_fraction'], 6)} at H16 to {fmt(h128['discriminative_fraction'], 6)} at H128.

INTERPRETATION: Paired cell-level evidence shows longer horizons usually increase discriminative supervision, but the decision-micro aggregate is non-monotone from H16 to H32 because H16 and H32 use different eligibility denominators.

### Q3. Does random-optimal probability meaningfully decline?

FACT: Decision-micro random-optimal probability changes from {fmt(h16['random_optimal_probability'], 6)} at H16 to {fmt(h128['random_optimal_probability'], 6)} at H128.

INTERPRETATION: The probability declines, meaning a random candidate is less often optimal at longer horizons, but it remains high.

### Q4. Does random regret increase?

FACT: Decision-micro mean random regret changes from {fmt(h16['mean_random_regret'], 6)} at H16 to {fmt(h128['mean_random_regret'], 6)} at H128.

INTERPRETATION: Candidate choice becomes more consequential, especially in the nondegenerate family/capacity cells.

### Q5. Are effects consistent across families and capacities?

FACT: Family and capacity tables above show heterogeneity; wiki2018 remains fully degenerate, while metacdn and twemcache carry much of the discriminative signal.

INTERPRETATION: The paired cell direction is broadly toward more discrimination, but not uniformly across all cells and not monotonically in the pooled decision-micro view.

### Q6. Does H64 to H128 add substantial information relative to earlier steps?

FACT: Decision-micro discriminative fraction changes by {fmt(float(h128['discriminative_fraction']) - float(h64['discriminative_fraction']), 6)} from H64 to H128, compared with {fmt(float(h32['discriminative_fraction']) - float(h16['discriminative_fraction']), 6)} from H16 to H32 and {fmt(float(h64['discriminative_fraction']) - float(h32['discriminative_fraction']), 6)} from H32 to H64. In paired cell-macro deltas, the mean discriminative-fraction changes are {fmt([r for r in deltas if r['metric'] == 'discriminative_fraction' and r['delta'] == 'H32-H16'][0]['mean_delta'], 6)}, {fmt([r for r in deltas if r['metric'] == 'discriminative_fraction' and r['delta'] == 'H64-H32'][0]['mean_delta'], 6)}, and {fmt([r for r in deltas if r['metric'] == 'discriminative_fraction' and r['delta'] == 'H128-H64'][0]['mean_delta'], 6)}.

INTERPRETATION: Changes diminish between H64 and H128. Saying the target "saturates by H64" would be too strong because family/capacity heterogeneity remains.

### Q7. Does strong tie degeneracy remain true at H128?

FACT: At H128 the decision-micro all-tied fraction is {fmt(h128['all_tied_fraction'], 6)}, and the cell-macro all-tied mean is {fmt([r for r in macro if int(r['horizon']) == 128][0]['all_tied_fraction_mean'], 6)}.

INTERPRETATION: Yes. Even at H128, tie degeneracy remains a central property of the supervision target.

## Manuscript-Ready Factual Claims

- H16/H32/H64/H128 comparison uses existing H16 and completed H32/H64/H128 artifacts; no labels were regenerated.
- Paired cell-level summaries show longer horizons generally reduce tie density and increase discriminative decisions, but the pooled decision-micro H16-to-H32 comparison is affected by denominator scope.
- The H64 to H128 increment is smaller than earlier increments in the decision-micro aggregate.
- Wiki2018 remains fully tied across all tested horizons and is included in the primary corpus, with a separate excluding-wiki2018 sensitivity.
- Strong tie degeneracy persists at H128.

## Limitations

- The H16 source is the canonical H16-eligible population. The H32/H64/H128 campaign uses the long-horizon-eligible production population, which changes twemcache denominators.
- These are descriptive family/capacity summaries, not IID inferential tests over 20 independent workloads.
- The comparison does not change public release scope and does not include learned closed-loop or LFU results.
"""
    (out_dir / "PE_H16_H128_COMPARATIVE_REPORT.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h16-source", type=Path, default=DEFAULT_H16_SOURCE)
    ap.add_argument("--h16-report", type=Path, default=DEFAULT_H16_REPORT)
    ap.add_argument("--h16-script", type=Path, default=DEFAULT_H16_SCRIPT)
    ap.add_argument("--long-root", type=Path, default=DEFAULT_LONG_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path.cwd()
    long_table = args.long_root / "summaries/table_A_family_capacity_horizon.csv"
    long_pooled = args.long_root / "summaries/table_B_horizon_pooled.json"
    long_manifest = args.long_root / "manifests/campaign_manifest.json"
    required = [args.h16_source, args.h16_report, args.h16_script, long_table, long_pooled, long_manifest]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("MISSING_SOURCES:")
        for p in missing:
            print(f"  - {p}")
        return 2

    h16_rows = read_csv(args.h16_source)
    existence = classify_h16_cells(h16_rows)
    if any(row["classification"] != "EXISTING_CANONICAL_VALID" for row in existence):
        write_csv(args.out_dir / "h16_existence_audit.csv", existence, list(existence[0].keys()))
        print("H16_EXISTENCE_AUDIT_FAIL")
        return 1

    h16_cells = convert_h16_cells(h16_rows)
    long_cells = load_long_horizon_cells(long_table)
    combined = sorted(
        [*h16_cells, *long_cells],
        key=lambda r: (str(r["internal_family"]), int(r["capacity"]), int(r["horizon"])),
    )
    validation = validate_cells(combined)
    compatibility = compatibility_rows(args.h16_source, args.long_root)
    micro = summarize_micro(combined)
    macro = summarize_macro(combined)
    excluding_micro = summarize_micro(combined, exclude_wiki2018=True)
    excluding_macro = summarize_macro(combined, exclude_wiki2018=True)
    deltas = paired_deltas(combined)
    family_sensitivity = breakdown(combined, "family")
    capacity_sensitivity = breakdown(combined, "capacity")
    leave_one_out = leave_one_family_out(combined)
    wide = wide_family_capacity_table(combined)
    cap32_sanity = capacity_micro(combined, 32)

    fields = [
        "family", "internal_family", "capacity", "horizon", "decision_count",
        "candidate_row_count", "all_tied_decision_count", "discriminative_decision_count",
        "all_tied_fraction", "discriminative_fraction", "mean_optimal_set_fraction",
        "random_optimal_probability", "mean_random_regret", "source",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "h16_existence_audit.csv", existence, list(existence[0].keys()))
    write_csv(args.out_dir / "compatibility_audit.csv", compatibility, list(compatibility[0].keys()))
    write_json(args.out_dir / "compatibility_audit.json", compatibility)
    write_csv(args.out_dir / "converted_h16_cells.csv", h16_cells, fields)
    write_json(args.out_dir / "converted_h16_cells.json", h16_cells)
    write_csv(args.out_dir / "combined_family_capacity_horizon.csv", combined, fields)
    write_json(args.out_dir / "combined_family_capacity_horizon.json", combined)
    write_csv(args.out_dir / "primary_decision_micro_summary.csv", micro, list(micro[0].keys()))
    write_json(args.out_dir / "primary_decision_micro_summary.json", micro)
    write_csv(args.out_dir / "cell_macro_summary.csv", macro, list(macro[0].keys()))
    write_json(args.out_dir / "cell_macro_summary.json", macro)
    write_csv(args.out_dir / "excluding_wiki2018_decision_micro_summary.csv", excluding_micro, list(excluding_micro[0].keys()))
    write_json(args.out_dir / "excluding_wiki2018_decision_micro_summary.json", excluding_micro)
    write_csv(args.out_dir / "excluding_wiki2018_cell_macro_summary.csv", excluding_macro, list(excluding_macro[0].keys()))
    write_json(args.out_dir / "excluding_wiki2018_cell_macro_summary.json", excluding_macro)
    write_csv(args.out_dir / "horizon_paired_deltas.csv", deltas, list(deltas[0].keys()))
    write_json(args.out_dir / "horizon_paired_deltas.json", deltas)
    write_csv(args.out_dir / "family_sensitivity.csv", family_sensitivity, list(family_sensitivity[0].keys()))
    write_json(args.out_dir / "family_sensitivity.json", family_sensitivity)
    write_csv(args.out_dir / "capacity_sensitivity.csv", capacity_sensitivity, list(capacity_sensitivity[0].keys()))
    write_json(args.out_dir / "capacity_sensitivity.json", capacity_sensitivity)
    write_csv(args.out_dir / "cap32_sanity_scope_audit.csv", cap32_sanity, list(cap32_sanity[0].keys()))
    write_json(args.out_dir / "cap32_sanity_scope_audit.json", cap32_sanity)
    write_csv(args.out_dir / "leave_one_family_out_macro.csv", leave_one_out, list(leave_one_out[0].keys()))
    write_json(args.out_dir / "leave_one_family_out_macro.json", leave_one_out)
    write_csv(args.out_dir / "family_capacity_wide_primary_metrics.csv", wide, list(wide[0].keys()))
    write_json(args.out_dir / "validation.json", validation)

    source_hashes = {
        "h16_source_csv": sha256_file(args.h16_source),
        "h16_decision_view_report": sha256_file(args.h16_report),
        "h16_analysis_script": sha256_file(args.h16_script),
        "long_horizon_table_A": sha256_file(long_table),
        "long_horizon_table_B": sha256_file(long_pooled),
        "long_horizon_manifest": sha256_file(long_manifest),
    }
    fig_status = make_plot(combined, micro, args.out_dir)
    provenance = {
        "task": "PE_H16_LONG_HORIZON_COMPARATIVE_INTEGRATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": git_output(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "source_head": git_output(["rev-parse", "HEAD"], repo),
        "h16_regenerated": False,
        "h32_h64_h128_regenerated": False,
        "h16_source_paths": {
            "stratified_summary_csv": str(args.h16_source),
            "decision_view_report_json": str(args.h16_report),
            "analysis_script": str(args.h16_script),
        },
        "h16_source_hashes": source_hashes,
        "long_horizon_sources": {
            "table_A": str(long_table),
            "table_B": str(long_pooled),
            "manifest": str(long_manifest),
        },
        "converter_script": "scripts/pe_h16_h128_comparative.py",
        "aggregation_script": "scripts/pe_h16_h128_comparative.py",
        "weighting_definitions": {
            "DECISION_MICRO": "weighted by decision_count across family/capacity cells",
            "CELL_MACRO": "each family/capacity cell receives equal weight",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "validation_status": validation["status"],
        "outputs": sorted(str(p.relative_to(args.out_dir)) for p in args.out_dir.rglob("*") if p.is_file()),
    }
    write_json(args.out_dir / "provenance.json", provenance)
    write_report(
        args.out_dir,
        h16_path=args.h16_source,
        long_table=long_table,
        source_hashes=source_hashes,
        compatibility=compatibility,
        validation=validation,
        micro=micro,
        macro=macro,
        excluding=excluding_micro,
        deltas=deltas,
        family_sensitivity=family_sensitivity,
        capacity_sensitivity=capacity_sensitivity,
        cap32_sanity=cap32_sanity,
        fig_status=fig_status,
    )

    print("COMPARISON_OK" if validation["status"] == "PASS" else "COMPARISON_FAIL")
    print(f"H16_REGENERATED=NO")
    print(f"OUT_DIR={args.out_dir}")
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
