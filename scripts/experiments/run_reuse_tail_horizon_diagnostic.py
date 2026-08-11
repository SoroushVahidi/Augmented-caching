"""Run the reuse-tail / horizon-exceedance diagnostic.

Scientific quantity:

    P(T > H | object is resident at the decision point)

where T is the number of future request positions until a currently resident
candidate object is next requested.  T is not classical reuse distance or
stack distance; those count distinct intervening objects, while this diagnostic
counts request positions.  A candidate never requested again has T = infinity.

The decision population mirrors the eviction-loss target construction:
full-cache misses under the LRU reference cache state, every resident candidate
present before inserting the incoming missed object, restricted by default to
the canonical held-out score window [10000, 50000).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_END, SCORE_START
from lafc.reuse_tail_horizon import (
    DEFAULT_HORIZONS,
    PRIMARY_HORIZON,
    CellAccumulator,
    ReuseTailObservation,
    iter_resident_candidate_observations,
    merge_summary_rows,
    stable_candidate_key,
    summarize_accumulator,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FOLDS_DIR = REPO_ROOT / "configs/fair_cross_family_v1/folds"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis/reuse_tail_horizon_diagnostic_v1"
DEFAULT_FAMILIES = ("brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018")
DEFAULT_CAPACITIES = (32, 64, 128)
PROTOCOL_ID = "reuse_tail_horizon_diagnostic_v1"


SUMMARY_FIELDS = [
    "family",
    "trace",
    "capacity",
    "horizon",
    "decision_points",
    "resident_candidate_observations",
    "finite_reuse_count",
    "never_reused_count",
    "t_gt_h_count_including_never",
    "t_gt_h_count_eventually_reused",
    "t_le_h_count",
    "never_reused_fraction",
    "p_t_gt_h_including_never",
    "p_t_gt_h_eventually_reused",
    "p_t_le_h_including_never",
    "finite_t_min",
    "finite_t_q50",
    "finite_t_q75",
    "finite_t_q90",
    "finite_t_q95",
    "finite_t_q99",
    "finite_t_max",
    "runtime_seconds",
]

AGG_FIELDS = [
    "aggregate_scope",
    "family",
    "capacity",
    "horizon",
    "decision_points",
    "resident_candidate_observations",
    "finite_reuse_count",
    "never_reused_count",
    "t_gt_h_count_including_never",
    "t_gt_h_count_eventually_reused",
    "t_le_h_count",
    "never_reused_fraction",
    "p_t_gt_h_including_never",
    "p_t_gt_h_eventually_reused",
    "p_t_le_h_including_never",
]

AUDIT_FIELDS = [
    "family",
    "trace",
    "capacity",
    "decision_index",
    "decision_id",
    "candidate_key",
    "candidate_page_id",
    "next_reuse_request_index",
    "t",
    "never_reused",
    "horizon",
    "t_gt_h",
    "t_le_h",
]


class DiagnosticError(RuntimeError):
    pass


def _git_output(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_fold(family: str) -> Mapping[str, object]:
    path = FOLDS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"fold config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_trace_path(fold: Mapping[str, object], data_read_root: Path) -> Path:
    rel = Path(str(fold["test_trace_path"]))
    candidates = [
        data_read_root / rel,
        REPO_ROOT / rel,
        REPO_ROOT.parent / "Augmented-caching" / rel,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"test trace not found for {rel}")


def _parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _prepare_output_dir(out_dir: Path, *, allow_existing: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"output directory is non-empty; refusing to overwrite existing "
            f"scientific outputs: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.12g}"
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _audit_rows_for_observation(
    obs: ReuseTailObservation,
    horizons: Sequence[int],
    *,
    include_raw_candidate: bool,
) -> Iterable[Dict[str, object]]:
    for h in horizons:
        never = obs.never_reused
        t_gt_h = bool(never or obs.t > h)
        t_le_h = bool((not never) and obs.t <= h)
        yield {
            "family": obs.family,
            "trace": obs.trace_name,
            "capacity": obs.capacity,
            "decision_index": obs.decision_index,
            "decision_id": obs.decision_id,
            "candidate_key": stable_candidate_key(obs.family, obs.trace_name, obs.candidate_page_id),
            "candidate_page_id": obs.candidate_page_id if include_raw_candidate else "",
            "next_reuse_request_index": "" if obs.next_reuse_request_index is None else obs.next_reuse_request_index,
            "t": "inf" if never else int(obs.t),
            "never_reused": "yes" if never else "no",
            "horizon": int(h),
            "t_gt_h": "yes" if t_gt_h else "no",
            "t_le_h": "yes" if t_le_h else "no",
        }


def _run_cell(
    *,
    family: str,
    fold: Mapping[str, object],
    trace_path: Path,
    capacity: int,
    horizons: Sequence[int],
    score_start: int,
    score_end: int,
    audit_sample_limit: int,
    include_raw_candidate: bool,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    trace_hash = _sha256_of_file(trace_path)
    expected_hash = str(fold.get("test_trace_sha256", ""))
    if expected_hash and trace_hash != expected_hash:
        raise DiagnosticError(f"trace hash mismatch for {family}: {trace_path}")

    requests, _pages, _source = load_trace_from_any(str(trace_path))
    trace_name = str(fold["test_trace_name"])
    acc = CellAccumulator(
        family=family,
        trace_name=trace_name,
        capacity=capacity,
        horizons=horizons,
    )
    audit_rows: List[Dict[str, object]] = []
    t0 = time.time()
    last_decision_id: Optional[str] = None
    sampled_observations: set[tuple[str, str]] = set()
    any_sample_count = 0
    finite_sample_count = 0
    never_sample_count = 0

    for obs in iter_resident_candidate_observations(
        requests,
        family=family,
        trace_name=trace_name,
        capacity=capacity,
        score_start=score_start,
        score_end=score_end,
    ):
        if obs.decision_id != last_decision_id:
            acc.decision_points += 1
            last_decision_id = obs.decision_id
        acc.record(obs)
        sample_key = (obs.decision_id, obs.candidate_page_id)
        should_sample = any_sample_count < audit_sample_limit
        if obs.never_reused:
            should_sample = should_sample or never_sample_count < audit_sample_limit
        else:
            should_sample = should_sample or finite_sample_count < audit_sample_limit
        if should_sample and sample_key not in sampled_observations:
            audit_rows.extend(
                _audit_rows_for_observation(
                    obs,
                    horizons,
                    include_raw_candidate=include_raw_candidate,
                )
            )
            sampled_observations.add(sample_key)
            any_sample_count += 1
            if obs.never_reused:
                never_sample_count += 1
            else:
                finite_sample_count += 1

    runtime = time.time() - t0
    summary_rows: List[Dict[str, object]] = []
    for h in horizons:
        row = summarize_accumulator(acc, h)
        row["runtime_seconds"] = runtime
        summary_rows.append(row)

    integrity = {
        "family": family,
        "trace": trace_name,
        "trace_path": str(trace_path),
        "trace_sha256": trace_hash,
        "capacity": capacity,
        "request_count": len(requests),
        "score_start": score_start,
        "score_end": score_end,
        "decision_points": acc.decision_points,
        "resident_candidate_observations": acc.observations,
        "observations_equal_decisions_times_capacity": acc.observations == acc.decision_points * capacity,
        "runtime_seconds": runtime,
    }
    return summary_rows, audit_rows, integrity


def _aggregate_rows(cell_horizon_rows: Sequence[Mapping[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    by_h: Dict[int, List[Mapping[str, object]]] = {}
    by_family_h: Dict[tuple[str, int], List[Mapping[str, object]]] = {}
    by_capacity_h: Dict[tuple[int, int], List[Mapping[str, object]]] = {}
    by_family_capacity_h: Dict[tuple[str, int, int], List[Mapping[str, object]]] = {}

    for row in cell_horizon_rows:
        h = int(row["horizon"])
        family = str(row["family"])
        capacity = int(row["capacity"])
        by_h.setdefault(h, []).append(row)
        by_family_h.setdefault((family, h), []).append(row)
        by_capacity_h.setdefault((capacity, h), []).append(row)
        by_family_capacity_h.setdefault((family, capacity, h), []).append(row)

    summary_by_horizon: List[Dict[str, object]] = []
    for h, rows in sorted(by_h.items()):
        summary_by_horizon.append(
            merge_summary_rows(rows, group={"aggregate_scope": "overall", "family": "", "capacity": ""})
        )
    summary_by_family: List[Dict[str, object]] = []
    for (family, _h), rows in sorted(by_family_h.items()):
        summary_by_family.append(
            merge_summary_rows(rows, group={"aggregate_scope": "family", "family": family, "capacity": ""})
        )
    summary_by_capacity: List[Dict[str, object]] = []
    for (capacity, _h), rows in sorted(by_capacity_h.items()):
        summary_by_capacity.append(
            merge_summary_rows(rows, group={"aggregate_scope": "capacity", "family": "", "capacity": capacity})
        )
    summary_by_family_capacity_horizon: List[Dict[str, object]] = []
    for (family, capacity, _h), rows in sorted(by_family_capacity_h.items()):
        summary_by_family_capacity_horizon.append(
            merge_summary_rows(
                rows,
                group={"aggregate_scope": "family_capacity", "family": family, "capacity": capacity},
            )
        )
    return {
        "summary_by_horizon": summary_by_horizon,
        "summary_by_family": summary_by_family,
        "summary_by_capacity": summary_by_capacity,
        "summary_by_family_capacity_horizon": summary_by_family_capacity_horizon,
    }


def _quantile_rows(primary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    fields = ["finite_t_min", "finite_t_q50", "finite_t_q75", "finite_t_q90", "finite_t_q95", "finite_t_q99", "finite_t_max"]
    out = []
    for row in primary_rows:
        out.append(
            {
                "family": row["family"],
                "trace": row["trace"],
                "capacity": row["capacity"],
                "finite_reuse_count": row["finite_reuse_count"],
                **{field: row.get(field, "") for field in fields},
            }
        )
    return out


def _write_report(
    path: Path,
    *,
    primary_rows: Sequence[Mapping[str, object]],
    summary_by_horizon: Sequence[Mapping[str, object]],
    horizons: Sequence[int],
    families: Sequence[str],
    capacities: Sequence[int],
) -> None:
    overall_h4 = [r for r in summary_by_horizon if int(r["horizon"]) == PRIMARY_HORIZON]
    overall_h4_row = overall_h4[0] if overall_h4 else {}
    lines = [
        "# Reuse-Tail Horizon Diagnostic",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "## Definition",
        "",
        "`T` is the number of future request positions until a currently resident",
        "candidate object is next requested. If the object is never requested again",
        "in the remaining trace, `T = infinity`. This is not classical reuse",
        "distance or stack distance; those count distinct intervening objects,",
        "while this diagnostic counts request positions.",
        "",
        "Primary quantity: `P(T > H | object is resident at the decision point)`.",
        "The conditioning population is every resident candidate at each full-cache",
        "miss decision under the same LRU reference state used by the eviction-loss",
        "candidate-row construction, restricted to the canonical score window.",
        "",
        "## Causal Guardrail",
        "",
        "This measures potential unseen future reuse. A reuse after `H` does not",
        "prove that evicting the object caused an avoidable miss; causal excess",
        "misses require a policy counterfactual. This diagnostic addresses only",
        "whether potentially relevant reuse lies outside the finite supervision",
        "horizon.",
        "",
        "## Scope",
        "",
        f"- Families: `{', '.join(families)}`",
        f"- Capacities: `{', '.join(str(c) for c in capacities)}`",
        f"- Horizons reported from the same T samples: `{', '.join(str(h) for h in horizons)}`",
        "",
        "## Overall H=4 Result",
        "",
    ]
    if overall_h4_row:
        lines.extend(
            [
                f"- Resident-candidate observations: `{overall_h4_row['resident_candidate_observations']}`",
                f"- Decision points: `{overall_h4_row['decision_points']}`",
                f"- `P(T > 4 | resident)`, including never-reused: `{overall_h4_row['p_t_gt_h_including_never']}`",
                f"- `P(T > 4 | resident, eventually reused)`: `{overall_h4_row['p_t_gt_h_eventually_reused']}`",
                f"- `P(T <= 4 | resident)`: `{overall_h4_row['p_t_le_h_including_never']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Capacity Trend at H=4",
            "",
            "Descriptive only. Do not infer an H/C law from this diagnostic.",
            "",
            "| Family | " + " | ".join(f"C={capacity}" for capacity in capacities) + " |",
            "| --- | " + " | ".join("---:" for _ in capacities) + " |",
        ]
    )
    by_family_capacity = {
        (str(r["family"]), int(r["capacity"])): r
        for r in primary_rows
    }
    for family in families:
        vals = []
        for capacity in capacities:
            row = by_family_capacity.get((family, capacity), {})
            vals.append(str(row.get("p_t_gt_h_including_never", "")))
        lines.append(f"| {family} | " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "Compare this trend qualitatively with the already observed target-",
            "degeneracy trend: zero-margin pair fraction and mean optimal-set",
            "fraction increased with capacity. Any co-movement here is exploratory",
            "correlation only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--capacities", default=",".join(str(c) for c in DEFAULT_CAPACITIES))
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--score-start", type=int, default=SCORE_START)
    parser.add_argument("--score-end", type=int, default=SCORE_END)
    parser.add_argument("--data-read-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-sample-limit-per-cell", type=int, default=20)
    parser.add_argument("--include-raw-candidate-id", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    args = parser.parse_args()

    families = _parse_csv_strings(args.families)
    capacities = _parse_csv_ints(args.capacities)
    horizons = _parse_csv_ints(args.horizons)
    if PRIMARY_HORIZON not in horizons:
        raise ValueError(f"primary horizon {PRIMARY_HORIZON} must be included in --horizons")

    out_dir = args.out_dir
    _prepare_output_dir(out_dir, allow_existing=args.allow_existing_output_dir)

    config_snapshot = {
        "protocol_id": PROTOCOL_ID,
        "scientific_definition": "T is future request-position delay to the next request for a resident candidate; not classical reuse/stack distance.",
        "primary_quantity": "P(T > H | object is resident at the decision point)",
        "conditioning_population": "full-cache misses under LRU reference state; every resident candidate before incoming missed object is inserted; canonical score window",
        "families": families,
        "capacities": capacities,
        "horizons": horizons,
        "score_start": args.score_start,
        "score_end": args.score_end,
        "primary_horizon": PRIMARY_HORIZON,
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True) + "\n")

    t0 = time.time()
    max_seconds = None if args.max_wall_hours is None else args.max_wall_hours * 3600.0
    cell_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    cell_integrity: List[Dict[str, object]] = []
    expected_cells = len(families) * len(capacities)
    completed_cells = 0
    status = "COMPLETE"

    for family in families:
        fold = _load_fold(family)
        trace_path = _resolve_trace_path(fold, args.data_read_root)
        for capacity in capacities:
            if args.max_cells is not None and completed_cells >= args.max_cells:
                status = "PARTIAL_MAX_CELLS"
                break
            if max_seconds is not None and completed_cells > 0 and (time.time() - t0) >= max_seconds:
                status = "PARTIAL_WALL_TIME_GUARD"
                break
            print(f"[cell] family={family} capacity={capacity}", flush=True)
            rows, samples, integrity = _run_cell(
                family=family,
                fold=fold,
                trace_path=trace_path,
                capacity=capacity,
                horizons=horizons,
                score_start=args.score_start,
                score_end=args.score_end,
                audit_sample_limit=args.audit_sample_limit_per_cell,
                include_raw_candidate=args.include_raw_candidate_id,
            )
            cell_rows.extend(rows)
            audit_rows.extend(samples)
            cell_integrity.append(integrity)
            completed_cells += 1
        if status != "COMPLETE":
            break

    primary_rows = [r for r in cell_rows if int(r["horizon"]) == PRIMARY_HORIZON]
    aggregates = _aggregate_rows(cell_rows) if cell_rows else {
        "summary_by_horizon": [],
        "summary_by_family": [],
        "summary_by_capacity": [],
        "summary_by_family_capacity_horizon": [],
    }

    _write_csv(out_dir / "summary_by_family_capacity.csv", SUMMARY_FIELDS, primary_rows)
    _write_csv(out_dir / "summary_by_family_capacity_horizon.csv", SUMMARY_FIELDS, cell_rows)
    _write_csv(out_dir / "summary_by_horizon.csv", AGG_FIELDS, aggregates["summary_by_horizon"])
    _write_csv(out_dir / "summary_by_family.csv", AGG_FIELDS, aggregates["summary_by_family"])
    _write_csv(out_dir / "summary_by_capacity.csv", AGG_FIELDS, aggregates["summary_by_capacity"])
    _write_csv(out_dir / "summary_by_family_capacity_aggregate.csv", AGG_FIELDS, aggregates["summary_by_family_capacity_horizon"])
    _write_csv(
        out_dir / "reuse_delay_quantiles.csv",
        ["family", "trace", "capacity", "finite_reuse_count", "finite_t_min", "finite_t_q50", "finite_t_q75", "finite_t_q90", "finite_t_q95", "finite_t_q99", "finite_t_max"],
        _quantile_rows(primary_rows),
    )
    _write_csv(out_dir / "audit_samples.csv", AUDIT_FIELDS, audit_rows)

    integrity_summary = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "expected_cells": expected_cells,
        "completed_cells": completed_cells,
        "families": families,
        "capacities": capacities,
        "horizons": horizons,
        "score_start": args.score_start,
        "score_end": args.score_end,
        "all_completed_cell_observation_counts_consistent": all(
            bool(row["observations_equal_decisions_times_capacity"]) for row in cell_integrity
        ),
        "cell_integrity": cell_integrity,
        "runtime_seconds": time.time() - t0,
    }
    (out_dir / "integrity_summary.json").write_text(json.dumps(integrity_summary, indent=2, sort_keys=True) + "\n")

    provenance = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git_output(["rev-parse", "HEAD"]),
            "branch": _git_output(["branch", "--show-current"]),
            "dirty_status": _git_output(["status", "--short"]),
        },
        "argv": sys.argv,
        "config_snapshot": config_snapshot,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _write_report(
        out_dir / "report.md",
        primary_rows=primary_rows,
        summary_by_horizon=aggregates["summary_by_horizon"],
        horizons=horizons,
        families=families,
        capacities=capacities,
    )
    print(json.dumps({"status": status, "completed_cells": completed_cells, "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    _main()
