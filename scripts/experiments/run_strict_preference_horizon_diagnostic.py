"""Resumable compact diagnostic for H4 target informativeness and stability.

The runner measures target resolution and horizon comparisons; it does not
train or deploy a policy. Unit metadata is rewritten with canonical finalized
paths before atomic directory rename, preventing ``.tmp-*`` references.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import tempfile
import time
from collections import OrderedDict, deque
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Sequence

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_END, SCORE_START
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    _build_distinct_suffix_counts,
    _build_occurrence_index,
    build_candidate_rows_for_full_cache_state,
)
from lafc.target_degeneracy import deterministic_exact_tiebreak, exact_tie_metrics, eviction_loss_values, numeric_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/strict_preference_horizon_diagnostic_v1.json"
DEFAULT_OUT = REPO_ROOT / "analysis/strict_preference_horizon_diagnostic_v1"
FOLDS_DIR = REPO_ROOT / "configs/fair_cross_family_v1/folds"
HORIZONS = (4, 8, 16, 32)


def _finalized_output_paths(outputs: Mapping[str, Any], final_dir: Path) -> Dict[str, str]:
    return {key: str(final_dir / Path(value).name) for key, value in outputs.items()}


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _fold(family: str) -> Mapping[str, Any]:
    return json.loads((FOLDS_DIR / f"{family}.json").read_text(encoding="utf-8"))


def _trace_path(fold: Mapping[str, Any], data_root: Path) -> Path:
    relative = Path(str(fold["test_trace_path"]))
    for candidate in (data_root / relative, REPO_ROOT / relative, REPO_ROOT.parent / "Augmented-caching" / relative):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"trace not found: {relative}")


def _finite(value: Any) -> bool:
    return value is None or not isinstance(value, float) or math.isfinite(value)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _quantiles(values: Sequence[float]) -> Dict[str, float | None]:
    return numeric_summary(values)


def _candidate_values(*, requests, t: int, trace_name: str, family: str, capacity: int, horizon: int, cache_order, recent_req_hist, recent_hit_hist, occurrence_index, distinct_suffix_counts) -> Dict[str, float]:
    rows = build_candidate_rows_for_full_cache_state(
        requests=requests,
        request_index=t,
        capacity=capacity,
        trace_name=trace_name,
        trace_family=family,
        cfg=ObjectiveAblationConfig(horizon=horizon),
        cache_order=cache_order,
        bucket_by_page={},
        confidence_by_page={},
        recent_req_hist=recent_req_hist,
        recent_hit_hist=recent_hit_hist,
        occurrence_index=occurrence_index,
        distinct_suffix_counts=distinct_suffix_counts,
        include_features=False,
    )
    return eviction_loss_values(rows)


def _empty_horizon(horizon: int) -> Dict[str, Any]:
    return {
        "horizon": horizon,
        "h4_multiple_events": 0,
        "long_unique_winner_count": 0,
        "long_reduces_optimal_set_count": 0,
        "optimal_set_size_reduction_sum": 0,
        "long_optimal_set_intersects_h4_set_count": 0,
        "long_unique_winner_in_h4_set_count": 0,
        "strict_member_resolution_count": 0,
        "strict_member_resolution_unique_count": 0,
        "h4_unique_events": 0,
        "h4_unique_agreement_count": 0,
        "h4_unique_disagreement_count": 0,
        "h4_unique_strict_reversal_count": 0,
        "all_event_count": 0,
        "all_event_h4_long_jaccard_sum": 0.0,
        "long_optimal_set_sizes": [],
        "h4_tie_long_spreads": [],
    }


def _finalize_horizon(metrics: Dict[str, Any]) -> Dict[str, Any]:
    tie_count = metrics["h4_multiple_events"]
    unique_count = metrics["h4_unique_events"]
    all_count = metrics["all_event_count"]
    sizes = metrics.pop("long_optimal_set_sizes")
    spreads = metrics.pop("h4_tie_long_spreads")
    return {
        "horizon": metrics["horizon"],
        "h4_multiple_event_count": tie_count,
        "long_unique_winner_fraction_given_h4_multiple": _fraction(metrics["long_unique_winner_count"], tie_count),
        "long_reduces_h4_optimal_set_fraction": _fraction(metrics["long_reduces_optimal_set_count"], tie_count),
        "mean_h4_minus_long_optimal_set_size": _fraction(metrics["optimal_set_size_reduction_sum"], tie_count),
        "long_optimal_set_intersects_h4_set_fraction": _fraction(metrics["long_optimal_set_intersects_h4_set_count"], tie_count),
        "long_unique_winner_in_h4_set_fraction": _fraction(metrics["long_unique_winner_in_h4_set_count"], tie_count),
        "h4_tied_set_long_unique_resolution_fraction": _fraction(metrics["strict_member_resolution_unique_count"], metrics["strict_member_resolution_count"]),
        "h4_tied_set_long_unique_resolution_count": metrics["strict_member_resolution_unique_count"],
        "h4_tied_set_long_resolution_denominator": metrics["strict_member_resolution_count"],
        "h4_unique_event_count": unique_count,
        "h4_unique_winner_agreement_fraction": _fraction(metrics["h4_unique_agreement_count"], unique_count),
        "h4_unique_winner_disagreement_fraction": _fraction(metrics["h4_unique_disagreement_count"], unique_count),
        "h4_unique_strict_reversal_fraction": _fraction(metrics["h4_unique_strict_reversal_count"], unique_count),
        "all_event_count": all_count,
        "optimal_set_jaccard_mean": _fraction(metrics["all_event_h4_long_jaccard_sum"], all_count),
        "long_optimal_set_size_summary": _quantiles(sizes),
        "long_values_on_h4_tie_set_spread_summary": _quantiles(spreads),
    }


def _validate_summary(summary: Mapping[str, Any], family: str, capacity: int, trace_sha: str) -> None:
    if summary.get("status") != "COMPLETE":
        raise ValueError("unit is not complete")
    protocol = summary["protocol"]
    if protocol["history"] != [0, 10000] or protocol["score"] != [10000, 50000] or protocol["horizons"] != list(HORIZONS):
        raise ValueError("unit protocol mismatch")
    if protocol["capacity"] != capacity or summary["trace"]["family"] != family or summary["trace"]["sha256"] != trace_sha:
        raise ValueError("unit identity mismatch")
    h4 = summary["h4_informativeness"]
    if h4["event_count"] <= 0 or not all(_finite(v) for v in h4.values() if not isinstance(v, dict)):
        raise ValueError("invalid H4 summary")
    for comparison in summary["horizon_comparisons"]:
        if comparison["horizon"] not in (8, 16, 32):
            raise ValueError("unexpected comparison horizon")


def _analyze_unit(*, requests, family: str, capacity: int, trace_name: str, trace_path: Path, trace_sha: str, fold: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order: OrderedDict[str, None] = OrderedDict()
    recent_req_hist = deque(maxlen=ObjectiveAblationConfig().history_window)
    recent_hit_hist = deque(maxlen=ObjectiveAblationConfig().history_window)
    occurrence_index = _build_occurrence_index(requests)
    distinct_suffix_counts = _build_distinct_suffix_counts(requests)
    h4_event_count = 0
    h4_unique = h4_multiple = h4_all_tied = 0
    optimal_sizes: list[int] = []
    optimal_fractions: list[float] = []
    ordinary_margins: list[float] = []
    positive_margins: list[float] = []
    strict_margins: list[float] = []
    decision_relevance = {"reused_within_h4": 0, "reused_after_h4": 0, "never_reused_observed_suffix": 0}
    horizon_metrics = {h: _empty_horizon(h) for h in (8, 16, 32)}
    for t, request in enumerate(requests):
        pid = str(request.page_id)
        if pid in order:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue
        if t < SCORE_START or t >= SCORE_END:
            # The trajectory still follows the H4 exact oracle outside the scored window.
            h4_values_for_trajectory = _candidate_values(requests=requests, t=t, trace_name=trace_name, family=family, capacity=capacity, horizon=4, cache_order=list(order), recent_req_hist=recent_req_hist, recent_hit_hist=recent_hit_hist, occurrence_index=occurrence_index, distinct_suffix_counts=distinct_suffix_counts)
            order.pop(deterministic_exact_tiebreak(tuple(sorted(k for k, v in h4_values_for_trajectory.items() if v == min(h4_values_for_trajectory.values())))))
            order[pid] = None
            recent_req_hist.append(pid)
            continue
        h4_values = _candidate_values(requests=requests, t=t, trace_name=trace_name, family=family, capacity=capacity, horizon=4, cache_order=list(order), recent_req_hist=recent_req_hist, recent_hit_hist=recent_hit_hist, occurrence_index=occurrence_index, distinct_suffix_counts=distinct_suffix_counts)
        h4_metrics = exact_tie_metrics(h4_values)
        h4_set = set(h4_metrics.optimal_candidates)
        h4_event_count += 1
        optimal_sizes.append(h4_metrics.optimal_set_size)
        optimal_fractions.append(h4_metrics.optimal_set_fraction)
        ordinary_margins.append(h4_metrics.ordinary_margin)
        if h4_metrics.ordinary_margin > 0:
            positive_margins.append(h4_metrics.ordinary_margin)
        if h4_metrics.strict_distinct_margin is not None:
            strict_margins.append(h4_metrics.strict_distinct_margin)
        if h4_metrics.optimal_set_size == 1:
            h4_unique += 1
            next_positions = occurrence_index.get(next(iter(h4_set)), [])
            index = bisect.bisect_right(next_positions, t)
            if index >= len(next_positions):
                decision_relevance["never_reused_observed_suffix"] += 1
            elif next_positions[index] <= t + 4:
                decision_relevance["reused_within_h4"] += 1
            else:
                decision_relevance["reused_after_h4"] += 1
        else:
            h4_multiple += 1
        if h4_metrics.distinct_value_count == 1:
            h4_all_tied += 1
        for horizon, metrics in horizon_metrics.items():
            long_values = _candidate_values(requests=requests, t=t, trace_name=trace_name, family=family, capacity=capacity, horizon=horizon, cache_order=list(order), recent_req_hist=recent_req_hist, recent_hit_hist=recent_hit_hist, occurrence_index=occurrence_index, distinct_suffix_counts=distinct_suffix_counts)
            long_metrics = exact_tie_metrics(long_values)
            long_set = set(long_metrics.optimal_candidates)
            metrics["all_event_count"] += 1
            metrics["all_event_h4_long_jaccard_sum"] += _jaccard(h4_set, long_set)
            metrics["long_optimal_set_sizes"].append(long_metrics.optimal_set_size)
            if h4_metrics.optimal_set_size > 1:
                metrics["h4_multiple_events"] += 1
                metrics["long_unique_winner_count"] += int(long_metrics.optimal_set_size == 1)
                metrics["long_reduces_optimal_set_count"] += int(long_metrics.optimal_set_size < h4_metrics.optimal_set_size)
                metrics["optimal_set_size_reduction_sum"] += h4_metrics.optimal_set_size - long_metrics.optimal_set_size
                metrics["long_optimal_set_intersects_h4_set_count"] += int(bool(h4_set & long_set))
                metrics["long_unique_winner_in_h4_set_count"] += int(long_metrics.optimal_set_size == 1 and bool(h4_set & long_set))
                tied_values = {candidate: long_values[candidate] for candidate in h4_set}
                tied_best = min(tied_values.values())
                tied_winners = [candidate for candidate, value in tied_values.items() if value == tied_best]
                metrics["strict_member_resolution_count"] += 1
                metrics["strict_member_resolution_unique_count"] += int(len(tied_winners) == 1)
                metrics["h4_tie_long_spreads"].append(max(tied_values.values()) - min(tied_values.values()))
            else:
                metrics["h4_unique_events"] += 1
                winner = next(iter(h4_set))
                agrees = winner in long_set
                metrics["h4_unique_agreement_count"] += int(agrees)
                metrics["h4_unique_disagreement_count"] += int(not agrees)
                metrics["h4_unique_strict_reversal_count"] += int(long_metrics.optimal_set_size == 1 and not agrees)
        trajectory_choice = deterministic_exact_tiebreak(h4_metrics.optimal_candidates)
        order.pop(trajectory_choice)
        order[pid] = None
        recent_req_hist.append(pid)
    h4_summary = {
        "event_count": h4_event_count,
        "unique_winner_count": h4_unique,
        "unique_winner_fraction": _fraction(h4_unique, h4_event_count),
        "multiple_optimum_count": h4_multiple,
        "multiple_optimum_fraction": _fraction(h4_multiple, h4_event_count),
        "all_candidates_tied_count": h4_all_tied,
        "all_candidates_tied_fraction": _fraction(h4_all_tied, h4_event_count),
        "mean_optimal_set_size": _fraction(sum(optimal_sizes), h4_event_count),
        "median_optimal_set_size": median(optimal_sizes),
        "mean_optimal_set_fraction": _fraction(sum(optimal_fractions), h4_event_count),
        "ordinary_margin_summary": _quantiles(ordinary_margins),
        "strict_positive_ordinary_margin_fraction": _fraction(len(positive_margins), h4_event_count),
        "positive_ordinary_margin_summary": _quantiles(positive_margins),
        "strict_distinct_margin_summary": _quantiles(strict_margins),
        "ordinary_margin_zero_fraction": _fraction(sum(m == 0.0 for m in ordinary_margins), h4_event_count),
    }
    summary = {
        "status": "COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"head": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"]), "dirty_status": _git(["status", "--short"])},
        "protocol": {"family": family, "capacity": capacity, "horizons": list(HORIZONS), "history": [0, 10000], "score": [10000, 50000], "trajectory": "exact H=4 eviction-loss oracle trajectory with lexicographic H4 tie break", "target_semantics": "shared build_candidate_rows_for_full_cache_state eviction_loss_label with LRU continuation", "optimal_set_semantics": "all candidates attaining the minimum target value; no total order imposed on ties"},
        "trace": {"family": family, "name": fold["test_trace_name"], "path": str(trace_path), "sha256": trace_sha, "fold": dict(fold)},
        "h4_informativeness": h4_summary,
        "horizon_comparisons": [_finalize_horizon(horizon_metrics[h]) for h in (8, 16, 32)],
        "decision_relevance_h4_unique": decision_relevance,
        "outputs": {"summary_json": str(out_dir / "summary.json"), "provenance_json": str(out_dir / "provenance.json")},
    }
    _atomic_json(out_dir / "summary.json", summary)
    _atomic_json(out_dir / "provenance.json", {"git": summary["git"], "protocol": summary["protocol"], "trace": summary["trace"], "hostname": socket.gethostname()})
    return summary


def _run_unit(family: str, capacity: int, data_root: Path, root: Path) -> Dict[str, Any]:
    fold = _fold(family)
    trace_path = _trace_path(fold, data_root)
    trace_sha = _sha256(trace_path)
    if trace_sha != fold["test_trace_sha256"]:
        raise ValueError(f"trace hash mismatch for {family}")
    requests, _pages, _source = load_trace_from_any(str(trace_path))
    final_dir = root / "units" / f"{family}_cap{capacity}"
    if (final_dir / "summary.json").exists():
        summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
        _validate_summary(summary, family, capacity, trace_sha)
        return summary
    temporary = root / "units" / f".{family}_cap{capacity}.tmp-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"temporary unit exists: {temporary}")
    summary = _analyze_unit(requests=requests, family=family, capacity=capacity, trace_name=str(fold["test_trace_name"]), trace_path=trace_path, trace_sha=trace_sha, fold=fold, out_dir=temporary)
    summary["outputs"] = _finalized_output_paths(summary.get("outputs", {}), final_dir)
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_summary(summary, family, capacity, trace_sha)
    if family == "brightkite" and capacity == 64:
        h4 = summary["h4_informativeness"]
        if h4["event_count"] != 19079 or h4["multiple_optimum_fraction"] != 1.0 or abs(h4["mean_optimal_set_fraction"] - 0.9932132514806856) > 1e-12 or h4["ordinary_margin_zero_fraction"] != 1.0:
            raise ValueError("Brightkite/cap64/H4 degeneracy regression mismatch")
    os.replace(temporary, final_dir)
    return summary


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-read-root", type=Path, default=REPO_ROOT.parent / "Augmented-caching")
    parser.add_argument("--families", default=None, help="comma-separated subset for regression or resume testing")
    parser.add_argument("--capacities", default=None, help="comma-separated subset for regression or resume testing")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    families = [x for x in (args.families.split(",") if args.families else cfg["families"])]
    capacities = [int(x) for x in (args.capacities.split(",") if args.capacities else cfg["capacities"])]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "config_snapshot.json", cfg)
    manifest_path = args.out_dir / "unit_completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"status": "RUNNING", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": {"head": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"])}, "protocol": cfg, "expected_units": len(families) * len(capacities), "completed_units": 0, "units": {}}
    for family in families:
        for capacity in capacities:
            key = f"{family}_cap{capacity}"
            summary = _run_unit(family, capacity, args.data_read_root, args.out_dir)
            manifest["units"][key] = {"status": "COMPLETE", "family": family, "capacity": capacity, "summary": str(args.out_dir / "units" / key / "summary.json"), "trace_sha256": summary["trace"]["sha256"]}
            manifest["completed_units"] = len(manifest["units"])
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "unit_complete", "unit": key, "completed_units": manifest["completed_units"]}, sort_keys=True), flush=True)
    if manifest["completed_units"] != manifest["expected_units"]:
        return
    cell_rows = []
    horizon_rows = []
    for family in families:
        for capacity in capacities:
            summary = json.loads((args.out_dir / "units" / f"{family}_cap{capacity}" / "summary.json").read_text(encoding="utf-8"))
            h4 = summary["h4_informativeness"]
            cell_rows.append({"family": family, "capacity": capacity, "status": "ok", "event_count": h4["event_count"], "unique_winner_fraction": h4["unique_winner_fraction"], "multiple_optimum_fraction": h4["multiple_optimum_fraction"], "all_candidates_tied_fraction": h4["all_candidates_tied_fraction"], "mean_optimal_set_size": h4["mean_optimal_set_size"], "median_optimal_set_size": h4["median_optimal_set_size"], "mean_optimal_set_fraction": h4["mean_optimal_set_fraction"], "ordinary_margin_zero_fraction": h4["ordinary_margin_zero_fraction"], "strict_positive_ordinary_margin_fraction": h4["strict_positive_ordinary_margin_fraction"]})
            for comparison in summary["horizon_comparisons"]:
                horizon_rows.append({"family": family, "capacity": capacity, "status": "ok", **comparison})
    _atomic_csv(args.out_dir / "cell_summary.csv", list(cell_rows[0]), cell_rows)
    _atomic_csv(args.out_dir / "horizon_comparison.csv", list(horizon_rows[0]), horizon_rows)
    _atomic_json(args.out_dir / "integrity_summary.json", {"status": "COMPLETE", "units": len(cell_rows), "cell_rows": len(cell_rows), "horizon_rows": len(horizon_rows), "expected_horizons_per_cell": 3, "unique_cell_keys": len({(r["family"], r["capacity"]) for r in cell_rows})})
    _atomic_json(args.out_dir / "provenance.json", {"git": manifest["source"], "protocol": cfg, "hostname": socket.gethostname(), "trace_hashes": manifest["units"]})
    manifest["status"] = "COMPLETE"
    _atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    _main()
