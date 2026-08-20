"""Read-only diagnostic for finite-horizon eviction-loss target degeneracy."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Deque, Dict, List, Mapping, Optional, Sequence

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_END, SCORE_START
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
    numeric_summary,
    resolve_tied_set_at_long_horizon,
)
from lafc.types import Page, PageId, Request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "analysis/eviction_loss_target_degeneracy_v1"
DEFAULT_REGISTRY = REPO_ROOT / "analysis/supervision_objective_ablation_v1/model_registry.json"
FOLDS_DIR = REPO_ROOT / "configs/fair_cross_family_v1/folds"
REGISTRY_OBJECTIVE = "objective_eviction_loss"


class ProvenanceError(RuntimeError):
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
    return json.loads((FOLDS_DIR / f"{family}.json").read_text(encoding="utf-8"))


def _resolve_trace_path(fold: Mapping[str, object], data_read_root: Path) -> Path:
    rel = Path(str(fold["test_trace_path"]))
    for path in (
        data_read_root / rel,
        REPO_ROOT / rel,
        REPO_ROOT.parent / "Augmented-caching" / rel,
    ):
        if path.exists():
            return path
    raise FileNotFoundError(f"test trace not found for {rel}")


def _verify_model(registry_path: Path, family: str) -> tuple[Optional[Path], Optional[Mapping[str, object]], Optional[str]]:
    if not registry_path.exists():
        return None, None, f"registry not found: {registry_path}"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("MODEL_SELECTION_FROZEN") is not True:
        return None, None, f"MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')}"
    record = None
    for rec in registry.get("records", []):
        if rec.get("objective") == REGISTRY_OBJECTIVE and rec.get("held_out_family") == family:
            record = rec
            break
    if record is None:
        return None, None, f"no {REGISTRY_OBJECTIVE} model record for {family}"
    if family in list(record.get("training_families", [])):
        return None, None, f"held-out family {family} is in training_families"
    if str(record.get("validation_family", "")) == family:
        return None, None, f"held-out family {family} is validation family"
    model_path = REPO_ROOT / str(record.get("model_artifact_path", ""))
    if not model_path.exists():
        return None, None, f"model artifact missing: {model_path}"
    actual_hash = _sha256_of_file(model_path)
    if actual_hash != record.get("model_artifact_sha256"):
        return None, None, "model hash mismatch"
    return model_path, record, None


def _load_model_scorer(model_path: Optional[Path]):
    if model_path is None:
        return None
    model = EvictValueV1Model.load(model_path)
    if list(model.feature_columns) != list(EVICT_VALUE_V1_FEATURE_COLUMNS):
        raise ProvenanceError(f"unexpected feature columns: {model.feature_columns}")

    def choose(rows: Sequence[Mapping[str, object]]) -> PageId:
        candidates = [str(row["candidate_page_id"]) for row in rows]
        feats = [{col: float(row[col]) for col in EVICT_VALUE_V1_FEATURE_COLUMNS} for row in rows]
        preds = dict(zip(candidates, model.predict_loss_batch(feats)))
        return min(candidates, key=lambda pid: (float(preds[pid]), candidates.index(pid)))

    return choose


def _prepare_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty; refusing overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _candidate_rows(
    *,
    requests: Sequence[Request],
    request_index: int,
    capacity: int,
    trace_name: str,
    trace_family: str,
    horizon: int,
    cache_order: Sequence[PageId],
    bucket_by_page: Mapping[PageId, int],
    confidence_by_page: Mapping[PageId, float],
    recent_req_hist: Sequence[PageId],
    recent_hit_hist: Sequence[PageId],
    occurrence_index,
    distinct_suffix_counts,
    candidate_subset: Optional[set[PageId]] = None,
    include_features: bool = True,
) -> List[Dict[str, object]]:
    return build_candidate_rows_for_full_cache_state(
        requests=requests,
        request_index=request_index,
        capacity=capacity,
        trace_name=trace_name,
        trace_family=trace_family,
        cfg=ObjectiveAblationConfig(horizon=horizon),
        cache_order=cache_order,
        bucket_by_page=bucket_by_page,
        confidence_by_page=confidence_by_page,
        recent_req_hist=recent_req_hist,
        recent_hit_hist=recent_hit_hist,
        occurrence_index=occurrence_index,
        distinct_suffix_counts=distinct_suffix_counts,
        candidate_subset=candidate_subset,
        include_features=include_features,
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fraction(numer: int, denom: int) -> Optional[float]:
    return (numer / denom) if denom else None


def _summary_for_events(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    n = len(rows)
    tie_sizes = [int(row["optimal_set_size"]) for row in rows]
    strict_margins = [float(row["strict_distinct_margin"]) for row in rows if row["strict_distinct_margin"] != ""]
    return {
        "event_count": n,
        "mean_candidate_count": float(mean(float(row["candidate_count"]) for row in rows)) if rows else None,
        "tie_event_count": sum(1 for row in rows if int(row["optimal_set_size"]) > 1),
        "tie_event_fraction": _fraction(sum(1 for row in rows if int(row["optimal_set_size"]) > 1), n),
        "median_optimal_set_size": float(median(tie_sizes)) if tie_sizes else None,
        "mean_optimal_set_fraction": float(mean(float(row["optimal_set_fraction"]) for row in rows)) if rows else None,
        "distinct_target_value_count": numeric_summary(float(row["distinct_value_count"]) for row in rows),
        "strict_distinct_margin": numeric_summary(strict_margins),
        "ordinary_margin_zero_fraction": _fraction(sum(1 for row in rows if float(row["ordinary_margin"]) == 0.0), n),
        "target_entropy_bits": numeric_summary(float(row["target_entropy_bits"]) for row in rows),
        "target_spread": numeric_summary(float(row["target_spread"]) for row in rows),
    }


def _summary_for_resolution(rows: Sequence[Mapping[str, object]], h_long: int) -> Dict[str, object]:
    subset = [row for row in rows if int(row["h_long"]) == h_long]
    n = len(subset)
    learned_subset = [row for row in subset if row["learned_choice_in_h4_tie"] == "yes"]
    return {
        "h_long": h_long,
        "tie_event_count": n,
        "fraction_ties_broken": _fraction(sum(1 for row in subset if row["tied_set_broken"] == "yes"), n),
        "longer_horizon_spread": numeric_summary(float(row["long_spread"]) for row in subset),
        "deterministic_longer_horizon_best_fraction": _fraction(
            sum(1 for row in subset if row["deterministic_is_long_best"] == "yes"), n
        ),
        "deterministic_longer_horizon_regret": numeric_summary(
            float(row["deterministic_long_regret"]) for row in subset
        ),
        "learned_choice_in_h4_tie_fraction": _fraction(len(learned_subset), n),
        "learned_longer_horizon_best_fraction": _fraction(
            sum(1 for row in learned_subset if row["learned_is_long_best"] == "yes"), len(learned_subset)
        ),
        "learned_longer_horizon_regret": numeric_summary(
            float(row["learned_long_regret"]) for row in learned_subset if row["learned_long_regret"] != ""
        ),
    }


def analyze(
    *,
    requests: Sequence[Request],
    pages: Mapping[PageId, Page],
    trace_name: str,
    trace_family: str,
    capacity: int,
    horizon: int,
    long_horizons: Sequence[int],
    score_start: int,
    score_end: int,
    learned_choose,
    out_dir: Path,
    overwrite: bool,
    trace_path: Optional[Path],
    trace_sha256: str,
    fold: Optional[Mapping[str, object]],
    learned_model: Mapping[str, object],
) -> Dict[str, object]:
    del pages
    _prepare_out_dir(out_dir, overwrite=overwrite)

    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=ObjectiveAblationConfig().history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=ObjectiveAblationConfig().history_window)
    occurrence_index = _build_occurrence_index(requests)
    distinct_suffix_counts = _build_distinct_suffix_counts(requests)

    event_rows: List[Dict[str, object]] = []
    resolution_rows: List[Dict[str, object]] = []
    t0 = time.time()

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        cache_order = list(order.keys())
        base_rows = _candidate_rows(
            requests=requests,
            request_index=t,
            capacity=capacity,
            trace_name=trace_name,
            trace_family=trace_family,
            horizon=horizon,
            cache_order=cache_order,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
            occurrence_index=occurrence_index,
            distinct_suffix_counts=distinct_suffix_counts,
            include_features=learned_choose is not None,
        )
        base_values = eviction_loss_values(base_rows)
        metrics = exact_tie_metrics(base_values)
        deterministic_choice = deterministic_exact_tiebreak(metrics.optimal_candidates)
        learned_choice = learned_choose(base_rows) if learned_choose is not None else None

        if score_start <= t < score_end:
            event_rows.append(
                {
                    "decision_id": str(base_rows[0]["decision_id"]),
                    "request_t": t,
                    "candidate_count": metrics.candidate_count,
                    "best_value": metrics.best_value,
                    "optimal_set_size": metrics.optimal_set_size,
                    "optimal_set_fraction": metrics.optimal_set_fraction,
                    "distinct_value_count": metrics.distinct_value_count,
                    "ordinary_margin": metrics.ordinary_margin,
                    "strict_distinct_margin": "" if metrics.strict_distinct_margin is None else metrics.strict_distinct_margin,
                    "target_entropy_bits": metrics.target_entropy_bits,
                    "target_spread": metrics.target_spread,
                    "deterministic_choice": deterministic_choice,
                    "learned_choice": learned_choice or "",
                    "learned_choice_h4_optimal": (
                        "" if learned_choice is None else ("yes" if learned_choice in metrics.optimal_candidates else "no")
                    ),
                }
            )
            if metrics.optimal_set_size > 1:
                tied_set = set(metrics.optimal_candidates)
                for h_long in long_horizons:
                    long_rows = _candidate_rows(
                        requests=requests,
                        request_index=t,
                        capacity=capacity,
                        trace_name=trace_name,
                        trace_family=trace_family,
                        horizon=h_long,
                        cache_order=cache_order,
                        bucket_by_page=bucket_by_page,
                        confidence_by_page=conf_by_page,
                        recent_req_hist=recent_req_hist,
                        recent_hit_hist=recent_hit_hist,
                        occurrence_index=occurrence_index,
                        distinct_suffix_counts=distinct_suffix_counts,
                        candidate_subset=tied_set,
                        include_features=False,
                    )
                    long_values = eviction_loss_values(long_rows)
                    learned_in_tie = learned_choice if learned_choice in tied_set else None
                    res = resolve_tied_set_at_long_horizon(
                        h_long=h_long,
                        h_tied_candidates=metrics.optimal_candidates,
                        long_values=long_values,
                        deterministic_choice=deterministic_choice,
                        learned_choice=learned_in_tie,
                    )
                    resolution_rows.append(
                        {
                            "decision_id": str(base_rows[0]["decision_id"]),
                            "request_t": t,
                            "h_long": h_long,
                            "tie_set_size": res.tie_set_size,
                            "long_min": res.long_min,
                            "long_max": res.long_max,
                            "long_spread": res.long_spread,
                            "tied_set_broken": "yes" if res.tied_set_broken else "no",
                            "deterministic_choice": res.deterministic_choice,
                            "deterministic_long_value": res.deterministic_long_value,
                            "deterministic_is_long_best": "yes" if res.deterministic_is_long_best else "no",
                            "deterministic_long_regret": res.deterministic_long_regret,
                            "learned_choice": learned_choice or "",
                            "learned_choice_in_h4_tie": "yes" if learned_in_tie is not None else "no",
                            "learned_long_value": "" if res.learned_long_value is None else res.learned_long_value,
                            "learned_is_long_best": "" if res.learned_is_long_best is None else ("yes" if res.learned_is_long_best else "no"),
                            "learned_long_regret": "" if res.learned_long_regret is None else res.learned_long_regret,
                        }
                    )
            if len(event_rows) % 1000 == 0:
                print(f"[degeneracy] scored_events={len(event_rows)} t={t}", flush=True)

        order.pop(deterministic_choice)
        order[pid] = None
        recent_req_hist.append(pid)

    event_fields = [
        "decision_id",
        "request_t",
        "candidate_count",
        "best_value",
        "optimal_set_size",
        "optimal_set_fraction",
        "distinct_value_count",
        "ordinary_margin",
        "strict_distinct_margin",
        "target_entropy_bits",
        "target_spread",
        "deterministic_choice",
        "learned_choice",
        "learned_choice_h4_optimal",
    ]
    resolution_fields = [
        "decision_id",
        "request_t",
        "h_long",
        "tie_set_size",
        "long_min",
        "long_max",
        "long_spread",
        "tied_set_broken",
        "deterministic_choice",
        "deterministic_long_value",
        "deterministic_is_long_best",
        "deterministic_long_regret",
        "learned_choice",
        "learned_choice_in_h4_tie",
        "learned_long_value",
        "learned_is_long_best",
        "learned_long_regret",
    ]
    _write_csv(out_dir / "event_metrics.csv", event_fields, event_rows)
    _write_csv(out_dir / "tie_resolution.csv", resolution_fields, resolution_rows)

    event_summary = _summary_for_events(event_rows)
    resolution_summary = {
        str(h_long): _summary_for_resolution(resolution_rows, h_long)
        for h_long in long_horizons
    }
    summary = {
        "status": "COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git_output(["rev-parse", "HEAD"]),
            "branch": _git_output(["branch", "--show-current"]),
            "dirty_status": _git_output(["status", "--short"]),
        },
        "predeclared_hypotheses": {
            "H1": "H=4 produces a high fraction of multi-candidate optimal ties.",
            "H2": "Many H=4 tied sets contain candidates with materially different longer-horizon costs.",
            "H3": "The current deterministic tie-breaker often fails to choose the best longer-horizon candidate within an H=4 tied set.",
            "H4": "Increasing diagnostic horizon reduces tie-set size / increases distinct-value resolution.",
        },
        "protocol": {
            "family": trace_family,
            "capacity": capacity,
            "horizon": horizon,
            "long_horizons": list(long_horizons),
            "score_start": score_start,
            "score_end": score_end,
            "request_count": len(requests),
            "trajectory": "exact H-step eviction-loss oracle trajectory with lexicographic H-tie break",
            "target_semantics": "shared build_candidate_rows_for_full_cache_state eviction_loss_label with LRU continuation",
            "ordinary_margin": "second sorted target value minus best target value; zero whenever the minimum is duplicated",
            "strict_distinct_margin": "second distinct target value minus best target value; null when all candidate target values are equal",
        },
        "trace": {
            "name": trace_name,
            "family": trace_family,
            "path": str(trace_path) if trace_path is not None else None,
            "sha256": trace_sha256,
            "fold": dict(fold) if fold is not None else None,
        },
        "learned_model": dict(learned_model),
        "event_summary": event_summary,
        "longer_horizon_resolution": resolution_summary,
        "runtime_seconds": time.time() - t0,
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "event_metrics_csv": str(out_dir / "event_metrics.csv"),
            "tie_resolution_csv": str(out_dir / "tie_resolution.csv"),
            "summary_md": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_dir / "summary.md", summary)
    return summary


def _write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Eviction-Loss Target Degeneracy Diagnostic",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a read-only mechanism diagnostic. It does not change the deployed policy.",
        "",
        "## Event Summary",
        "",
    ]
    for key, value in summary["event_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Longer-Horizon Resolution", ""])
    for h_long, payload in summary["longer_horizon_resolution"].items():
        lines.append(f"### H_long={h_long}")
        for key, value in payload.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="brightkite")
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--long-horizons", default="8,16,32")
    ap.add_argument("--score-start", type=int, default=SCORE_START)
    ap.add_argument("--score-end", type=int, default=SCORE_END)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--data-read-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-learned", action="store_true")
    args = ap.parse_args()

    long_horizons = [int(x) for x in args.long_horizons.split(",") if x.strip()]
    fold = _load_fold(args.family)
    trace_path = _resolve_trace_path(fold, args.data_read_root)
    trace_hash = _sha256_of_file(trace_path)
    if fold.get("test_trace_sha256") and trace_hash != fold["test_trace_sha256"]:
        raise ProvenanceError("trace hash mismatch")
    requests, pages, _source = load_trace_from_any(str(trace_path))
    if args.max_requests is not None:
        requests = requests[: args.max_requests]
        live_pages = {req.page_id for req in requests}
        pages = {pid: page for pid, page in pages.items() if pid in live_pages}

    model_path: Optional[Path] = None
    model_record: Optional[Mapping[str, object]] = None
    model_error: Optional[str] = "disabled by --no-learned" if args.no_learned else None
    if not args.no_learned:
        model_path, model_record, model_error = _verify_model(args.registry, args.family)
    learned_choose = _load_model_scorer(model_path) if model_path is not None else None
    learned_model = {
        "status": "USED" if model_path is not None else "NOT_AVAILABLE",
        "reason": model_error,
        "model_path": str(model_path) if model_path is not None else None,
        "model_sha256": model_record.get("model_artifact_sha256") if model_record is not None else None,
        "training_families": model_record.get("training_families") if model_record is not None else None,
        "validation_family": model_record.get("validation_family") if model_record is not None else None,
        "held_out_family": model_record.get("held_out_family") if model_record is not None else None,
    }

    out_dir = args.out_dir / f"{args.family}_cap{args.capacity}_h{args.horizon}"
    summary = analyze(
        requests=requests,
        pages=pages,
        trace_name=str(fold["test_trace_name"]),
        trace_family=args.family,
        capacity=args.capacity,
        horizon=args.horizon,
        long_horizons=long_horizons,
        score_start=args.score_start,
        score_end=args.score_end,
        learned_choose=learned_choose,
        out_dir=out_dir,
        overwrite=args.overwrite,
        trace_path=trace_path,
        trace_sha256=trace_hash,
        fold=fold,
        learned_model=learned_model,
    )
    print(json.dumps({"status": summary["status"], "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    _main()
