"""Small exact-target-oracle diagnostic for eviction-loss supervision.

This runner decomposes one held-out trace/capacity/window into:

  LRU
  exact finite-horizon eviction-loss oracle
  learned eviction-loss scalar policy, if frozen provenance is valid
  offline Belady/MIN as a future-aware reference

It never trains a model and fails closed rather than substituting an
ineligible learned artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import (
    HISTORY_START,
    SCORE_END,
    SCORE_START,
    WindowScore,
    score_window,
)
from lafc.oracle_diagnostics import (
    ExactOracleDecision,
    OracleReplaySummary,
    get_exact_oracle_objective_spec,
    replay_exact_target_policy,
    replay_score_driven_policy,
    summarize_decision_diagnostics,
)
from lafc.policies.lru import LRUPolicy
from lafc.policies.offline_belady import OfflineBeladyPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import ObjectiveAblationConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "analysis/supervision_objective_ablation_v1/model_registry.json"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis/exact_target_oracle_diagnostic_v1"
FOLDS_DIR = REPO_ROOT / "configs/fair_cross_family_v1/folds"
OBJECTIVE = "eviction_loss"
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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_registry(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise ProvenanceError(f"model registry not found: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("MODEL_SELECTION_FROZEN") is not True:
        raise ProvenanceError(
            f"MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')}; refusing learned comparison"
        )
    return registry


def _find_registry_record(registry: Mapping[str, object], family: str) -> Mapping[str, object]:
    for rec in registry.get("records", []):
        if rec.get("objective") == REGISTRY_OBJECTIVE and rec.get("held_out_family") == family:
            return rec
    raise ProvenanceError(f"no frozen {REGISTRY_OBJECTIVE} model record for held-out family {family!r}")


def _verify_eligible_model(registry_path: Path, family: str) -> Tuple[Path, Mapping[str, object]]:
    registry = _load_registry(registry_path)
    record = _find_registry_record(registry, family)
    training_families = list(record.get("training_families", []))
    validation_family = str(record.get("validation_family", ""))
    if family in training_families:
        raise ProvenanceError(f"held-out family {family!r} appears in training_families")
    if validation_family == family:
        raise ProvenanceError(f"held-out family {family!r} is also the validation family")
    model_path = REPO_ROOT / str(record.get("model_artifact_path", ""))
    expected_parent = REPO_ROOT / f"models/supervision_objective_ablation_v1/{REGISTRY_OBJECTIVE}"
    if model_path.name != f"{family}.pkl" or model_path.parent != expected_parent:
        raise ProvenanceError(f"model path does not match frozen fold convention: {model_path}")
    if not model_path.exists():
        raise ProvenanceError(f"model artifact missing: {model_path}")
    actual_hash = _sha256_of_file(model_path)
    if actual_hash != record.get("model_artifact_sha256"):
        raise ProvenanceError(
            f"model hash mismatch: registry={record.get('model_artifact_sha256')} disk={actual_hash}"
        )
    return model_path, record


def _load_fold(family: str) -> Mapping[str, object]:
    path = FOLDS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"fold file not found: {path}")
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
    raise FileNotFoundError("test trace not found in: " + ", ".join(str(p) for p in candidates))


def _score_hit_sequence(hit_sequence: Sequence[bool], score_start: int, score_end: int) -> WindowScore:
    n = len(hit_sequence)
    if not (0 <= score_start <= score_end <= n):
        raise ValueError(f"invalid hit window [{score_start}, {score_end}) for {n} requests")
    window = hit_sequence[score_start:score_end]
    misses = sum(1 for hit in window if not hit)
    hits = len(window) - misses
    return WindowScore(
        history_requests=score_start,
        scored_requests=len(window),
        score_start=score_start,
        score_end=score_end,
        hits=hits,
        misses=misses,
        miss_ratio=(misses / len(window)) if window else math.nan,
    )


def _policy_metrics(score: WindowScore, lru_misses: int, belady_misses: int) -> Dict[str, object]:
    return {
        "score_start": score.score_start,
        "score_end": score.score_end,
        "history_requests": score.history_requests,
        "scored_requests": score.scored_requests,
        "hits": score.hits,
        "misses": score.misses,
        "miss_ratio": score.miss_ratio,
        "excess_misses_relative_to_lru": score.misses - lru_misses,
        "gap_to_belady_misses": score.misses - belady_misses,
    }


def _target_margin(decision: ExactOracleDecision) -> float:
    values = sorted(float(v) for v in decision.candidate_values.values())
    if len(values) < 2:
        return 0.0
    return values[1] - values[0]


def _margin_bin(margin: float) -> str:
    if margin == 0.0:
        return "0"
    if margin <= 0.5:
        return "(0,0.5]"
    if margin <= 1.0:
        return "(0.5,1]"
    if margin <= 2.0:
        return "(1,2]"
    return ">2"


def _summarize_margin_bins(decisions: Sequence[ExactOracleDecision]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[ExactOracleDecision]] = {}
    for decision in decisions:
        grouped.setdefault(_margin_bin(_target_margin(decision)), []).append(decision)
    rows: List[Dict[str, object]] = []
    for bin_name in ["0", "(0,0.5]", "(0.5,1]", "(1,2]", ">2"]:
        group = grouped.get(bin_name, [])
        count = len(group)
        regrets = [float(d.target_regret) for d in group]
        non_opt = [d for d in group if not d.agrees_with_exact]
        rows.append(
            {
                "margin_bin": bin_name,
                "decisions": count,
                "agreement_rate": (sum(1 for d in group if d.agrees_with_exact) / count) if count else math.nan,
                "non_optimal_fraction": (len(non_opt) / count) if count else math.nan,
                "mean_target_regret": (sum(regrets) / count) if count else math.nan,
                "total_target_regret": sum(regrets),
            }
        )
    return rows


def _decisions_in_window(decisions: Iterable[ExactOracleDecision], score_start: int, score_end: int) -> List[ExactOracleDecision]:
    return [d for d in decisions if score_start <= d.request_t < score_end]


def _write_decisions_csv(path: Path, decisions: Sequence[ExactOracleDecision]) -> None:
    fields = [
        "decision_id",
        "request_t",
        "exact_candidate",
        "chosen_candidate",
        "exact_value",
        "chosen_value",
        "target_regret",
        "agrees_with_exact",
        "target_margin",
        "candidate_count",
        "optimal_candidate_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {
                    "decision_id": decision.decision_id,
                    "request_t": decision.request_t,
                    "exact_candidate": decision.exact_candidate,
                    "chosen_candidate": decision.chosen_candidate,
                    "exact_value": decision.exact_value,
                    "chosen_value": decision.chosen_value,
                    "target_regret": decision.target_regret,
                    "agrees_with_exact": decision.agrees_with_exact,
                    "target_margin": _target_margin(decision),
                    "candidate_count": len(decision.candidate_values),
                    "optimal_candidate_count": len(decision.optimal_candidates),
                }
            )


def _write_margin_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "margin_bin",
        "decisions",
        "agreement_rate",
        "non_optimal_fraction",
        "mean_target_regret",
        "total_target_regret",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _learned_scorer(model_path: Path):
    model = EvictValueV1Model.load(model_path)
    if list(model.feature_columns) != list(EVICT_VALUE_V1_FEATURE_COLUMNS):
        raise ProvenanceError(
            f"model feature columns do not match evict_value_v1 schema: {model.feature_columns}"
        )

    def score(rows: Sequence[Mapping[str, object]]) -> Mapping[str, float]:
        feature_rows = [
            {col: float(row[col]) for col in EVICT_VALUE_V1_FEATURE_COLUMNS}
            for row in rows
        ]
        preds = model.predict_loss_batch(feature_rows)
        return {
            str(row["candidate_page_id"]): float(pred)
            for row, pred in zip(rows, preds)
        }

    return score


def _prepare_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty; refusing to overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_synthetic_smoke(args: argparse.Namespace) -> Dict[str, object]:
    page_ids = ["a", "b", "c", "c", "c", "d", "a", "b", "e", "a", "f", "b"]
    requests, pages = build_requests_from_lists(page_ids=page_ids)
    return _run_diagnostic(
        requests=requests,
        pages=pages,
        trace_name="synthetic_exact_target_oracle_smoke",
        trace_family="synthetic",
        trace_sha256="synthetic",
        trace_path=None,
        fold=None,
        learned_model_path=None,
        learned_model_record=None,
        learned_provenance_error="synthetic smoke skips learned model",
        capacity=2,
        horizon=int(args.horizon),
        score_start=0,
        score_end=len(requests),
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        determinism_check=args.determinism_check,
    )


def _run_diagnostic(
    *,
    requests,
    pages,
    trace_name: str,
    trace_family: str,
    trace_sha256: str,
    trace_path: Optional[Path],
    fold: Optional[Mapping[str, object]],
    learned_model_path: Optional[Path],
    learned_model_record: Optional[Mapping[str, object]],
    learned_provenance_error: Optional[str],
    capacity: int,
    horizon: int,
    score_start: int,
    score_end: int,
    out_dir: Path,
    overwrite: bool,
    determinism_check: bool,
) -> Dict[str, object]:
    if score_end > len(requests):
        raise ValueError(f"score_end={score_end} exceeds request count {len(requests)}")
    _prepare_out_dir(out_dir, overwrite=overwrite)

    cfg = ObjectiveAblationConfig(horizon=horizon)
    timings: Dict[str, float] = {}

    t0 = time.time()
    lru_result = run_policy(LRUPolicy(), requests, pages, capacity)
    timings["lru_seconds"] = time.time() - t0
    lru_score = score_window(lru_result.events, score_start, score_end)

    t0 = time.time()
    belady_result = run_policy(OfflineBeladyPolicy(), requests, pages, capacity)
    timings["offline_belady_seconds"] = time.time() - t0
    belady_score = score_window(belady_result.events, score_start, score_end)

    t0 = time.time()
    exact = replay_exact_target_policy(
        requests=requests,
        capacity=capacity,
        trace_name=trace_name,
        trace_family=trace_family,
        cfg=cfg,
        objective=OBJECTIVE,
    )
    timings["exact_target_oracle_seconds"] = time.time() - t0
    exact_score = _score_hit_sequence(exact.hit_sequence, score_start, score_end)

    if determinism_check:
        exact_again = replay_exact_target_policy(
            requests=requests,
            capacity=capacity,
            trace_name=trace_name,
            trace_family=trace_family,
            cfg=cfg,
            objective=OBJECTIVE,
        )
        if exact.hit_sequence != exact_again.hit_sequence or exact.decisions != exact_again.decisions:
            raise RuntimeError("determinism check failed for exact oracle replay")

    learned: Optional[OracleReplaySummary] = None
    learned_score: Optional[WindowScore] = None
    learned_decisions_scored: List[ExactOracleDecision] = []
    margin_rows: List[Dict[str, object]] = []
    if learned_model_path is not None:
        t0 = time.time()
        learned = replay_score_driven_policy(
            requests=requests,
            capacity=capacity,
            trace_name=trace_name,
            trace_family=trace_family,
            cfg=cfg,
            objective=OBJECTIVE,
            scorer=_learned_scorer(learned_model_path),
            policy_name="learned_eviction_loss_scalar",
        )
        timings["learned_eviction_loss_seconds"] = time.time() - t0
        learned_score = _score_hit_sequence(learned.hit_sequence, score_start, score_end)
        learned_decisions_scored = _decisions_in_window(learned.decisions, score_start, score_end)
        margin_rows = _summarize_margin_bins(learned_decisions_scored)

        if determinism_check:
            learned_again = replay_score_driven_policy(
                requests=requests,
                capacity=capacity,
                trace_name=trace_name,
                trace_family=trace_family,
                cfg=cfg,
                objective=OBJECTIVE,
                scorer=_learned_scorer(learned_model_path),
                policy_name="learned_eviction_loss_scalar",
            )
            if learned.hit_sequence != learned_again.hit_sequence or learned.decisions != learned_again.decisions:
                raise RuntimeError("determinism check failed for learned replay")

    exact_decisions_scored = _decisions_in_window(exact.decisions, score_start, score_end)
    policies: Dict[str, object] = {
        "lru": {
            "information_class": "ONLINE_DEPLOYABLE",
            **_policy_metrics(lru_score, lru_score.misses, belady_score.misses),
        },
        "exact_finite_horizon_eviction_loss_oracle": {
            "information_class": "FUTURE_AWARE_DIAGNOSTIC_TARGET_ORACLE",
            "decision_count_scored": len(exact_decisions_scored),
            "agreement_with_self_rate": 1.0,
            **_policy_metrics(exact_score, lru_score.misses, belady_score.misses),
        },
        "offline_belady": {
            "information_class": "FUTURE_AWARE_REFERENCE",
            **_policy_metrics(belady_score, lru_score.misses, belady_score.misses),
        },
    }

    learned_metrics: Dict[str, object] = {
        "status": "NOT_AVAILABLE",
        "reason": learned_provenance_error or "not requested",
    }
    if learned is not None and learned_score is not None and learned_model_record is not None:
        learned_diag = summarize_decision_diagnostics(learned_decisions_scored)
        learned_metrics = {
            "status": "USED",
            "model_path": str(learned_model_path),
            "model_sha256": str(learned_model_record["model_artifact_sha256"]),
            "registry_objective": str(learned_model_record["objective"]),
            "selected_hyperparameters": str(learned_model_record.get("selected_hyperparameters", "")),
            "training_families": list(learned_model_record.get("training_families", [])),
            "validation_family": str(learned_model_record.get("validation_family", "")),
            "held_out_family": str(learned_model_record.get("held_out_family", "")),
            "random_seed": learned_model_record.get("random_seed", 0),
            "decision_diagnostics_scored_window": learned_diag,
        }
        policies["learned_eviction_loss_scalar"] = {
            "information_class": "TRAINED_WITH_FUTURE_LABELS_BUT_ONLINE_AT_INFERENCE",
            "decision_count_scored": len(learned_decisions_scored),
            **_policy_metrics(learned_score, lru_score.misses, belady_score.misses),
            **learned_diag,
        }

    summary: Dict[str, object] = {
        "status": "COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git_output(["rev-parse", "HEAD"]),
            "branch": _git_output(["branch", "--show-current"]),
            "dirty_status": _git_output(["status", "--short"]),
        },
        "protocol": {
            "purpose": "decompose LRU vs exact finite-horizon eviction-loss oracle vs learned eviction-loss policy vs Belady",
            "objective": OBJECTIVE,
            "capacity": capacity,
            "horizon": horizon,
            "history_start": HISTORY_START,
            "score_start": score_start,
            "score_end": score_end,
            "request_count": len(requests),
            "target_semantics": get_exact_oracle_objective_spec(OBJECTIVE).explanation,
            "continuation_policy": "LRU within finite H-step label suffix",
            "exact_oracle_tie_break": "lexicographically first among exact-target optimal candidates",
            "learned_policy_tie_break": "deployed scalar policy semantics: cache-order tie for min direction",
            "capacity_semantics": "object_slots",
            "object_size_semantics": "unit",
        },
        "trace": {
            "name": trace_name,
            "family": trace_family,
            "path": str(trace_path) if trace_path is not None else None,
            "sha256": trace_sha256,
            "fold": dict(fold) if fold is not None else None,
        },
        "learned_model": learned_metrics,
        "policies": policies,
        "margin_bins_scored_window": margin_rows,
        "timings": timings,
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "provenance_json": str(out_dir / "provenance.json"),
            "learned_decisions_csv": str(out_dir / "learned_decisions.csv") if learned is not None else None,
            "margin_bins_csv": str(out_dir / "margin_bins.csv") if learned is not None else None,
            "report_md": str(out_dir / "report.md"),
        },
    }

    _write_json(out_dir / "summary.json", summary)
    _write_json(
        out_dir / "provenance.json",
        {
            "git": summary["git"],
            "trace": summary["trace"],
            "protocol": summary["protocol"],
            "learned_model": learned_metrics,
        },
    )
    if learned is not None:
        _write_decisions_csv(out_dir / "learned_decisions.csv", learned_decisions_scored)
        _write_margin_csv(out_dir / "margin_bins.csv", margin_rows)
    _write_report(out_dir / "report.md", summary)
    return summary


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    policies = summary["policies"]
    lines = [
        "# Exact-Target-Oracle Diagnostic",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This diagnostic compares LRU, the exact finite-horizon eviction-loss target oracle, "
        "the eligible learned eviction-loss scalar policy when available, and offline Belady as "
        "a separate future-aware reference.",
        "",
        "## Scores",
        "",
        "| policy | information class | misses | miss ratio | excess vs LRU | gap to Belady |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, payload in policies.items():
        row = payload
        lines.append(
            f"| {name} | {row['information_class']} | {row['misses']} | "
            f"{float(row['miss_ratio']):.6f} | {row['excess_misses_relative_to_lru']} | "
            f"{row['gap_to_belady_misses']} |"
        )
    learned = summary["learned_model"]
    lines.extend(["", "## Learned-Model Eligibility", "", f"Status: `{learned['status']}`"])
    if learned["status"] == "USED":
        lines.extend(
            [
                "",
                f"Model: `{learned['model_path']}`",
                f"Training families: `{','.join(learned['training_families'])}`",
                f"Validation family: `{learned['validation_family']}`",
                f"Held-out family: `{learned['held_out_family']}`",
            ]
        )
    else:
        lines.extend(["", f"Reason: `{learned['reason']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="brightkite")
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--score-start", type=int, default=SCORE_START)
    ap.add_argument("--score-end", type=int, default=SCORE_END)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--data-read-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-learned", action="store_true")
    ap.add_argument("--synthetic-smoke", action="store_true")
    ap.add_argument("--determinism-check", action="store_true")
    args = ap.parse_args()

    if args.synthetic_smoke:
        summary = _run_synthetic_smoke(args)
        print(json.dumps({"status": summary["status"], "out_dir": str(args.out_dir)}, sort_keys=True))
        return

    fold = _load_fold(args.family)
    trace_path = _resolve_trace_path(fold, args.data_read_root)
    trace_hash = _sha256_of_file(trace_path)
    expected_hash = str(fold.get("test_trace_sha256", ""))
    if expected_hash and trace_hash != expected_hash:
        raise ProvenanceError(f"trace hash mismatch for {trace_path}: fold={expected_hash} disk={trace_hash}")

    requests, pages, _dataset_source = load_trace_from_any(str(trace_path))
    if args.max_requests is not None:
        requests = requests[: args.max_requests]
        pages = {pid: page for pid, page in pages.items() if pid in {req.page_id for req in requests}}
    learned_model_path: Optional[Path] = None
    learned_record: Optional[Mapping[str, object]] = None
    learned_error: Optional[str] = None
    if args.no_learned:
        learned_error = "disabled by --no-learned"
    else:
        try:
            learned_model_path, learned_record = _verify_eligible_model(args.registry, args.family)
        except ProvenanceError as exc:
            learned_error = str(exc)

    out_dir = args.out_dir / f"{args.family}_cap{args.capacity}_h{args.horizon}"
    summary = _run_diagnostic(
        requests=requests,
        pages=pages,
        trace_name=str(fold["test_trace_name"]),
        trace_family=args.family,
        trace_sha256=trace_hash,
        trace_path=trace_path,
        fold=fold,
        learned_model_path=learned_model_path,
        learned_model_record=learned_record,
        learned_provenance_error=learned_error,
        capacity=args.capacity,
        horizon=args.horizon,
        score_start=args.score_start,
        score_end=args.score_end,
        out_dir=out_dir,
        overwrite=args.overwrite,
        determinism_check=args.determinism_check,
    )
    print(json.dumps({"status": summary["status"], "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    _main()
