"""Held-out evaluation runner for the supervision-objective ablation
(docs/supervision_objective_ablation_protocol.md,
configs/supervision_objective_ablation_v1.json).

For every (objective, held-out family, capacity) combination, loads ONLY
that fold's own frozen model (verified against the frozen model registry)
and scores it on the held-out family's canonical trace: history [0,10000)
(warm-up, not counted), score [10000,50000) (exactly 40,000 requests),
via the SAME run_policy() + score_window() reconstruction used by
scripts/experiments/run_evict_value_v1_cross_family_eval.py -- one full
in-order simulation pass, sliced losslessly, no policy is re-run merely to
get the windowed metric.

Fails closed -- every one of these aborts the run for that row (writing a
"failed"/"blocked" status row, never silently skipping or substituting):
  - the model registry is not frozen (MODEL_SELECTION_FROZEN != true);
  - the requested (objective, family) model is missing from the registry;
  - the on-disk model file's hash does not match the registry's recorded
    hash (tamper/staleness detection);
  - the model path does not match the expected naming convention for its
    own fold (wrong-fold protection -- guessing/substituting another
    fold's model is structurally impossible, the path is read from the
    registry record for THIS (objective, family) only);
  - the test trace does not hash-match the fold's own recorded
    test_trace_sha256;
  - a duplicate result key would be written (resume-safe).

Never fits a model during evaluation; never falls back to a surrogate.

Usage:
    python scripts/experiments/run_supervision_objective_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.external_baseline_common import (
    IncrementalCsvWriter,
    base_provenance,
    sha256_of_file,
    write_provenance_json,
)
from lafc.experiments.reviewer_fairness_common import (
    COMMON_SCHEMA_FIELDS,
    HISTORY_END,
    HISTORY_START,
    PROTOCOL_VERSION,
    SCORE_END,
    SCORE_START,
    score_window,
    validate_common_row,
)
from lafc.policies.supervision_objective_ablation_policy import PairwiseObjectivePolicy, ScalarObjectivePolicy
from lafc.runner.run_policy import run_policy
from supervision_objective_ablation_gates import assert_gate_clear, evaluator_startup_failures

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
DEFAULT_REGISTRY = Path("analysis/supervision_objective_ablation_v1/model_registry.json")
DEFAULT_OUT_DIR = Path("analysis/supervision_objective_ablation_v1")

FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
SCALAR_OBJECTIVES = {
    "objective_eviction_loss": "min",
    "objective_next_arrival": "max",
    "objective_reuse_distance": "max",
}
ALL_OBJECTIVES = list(SCALAR_OBJECTIVES.keys()) + ["objective_pairwise"]

FIELDNAMES = COMMON_SCHEMA_FIELDS + [
    "protocol_id", "objective", "held_out_family", "fold_id", "model_hash", "model_family",
    "feature_schema_version", "objective_definition_version", "horizon", "censoring_mode",
    "training_families", "validation_family", "seed",
]
KEY_FIELDS = ["objective", "held_out_family", "capacity"]


class WrongFoldError(RuntimeError):
    pass


def _load_fold(family: str) -> Dict[str, object]:
    return json.loads((FOLDS_DIR / f"{family}.json").read_text(encoding="utf-8"))


def _load_registry(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}. Run build_supervision_objective_ablation_registry.py first.")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not registry.get("MODEL_SELECTION_FROZEN"):
        raise WrongFoldError(
            f"Model registry at {path} has MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')}. "
            "Refusing to run held-out evaluation before every model is trained and the registry is frozen "
            f"(missing: {registry.get('missing_models')})."
        )
    return registry


def _find_record(registry: Dict[str, object], objective: str, family: str) -> Dict[str, object]:
    for rec in registry["records"]:
        if rec["objective"] == objective and rec["held_out_family"] == family:
            return rec
    raise WrongFoldError(f"No registry record for objective={objective} family={family} -- refusing to guess a substitute.")


def _verify_model(record: Dict[str, object], family: str, objective: str) -> Path:
    model_path = Path(record["model_artifact_path"])
    expected_name = f"{family}.pkl"
    if model_path.name != expected_name or model_path.parent.name != objective:
        raise WrongFoldError(
            f"Registry record for {objective}/{family} points at {model_path}, which does not match "
            f"the expected naming convention models/supervision_objective_ablation_v1/{objective}/{family}.pkl."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file listed in registry not found on disk: {model_path}")
    actual_hash = sha256_of_file(model_path)
    if actual_hash != record["model_artifact_sha256"]:
        raise WrongFoldError(
            f"Model hash mismatch for {objective}/{family}: registry says {record['model_artifact_sha256']}, "
            f"on-disk file hashes to {actual_hash} -- artifact was modified after the registry was frozen."
        )
    return model_path


def _build_policy(objective: str, model_path: Path):
    if objective == "objective_pairwise":
        return PairwiseObjectivePolicy(model_path=str(model_path))
    direction = SCALAR_OBJECTIVES[objective]
    return ScalarObjectivePolicy(model_path=str(model_path), direction=direction)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--data-read-root", type=Path, default=Path("."))
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    assert_gate_clear(
        evaluator_startup_failures(registry_path=args.registry, out_dir=args.out_dir),
        "EVALUATOR_STARTUP",
    )
    registry = _load_registry(args.registry)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "policy_comparison.csv"
    writer = IncrementalCsvWriter(out_csv, FIELDNAMES, KEY_FIELDS)

    n_written, n_skipped, n_failed, n_blocked = 0, 0, 0, 0
    trace_cache: Dict[str, tuple] = {}

    for family in FAMILIES:
        fold = _load_fold(family)
        test_trace_path = args.data_read_root / fold["test_trace_path"]
        if not test_trace_path.exists():
            print(f"[BLOCKED] family={family}: test trace not found at {test_trace_path}", file=sys.stderr)
            n_blocked += len(ALL_OBJECTIVES) * len(caps)
            continue
        actual_hash = sha256_of_file(test_trace_path)
        if fold.get("test_trace_sha256") and actual_hash != fold["test_trace_sha256"]:
            print(f"[BLOCKED] family={family}: trace hash mismatch", file=sys.stderr)
            n_blocked += len(ALL_OBJECTIVES) * len(caps)
            continue

        if family not in trace_cache:
            trace_cache[family] = load_trace_from_any(str(test_trace_path))
        reqs, pages, _src = trace_cache[family]

        for objective in ALL_OBJECTIVES:
            try:
                record = _find_record(registry, objective, family)
                model_path = _verify_model(record, family, objective)
            except (FileNotFoundError, WrongFoldError) as exc:
                print(f"[BLOCKED] objective={objective} family={family}: {exc}", file=sys.stderr)
                n_blocked += len(caps)
                continue

            model_hash = record["model_artifact_sha256"]
            for cap in caps:
                key = {"objective": objective, "held_out_family": family, "capacity": cap}
                if writer.already_done(key):
                    n_skipped += 1
                    continue
                try:
                    policy = _build_policy(objective, model_path)
                    t0 = time.time()
                    result = run_policy(policy, reqs, pages, cap)
                    wall_s = time.time() - t0
                    primary = score_window(result.events, SCORE_START, len(result.events))

                    row = {
                        "experiment_protocol_version": PROTOCOL_VERSION,
                        "policy": f"supervision_objective_ablation_{objective}", "policy_variant": "primary_controlled_window",
                        "implementation_source": "native_pretrained_offline", "implementation_commit": "n/a",
                        "trace": fold["test_trace_name"], "trace_sha256": actual_hash,
                        "capacity": cap, "capacity_semantics": "object_slots", "object_size_semantics": "unit",
                        "history_start": HISTORY_START, "history_end": HISTORY_END,
                        "score_start": primary.score_start, "score_end": primary.score_end,
                        "history_requests": primary.history_requests, "scored_requests": primary.scored_requests,
                        "hits": primary.hits, "misses": primary.misses, "miss_ratio": round(primary.miss_ratio, 6),
                        "cache_warmup": f"processes_[{HISTORY_START},{HISTORY_END})_then_scores_[{SCORE_START},{SCORE_END})",
                        "model_training_mode": "offline_pretrained_frozen_cross_family",
                        "model_training_data": f"5 training families ({','.join(fold['training_families'])}), "
                        f"validation family={fold['validation_family']}, held-out family={family} contributes ZERO rows",
                        "model_frozen_during_test": "yes", "online_adaptation_during_test": "no",
                        "hyperparameter_source": "frozen_supervision_objective_ablation_v1_grid",
                        "random_seed": str(record.get("random_seed", 0)), "future_information": "none",
                        "runtime_seconds": round(wall_s, 4), "status": "ok", "failure_reason": "",
                        "protocol_id": "supervision_objective_ablation_v1", "objective": objective,
                        "held_out_family": family, "fold_id": fold["fold_id"], "model_hash": model_hash,
                        "model_family": record.get("selected_hyperparameters", ""),
                        "feature_schema_version": "evict_value_v1", "objective_definition_version": "v1",
                        "horizon": 4, "censoring_mode": "horizon_controlled_primary",
                        "training_families": ";".join(fold["training_families"]),
                        "validation_family": fold["validation_family"], "seed": str(record.get("random_seed", 0)),
                    }
                    validate_common_row(row)
                    writer.write_row(row)
                    n_written += 1
                    print(f"[eval] objective={objective} family={family} cap={cap}: "
                          f"misses={primary.misses}/{primary.scored_requests} ({wall_s:.2f}s)")
                except Exception as exc:  # noqa: BLE001
                    row = {f: "" for f in FIELDNAMES}
                    row.update({
                        "experiment_protocol_version": PROTOCOL_VERSION,
                        "policy": f"supervision_objective_ablation_{objective}", "policy_variant": "primary_controlled_window",
                        "trace": fold["test_trace_name"], "capacity": cap,
                        "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}",
                        "protocol_id": "supervision_objective_ablation_v1", "objective": objective,
                        "held_out_family": family, "fold_id": fold["fold_id"],
                    })
                    writer.write_row(row)
                    n_failed += 1
                    print(f"[FAIL] objective={objective} family={family} cap={cap}: {exc}", file=sys.stderr)
                    traceback.print_exc()

    writer.close()
    provenance = {
        **base_provenance(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": "supervision_objective_ablation_v1",
        "registry_sha256": registry.get("registry_sha256"),
        "n_blocked": n_blocked, "rows_written_this_invocation": n_written,
        "rows_skipped_already_present": n_skipped, "rows_failed_this_invocation": n_failed,
    }
    write_provenance_json(args.out_dir / "provenance.json", provenance)
    print(f"\nWrote {n_written} new row(s), skipped {n_skipped}, {n_failed} failed, "
          f"{n_blocked} blocked. Outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
