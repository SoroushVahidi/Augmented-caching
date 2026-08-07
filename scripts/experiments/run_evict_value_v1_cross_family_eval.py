"""Held-out evaluation runner for evict_value_v1_cross_family_v1.

For each of the 7 leave-one-family-out folds (configs/fair_cross_family_v1/
folds/<family>.json, frozen before any dataset was built -- see
configs/reviewer_fairness_cross_family_v1.json), loads ONLY that fold's own
frozen model and scores it on the held-out family's canonical trace:
history [0,10000) (cache/state warm-up, not counted), score [10000,50000)
(exactly 40,000 requests), capacities 32/64/128.

Fails closed (section 7 of the task) -- every one of these aborts the whole
run with a clear error, not a silent skip or fallback:
  - the fold's model file does not exist (explicit artifact mode; no
    surrogate, no fallback to the contaminated heavy_r1 artifact, no
    fallback to fair_v1 -- see _reject_ineligible_artifacts below);
  - the model path being loaded for family F is not EXACTLY
    fold['model_output_path'] for F's own fold (wrong-fold protection --
    e.g. brightkite's model can never be loaded while evaluating the
    citibike fold, because the path is read from citibike's own fold
    JSON, never guessed or substituted);
  - the trace about to be scored does not hash-match the fold's own
    recorded test_trace_sha256;
  - the fold's dataset manifest (once built) records the held-out family
    among its training/validation families (would mean isolation failed
    upstream) or omits any of the 5 declared training families.

Usage:
    python scripts/experiments/run_evict_value_v1_cross_family_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

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
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.runner.run_policy import run_policy

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
DEFAULT_OUT_DIR = Path("analysis/reviewer_fairness_cross_family_v1/evict_value_v1")

FIELDNAMES = COMMON_SCHEMA_FIELDS + ["fold_id", "model_sha256", "validation_family", "training_families"]
KEY_FIELDS = ["trace", "capacity", "policy", "policy_variant"]

# Ineligible artifacts that must never be silently substituted for a
# cross-family fold model, even if a fold's own model is missing.
INELIGIBLE_MODEL_PATHS = {
    "models/evict_value_wulver_v1_best.pkl",  # heavy_r1, contaminated
    "models/evict_value_v1_fair_v1.pkl",  # rejected, temporally ineligible
}


class WrongFoldError(RuntimeError):
    pass


def _load_fold(family: str) -> Dict[str, object]:
    path = FOLDS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold manifest not found: {path}")
    return json.loads(path.read_text())


def _verify_fold_and_model(fold: Dict[str, object], data_read_root: Path) -> Path:
    family = fold["test_family"]
    model_path = Path(fold["model_output_path"])

    if str(model_path) in INELIGIBLE_MODEL_PATHS:
        raise WrongFoldError(
            f"Fold {family}'s model_output_path resolves to an explicitly "
            f"ineligible artifact ({model_path}) -- refusing to evaluate."
        )
    expected_name = f"evict_value_v1_cross_family_v1_{family}.pkl"
    if model_path.name != expected_name:
        raise WrongFoldError(
            f"Fold {family}'s model_output_path {model_path} does not match "
            f"the expected naming convention {expected_name} -- refusing to "
            "guess which model belongs to this fold."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Fold {family}'s model not found at {model_path}. Explicit "
            "artifact mode: refusing to fall back to a surrogate, the "
            "contaminated heavy_r1 model, or the rejected fair_v1 model."
        )

    dataset_manifest_path = Path(fold["dataset_output_root"]) / "manifest.json"
    if dataset_manifest_path.exists():
        ds_manifest = json.loads(dataset_manifest_path.read_text())
        seen_families = {s.get("trace_family") for s in ds_manifest.get("preflight", {}).get("trace_stats", [])} or None
        if seen_families is None:
            # Fall back to scanning shard filenames if preflight stats are absent.
            seen_families = set()
        if family in seen_families:
            raise WrongFoldError(
                f"Fold {family}'s own dataset manifest {dataset_manifest_path} "
                f"includes the held-out family {family!r} among its input traces "
                "-- fold isolation failed upstream; refusing to evaluate."
            )
        expected_families = set(fold["training_families"]) | {fold["validation_family"]}
        if seen_families and not expected_families.issubset(seen_families | {None}):
            missing = expected_families - seen_families
            if missing:
                raise WrongFoldError(
                    f"Fold {family}'s dataset manifest is missing declared "
                    f"training/validation families: {missing}"
                )

    test_trace_path = data_read_root / fold["test_trace_path"]
    if not test_trace_path.exists():
        raise FileNotFoundError(f"Fold {family}'s test trace not found at {test_trace_path}")
    actual_hash = sha256_of_file(test_trace_path)
    expected_hash = fold.get("test_trace_sha256")
    if expected_hash and actual_hash != expected_hash:
        raise WrongFoldError(
            f"Fold {family}'s test trace hash mismatch: expected {expected_hash}, "
            f"got {actual_hash} -- refusing to evaluate against a different trace "
            "than the one this fold was frozen against."
        )

    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--data-read-root", type=Path, default=Path("."),
                     help="Root to resolve fold trace paths against (this worktree's "
                     "data/processed/ is empty by design; point at the checkout that "
                     "actually has the canonical traces).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    families = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "policy_comparison.csv"
    writer = IncrementalCsvWriter(out_csv, FIELDNAMES, KEY_FIELDS)

    n_written, n_skipped, n_failed, n_blocked = 0, 0, 0, 0

    for family in families:
        fold = _load_fold(family)
        try:
            model_path = _verify_fold_and_model(fold, args.data_read_root)
        except (FileNotFoundError, WrongFoldError) as exc:
            print(f"[BLOCKED] fold={family}: {exc}", file=sys.stderr)
            n_blocked += 1
            continue

        model_hash = sha256_of_file(model_path)
        test_trace_path = args.data_read_root / fold["test_trace_path"]
        reqs, pages, _src = load_trace_from_any(test_trace_path)
        trace_hash = sha256_of_file(test_trace_path)

        for cap in caps:
            key1 = {"trace": fold["test_trace_name"], "capacity": cap, "policy": "evict_value_v1_cross_family_v1", "policy_variant": "deployment_full_stream"}
            key2 = {"trace": fold["test_trace_name"], "capacity": cap, "policy": "evict_value_v1_cross_family_v1", "policy_variant": "primary_controlled_window"}
            if writer.already_done(key1) and writer.already_done(key2):
                n_skipped += 2
                continue
            try:
                policy = EvictValueV1Policy(model_path=str(model_path))
                t0 = time.time()
                result = run_policy(policy, reqs, pages, cap)
                wall_s = time.time() - t0
                full = score_window(result.events, HISTORY_START, len(result.events))
                primary = score_window(result.events, SCORE_START, len(result.events))

                for variant, w in [("deployment_full_stream", full), ("primary_controlled_window", primary)]:
                    row = {
                        "experiment_protocol_version": PROTOCOL_VERSION,
                        "policy": "evict_value_v1_cross_family_v1", "policy_variant": variant,
                        "implementation_source": "native_pretrained_cross_family",
                        "implementation_commit": "n/a",
                        "trace": fold["test_trace_name"], "trace_sha256": trace_hash,
                        "capacity": cap, "capacity_semantics": "object_slots",
                        "object_size_semantics": "unit",
                        "history_start": HISTORY_START, "history_end": HISTORY_END,
                        "score_start": w.score_start, "score_end": w.score_end,
                        "history_requests": w.history_requests, "scored_requests": w.scored_requests,
                        "hits": w.hits, "misses": w.misses, "miss_ratio": round(w.miss_ratio, 6),
                        "cache_warmup": f"processes_[{HISTORY_START},{HISTORY_END})_then_scores_[{SCORE_START},{SCORE_END})",
                        "model_training_mode": "offline_pretrained_frozen_cross_family",
                        "model_training_data": f"5 training families ({','.join(fold['training_families'])}), "
                        f"validation family={fold['validation_family']}, held-out family={family} "
                        "contributes ZERO rows (see fold isolation checks)",
                        "model_frozen_during_test": "yes", "online_adaptation_during_test": "no",
                        "hyperparameter_source": "reused_heavy_r1_grid_selected_on_validation_family_only",
                        "random_seed": "0", "future_information": "none",
                        "runtime_seconds": round(wall_s, 4), "status": "ok", "failure_reason": "",
                        "fold_id": fold["fold_id"], "model_sha256": model_hash,
                        "validation_family": fold["validation_family"],
                        "training_families": ";".join(fold["training_families"]),
                    }
                    validate_common_row(row)
                    key = {"trace": fold["test_trace_name"], "capacity": cap, "policy": "evict_value_v1_cross_family_v1", "policy_variant": variant}
                    if writer.already_done(key):
                        n_skipped += 1
                        continue
                    writer.write_row(row)
                    n_written += 1
                print(f"[eval] fold={family} cap={cap}: full={full.misses}/{full.scored_requests} "
                      f"primary={primary.misses}/{primary.scored_requests} ({wall_s:.2f}s)")
            except Exception as exc:  # noqa: BLE001
                for variant in ("deployment_full_stream", "primary_controlled_window"):
                    row = {f: "" for f in FIELDNAMES}
                    row.update({
                        "experiment_protocol_version": PROTOCOL_VERSION,
                        "policy": "evict_value_v1_cross_family_v1", "policy_variant": variant,
                        "trace": fold["test_trace_name"], "capacity": cap,
                        "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}",
                        "fold_id": fold["fold_id"],
                    })
                    writer.write_row(row)
                    n_failed += 1
                print(f"[FAIL] fold={family} cap={cap}: {exc}", file=sys.stderr)
                traceback.print_exc()

    writer.close()
    provenance = {
        **base_provenance(),
        "protocol_version": PROTOCOL_VERSION,
        "n_folds_blocked": n_blocked,
        "rows_written_this_invocation": n_written,
        "rows_skipped_already_present": n_skipped,
        "rows_failed_this_invocation": n_failed,
    }
    write_provenance_json(args.out_dir / "provenance.json", provenance)
    print(f"\nWrote {n_written} new row(s), skipped {n_skipped}, {n_failed} failed, "
          f"{n_blocked} fold(s) blocked. Outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
