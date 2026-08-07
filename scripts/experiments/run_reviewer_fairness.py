"""Unified reviewer-fairness runner.

For each (trace, capacity, policy) combination, runs the FULL canonical
50,000-request replay **once**, then derives BOTH of the two fairness
questions from that single execution's per-request event log (see
`lafc.experiments.reviewer_fairness_common` for why this is lossless and
needs no second simulation):

  - "deployment_full_stream": misses over [0, 50000) -- matches this
    repository's existing, already-published convention for
    evict_value_v1 and the classical baselines (see
    docs/reviewer_fairness_protocol.md section 3).
  - "primary_controlled_window": misses over [10000, 50000) -- an
    identical held-out suffix for every policy, after an identical
    history/warm-up prefix processed under each policy's own legitimate
    online behavior. See docs/reviewer_fairness_protocol.md sections 4-5
    for why this is the recommended PRIMARY reviewer-facing comparison.

Supports (see docs/reviewer_fairness_protocol.md, "Policy inventory"):
    lru, sieve, fifo_reinsertion, cacheus, halp, three_l_cache, evict_value_v1

Deliberately does NOT support --policy lrb: LRB may still be running in
the primary checkout as of this protocol's design (see
docs/reviewer_fairness_protocol.md section "LRB fairness audit"); do not
duplicate it here. A --policy-only lrb mode already exists in
scripts/experiments/run_lrb_external_baseline.py via --skip-baselines and
should be pointed at this same manifest/window once LRB is free.

evict_value_v1 rows ARE computed (the model is read-only accessible) but
are marked ineligible for the primary comparison
(future_information="TRAIN_TEST_OVERLAP") per the CRITICAL finding in
docs/reviewer_fairness_protocol.md section "evict_value_v1 training
provenance audit" -- included for documentation/diagnostic completeness,
not silently omitted.

Writes incrementally, resumable, explicit failure rows. Outputs under
analysis/reviewer_fairness/, never touching *_heavy_r1, lrb/,
three_l_cache/, halp/, or cacheus/ artifacts.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.external_baseline_common import (
    IncrementalCsvWriter,
    base_provenance,
    read_trace_manifest,
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
from lafc.runner.run_policy import run_policy

DEFAULT_MANIFEST = Path("analysis/wulver_trace_manifest_full.csv")
DEFAULT_OUT_DIR = Path("analysis/reviewer_fairness")

FIELDNAMES = COMMON_SCHEMA_FIELDS + [
    "batch_size_or_equivalent", "n_history_events", "n_scored_events",
]
KEY_FIELDS = ["trace", "capacity", "policy", "policy_variant"]


def _make_policy(name: str, capacity: int):
    """Returns (policy_instance, metadata_dict) for the given policy name.
    metadata_dict fills in the policy-specific common-schema fields.
    """
    if name == "lru":
        from lafc.policies.lru import LRUPolicy
        return LRUPolicy(), dict(
            implementation_source="native", implementation_commit="n/a",
            model_training_mode="none", model_training_data="n/a",
            model_frozen_during_test="n/a", online_adaptation_during_test="no",
            hyperparameter_source="n/a (parameter-free)", random_seed="n/a",
            future_information="none", batch_size_or_equivalent="",
        )
    if name == "sieve":
        from lafc.policies.sieve import SievePolicy
        return SievePolicy(), dict(
            implementation_source="native", implementation_commit="n/a",
            model_training_mode="none", model_training_data="n/a",
            model_frozen_during_test="n/a", online_adaptation_during_test="no",
            hyperparameter_source="n/a (parameter-free)", random_seed="n/a",
            future_information="none", batch_size_or_equivalent="",
        )
    if name == "fifo_reinsertion":
        from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy
        return FIFOReinsertionPolicy(), dict(
            implementation_source="native", implementation_commit="n/a",
            model_training_mode="none", model_training_data="n/a",
            model_frozen_during_test="n/a", online_adaptation_during_test="no",
            hyperparameter_source="n/a (parameter-free)", random_seed="n/a",
            future_information="none", batch_size_or_equivalent="",
        )
    if name == "cacheus":
        from lafc.cacheus_official_loader import EXPECTED_COMMIT
        from lafc.policies.cacheus import CacheusPolicy
        return CacheusPolicy(), dict(
            implementation_source="official_source_unmodified",
            implementation_commit=EXPECTED_COMMIT,
            model_training_mode="online_continuous_no_explicit_phase",
            model_training_data="in_trace_only",
            model_frozen_during_test="no", online_adaptation_during_test="yes",
            hyperparameter_source="official_defaults", random_seed="123_hardcoded_upstream",
            future_information="none", batch_size_or_equivalent="",
        )
    if name == "lrb":
        from lafc.policies.lrb import LRBConfig, LRBPolicy
        return LRBPolicy(LRBConfig(memory_window=4096, batch_size=2048, seed=0)), dict(
            implementation_source="repository_reimplementation",
            implementation_commit="9e8b4423383c01c4528deb447f152f0437a37c3a",
            model_training_mode="online_batched_retraining",
            model_training_data="in_trace_only",
            model_frozen_during_test="no", online_adaptation_during_test="yes",
            hyperparameter_source="repository_documented_defaults_not_regridsearched_this_run "
            "(memory_window=4096, batch_size=2048 -- same values used elsewhere in this "
            "repository's LRB comparisons, e.g. run_three_l_cache_comparison.py's LRB "
            "reference row; official CDN-scale defaults never fire at this trace length)",
            random_seed="0", future_information="none", batch_size_or_equivalent="2048",
        )
    if name == "halp":
        from lafc.policies.halp import HALPConfig, HALPPolicy
        return HALPPolicy(HALPConfig(training_trigger=HISTORY_END, seed=0)), dict(
            implementation_source="repository_reimplementation",
            implementation_commit="n/a (no official code)",
            model_training_mode="offline_frozen_single_split",
            model_training_data=f"in_trace_prefix_[0,{HISTORY_END})",
            model_frozen_during_test="yes", online_adaptation_during_test="no",
            hyperparameter_source="repository_chosen_no_official_defaults",
            random_seed="0", future_information="none",
            batch_size_or_equivalent="",
        )
    if name == "three_l_cache":
        from lafc.policies.three_l_cache import ThreeLCacheConfig, ThreeLCachePolicy
        return ThreeLCachePolicy(ThreeLCacheConfig(batch_size=4096, seed=0)), dict(
            implementation_source="repository_reimplementation",
            implementation_commit="134cd159b635cdab75419a4281bed1a330fef31f",
            model_training_mode="online_batched_retraining",
            model_training_data="in_trace_only",
            model_frozen_during_test="no", online_adaptation_during_test="yes",
            hyperparameter_source="class_default_batch_size_4096_NOT_validation_tuned_this_run",
            random_seed="0", future_information="none",
            batch_size_or_equivalent="4096",
        )
    if name == "evict_value_v1":
        from lafc.policies.evict_value_v1 import EvictValueV1Policy
        model_path = _EVICT_VALUE_MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"evict_value_v1 model not found at {model_path}; pass "
                "--evict-value-model with an accessible path."
            )
        return EvictValueV1Policy(model_path=str(model_path)), dict(
            implementation_source="native_pretrained",
            implementation_commit="n/a",
            model_training_mode="offline_pretrained_frozen",
            model_training_data="CRITICAL: chunk-level split over the SAME 7 "
            "canonical evaluation traces (data/derived/evict_value_v1_wulver_"
            "heavy_r1/manifest.json) -- see docs/reviewer_fairness_protocol.md, "
            "'evict_value_v1 training provenance audit'",
            model_frozen_during_test="yes", online_adaptation_during_test="no",
            hyperparameter_source="offline_validation_selected",
            random_seed="n/a",
            future_information="TRAIN_TEST_OVERLAP -- see audit doc, NOT eligible for primary comparison",
            batch_size_or_equivalent="",
        )
    raise ValueError(f"Unsupported --policy {name!r}")


_EVICT_VALUE_MODEL_PATH = "models/evict_value_wulver_v1_best.pkl"


def main() -> None:
    global _EVICT_VALUE_MODEL_PATH
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True,
                     choices=["lru", "sieve", "fifo_reinsertion", "cacheus", "halp",
                              "three_l_cache", "evict_value_v1", "lrb"])
    ap.add_argument("--trace-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--evict-value-model", type=Path, default=Path(_EVICT_VALUE_MODEL_PATH))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    _EVICT_VALUE_MODEL_PATH = str(args.evict_value_model)

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    traces = read_trace_manifest(args.trace_manifest)
    if not traces:
        raise SystemExit(f"No traces found in manifest {args.trace_manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"policy_comparison_{args.policy}.csv"
    writer = IncrementalCsvWriter(out_csv, FIELDNAMES, KEY_FIELDS)

    trace_hashes: Dict[str, str] = {}
    n_written, n_skipped, n_failed = 0, 0, 0

    for path, trace_name, family in traces:
        trace_hash = sha256_of_file(Path(path))
        trace_hashes[trace_name] = trace_hash
        reqs, pages, _src = load_trace_from_any(path)
        if len(reqs) != SCORE_END:
            print(
                f"WARNING: {trace_name} has {len(reqs)} requests, expected "
                f"exactly {SCORE_END} for the canonical protocol.", file=sys.stderr,
            )

        for cap in caps:
            row_key = {"trace": trace_name, "capacity": cap, "policy": args.policy,
                       "policy_variant": "deployment_full_stream"}
            row_key_2 = {"trace": trace_name, "capacity": cap, "policy": args.policy,
                         "policy_variant": "primary_controlled_window"}
            if writer.already_done(row_key) and writer.already_done(row_key_2):
                n_skipped += 2
                continue

            try:
                policy, meta = _make_policy(args.policy, cap)
                t0 = time.time()
                result = run_policy(policy, reqs, pages, cap)
                wall_s = time.time() - t0

                full = score_window(result.events, HISTORY_START, len(result.events))
                primary = score_window(result.events, SCORE_START, len(result.events))

                for variant, w, warmup in [
                    ("deployment_full_stream", full, "none_starts_empty"),
                    ("primary_controlled_window", primary,
                     f"processes_[{HISTORY_START},{HISTORY_END})_then_scores_[{SCORE_START},{SCORE_END})"),
                ]:
                    row = {
                        "experiment_protocol_version": PROTOCOL_VERSION,
                        "policy": args.policy, "policy_variant": variant,
                        "trace": trace_name, "trace_sha256": trace_hash,
                        "capacity": cap, "capacity_semantics": "object_slots",
                        "object_size_semantics": "unit",
                        "history_start": HISTORY_START, "history_end": HISTORY_END,
                        "score_start": w.score_start, "score_end": w.score_end,
                        "history_requests": w.history_requests, "scored_requests": w.scored_requests,
                        "hits": w.hits, "misses": w.misses, "miss_ratio": round(w.miss_ratio, 6),
                        "cache_warmup": warmup,
                        "runtime_seconds": round(wall_s, 4), "status": "ok", "failure_reason": "",
                        "n_history_events": w.history_requests, "n_scored_events": w.scored_requests,
                        **meta,
                    }
                    validate_common_row(row)
                    key = {"trace": trace_name, "capacity": cap, "policy": args.policy, "policy_variant": variant}
                    if writer.already_done(key):
                        n_skipped += 1
                        continue
                    writer.write_row(row)
                    n_written += 1
                print(f"[eval] {trace_name} cap={cap} {args.policy}: "
                      f"full={full.misses}/{full.scored_requests} "
                      f"primary={primary.misses}/{primary.scored_requests} ({wall_s:.2f}s)")
            except Exception as exc:  # noqa: BLE001 -- explicit failure row
                for variant in ("deployment_full_stream", "primary_controlled_window"):
                    row = {f: "" for f in FIELDNAMES}
                    row.update({
                        "experiment_protocol_version": PROTOCOL_VERSION,
                        "policy": args.policy, "policy_variant": variant,
                        "trace": trace_name, "trace_sha256": trace_hash,
                        "capacity": cap, "capacity_semantics": "object_slots",
                        "object_size_semantics": "unit",
                        "history_start": HISTORY_START, "history_end": HISTORY_END,
                        "score_start": "", "score_end": "",
                        "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}",
                    })
                    writer.write_row(row)
                    n_failed += 1
                print(f"[FAIL] {trace_name} cap={cap} {args.policy}: {exc}", file=sys.stderr)
                traceback.print_exc()

    writer.close()

    provenance = {
        **base_provenance(),
        "protocol_version": PROTOCOL_VERSION,
        "policy": args.policy,
        "history_start": HISTORY_START, "history_end": HISTORY_END,
        "score_start": SCORE_START, "score_end": SCORE_END,
        "trace_manifest": str(args.trace_manifest),
        "trace_hashes_sha256": trace_hashes,
        "capacities": caps,
        "rows_written_this_invocation": n_written,
        "rows_skipped_already_present": n_skipped,
        "rows_failed_this_invocation": n_failed,
    }
    write_provenance_json(args.out_dir / f"provenance_{args.policy}.json", provenance)
    print(f"\nWrote {n_written} new row(s), skipped {n_skipped}, {n_failed} failed. "
          f"Outputs under {args.out_dir}/")


if __name__ == "__main__":
    main()
