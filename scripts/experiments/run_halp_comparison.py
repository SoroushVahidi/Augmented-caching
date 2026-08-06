"""External-baseline experiment: HALP (Song et al., NSDI 2023) vs. LRU
and, optionally, the classical baseline pool, under the same 7 manuscript
trace families, capacities (32/64/128), and 50,000-request/trace budget as
the LRB and 3L-Cache external-baseline comparisons
(`scripts/experiments/run_lrb_external_baseline.py`,
`run_three_l_cache_comparison.py`).

No official HALP code exists (see `docs/halp_provenance.md`), so unlike the
LRB/3L-Cache runners there is no official commit to record for HALP itself;
this script records that fact explicitly in provenance.json rather than
omitting the field silently.

Writes each completed (trace, capacity, policy, variant) row to disk
immediately (`lafc.experiments.external_baseline_common.IncrementalCsvWriter`)
and is resumable: re-running with the same `--out-dir` skips rows already
present in `policy_comparison.csv`.

HALP's training protocol is a frozen temporal split (see
`docs/halp_method_spec.md`, "Training and evaluation protocol"): the first
`--halp-training-trigger` requests of each trace are the training/cold-start
window, and the model is frozen for the remainder. `--halp-training-trigger`
defaults to 20% of `--max-requests-per-trace` when the latter is set,
otherwise 10,000 (20% of the canonical 50,000-request traces), matching
`HALPConfig`'s own default.

Outputs (all under analysis/external_learned_baselines/halp/, never
touching canonical *_heavy_r1 artifacts or the separate
analysis/external_learned_baselines/{lrb,three_l_cache}/ directories):
    policy_comparison.csv  -- per trace/capacity/policy misses + hit rate (incremental)
    provenance.json          -- commit, versions, seeds, trace hashes, train/eval split
    summary.md                -- human-readable aggregate summary (written at the end)

See docs/halp_method_spec.md for the method specification and
docs/baselines.md (Baseline 8) for the standard write-up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.external_baseline_common import (
    IncrementalCsvWriter,
    base_provenance,
    package_version,
    read_trace_manifest,
    sha256_of_file,
    write_provenance_json,
)
from lafc.metrics.cost import hit_rate
from lafc.policies.halp import HALPConfig, HALPPolicy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy

DEFAULT_MANIFEST = Path("analysis/wulver_trace_manifest_full.csv")
DEFAULT_OUT_DIR = Path("analysis/external_learned_baselines/halp")

FIELDNAMES = [
    "trace_name", "trace_family", "capacity", "policy", "variant",
    "requests", "misses", "hit_rate", "training_trigger", "hidden_units",
    "alpha", "seed", "n_cold_start_evictions", "n_model_ranked_evictions",
    "model_trained", "wall_s",
]
KEY_FIELDS = ["trace_name", "capacity", "policy", "variant"]


def main() -> None:
    ap = argparse.ArgumentParser(description="External HALP baseline comparison (Song et al., NSDI 2023).")
    ap.add_argument("--trace-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--max-requests-per-trace", type=int, default=None)
    ap.add_argument(
        "--halp-training-trigger", type=int, default=None,
        help="Requests before HALP freezes training. Default: 20%% of the "
        "(possibly truncated) trace length, matching HALPConfig's own "
        "10,000-of-50,000 default.",
    )
    ap.add_argument("--halp-hidden-units", type=int, default=8)
    ap.add_argument("--halp-alpha", type=float, default=1e-4)
    ap.add_argument("--halp-seed", type=int, default=0)
    ap.add_argument(
        "--halp-only", action="store_true",
        help="Run only HALP (skip the LRU reference row).",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    traces = read_trace_manifest(args.trace_manifest)
    if not traces:
        raise SystemExit(f"No traces found in manifest {args.trace_manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "policy_comparison.csv"
    writer = IncrementalCsvWriter(out_csv, FIELDNAMES, KEY_FIELDS)

    trace_hashes: Dict[str, str] = {}
    n_rows_written = 0
    n_rows_skipped = 0

    for path, trace_name, family in traces:
        reqs, pages, _src = load_trace_from_any(path)
        if args.max_requests_per_trace:
            reqs = reqs[: args.max_requests_per_trace]
        trace_hashes[trace_name] = sha256_of_file(Path(path))
        n = len(reqs)

        training_trigger = args.halp_training_trigger
        if training_trigger is None:
            training_trigger = max(1, int(n * 0.2))

        for cap in caps:
            row_key = {"trace_name": trace_name, "capacity": cap, "policy": "halp", "variant": "canonical"}
            if writer.already_done(row_key):
                n_rows_skipped += 1
            else:
                cfg = HALPConfig(
                    training_trigger=training_trigger,
                    hidden_units=args.halp_hidden_units,
                    alpha=args.halp_alpha,
                    seed=args.halp_seed,
                )
                t0 = time.time()
                result = run_policy(HALPPolicy(cfg), reqs, pages, cap)
                wall_s = time.time() - t0
                summary = result.extra_diagnostics["halp"]["summary"]
                writer.write_row(
                    {
                        "trace_name": trace_name, "trace_family": family, "capacity": cap,
                        "policy": "halp", "variant": "canonical",
                        "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                        "training_trigger": training_trigger, "hidden_units": args.halp_hidden_units,
                        "alpha": args.halp_alpha, "seed": args.halp_seed,
                        "n_cold_start_evictions": int(summary["n_cold_start_evictions"]),
                        "n_model_ranked_evictions": int(summary["n_model_ranked_evictions"]),
                        "model_trained": int(summary["model_trained"]),
                        "wall_s": round(wall_s, 3),
                    }
                )
                n_rows_written += 1
                print(f"[eval] {trace_name} cap={cap} halp: misses={result.total_misses} ({wall_s:.2f}s)")

            if args.halp_only:
                continue

            row_key = {"trace_name": trace_name, "capacity": cap, "policy": "lru", "variant": "canonical"}
            if writer.already_done(row_key):
                n_rows_skipped += 1
                continue
            t0 = time.time()
            result = run_policy(LRUPolicy(), reqs, pages, cap)
            wall_s = time.time() - t0
            writer.write_row(
                {
                    "trace_name": trace_name, "trace_family": family, "capacity": cap,
                    "policy": "lru", "variant": "canonical",
                    "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                    "training_trigger": "", "hidden_units": "", "alpha": "", "seed": "",
                    "n_cold_start_evictions": "", "n_model_ranked_evictions": "", "model_trained": "",
                    "wall_s": round(wall_s, 3),
                }
            )
            n_rows_written += 1

    writer.close()

    provenance = {
        **base_provenance(),
        "official_halp_commit": None,
        "official_halp_repo": None,
        "official_halp_availability_note": (
            "No official HALP code, simulator, or artifact is public; "
            "see docs/halp_provenance.md. This is an independent "
            "reimplementation of the paper/blog algorithmic description."
        ),
        "sklearn_version": package_version("sklearn"),
        "trace_manifest": str(args.trace_manifest),
        "trace_hashes_sha256": trace_hashes,
        "capacities": caps,
        "max_requests_per_trace": args.max_requests_per_trace,
        "halp_training_trigger_arg": args.halp_training_trigger,
        "halp_hidden_units": args.halp_hidden_units,
        "halp_alpha": args.halp_alpha,
        "halp_seed": args.halp_seed,
        "rows_written_this_invocation": n_rows_written,
        "rows_skipped_already_present": n_rows_skipped,
    }
    write_provenance_json(args.out_dir / "provenance.json", provenance)

    rows_out: List[Dict[str, str]] = []
    if out_csv.exists():
        import csv as _csv

        with out_csv.open() as fh:
            rows_out = list(_csv.DictReader(fh))

    def mean_misses(policy: str) -> float:
        vals = [float(r["misses"]) for r in rows_out if r["policy"] == policy]
        return mean(vals) if vals else float("nan")

    lines = ["# HALP external-baseline comparison", ""]
    lines.append(f"Repository commit: `{provenance['repository_commit']}` | Official HALP commit: none (no public code)")
    lines.append("")
    lines.append("## Mean misses across all trace/capacity rows currently in policy_comparison.csv")
    lines.append("")
    for pname in sorted({r["policy"] for r in rows_out}):
        lines.append(f"- **{pname}:** {mean_misses(pname):.2f}")
    lines.append("")
    lines.append(
        "This file is generated data, not a manuscript claim. Do not cite these numbers "
        "in the manuscript until independently reviewed; see docs/halp_method_spec.md "
        "'Fidelity summary' and docs/baselines.md Baseline 8 'Faithfulness assessment'."
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(
        f"\nWrote {n_rows_written} new row(s), skipped {n_rows_skipped} already-present row(s). "
        f"Outputs under {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
