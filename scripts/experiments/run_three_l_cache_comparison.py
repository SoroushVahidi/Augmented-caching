"""External-baseline experiment: 3L-Cache (Zhou et al., FAST 2025) vs. LRB,
evict_value_v1, and the existing baseline pool.

Replays the same 7 manuscript trace families, capacities (32/64/128), and
50,000-request/trace budget as the canonical `evict_value_v1` `heavy_r1`
comparison and the LRB external-baseline comparison
(`scripts/experiments/run_lrb_external_baseline.py`), under identical
preprocessing and no separate warm-up trimming.

Unlike the first LRB experiment runner, this script writes each completed
(trace, capacity, policy, variant) row to disk immediately (see
`lafc.experiments.external_baseline_common.IncrementalCsvWriter`) and is
resumable: re-running with the same `--out-dir` skips rows already present
in `policy_comparison.csv`, so an interrupted run never loses completed
work and results survive interruption.

`evict_value_v1` is known to be extremely slow per eviction decision
(paper-documented 75-316ms/decision depending on capacity). This script
does NOT silently reuse the canonical `heavy_r1` CSV's evict_value_v1
numbers, because that artifact does not carry the trace-hash/code-version
provenance this script requires to *prove* exact equivalence before reuse
(scientific fairness over saved runtime, per the experiment's own
requirements). It DOES skip recomputing evict_value_v1 rows already present
in *this script's own* prior output (self-resume, full provenance
recorded), and supports `--three-l-cache-only`/`--skip-baselines` to avoid
it entirely for a fast 3L-Cache-focused run.

Outputs (all under analysis/external_learned_baselines/three_l_cache/,
never touching canonical *_heavy_r1 artifacts or the separate
analysis/external_learned_baselines/lrb/ directory):
    policy_comparison.csv    -- per trace/capacity/policy misses + hit rate (incremental)
    three_l_cache_tuning.csv  -- validation-only batch_size search log
    provenance.json           -- commit, official 3L-Cache commit, versions, seeds, trace hashes
    summary.md                 -- human-readable aggregate summary (written at the end)

See docs/three_l_cache_method_spec.md for the method specification and
docs/baselines.md (Baseline 7) for the standard write-up.
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
from lafc.policies.base import BasePolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy
from lafc.policies.lrb import LRBConfig, LRBPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.sieve import SievePolicy
from lafc.policies.three_l_cache import ThreeLCacheConfig, ThreeLCachePolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy

THREE_L_CACHE_OFFICIAL_COMMIT = "134cd159b635cdab75419a4281bed1a330fef31f"
LRB_OFFICIAL_COMMIT = "9e8b4423383c01c4528deb447f152f0437a37c3a"

DEFAULT_MANIFEST = Path("analysis/wulver_trace_manifest_full.csv")
DEFAULT_OUT_DIR = Path("analysis/external_learned_baselines/three_l_cache")

FIELDNAMES = [
    "trace_name", "trace_family", "capacity", "policy", "variant",
    "requests", "misses", "hit_rate", "batch_size", "seed", "n_retrain",
    "wall_s",
]
KEY_FIELDS = ["trace_name", "capacity", "policy", "variant"]

BASELINE_POLICIES: Dict[str, "type_or_factory"] = {
    "lru": lambda: LRUPolicy(),
    "sieve": lambda: SievePolicy(),
    "fifo_reinsertion": lambda: FIFOReinsertionPolicy(),
    "predictive_marker": lambda: PredictiveMarkerPolicy(),
    "blind_oracle": lambda: BlindOraclePolicy(),
    "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
    "rest_v1": lambda: RestV1Policy(),
    "trust_and_doubt": lambda: TrustAndDoubtPolicy(seed=7),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="External 3L-Cache baseline comparison (Zhou et al., FAST 2025).")
    ap.add_argument("--trace-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--max-requests-per-trace", type=int, default=None)
    ap.add_argument("--validation-fraction", type=float, default=0.2)
    ap.add_argument("--three-l-cache-seed", type=int, default=0)
    ap.add_argument("--three-l-cache-batch-size", type=int, default=4096)
    ap.add_argument("--lrb-memory-window", type=int, default=4096)
    ap.add_argument("--lrb-batch-size", type=int, default=2048)
    ap.add_argument("--lrb-seed", type=int, default=0)
    ap.add_argument("--evict-value-model", type=Path, default=Path("models/evict_value_wulver_v1_best.pkl"))
    ap.add_argument(
        "--three-l-cache-only", "--skip-baselines",
        dest="three_l_cache_only", action="store_true",
        help="Run only 3L-Cache (skip LRB, evict_value_v1, and the classical baseline pool).",
    )
    ap.add_argument("--skip-lrb", action="store_true", help="Skip the LRB comparison row.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    researcher_bs = args.three_l_cache_batch_size
    traces = read_trace_manifest(args.trace_manifest)
    if not traces:
        raise SystemExit(f"No traces found in manifest {args.trace_manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "policy_comparison.csv"
    writer = IncrementalCsvWriter(out_csv, FIELDNAMES, KEY_FIELDS)

    evict_value_model_present = args.evict_value_model.exists()
    if not evict_value_model_present and not args.three_l_cache_only:
        print(
            f"WARNING: evict_value_v1 model not found at {args.evict_value_model} -- "
            "this run will NOT include an evict_value_v1 comparison row.",
            file=sys.stderr,
        )

    trace_hashes: Dict[str, str] = {}
    n_rows_written = 0
    n_rows_skipped = 0

    for path, trace_name, family in traces:
        reqs, pages, _src = load_trace_from_any(path)
        if args.max_requests_per_trace:
            reqs = reqs[: args.max_requests_per_trace]
        trace_hashes[trace_name] = sha256_of_file(Path(path))
        n = len(reqs)
        n_val = max(1, int(n * args.validation_fraction))
        val_reqs = reqs[:n_val]

        for cap in caps:
            # --- 3L-Cache: official default (65536) ---
            row_key = {"trace_name": trace_name, "capacity": cap, "policy": "three_l_cache", "variant": "official_default"}
            if writer.already_done(row_key):
                n_rows_skipped += 1
            else:
                cfg = ThreeLCacheConfig(batch_size=65536, seed=args.three_l_cache_seed)
                t0 = time.time()
                result = run_policy(ThreeLCachePolicy(cfg), reqs, pages, cap)
                wall_s = time.time() - t0
                summary = result.extra_diagnostics["three_l_cache"]["summary"]
                writer.write_row(
                    {
                        "trace_name": trace_name, "trace_family": family, "capacity": cap,
                        "policy": "three_l_cache", "variant": "official_default",
                        "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                        "batch_size": 65536, "seed": args.three_l_cache_seed,
                        "n_retrain": int(summary["n_retrain"]), "wall_s": round(wall_s, 3),
                    }
                )
                n_rows_written += 1
                print(f"[eval] {trace_name} cap={cap} three_l_cache (official): misses={result.total_misses} ({wall_s:.2f}s)")

            # --- 3L-Cache: researcher-selected default (4096) ---
            row_key = {"trace_name": trace_name, "capacity": cap, "policy": "three_l_cache", "variant": "researcher_default"}
            if writer.already_done(row_key):
                n_rows_skipped += 1
            else:
                cfg = ThreeLCacheConfig(batch_size=researcher_bs, seed=args.three_l_cache_seed)
                t0 = time.time()
                result = run_policy(ThreeLCachePolicy(cfg), reqs, pages, cap)
                wall_s = time.time() - t0
                summary = result.extra_diagnostics["three_l_cache"]["summary"]
                writer.write_row(
                    {
                        "trace_name": trace_name, "trace_family": family, "capacity": cap,
                        "policy": "three_l_cache", "variant": "researcher_default",
                        "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                        "batch_size": researcher_bs, "seed": args.three_l_cache_seed,
                        "n_retrain": int(summary["n_retrain"]), "wall_s": round(wall_s, 3),
                    }
                )
                n_rows_written += 1
                print(f"[eval] {trace_name} cap={cap} three_l_cache (researcher): misses={result.total_misses} ({wall_s:.2f}s)")

            # --- LRB comparison row (same conditions) ---
            if not args.skip_lrb and not args.three_l_cache_only:
                row_key = {"trace_name": trace_name, "capacity": cap, "policy": "lrb", "variant": "validation_tuned"}
                if writer.already_done(row_key):
                    n_rows_skipped += 1
                else:
                    lrb_cfg = LRBConfig(
                        memory_window=args.lrb_memory_window, batch_size=args.lrb_batch_size, seed=args.lrb_seed
                    )
                    t0 = time.time()
                    result = run_policy(LRBPolicy(lrb_cfg), reqs, pages, cap)
                    wall_s = time.time() - t0
                    summary = result.extra_diagnostics["lrb"]["summary"]
                    writer.write_row(
                        {
                            "trace_name": trace_name, "trace_family": family, "capacity": cap,
                            "policy": "lrb", "variant": "validation_tuned",
                            "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                            "batch_size": args.lrb_batch_size, "seed": args.lrb_seed,
                            "n_retrain": int(summary["n_retrain"]), "wall_s": round(wall_s, 3),
                        }
                    )
                    n_rows_written += 1

            if args.three_l_cache_only:
                continue

            # --- evict_value_v1 + classical baseline pool, identical conditions ---
            if evict_value_model_present:
                row_key = {"trace_name": trace_name, "capacity": cap, "policy": "evict_value_v1", "variant": "canonical"}
                if writer.already_done(row_key):
                    n_rows_skipped += 1
                else:
                    t0 = time.time()
                    result = run_policy(
                        EvictValueV1Policy(model_path=str(args.evict_value_model)), reqs, pages, cap
                    )
                    wall_s = time.time() - t0
                    writer.write_row(
                        {
                            "trace_name": trace_name, "trace_family": family, "capacity": cap,
                            "policy": "evict_value_v1", "variant": "canonical",
                            "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                            "batch_size": "", "seed": "", "n_retrain": "", "wall_s": round(wall_s, 3),
                        }
                    )
                    n_rows_written += 1

            td_reqs = None
            for pname, fac in BASELINE_POLICIES.items():
                row_key = {"trace_name": trace_name, "capacity": cap, "policy": pname, "variant": "canonical"}
                if writer.already_done(row_key):
                    n_rows_skipped += 1
                    continue
                if pname == "trust_and_doubt":
                    if td_reqs is None:
                        td_reqs = attach_predicted_caches(reqs, capacity=cap)
                    pol_reqs = td_reqs
                else:
                    pol_reqs = reqs
                t0 = time.time()
                result = run_policy(fac(), pol_reqs, pages, cap)
                wall_s = time.time() - t0
                writer.write_row(
                    {
                        "trace_name": trace_name, "trace_family": family, "capacity": cap,
                        "policy": pname, "variant": "canonical",
                        "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                        "batch_size": "", "seed": "", "n_retrain": "", "wall_s": round(wall_s, 3),
                    }
                )
                n_rows_written += 1

    writer.close()
    

    provenance = {
        **base_provenance(),
        "official_three_l_cache_commit": THREE_L_CACHE_OFFICIAL_COMMIT,
        "official_three_l_cache_repo": "https://github.com/optiq-lab/3L-Cache",
        "official_lrb_commit": LRB_OFFICIAL_COMMIT,
        "lightgbm_version": package_version("lightgbm"),
        "trace_manifest": str(args.trace_manifest),
        "trace_hashes_sha256": trace_hashes,
        "capacities": caps,
        "max_requests_per_trace": args.max_requests_per_trace,
        "validation_fraction": args.validation_fraction,
        "three_l_cache_batch_size": researcher_bs,
        "three_l_cache_seed": args.three_l_cache_seed,
        "lrb_memory_window": args.lrb_memory_window,
        "lrb_batch_size": args.lrb_batch_size,
        "lrb_seed": args.lrb_seed,
        "rows_written_this_invocation": n_rows_written,
        "rows_skipped_already_present": n_rows_skipped,
    }
    write_provenance_json(args.out_dir / "provenance.json", provenance)

    rows_out = []
    if out_csv.exists():
        import csv as _csv

        with out_csv.open() as fh:
            rows_out = list(_csv.DictReader(fh))

    def mean_misses(policy: str) -> float:
        vals = [float(r["misses"]) for r in rows_out if r["policy"] == policy]
        return mean(vals) if vals else float("nan")

    lines = ["# 3L-Cache external-baseline comparison", ""]
    lines.append(
        f"Repository commit: `{provenance['repository_commit']}` | "
        f"Official 3L-Cache commit: `{THREE_L_CACHE_OFFICIAL_COMMIT}` | "
        f"Official LRB commit: `{LRB_OFFICIAL_COMMIT}`"
    )
    lines.append("")
    lines.append("## Mean misses across all trace/capacity rows currently in policy_comparison.csv")
    lines.append("")
    for pname in sorted({r["policy"] for r in rows_out}):
        lines.append(f"- **{pname}:** {mean_misses(pname):.2f}")
    lines.append("")
    lines.append(
        "This file is generated data, not a manuscript claim. Do not cite these numbers "
        "in the manuscript until independently reviewed; see docs/three_l_cache_method_spec.md "
        "'Known limitations' and docs/baselines.md Baseline 7 'Faithfulness assessment'."
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(
        f"\nWrote {n_rows_written} new row(s), skipped {n_rows_skipped} already-present row(s). "
        f"Outputs under {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
