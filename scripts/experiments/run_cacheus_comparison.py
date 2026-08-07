"""External-baseline experiment: CACHEUS (Rodriguez et al., FAST 2021) vs.
LRU, under the same 7 manuscript trace families, capacities (32/64/128),
and 50,000-request/trace budget as the LRB/3L-Cache/HALP external-baseline
comparisons.

CACHEUS is run via the **official, unmodified source**
(github.com/sylab/cacheus, fetched externally by
`scripts/setup/fetch_cacheus_official.py` -- never vendored, see
docs/cacheus_provenance.md), not a reimplementation. Requires the fetch
script to have been run; this script errors clearly (not silently) and
exits before writing anything if the official source is missing.

Writes each completed (trace, capacity, policy) row to disk immediately
(`lafc.experiments.external_baseline_common.IncrementalCsvWriter`) and is
resumable: re-running with the same `--out-dir` skips rows already present
in `policy_comparison.csv`. Explicit failure rows are written (not silently
skipped) if a specific trace/capacity combination raises.

Does NOT rerun evict_value_v1, LRB, 3L-Cache, or HALP -- use --cacheus-only
(the default; kept as an explicit flag for parity with the other runners'
CLI conventions) to make that explicit.

Outputs (all under analysis/external_learned_baselines/cacheus/, never
touching canonical *_heavy_r1 artifacts or the separate
analysis/external_learned_baselines/{lrb,three_l_cache,halp}/ directories):
    policy_comparison.csv  -- per trace/capacity/policy misses + hit rate (incremental)
    provenance.json          -- commit, versions, seeds, trace hashes, official-source commit
    summary.md                -- human-readable aggregate summary (written at the end)

See docs/cacheus_method_spec.md for the method specification and
docs/baselines.md (Baseline 9) for the standard write-up.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from statistics import mean
from typing import Dict, List

import argparse

from lafc.cacheus_official_loader import EXPECTED_COMMIT, EXTERNAL_CODE_DIR
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.external_baseline_common import (
    IncrementalCsvWriter,
    base_provenance,
    read_trace_manifest,
    sha256_of_file,
    write_provenance_json,
)
from lafc.metrics.cost import hit_rate
from lafc.policies.cacheus import CacheusConfig, CacheusPolicy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy

DEFAULT_MANIFEST = Path("analysis/wulver_trace_manifest_full.csv")
DEFAULT_OUT_DIR = Path("analysis/external_learned_baselines/cacheus")

FIELDNAMES = [
    "trace_name", "trace_family", "capacity", "policy", "variant",
    "requests", "misses", "hit_rate", "window_size",
    "final_weight_srlru", "final_weight_crlfu", "final_learning_rate",
    "n_history_hits_lru", "n_history_hits_lfu", "wall_s", "status", "error",
]
KEY_FIELDS = ["trace_name", "capacity", "policy", "variant"]


def main() -> None:
    ap = argparse.ArgumentParser(description="External CACHEUS baseline comparison (Rodriguez et al., FAST 2021).")
    ap.add_argument("--trace-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--max-requests-per-trace", type=int, default=None)
    ap.add_argument("--cacheus-window-size", type=int, default=100)
    ap.add_argument("--cacheus-only", action="store_true", default=True,
                     help="Run only CACHEUS (default; kept explicit for CLI parity "
                     "with the other external-baseline runners -- this script never "
                     "reruns evict_value_v1/LRB/3L-Cache/HALP regardless).")
    ap.add_argument("--skip-lru", action="store_true", help="Skip the LRU reference row.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if not (EXTERNAL_CODE_DIR / "algs" / "cacheus.py").exists():
        raise SystemExit(
            "Official CACHEUS source not found. Run:\n\n"
            "    python scripts/setup/fetch_cacheus_official.py\n\n"
            "before running this experiment. Refusing to start (no rows written)."
        )

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
    n_rows_failed = 0

    for path, trace_name, family in traces:
        reqs, pages, _src = load_trace_from_any(path)
        if args.max_requests_per_trace:
            reqs = reqs[: args.max_requests_per_trace]
        trace_hashes[trace_name] = sha256_of_file(Path(path))
        n = len(reqs)

        for cap in caps:
            row_key = {"trace_name": trace_name, "capacity": cap, "policy": "cacheus", "variant": "official_source"}
            if writer.already_done(row_key):
                n_rows_skipped += 1
            else:
                try:
                    cfg = CacheusConfig(window_size=args.cacheus_window_size)
                    t0 = time.time()
                    result = run_policy(CacheusPolicy(cfg), reqs, pages, cap)
                    wall_s = time.time() - t0
                    summary = result.extra_diagnostics["cacheus"]["summary"]
                    writer.write_row(
                        {
                            "trace_name": trace_name, "trace_family": family, "capacity": cap,
                            "policy": "cacheus", "variant": "official_source",
                            "requests": n, "misses": result.total_misses, "hit_rate": hit_rate(result.events),
                            "window_size": args.cacheus_window_size,
                            "final_weight_srlru": round(summary["final_weight_srlru"], 6),
                            "final_weight_crlfu": round(summary["final_weight_crlfu"], 6),
                            "final_learning_rate": round(summary["final_learning_rate"], 6),
                            "n_history_hits_lru": int(summary["n_history_hits_lru"]),
                            "n_history_hits_lfu": int(summary["n_history_hits_lfu"]),
                            "wall_s": round(wall_s, 3), "status": "ok", "error": "",
                        }
                    )
                    n_rows_written += 1
                    print(f"[eval] {trace_name} cap={cap} cacheus: misses={result.total_misses} ({wall_s:.2f}s)")
                except Exception as exc:  # noqa: BLE001 -- explicit failure row, not a silent skip
                    writer.write_row(
                        {
                            "trace_name": trace_name, "trace_family": family, "capacity": cap,
                            "policy": "cacheus", "variant": "official_source",
                            "requests": n, "misses": "", "hit_rate": "", "window_size": args.cacheus_window_size,
                            "final_weight_srlru": "", "final_weight_crlfu": "", "final_learning_rate": "",
                            "n_history_hits_lru": "", "n_history_hits_lfu": "",
                            "wall_s": "", "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    n_rows_failed += 1
                    print(f"[FAIL] {trace_name} cap={cap} cacheus: {type(exc).__name__}: {exc}", file=sys.stderr)
                    traceback.print_exc()

            if args.skip_lru:
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
                    "window_size": "", "final_weight_srlru": "", "final_weight_crlfu": "",
                    "final_learning_rate": "", "n_history_hits_lru": "", "n_history_hits_lfu": "",
                    "wall_s": round(wall_s, 3), "status": "ok", "error": "",
                }
            )
            n_rows_written += 1

    writer.close()

    provenance = {
        **base_provenance(),
        "official_cacheus_repo": "https://github.com/sylab/cacheus",
        "official_cacheus_commit": EXPECTED_COMMIT,
        "official_cacheus_license": None,
        "official_cacheus_execution_mode": "external, non-vendored clone; imported and executed unmodified",
        "trace_manifest": str(args.trace_manifest),
        "trace_hashes_sha256": trace_hashes,
        "capacities": caps,
        "max_requests_per_trace": args.max_requests_per_trace,
        "cacheus_window_size": args.cacheus_window_size,
        "rows_written_this_invocation": n_rows_written,
        "rows_skipped_already_present": n_rows_skipped,
        "rows_failed_this_invocation": n_rows_failed,
    }
    write_provenance_json(args.out_dir / "provenance.json", provenance)

    rows_out: List[Dict[str, str]] = []
    if out_csv.exists():
        import csv as _csv

        with out_csv.open() as fh:
            rows_out = list(_csv.DictReader(fh))

    def mean_misses(policy: str) -> float:
        vals = [float(r["misses"]) for r in rows_out if r["policy"] == policy and r.get("status") == "ok" and r["misses"]]
        return mean(vals) if vals else float("nan")

    lines = ["# CACHEUS external-baseline comparison", ""]
    lines.append(f"Repository commit: `{provenance['repository_commit']}` | Official CACHEUS commit: `{EXPECTED_COMMIT}` (no LICENSE file; executed externally, see docs/cacheus_provenance.md)")
    lines.append("")
    lines.append("## Mean misses across all trace/capacity rows currently in policy_comparison.csv (status=ok only)")
    lines.append("")
    for pname in sorted({r["policy"] for r in rows_out}):
        lines.append(f"- **{pname}:** {mean_misses(pname):.2f}")
    n_failed_rows = sum(1 for r in rows_out if r.get("status") == "failed")
    if n_failed_rows:
        lines.append("")
        lines.append(f"**{n_failed_rows} row(s) failed** -- see `status`/`error` columns in policy_comparison.csv.")
    lines.append("")
    lines.append(
        "This file is generated data, not a manuscript claim. Do not cite these numbers "
        "in the manuscript until independently reviewed; see docs/cacheus_method_spec.md "
        "'Fidelity summary' and docs/baselines.md Baseline 9 'Faithfulness assessment'."
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(
        f"\nWrote {n_rows_written} new row(s), skipped {n_rows_skipped} already-present row(s), "
        f"{n_rows_failed} failed row(s). Outputs under {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
