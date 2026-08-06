"""External-baseline experiment: LRB (Song et al., NSDI 2020) vs. the
existing baseline pool and evict_value_v1.

Replays the same 7 manuscript trace families (data/processed/*/trace.jsonl,
via analysis/wulver_trace_manifest_full.csv) at the same capacities
(default 32, 64, 128) and the same 50,000-request/trace budget as the
canonical `evict_value_v1` `heavy_r1` comparison
(scripts/run_policy_comparison_wulver_v1.py), under identical preprocessing
and no separate warm-up trimming -- matching that script's protocol exactly.

LRB's `memory_window` and `batch_size` are validation-tuned per trace and
capacity on a leading held-out fraction of each trace (default 20%, mirroring
the paper's own Section 4.1/6.6 validation-prefix protocol), never on the
evaluated region. One explicit *untuned paper-default* LRB run
(`memory_window=67108864, batch_size=131072`, the official code's literal
defaults) is also included per trace/capacity, to make visible -- rather than
silently hide -- that those CDN-scale defaults never trigger a single retrain
within a 50,000-request trace and degenerate to permanent cold-start LRU.

Outputs (all under analysis/external_learned_baselines/lrb/, never touching
the canonical *_heavy_r1 artifacts):
    policy_comparison.csv   -- per trace/capacity/policy misses + hit rate
    lrb_tuning.csv           -- validation-only memory_window/batch_size search log
    provenance.json          -- commit, official LRB commit, versions, seeds, trace hashes
    summary.md                -- human-readable aggregate summary

See docs/lrb_method_spec.md for the full method specification and
docs/baselines.md (Baseline 6) for the standard write-up.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Tuple

from lafc.evict_value_wulver_v1 import load_trace_from_any
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
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy

LRB_OFFICIAL_COMMIT = "9e8b4423383c01c4528deb447f152f0437a37c3a"
LRB_PAPER_DEFAULT_MEMORY_WINDOW = 67_108_864
LRB_PAPER_DEFAULT_BATCH_SIZE = 131_072

DEFAULT_MANIFEST = Path("analysis/wulver_trace_manifest_full.csv")
DEFAULT_OUT_DIR = Path("analysis/external_learned_baselines/lrb")

BASELINE_POLICIES: Dict[str, Callable[[], BasePolicy]] = {
    "lru": lambda: LRUPolicy(),
    "sieve": lambda: SievePolicy(),
    "fifo_reinsertion": lambda: FIFOReinsertionPolicy(),
    "predictive_marker": lambda: PredictiveMarkerPolicy(),
    "blind_oracle": lambda: BlindOraclePolicy(),
    "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
    "rest_v1": lambda: RestV1Policy(),
    "trust_and_doubt": lambda: TrustAndDoubtPolicy(seed=7),
}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read_manifest(manifest_csv: Path) -> List[Tuple[str, str, str]]:
    rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
    return [
        (
            r["path"].strip(),
            r.get("trace_name", "").strip() or r["path"],
            r.get("trace_family", "").strip() or "unknown",
        )
        for r in rows
    ]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _lightgbm_version() -> str:
    try:
        import lightgbm

        return lightgbm.__version__
    except Exception:
        return "not-installed"


def _tune_memory_window_and_batch_size(
    val_requests,
    pages,
    capacity: int,
    *,
    sample_rate: int,
    seed: int,
    memory_window_grid: List[int],
    batch_size_grid: List[int],
) -> Tuple[int, int, int]:
    """Validation-only grid search: minimizes misses on ``val_requests`` only.

    ``val_requests`` must be a strict prefix of the trace, disjoint from
    whatever is later evaluated as the reported result (see ``main``, which
    tunes on the leading ``validation_fraction`` and then evaluates on the
    full trace -- matching the paper's own validation-prefix protocol).
    """
    best_misses = None
    best_mw, best_bs = memory_window_grid[0], batch_size_grid[0]
    for mw in memory_window_grid:
        for bs in batch_size_grid:
            cfg = LRBConfig(sample_rate=sample_rate, memory_window=mw, batch_size=bs, seed=seed)
            result = run_policy(LRBPolicy(cfg), val_requests, pages, capacity)
            if best_misses is None or result.total_misses < best_misses:
                best_misses = result.total_misses
                best_mw, best_bs = mw, bs
    return best_mw, best_bs, int(best_misses or 0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="External LRB baseline comparison (Song et al., NSDI 2020)."
    )
    ap.add_argument("--trace-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument(
        "--max-requests-per-trace",
        type=int,
        default=None,
        help="Smoke-test override; omit for the full 50,000-request canonical budget.",
    )
    ap.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Leading fraction of each trace used only for LRB memory_window/"
        "batch_size tuning (mirrors the paper's Section 4.1/6.6 protocol).",
    )
    ap.add_argument("--lrb-sample-rate", type=int, default=64)
    ap.add_argument("--lrb-seed", type=int, default=0)
    ap.add_argument("--memory-window-grid", default="1024,4096,16384")
    ap.add_argument("--batch-size-grid", default="512,2048")
    ap.add_argument(
        "--evict-value-model", type=Path, default=Path("models/evict_value_wulver_v1_best.pkl")
    )
    ap.add_argument(
        "--skip-untuned-default",
        action="store_true",
        help="Skip the paper-default (untuned) LRB run.",
    )
    ap.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the lru/sieve/... baseline pool and evict_value_v1 (LRB-only smoke run).",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    caps = [int(x) for x in args.capacities.split(",") if x.strip()]
    mw_grid = [int(x) for x in args.memory_window_grid.split(",") if x.strip()]
    bs_grid = [int(x) for x in args.batch_size_grid.split(",") if x.strip()]
    traces = _read_manifest(args.trace_manifest)
    if not traces:
        raise SystemExit(f"No traces found in manifest {args.trace_manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    trace_hashes: Dict[str, str] = {}

    evict_value_model_present = args.evict_value_model.exists()
    if not evict_value_model_present and not args.skip_baselines:
        print(
            f"WARNING: evict_value_v1 model not found at {args.evict_value_model} -- "
            "this run will NOT include an evict_value_v1 comparison row for any "
            "trace/capacity. This is a silent gap unless you read this warning; "
            "pass --evict-value-model or ensure the canonical model artifact exists "
            "before citing this run as a fairness-complete comparison.",
            file=sys.stderr,
        )

    for path, trace_name, family in traces:
        reqs, pages, _src = load_trace_from_any(path)
        if args.max_requests_per_trace:
            reqs = reqs[: args.max_requests_per_trace]
        trace_hashes[trace_name] = _sha256_of_file(Path(path))

        n = len(reqs)
        n_val = max(1, int(n * args.validation_fraction))
        val_reqs = reqs[:n_val]

        for cap in caps:
            t0 = time.time()
            best_mw, best_bs, val_misses = _tune_memory_window_and_batch_size(
                val_reqs,
                pages,
                cap,
                sample_rate=args.lrb_sample_rate,
                seed=args.lrb_seed,
                memory_window_grid=mw_grid,
                batch_size_grid=bs_grid,
            )
            tuning_wall_s = time.time() - t0
            tuning_rows.append(
                {
                    "trace_name": trace_name,
                    "trace_family": family,
                    "capacity": cap,
                    "validation_requests": n_val,
                    "chosen_memory_window": best_mw,
                    "chosen_batch_size": best_bs,
                    "validation_misses": val_misses,
                    "tuning_wall_s": round(tuning_wall_s, 3),
                }
            )
            print(
                f"[tune] {trace_name} cap={cap}: memory_window={best_mw} "
                f"batch_size={best_bs} (val_misses={val_misses}, {tuning_wall_s:.2f}s)"
            )

            cfg_tuned = LRBConfig(
                sample_rate=args.lrb_sample_rate,
                memory_window=best_mw,
                batch_size=best_bs,
                seed=args.lrb_seed,
            )
            t0 = time.time()
            result = run_policy(LRBPolicy(cfg_tuned), reqs, pages, cap)
            wall_s = time.time() - t0
            summary = result.extra_diagnostics["lrb"]["summary"]
            rows_out.append(
                {
                    "trace_name": trace_name,
                    "trace_family": family,
                    "capacity": cap,
                    "policy": "lrb",
                    "variant": "validation_tuned",
                    "requests": n,
                    "misses": result.total_misses,
                    "hit_rate": hit_rate(result.events),
                    "memory_window": best_mw,
                    "batch_size": best_bs,
                    "sample_rate": args.lrb_sample_rate,
                    "seed": args.lrb_seed,
                    "n_retrain": int(summary["n_retrain"]),
                    "wall_s": round(wall_s, 3),
                }
            )
            print(
                f"[eval] {trace_name} cap={cap} lrb(validation_tuned): "
                f"misses={result.total_misses} n_retrain={int(summary['n_retrain'])} ({wall_s:.2f}s)"
            )

            if not args.skip_untuned_default:
                cfg_default = LRBConfig(
                    sample_rate=args.lrb_sample_rate,
                    memory_window=LRB_PAPER_DEFAULT_MEMORY_WINDOW,
                    batch_size=LRB_PAPER_DEFAULT_BATCH_SIZE,
                    seed=args.lrb_seed,
                )
                t0 = time.time()
                result_d = run_policy(LRBPolicy(cfg_default), reqs, pages, cap)
                wall_s_d = time.time() - t0
                summary_d = result_d.extra_diagnostics["lrb"]["summary"]
                rows_out.append(
                    {
                        "trace_name": trace_name,
                        "trace_family": family,
                        "capacity": cap,
                        "policy": "lrb",
                        "variant": "untuned_paper_default",
                        "requests": n,
                        "misses": result_d.total_misses,
                        "hit_rate": hit_rate(result_d.events),
                        "memory_window": LRB_PAPER_DEFAULT_MEMORY_WINDOW,
                        "batch_size": LRB_PAPER_DEFAULT_BATCH_SIZE,
                        "sample_rate": args.lrb_sample_rate,
                        "seed": args.lrb_seed,
                        "n_retrain": int(summary_d["n_retrain"]),
                        "wall_s": round(wall_s_d, 3),
                    }
                )
                print(
                    f"[eval] {trace_name} cap={cap} lrb(untuned_paper_default): "
                    f"misses={result_d.total_misses} n_retrain={int(summary_d['n_retrain'])} "
                    f"({wall_s_d:.2f}s)"
                )

            if not args.skip_baselines:
                td_reqs = attach_predicted_caches(reqs, capacity=cap)
                if evict_value_model_present:
                    t0 = time.time()
                    result_ev = run_policy(
                        EvictValueV1Policy(model_path=str(args.evict_value_model)),
                        reqs,
                        pages,
                        cap,
                    )
                    wall_s_ev = time.time() - t0
                    rows_out.append(
                        {
                            "trace_name": trace_name,
                            "trace_family": family,
                            "capacity": cap,
                            "policy": "evict_value_v1",
                            "variant": "canonical",
                            "requests": n,
                            "misses": result_ev.total_misses,
                            "hit_rate": hit_rate(result_ev.events),
                            "memory_window": "",
                            "batch_size": "",
                            "sample_rate": "",
                            "seed": "",
                            "n_retrain": "",
                            "wall_s": round(wall_s_ev, 3),
                        }
                    )

                for pname, fac in BASELINE_POLICIES.items():
                    # trust_and_doubt requires predicted_cache metadata, matching
                    # the canonical scripts/run_policy_comparison_wulver_v1.py handling.
                    pol_reqs = td_reqs if pname == "trust_and_doubt" else reqs
                    t0 = time.time()
                    result_b = run_policy(fac(), pol_reqs, pages, cap)
                    wall_s_b = time.time() - t0
                    rows_out.append(
                        {
                            "trace_name": trace_name,
                            "trace_family": family,
                            "capacity": cap,
                            "policy": pname,
                            "variant": "canonical",
                            "requests": n,
                            "misses": result_b.total_misses,
                            "hit_rate": hit_rate(result_b.events),
                            "memory_window": "",
                            "batch_size": "",
                            "sample_rate": "",
                            "seed": "",
                            "n_retrain": "",
                            "wall_s": round(wall_s_b, 3),
                        }
                    )

    fieldnames = list(rows_out[0].keys())
    out_csv = args.out_dir / "policy_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    tuning_csv = args.out_dir / "lrb_tuning.csv"
    with tuning_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tuning_rows[0].keys()))
        w.writeheader()
        w.writerows(tuning_rows)

    provenance = {
        "repository_commit": _git_commit(),
        "official_lrb_commit": LRB_OFFICIAL_COMMIT,
        "official_lrb_repo": "https://github.com/sunnyszy/lrb",
        "python_version": sys.version,
        "platform": platform.platform(),
        "lightgbm_version": _lightgbm_version(),
        "trace_manifest": str(args.trace_manifest),
        "trace_hashes_sha256": trace_hashes,
        "capacities": caps,
        "max_requests_per_trace": args.max_requests_per_trace,
        "validation_fraction": args.validation_fraction,
        "memory_window_grid": mw_grid,
        "batch_size_grid": bs_grid,
        "lrb_sample_rate": args.lrb_sample_rate,
        "lrb_seed": args.lrb_seed,
        "lrb_paper_default_memory_window": LRB_PAPER_DEFAULT_MEMORY_WINDOW,
        "lrb_paper_default_batch_size": LRB_PAPER_DEFAULT_BATCH_SIZE,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def mean_misses(policy: str, variant: str | None = None) -> float:
        vals = [
            float(r["misses"])
            for r in rows_out
            if r["policy"] == policy and (variant is None or r["variant"] == variant)
        ]
        return mean(vals) if vals else float("nan")

    lines: List[str] = []
    lines.append("# LRB external-baseline comparison")
    lines.append("")
    lines.append(
        f"Repository commit: `{provenance['repository_commit']}` | "
        f"Official LRB commit: `{LRB_OFFICIAL_COMMIT}` | "
        f"LightGBM: `{provenance['lightgbm_version']}`"
    )
    lines.append("")
    lines.append(f"Capacities: {caps} | Validation fraction: {args.validation_fraction}")
    lines.append("")
    lines.append("## Mean misses across all trace/capacity rows in this run")
    lines.append("")
    lines.append(f"- **lrb (validation_tuned):** {mean_misses('lrb', 'validation_tuned'):.2f}")
    if not args.skip_untuned_default:
        lines.append(
            f"- **lrb (untuned_paper_default):** {mean_misses('lrb', 'untuned_paper_default'):.2f} "
            "(expected to degenerate toward permanent cold-start LRU at this request scale -- see docs/lrb_method_spec.md)"
        )
    if not args.skip_baselines:
        if evict_value_model_present:
            lines.append(f"- **evict_value_v1:** {mean_misses('evict_value_v1'):.2f}")
        for pname in BASELINE_POLICIES:
            lines.append(f"- **{pname}:** {mean_misses(pname):.2f}")
    lines.append("")
    lines.append(
        "This file is generated data, not a manuscript claim. Do not cite these numbers "
        "in the manuscript until independently reviewed; see docs/lrb_method_spec.md "
        "'Known limitations' and docs/baselines.md Baseline 6 'Faithfulness assessment'."
    )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nWrote {out_csv}, {tuning_csv}, provenance.json, summary.md under {args.out_dir}/")


if __name__ == "__main__":
    main()
