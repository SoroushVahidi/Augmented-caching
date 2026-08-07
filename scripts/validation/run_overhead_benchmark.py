"""Controlled per-decision latency benchmark (non-canonical, timing-only).

This script is intentionally separate from scripts/run_policy_comparison_wulver_v1.py
(the canonical KBS cap32/64/128 sweep). It exists only to answer R2-MC2 / R3-Issue5 /
R3-Rec3 (computational overhead): how much wall-clock time does evict_value_v1's
per-candidate scoring cost relative to LRU/SIEVE/FIFO-Reinsertion/REST, as a function
of cache capacity, on one representative trace.

Timing boundary: each call to policy.on_request(req) inside the request loop. Trace
loading (load_trace_from_any) and per-policy/per-capacity policy.reset(...) happen
outside the timed loop, so setup cost is excluded. Decisions are additionally split
into "all requests" (hits + misses) and "eviction decisions only" (event.evicted is
not None), since the latter isolates the O(k) candidate-scan cost that LRU/SIEVE/
FIFO-Reinsertion do not pay per-candidate.

This must never be confused with, or overwrite, canonical cap32/cap64/cap128 outputs
or analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv. All outputs use the
`kbs_overhead_benchmark_` prefix.

Run on the author's local/cloud machine via tmux. Not run on Wulver or under Slurm.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.sieve import SievePolicy

BENCHMARK_ENVIRONMENT = "local/cloud machine via tmux (not Wulver, not Slurm)"

POLICY_FACTORIES = {
    "lru": lambda m: LRUPolicy(),
    "sieve": lambda m: SievePolicy(),
    "fifo_reinsertion": lambda m: FIFOReinsertionPolicy(),
    "rest_v1": lambda m: RestV1Policy(),
    "evict_value_v1": lambda m: EvictValueV1Policy(model_path=m),
}


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _stats(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "total_s": 0.0, "n": 0}
    sorted_vals = sorted(values_ms)
    return {
        "mean_ms": statistics.fmean(values_ms),
        "median_ms": statistics.median(values_ms),
        "p95_ms": _percentile(sorted_vals, 0.95),
        "total_s": sum(values_ms) / 1000.0,
        "n": len(values_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled per-decision latency benchmark (non-canonical).")
    ap.add_argument("--trace-path", default="data/processed/brightkite/trace.jsonl")
    ap.add_argument("--trace-name", default="brightkite_50k")
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--policies", default="lru,sieve,fifo_reinsertion,rest_v1,evict_value_v1")
    ap.add_argument("--evict-value-model", default="models/evict_value_wulver_v1_best_heavy_r1.pkl")
    ap.add_argument("--max-requests", type=int, default=5000)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    args = ap.parse_args()

    command_str = "python " + " ".join([sys.argv[0]] + sys.argv[1:])
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    caps = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    pol_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = sorted([p for p in pol_names if p not in POLICY_FACTORIES])
    if unknown:
        raise SystemExit(f"Unknown policy name(s): {', '.join(unknown)}")

    requests, pages, _src = load_trace_from_any(args.trace_path)
    if args.max_requests:
        requests = requests[: args.max_requests]
    n_requests = len(requests)
    if n_requests == 0:
        raise SystemExit("Trace produced zero requests after truncation.")

    rows: List[Dict[str, object]] = []
    failures: List[str] = []

    for cap in caps:
        for pname in pol_names:
            fac = POLICY_FACTORIES[pname]
            model_arg = args.evict_value_model if pname == "evict_value_v1" else ""
            try:
                policy = fac(model_arg)
                policy.reset(cap, pages)

                all_ms: List[float] = []
                evict_ms: List[float] = []
                n_misses = 0

                for req in requests:
                    t0 = time.perf_counter()
                    event = policy.on_request(req)
                    t1 = time.perf_counter()
                    elapsed_ms = (t1 - t0) * 1000.0
                    all_ms.append(elapsed_ms)
                    if not event.hit:
                        n_misses += 1
                    if event.evicted is not None:
                        evict_ms.append(elapsed_ms)

                all_stats = _stats(all_ms)
                evict_stats = _stats(evict_ms)

                rows.append(
                    {
                        "trace_name": args.trace_name,
                        "trace_path": args.trace_path,
                        "capacity": cap,
                        "policy": pname,
                        "n_requests": n_requests,
                        "n_misses": n_misses,
                        "n_eviction_decisions": evict_stats["n"],
                        "total_timed_seconds_all_requests": round(all_stats["total_s"], 6),
                        "mean_ms_per_request_all": round(all_stats["mean_ms"], 6),
                        "median_ms_per_request_all": round(all_stats["median_ms"], 6),
                        "p95_ms_per_request_all": round(all_stats["p95_ms"], 6),
                        "mean_ms_per_eviction_decision": round(evict_stats["mean_ms"], 6),
                        "median_ms_per_eviction_decision": round(evict_stats["median_ms"], 6),
                        "p95_ms_per_eviction_decision": round(evict_stats["p95_ms"], 6),
                        "benchmark_environment": BENCHMARK_ENVIRONMENT,
                        "timestamp_utc": timestamp_utc,
                        "command": command_str,
                        "status": "ok",
                    }
                )
                print(
                    f"[ok] capacity={cap} policy={pname} "
                    f"mean_ms_per_eviction_decision={evict_stats['mean_ms']:.4f} "
                    f"(n_eviction_decisions={evict_stats['n']}, n_misses={n_misses})"
                )
            except Exception as exc:  # noqa: BLE001 - deliberately broad; this is a benchmark probe, report and continue
                msg = f"capacity={cap} policy={pname}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                print(f"[FAILED] {msg}")
                rows.append(
                    {
                        "trace_name": args.trace_name,
                        "trace_path": args.trace_path,
                        "capacity": cap,
                        "policy": pname,
                        "n_requests": n_requests,
                        "n_misses": "",
                        "n_eviction_decisions": "",
                        "total_timed_seconds_all_requests": "",
                        "mean_ms_per_request_all": "",
                        "median_ms_per_request_all": "",
                        "p95_ms_per_request_all": "",
                        "mean_ms_per_eviction_decision": "",
                        "median_ms_per_eviction_decision": "",
                        "p95_ms_per_eviction_decision": "",
                        "benchmark_environment": BENCHMARK_ENVIRONMENT,
                        "timestamp_utc": timestamp_utc,
                        "command": command_str,
                        "status": f"FAILED: {type(exc).__name__}: {exc}",
                    }
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines: List[str] = []
    lines.append("# KBS overhead benchmark (non-canonical, timing-only)")
    lines.append("")
    lines.append(f"- Trace: `{args.trace_name}` (`{args.trace_path}`)")
    lines.append(f"- Requests used: {n_requests} (of the trace's full length; not the full 50,000-request canonical scale)")
    lines.append(f"- Capacities: {', '.join(str(c) for c in caps)}")
    lines.append(f"- Policies: {', '.join(pol_names)}")
    lines.append(f"- Benchmark environment: {BENCHMARK_ENVIRONMENT}")
    lines.append(f"- Timestamp (UTC): {timestamp_utc}")
    lines.append(f"- Command: `{command_str}`")
    lines.append("")
    lines.append(
        "Timing boundary: wall-clock around each `policy.on_request(req)` call only; "
        "trace loading and `policy.reset(...)` are excluded. \"Eviction decision\" rows "
        "are the subset of calls where `event.evicted is not None` (i.e. the cache was "
        "full and a victim was actually chosen) -- this isolates the O(k) candidate-scan "
        "cost from O(1) hit-path bookkeeping."
    )
    lines.append("")
    lines.append("## Mean ms per eviction decision, by policy and capacity")
    lines.append("")
    lines.append("| capacity | policy | n_eviction_decisions | mean_ms | median_ms | p95_ms | status |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['capacity']} | {r['policy']} | {r['n_eviction_decisions']} | "
            f"{r['mean_ms_per_eviction_decision']} | {r['median_ms_per_eviction_decision']} | "
            f"{r['p95_ms_per_eviction_decision']} | {r['status']} |"
        )
    lines.append("")
    if failures:
        lines.append("## Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {args.out_csv} and {args.out_md}")
    if failures:
        print(f"{len(failures)} (capacity, policy) pair(s) FAILED -- see {args.out_md}")


if __name__ == "__main__":
    main()
