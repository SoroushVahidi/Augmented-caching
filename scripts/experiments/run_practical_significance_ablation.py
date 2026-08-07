"""Practical-significance ablation runner (Reviewer 1, Concern 4).

Implements configs/practical_significance_ablation_v1.json /
docs/practical_significance_ablation_protocol.md. Non-API only.

Modes (any combination, or --all):
    --profile             component-level latency breakdown by trace/capacity/variant
    --exact-optimizations  decision-equivalence check + speedup for the exact variants
    --selective            selective-invocation quality/invocation-rate outputs
    --top-k                top-k candidate-pruning quality/cost curve
    --model-complexity     reuses the frozen h4 ridge/random_forest/hist_gb artifacts
    --break-even           break-even miss-cost analysis vs certified fair baselines
    --miss-cost-sweep      synthetic log-spaced miss-cost grid (no timing needed)
    --weighted-cost         synthetic heterogeneous-cost sensitivity analysis
    --pareto               quality-latency Pareto frontier

All outputs are written incrementally under analysis/practical_significance_ablation_v1/
and are resumable (--resume skips rows already present in the output CSV).

IMPORTANT: --profile timing numbers are only reviewer-facing evidence when
collected on an idle machine (see protocol doc Section 10). Pass
--smoke-scale (the default) for correctness-only runs while other Concern
1-3 jobs may be active; only drop --smoke-scale for the final controlled
campaign on an idle machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lafc.evict_value_wulver_v1 import load_trace_from_any  # noqa: E402
from lafc.policies.evict_value_v1 import EvictValueV1Policy  # noqa: E402
from lafc.policies.evict_value_v1_optimized import (  # noqa: E402
    EvictValueV1CachedExactPolicy,
    EvictValueV1VectorizedCachedExactPolicy,
    EvictValueV1VectorizedExactPolicy,
    score_candidates_vectorized_cached_exact,
)
from lafc.policies.evict_value_v1_selective import (  # noqa: E402
    EvictValueV1SelectiveDisagreementPolicy,
    EvictValueV1SelectivePeriodicPolicy,
    EvictValueV1TopKPolicy,
    canonical_victim_would_be_pruned,
)
from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy  # noqa: E402
from lafc.policies.lru import LRUPolicy  # noqa: E402
from lafc.policies.sieve import SievePolicy  # noqa: E402
from lafc.types import Page, PageId, Request  # noqa: E402

PROTOCOL_PATH = Path("configs/practical_significance_ablation_v1.json")
OUT_DIR = Path("analysis/practical_significance_ablation_v1")

EXACT_VARIANT_CLASSES = {
    "canonical": EvictValueV1Policy,
    "cached_exact": EvictValueV1CachedExactPolicy,
    "vectorized_exact": EvictValueV1VectorizedExactPolicy,
    "vectorized_cached_exact": EvictValueV1VectorizedCachedExactPolicy,
}

FIXED_BREAK_EVEN_BASELINES = ["lru", "sieve", "fifo_reinsertion"]


def load_protocol() -> Dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def command_str() -> str:
    return "python " + " ".join([sys.argv[0]] + sys.argv[1:])


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Incremental / resumable CSV writer
# ---------------------------------------------------------------------------


def read_existing_keys(path: Path, key_fields: Sequence[str]) -> set:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys.add(tuple(row.get(f, "") for f in key_fields))
    return keys


def append_rows(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Trace / model loading
# ---------------------------------------------------------------------------


@dataclass
class TraceHandle:
    trace_name: str
    trace_family: str
    requests: List[Request]
    pages: Dict[PageId, Page]


def load_traces(protocol: Dict[str, object], data_read_root: str) -> Dict[str, TraceHandle]:
    trace_hashes = protocol["traces"]["trace_hashes_sha256"]
    family_by_trace_name = {
        "brightkite_50k": "brightkite",
        "citibike_202401_50k": "citibike",
        "wiki2018_pageviews_en_50k": "wiki2018",
        "twemcache_cluster26_sample100_50k": "twemcache",
        "metakv_kvcache_202206_head_50k": "metakv",
        "metacdn_cdn_202303_head_50k": "metacdn",
        "cloudphysics_alibaba_block_head_50k": "cloudphysics",
    }
    out: Dict[str, TraceHandle] = {}
    for trace_name in trace_hashes:
        family = family_by_trace_name[trace_name]
        path = f"{data_read_root}/data/processed/{family}/trace.jsonl"
        requests, pages, _src = load_trace_from_any(path)
        out[trace_name] = TraceHandle(trace_name=trace_name, trace_family=family, requests=requests, pages=pages)
    return out


def _stats_ms(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "n": 0}
    s = sorted(values_ms)

    def pct(p: float) -> float:
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)

    return {
        "mean_ms": statistics.fmean(values_ms),
        "median_ms": statistics.median(values_ms),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "n": len(values_ms),
    }


# ---------------------------------------------------------------------------
# --profile
# ---------------------------------------------------------------------------


def _profiled_eviction_decision(policy, request: Request) -> Tuple[PageId, Dict[str, float]]:
    """Component-timed replica of vectorized_cached_exact's decision path.
    Returns (victim, {component_name: elapsed_ms}). Deliberately a separate
    read-only instrumentation path -- never used for the actual replay
    (canonical/exact-variant classes remain untouched by profiling)."""
    t0 = time.perf_counter()
    candidates = list(policy._order.keys())
    t1 = time.perf_counter()
    pred_losses = score_candidates_vectorized_cached_exact(policy, request, candidates)
    t2 = time.perf_counter()
    idx_of = {p: i for i, p in enumerate(candidates)}
    victim = min(candidates, key=lambda p: (pred_losses[p], idx_of[p]))
    t3 = time.perf_counter()
    return victim, {
        "B_candidate_construction_ms": (t1 - t0) * 1000.0,
        "CD_feature_and_prediction_ms": (t2 - t1) * 1000.0,
        "E_ranking_ms": (t3 - t2) * 1000.0,
    }


def mode_profile(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "profiler_breakdown.csv"
    fields = [
        "trace_name", "capacity", "repetition", "n_decisions_timed",
        "mean_ms_per_eviction_decision_whole", "median_ms_per_eviction_decision_whole",
        "p90_ms_per_eviction_decision_whole", "p95_ms_per_eviction_decision_whole", "p99_ms_per_eviction_decision_whole",
        "mean_ms_B_candidate_construction", "mean_ms_CD_feature_and_prediction", "mean_ms_E_ranking", "mean_ms_A_F_other",
        "n_requests_used", "warmup_discarded", "smoke_scale", "timestamp_utc", "command",
    ]
    key_fields = ["trace_name", "capacity", "repetition"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    model_path = args.evict_value_model
    traces = load_traces(protocol, args.data_read_root)
    warmup = 0 if args.smoke_scale else 200
    reps = 1 if args.smoke_scale else int(protocol["timing_methodology"]["repetitions"])

    rows: List[Dict[str, object]] = []
    for trace_name, th in traces.items():
        requests = th.requests[: args.max_requests]
        for cap in args.capacities:
            for rep in range(reps):
                key = (trace_name, str(cap), str(rep))
                if key in existing:
                    continue
                policy = EvictValueV1VectorizedCachedExactPolicy(model_path=model_path)
                policy.reset(cap, th.pages)
                whole_ms: List[float] = []
                comp_totals = {"B_candidate_construction_ms": 0.0, "CD_feature_and_prediction_ms": 0.0, "E_ranking_ms": 0.0}
                n_evict = 0
                for i, req in enumerate(requests):
                    pid = req.page_id
                    t0 = time.perf_counter()
                    if policy.in_cache(pid):
                        policy._order.move_to_end(pid)
                        policy._record_hit()
                        policy._recent_req_hist.append(pid)
                        policy._recent_hit_hist.append(pid)
                        t1 = time.perf_counter()
                    else:
                        policy._record_miss(1.0)
                        if policy._cache.is_full():
                            victim, comps = _profiled_eviction_decision(policy, req)
                            policy._evict(victim)
                            policy._order.pop(victim, None)
                            policy._evictions += 1
                            if i >= warmup:
                                for k, v in comps.items():
                                    comp_totals[k] += v
                                n_evict += 1
                        policy._add(pid)
                        policy._order[pid] = None
                        policy._recent_req_hist.append(pid)
                        t1 = time.perf_counter()
                    if i >= warmup:
                        whole_ms.append((t1 - t0) * 1000.0)

                stats = _stats_ms(whole_ms)
                mean_b = comp_totals["B_candidate_construction_ms"] / n_evict if n_evict else 0.0
                mean_cd = comp_totals["CD_feature_and_prediction_ms"] / n_evict if n_evict else 0.0
                mean_e = comp_totals["E_ranking_ms"] / n_evict if n_evict else 0.0
                mean_whole_evict = stats["mean_ms"]
                mean_a_f_other = max(0.0, mean_whole_evict - mean_b - mean_cd - mean_e)
                row = {
                    "trace_name": trace_name,
                    "capacity": cap,
                    "repetition": rep,
                    "n_decisions_timed": stats["n"],
                    "mean_ms_per_eviction_decision_whole": round(stats["mean_ms"], 6),
                    "median_ms_per_eviction_decision_whole": round(stats["median_ms"], 6),
                    "p90_ms_per_eviction_decision_whole": round(stats["p90_ms"], 6),
                    "p95_ms_per_eviction_decision_whole": round(stats["p95_ms"], 6),
                    "p99_ms_per_eviction_decision_whole": round(stats["p99_ms"], 6),
                    "mean_ms_B_candidate_construction": round(mean_b, 6),
                    "mean_ms_CD_feature_and_prediction": round(mean_cd, 6),
                    "mean_ms_E_ranking": round(mean_e, 6),
                    "mean_ms_A_F_other": round(mean_a_f_other, 6),
                    "n_requests_used": len(requests),
                    "warmup_discarded": warmup,
                    "smoke_scale": args.smoke_scale,
                    "timestamp_utc": timestamp_utc(),
                    "command": command_str(),
                }
                rows.append(row)
                print(f"[profile] trace={trace_name} cap={cap} rep={rep} mean_ms={stats['mean_ms']:.4f} n_evict={n_evict}")
                append_rows(out_csv, [row], fields)
                rows = []
    print(f"Wrote {out_csv}")


# ---------------------------------------------------------------------------
# --exact-optimizations
# ---------------------------------------------------------------------------


def mode_exact_optimizations(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_json = OUT_DIR / "exact_optimization_equivalence.json"
    model_path = args.evict_value_model
    traces = load_traces(protocol, args.data_read_root)

    results: Dict[str, object] = {
        "protocol_id": protocol["protocol_id"],
        "model_path": model_path,
        "smoke_scale": args.smoke_scale,
        "timestamp_utc": timestamp_utc(),
        "command": command_str(),
        "per_trace_capacity": [],
    }
    all_exact = True
    for trace_name, th in traces.items():
        requests = th.requests[: args.max_requests]
        for cap in args.capacities:
            def decisions(cls):
                pol = cls(model_path=model_path)
                pol.reset(cap, th.pages)
                seq = []
                timings = []
                for req in requests:
                    t0 = time.perf_counter()
                    ev = pol.on_request(req)
                    t1 = time.perf_counter()
                    seq.append((ev.hit, ev.evicted))
                    if ev.evicted is not None:
                        timings.append((t1 - t0) * 1000.0)
                return seq, timings

            base_seq, base_t = decisions(EXACT_VARIANT_CLASSES["canonical"])
            entry = {
                "trace_name": trace_name,
                "capacity": cap,
                "n_requests": len(requests),
                "canonical_mean_ms_per_eviction": round(statistics.fmean(base_t), 6) if base_t else 0.0,
                "variants": {},
            }
            for name, cls in EXACT_VARIANT_CLASSES.items():
                if name == "canonical":
                    continue
                seq, t = decisions(cls)
                exact = seq == base_seq
                all_exact = all_exact and exact
                mean_ms = statistics.fmean(t) if t else 0.0
                speedup = (entry["canonical_mean_ms_per_eviction"] / mean_ms) if mean_ms > 0 else None
                entry["variants"][name] = {
                    "exact_match": exact,
                    "mean_ms_per_eviction": round(mean_ms, 6),
                    "speedup_vs_canonical": round(speedup, 4) if speedup is not None else None,
                }
                print(f"[exact-opt] trace={trace_name} cap={cap} variant={name} exact={exact} speedup={speedup}")
            results["per_trace_capacity"].append(entry)

    results["all_variants_exact_across_all_trace_capacity_pairs"] = all_exact
    results["speedup_numbers_are_final_reviewer_evidence"] = not args.smoke_scale
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} (all_exact={all_exact})")


# ---------------------------------------------------------------------------
# --selective
# ---------------------------------------------------------------------------


def mode_selective(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "selective_invocation.csv"
    fields = [
        "trace_name", "capacity", "rule", "n_requests", "n_misses", "misses_canonical",
        "n_eviction_decisions", "n_learned_scorer_invocations", "invocation_rate",
        "quality_retained_top1_match_rate", "timestamp_utc", "command",
    ]
    key_fields = ["trace_name", "capacity", "rule"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    model_path = args.evict_value_model
    traces = load_traces(protocol, args.data_read_root)
    rules = protocol["selective_invocation_variants"]
    period_k = int(rules["periodic"]["predeclared_parameters"]["K"])

    rows: List[Dict[str, object]] = []
    for trace_name, th in traces.items():
        requests = th.requests[: args.max_requests]
        for cap in args.capacities:
            canonical = EvictValueV1Policy(model_path=model_path)
            canonical.reset(cap, th.pages)
            canonical_events = [canonical.on_request(r) for r in requests]
            misses_canonical = sum(1 for e in canonical_events if not e.hit)
            canonical_victims = {i: e.evicted for i, e in enumerate(canonical_events) if e.evicted is not None}

            for rule, cls, kwargs in [
                ("disagreement_lru_sieve", EvictValueV1SelectiveDisagreementPolicy, {}),
                ("periodic", EvictValueV1SelectivePeriodicPolicy, {"period_k": period_k}),
            ]:
                key = (trace_name, str(cap), rule)
                if key in existing:
                    continue
                pol = cls(model_path=model_path, **kwargs)
                pol.reset(cap, th.pages)
                match = 0
                evict_idx = 0
                n_misses = 0
                for i, req in enumerate(requests):
                    ev = pol.on_request(req)
                    if not ev.hit:
                        n_misses += 1
                    if ev.evicted is not None:
                        if evict_idx in canonical_victims and canonical_victims[evict_idx] == ev.evicted:
                            match += 1
                        evict_idx += 1
                row = {
                    "trace_name": trace_name,
                    "capacity": cap,
                    "rule": rule,
                    "n_requests": len(requests),
                    "n_misses": n_misses,
                    "misses_canonical": misses_canonical,
                    "n_eviction_decisions": pol.n_eviction_decisions,
                    "n_learned_scorer_invocations": pol.n_learned_scorer_invocations,
                    "invocation_rate": round(pol.invocation_rate(), 6),
                    "quality_retained_top1_match_rate": round(match / evict_idx, 6) if evict_idx else 0.0,
                    "timestamp_utc": timestamp_utc(),
                    "command": command_str(),
                }
                rows.append(row)
                print(f"[selective] trace={trace_name} cap={cap} rule={rule} invocation_rate={row['invocation_rate']:.3f} misses={n_misses} (canonical={misses_canonical})")
    append_rows(out_csv, rows, fields)
    print(f"Wrote {out_csv} ({len(rows)} new rows)")


# ---------------------------------------------------------------------------
# --top-k
# ---------------------------------------------------------------------------


def mode_topk(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "topk_tradeoff.csv"
    fields = [
        "trace_name", "capacity", "k", "n_requests", "n_misses", "misses_canonical",
        "n_eviction_decisions", "victim_retention_rate", "quality_delta_misses_vs_canonical",
        "mean_ms_per_eviction_decision", "speedup_vs_canonical", "timestamp_utc", "command",
    ]
    key_fields = ["trace_name", "capacity", "k"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    model_path = args.evict_value_model
    traces = load_traces(protocol, args.data_read_root)
    k_grid = protocol["topk_pruning"]["k_grid"]

    rows: List[Dict[str, object]] = []
    for trace_name, th in traces.items():
        requests = th.requests[: args.max_requests]
        for cap in args.capacities:
            canonical = EvictValueV1VectorizedCachedExactPolicy(model_path=model_path)
            canonical.reset(cap, th.pages)
            canonical_events = []
            canonical_times = []
            for req in requests:
                t0 = time.perf_counter()
                ev = canonical.on_request(req)
                t1 = time.perf_counter()
                canonical_events.append(ev)
                if ev.evicted is not None:
                    canonical_times.append((t1 - t0) * 1000.0)
            misses_canonical = sum(1 for e in canonical_events if not e.hit)
            canonical_mean_ms = statistics.fmean(canonical_times) if canonical_times else 0.0
            canonical_victims = {i: e.evicted for i, e in enumerate(canonical_events) if e.evicted is not None}
            canonical_candidates_at_decision: Dict[int, List[PageId]] = {}
            # Replay canonical again to snapshot candidate order pre-eviction (cheap; smoke scale).
            shadow = EvictValueV1Policy(model_path=model_path)
            shadow.reset(cap, th.pages)
            evict_idx = 0
            for req in requests:
                pid = req.page_id
                if shadow.in_cache(pid):
                    shadow.on_request(req)
                    continue
                if shadow._cache.is_full():
                    canonical_candidates_at_decision[evict_idx] = list(shadow._order.keys())
                    evict_idx += 1
                shadow.on_request(req)

            for k in k_grid:
                if k >= cap:
                    continue
                key = (trace_name, str(cap), str(k))
                if key in existing:
                    continue
                pol = EvictValueV1TopKPolicy(model_path=model_path, k=k)
                pol.reset(cap, th.pages)
                n_misses = 0
                evict_idx2 = 0
                times = []
                retained = 0
                for req in requests:
                    t0 = time.perf_counter()
                    ev = pol.on_request(req)
                    t1 = time.perf_counter()
                    if not ev.hit:
                        n_misses += 1
                    if ev.evicted is not None:
                        times.append((t1 - t0) * 1000.0)
                        cand_list = canonical_candidates_at_decision.get(evict_idx2)
                        if cand_list is not None and evict_idx2 in canonical_victims:
                            if not canonical_victim_would_be_pruned(canonical_victims[evict_idx2], cand_list, k):
                                retained += 1
                        evict_idx2 += 1
                mean_ms = statistics.fmean(times) if times else 0.0
                speedup = (canonical_mean_ms / mean_ms) if mean_ms > 0 else None
                row = {
                    "trace_name": trace_name,
                    "capacity": cap,
                    "k": k,
                    "n_requests": len(requests),
                    "n_misses": n_misses,
                    "misses_canonical": misses_canonical,
                    "n_eviction_decisions": evict_idx2,
                    "victim_retention_rate": round(retained / evict_idx2, 6) if evict_idx2 else 0.0,
                    "quality_delta_misses_vs_canonical": n_misses - misses_canonical,
                    "mean_ms_per_eviction_decision": round(mean_ms, 6),
                    "speedup_vs_canonical": round(speedup, 4) if speedup is not None else None,
                    "timestamp_utc": timestamp_utc(),
                    "command": command_str(),
                }
                rows.append(row)
                print(f"[topk] trace={trace_name} cap={cap} k={k} retention={row['victim_retention_rate']:.3f} misses={n_misses} (canonical={misses_canonical})")
    append_rows(out_csv, rows, fields)
    print(f"Wrote {out_csv} ({len(rows)} new rows)")


# ---------------------------------------------------------------------------
# --model-complexity
# ---------------------------------------------------------------------------


def mode_model_complexity(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "model_complexity_tradeoff.csv"
    fields = [
        "trace_name", "capacity", "model_family", "n_requests", "n_misses",
        "n_eviction_decisions", "mean_ms_per_eviction_decision", "median_ms_per_eviction_decision",
        "p95_ms_per_eviction_decision", "timestamp_utc", "command",
    ]
    key_fields = ["trace_name", "capacity", "model_family"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    families = protocol["model_complexity_variants"]["families"]
    traces = load_traces(protocol, args.data_read_root)
    rows: List[Dict[str, object]] = []
    for family in families:
        model_path = f"{args.data_read_root}/models/evict_value_wulver_v1_h4_{family}.pkl"
        if not Path(model_path).exists():
            print(f"[model-complexity] SKIP family={family}: artifact not found at {model_path}")
            continue
        for trace_name, th in traces.items():
            requests = th.requests[: args.max_requests]
            for cap in args.capacities:
                key = (trace_name, str(cap), family)
                if key in existing:
                    continue
                pol = EvictValueV1VectorizedCachedExactPolicy(model_path=model_path)
                pol.reset(cap, th.pages)
                n_misses = 0
                times = []
                for req in requests:
                    t0 = time.perf_counter()
                    ev = pol.on_request(req)
                    t1 = time.perf_counter()
                    if not ev.hit:
                        n_misses += 1
                    if ev.evicted is not None:
                        times.append((t1 - t0) * 1000.0)
                s = _stats_ms(times)
                row = {
                    "trace_name": trace_name,
                    "capacity": cap,
                    "model_family": family,
                    "n_requests": len(requests),
                    "n_misses": n_misses,
                    "n_eviction_decisions": s["n"],
                    "mean_ms_per_eviction_decision": round(s["mean_ms"], 6),
                    "median_ms_per_eviction_decision": round(s["median_ms"], 6),
                    "p95_ms_per_eviction_decision": round(s["p95_ms"], 6),
                    "timestamp_utc": timestamp_utc(),
                    "command": command_str(),
                }
                rows.append(row)
                print(f"[model-complexity] trace={trace_name} cap={cap} family={family} mean_ms={s['mean_ms']:.4f} misses={n_misses}")
    append_rows(out_csv, rows, fields)
    print(f"Wrote {out_csv} ({len(rows)} new rows)")


# ---------------------------------------------------------------------------
# Certified fair-window miss counts (Concern 1 artifacts, read-only)
# ---------------------------------------------------------------------------

_FAIR_DIR = Path("analysis/reviewer_fairness")
_ORIGINAL_IMPLEMENTABLE_BASELINES = [
    "lru", "sieve", "fifo_reinsertion", "blind_oracle_lru_combiner",
    "rest_v1", "trust_and_doubt", "predictive_marker",
]


def load_certified_fair_rows(policy: str) -> List[Dict[str, str]]:
    path = _FAIR_DIR / f"policy_comparison_{policy}.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get("score_start") == "10000" and r.get("score_end") == "50000" and r.get("scored_requests") == "40000"]


def certified_mean_misses(policy: str) -> Optional[float]:
    rows = load_certified_fair_rows(policy)
    misses = [float(r["misses"]) for r in rows if r.get("status") == "ok" and r.get("misses")]
    return statistics.fmean(misses) if misses else None


def pick_fourth_baseline() -> Optional[str]:
    best_name, best_val = None, None
    for name in _ORIGINAL_IMPLEMENTABLE_BASELINES:
        if name in FIXED_BREAK_EVEN_BASELINES:
            continue
        m = certified_mean_misses(name)
        if m is None:
            continue
        if best_val is None or m < best_val:
            best_name, best_val = name, m
    return best_name


# ---------------------------------------------------------------------------
# --break-even
# ---------------------------------------------------------------------------


def _decision_compute_cost_ms(policy: str, capacity: int, profiler_rows: List[Dict[str, str]]) -> Optional[float]:
    """Mean ms/eviction-decision for `policy` at `capacity`, averaged across
    whatever traces are present in the profiler output. `policy` in
    {"evict_value_v1", "lru", "sieve", "fifo_reinsertion"}. For the
    non-learned baselines, cost is taken from the same canonical overhead
    benchmark artifact this protocol extends (kbs_overhead_benchmark), since
    those policies are not part of this session's --profile scope."""
    if policy == "evict_value_v1":
        vals = [float(r["mean_ms_per_eviction_decision_whole"]) for r in profiler_rows if r["capacity"] == str(capacity)]
        return statistics.fmean(vals) if vals else None
    return None


def break_even_cmiss(
    compute_learned_total_ms: float, compute_baseline_total_ms: float, misses_baseline: float, misses_learned: float
) -> Optional[float]:
    """Pure implementation of the frozen break-even formula (protocol
    Section 9): Cmiss* = (ComputeCost_learned - ComputeCost_b) / (Misses_b - Misses_learned),
    defined only when Misses_b > Misses_learned. Returns None (no crossover)
    otherwise -- callers must report that explicitly, never silently drop it.
    """
    if misses_baseline <= misses_learned:
        return None
    return (compute_learned_total_ms - compute_baseline_total_ms) / (misses_baseline - misses_learned)


def mode_break_even(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "break_even_miss_cost.csv"
    fields = [
        "capacity", "baseline", "misses_learned", "misses_baseline",
        "compute_cost_learned_ms_total", "compute_cost_baseline_ms_total",
        "break_even_cmiss_ms", "crossover_exists", "note", "timestamp_utc", "command",
    ]
    key_fields = ["capacity", "baseline"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    profiler_csv = OUT_DIR / "profiler_breakdown.csv"
    profiler_rows = []
    if profiler_csv.exists():
        with profiler_csv.open("r", newline="", encoding="utf-8") as fh:
            profiler_rows = list(csv.DictReader(fh))

    baseline_overhead_ms = {"lru": 0.001, "sieve": 0.003, "fifo_reinsertion": 0.001}
    baseline_overhead_source = (
        "scripts/run_overhead_benchmark.py canonical single-trace probe "
        "(analysis/kbs_overhead_benchmark_local_tmux_20260621.md); not re-measured this session"
    )

    fourth = pick_fourth_baseline()
    baselines = list(FIXED_BREAK_EVEN_BASELINES)
    if fourth:
        baselines.append(fourth)

    rows: List[Dict[str, object]] = []
    for cap in args.capacities:
        misses_learned = certified_mean_misses("evict_value_v1")
        compute_learned_per_decision = _decision_compute_cost_ms("evict_value_v1", cap, profiler_rows)
        for baseline in baselines:
            key = (str(cap), baseline)
            if key in existing:
                continue
            misses_baseline = certified_mean_misses(baseline)
            compute_baseline_per_decision = baseline_overhead_ms.get(baseline)
            if misses_learned is None or misses_baseline is None or compute_learned_per_decision is None or compute_baseline_per_decision is None:
                row = {
                    "capacity": cap, "baseline": baseline,
                    "misses_learned": misses_learned, "misses_baseline": misses_baseline,
                    "compute_cost_learned_ms_total": None, "compute_cost_baseline_ms_total": None,
                    "break_even_cmiss_ms": None, "crossover_exists": None,
                    "note": "INCOMPLETE_INPUTS: run --profile first (evict_value_v1 compute cost) or missing certified fair-window data",
                    "timestamp_utc": timestamp_utc(), "command": command_str(),
                }
                rows.append(row)
                continue
            # DecisionComputeCost applied on evictions only, approximated as total misses
            # (post-warmup miss ~= eviction for a full cache at these capacities/scales).
            compute_learned_total = compute_learned_per_decision * misses_learned
            compute_baseline_total = compute_baseline_per_decision * misses_baseline
            cmiss_star = break_even_cmiss(compute_learned_total, compute_baseline_total, misses_baseline, misses_learned)
            if cmiss_star is not None:
                crossover = True
                note = f"break-even miss cost vs {baseline} (baseline_overhead_source: {baseline_overhead_source})"
            else:
                crossover = False
                note = (
                    f"no positive miss penalty makes evict_value_v1 preferable to {baseline} "
                    f"under this additive model (misses_learned={misses_learned:.1f} >= misses_baseline={misses_baseline:.1f})"
                )
            row = {
                "capacity": cap, "baseline": baseline,
                "misses_learned": round(misses_learned, 3), "misses_baseline": round(misses_baseline, 3),
                "compute_cost_learned_ms_total": round(compute_learned_total, 3),
                "compute_cost_baseline_ms_total": round(compute_baseline_total, 3),
                "break_even_cmiss_ms": round(cmiss_star, 6) if cmiss_star is not None else None,
                "crossover_exists": crossover,
                "note": note,
                "timestamp_utc": timestamp_utc(), "command": command_str(),
            }
            rows.append(row)
            print(f"[break-even] cap={cap} baseline={baseline} crossover={crossover} cmiss*={cmiss_star}")
    append_rows(out_csv, rows, fields)
    print(f"Wrote {out_csv} ({len(rows)} new rows); fourth baseline chosen = {fourth}")


# ---------------------------------------------------------------------------
# --miss-cost-sweep
# ---------------------------------------------------------------------------

_SWEEP_UNIT_TO_MS = {"us": 0.001, "ms": 1.0, "s": 1000.0}


def _parse_sweep_value_ms(token: str) -> float:
    for unit, mult in _SWEEP_UNIT_TO_MS.items():
        if token.endswith(unit):
            return float(token[: -len(unit)]) * mult
    raise ValueError(f"unrecognized miss-cost sweep token: {token}")


def mode_miss_cost_sweep(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "miss_cost_sweep.csv"
    fields = ["capacity", "policy", "cmiss_token", "cmiss_ms", "misses", "total_cost_ms", "timestamp_utc", "command"]
    key_fields = ["capacity", "policy", "cmiss_token"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    grid = protocol["miss_cost_sweep"]["grid"]
    profiler_csv = OUT_DIR / "profiler_breakdown.csv"
    profiler_rows = []
    if profiler_csv.exists():
        with profiler_csv.open("r", newline="", encoding="utf-8") as fh:
            profiler_rows = list(csv.DictReader(fh))
    baseline_overhead_ms = {"lru": 0.001, "sieve": 0.003, "fifo_reinsertion": 0.001}

    rows: List[Dict[str, object]] = []
    for cap in args.capacities:
        policies = {"evict_value_v1": _decision_compute_cost_ms("evict_value_v1", cap, profiler_rows)}
        policies.update(baseline_overhead_ms)
        for policy, per_decision_ms in policies.items():
            misses = certified_mean_misses(policy)
            if misses is None or per_decision_ms is None:
                continue
            for token in grid:
                key = (str(cap), policy, token)
                if key in existing:
                    continue
                cmiss_ms = _parse_sweep_value_ms(token)
                total_cost = per_decision_ms * misses + cmiss_ms * misses
                row = {
                    "capacity": cap, "policy": policy, "cmiss_token": token, "cmiss_ms": cmiss_ms,
                    "misses": round(misses, 3), "total_cost_ms": round(total_cost, 3),
                    "timestamp_utc": timestamp_utc(), "command": command_str(),
                }
                rows.append(row)
    append_rows(out_csv, rows, fields)
    print(f"Wrote {out_csv} ({len(rows)} new rows) -- synthetic grid, not a real-system cost claim")


# ---------------------------------------------------------------------------
# --weighted-cost (synthetic sensitivity only; see protocol Section 8)
# ---------------------------------------------------------------------------


def lognormal_multiplier(unique_page_ids: List[PageId], seed: int) -> Dict[PageId, float]:
    """Deterministic per-page synthetic cost multiplier (protocol Section 8):
    exp(Normal(0,1)), one draw per unique page_id, consumed in sorted page_id
    order so the result is independent of dict/set iteration order and fully
    reproducible given the same (unique_page_ids, seed)."""
    rng = np.random.default_rng(seed)
    return {pid: float(np.exp(rng.normal(0.0, 1.0))) for pid in sorted(unique_page_ids)}


def weighted_miss_cost(missed_page_ids: List[PageId], multiplier: Dict[PageId, float], base_cmiss_ms: float) -> float:
    return sum(base_cmiss_ms * multiplier[pid] for pid in missed_page_ids)


def mode_weighted_cost(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "weighted_cost.csv"
    fields = [
        "trace_name", "capacity", "policy", "n_requests", "n_misses",
        "base_cmiss_ms", "unweighted_total_cost_ms", "weighted_total_cost_ms",
        "weighting_scheme", "seed", "timestamp_utc", "command",
    ]
    key_fields = ["trace_name", "capacity", "policy"]
    existing = read_existing_keys(out_csv, key_fields) if args.resume else set()

    scheme = protocol["weighted_cost_analysis"]["synthetic_scheme"]
    seed = int(scheme["seed"])
    base_cmiss_ms = 1.0  # arbitrary synthetic unit; sensitivity is about relative ranking, not absolute value

    model_path = args.evict_value_model
    traces = load_traces(protocol, args.data_read_root)
    rows: List[Dict[str, object]] = []
    for trace_name, th in traces.items():
        requests = th.requests[: args.max_requests]
        unique_pages = sorted({r.page_id for r in requests})
        multiplier = lognormal_multiplier(unique_pages, seed)
        for cap in args.capacities:
            policy_factories = {
                "lru": lambda: LRUPolicy(),
                "sieve": lambda: SievePolicy(),
                "fifo_reinsertion": lambda: FIFOReinsertionPolicy(),
                "evict_value_v1": lambda: EvictValueV1VectorizedCachedExactPolicy(model_path=model_path),
            }
            for name, factory in policy_factories.items():
                key = (trace_name, str(cap), name)
                if key in existing:
                    continue
                pol = factory()
                pol.reset(cap, th.pages)
                n_misses = 0
                weighted_cost = 0.0
                for req in requests:
                    ev = pol.on_request(req)
                    if not ev.hit:
                        n_misses += 1
                        weighted_cost += base_cmiss_ms * multiplier[req.page_id]
                unweighted_cost = base_cmiss_ms * n_misses
                row = {
                    "trace_name": trace_name, "capacity": cap, "policy": name,
                    "n_requests": len(requests), "n_misses": n_misses,
                    "base_cmiss_ms": base_cmiss_ms,
                    "unweighted_total_cost_ms": round(unweighted_cost, 3),
                    "weighted_total_cost_ms": round(weighted_cost, 3),
                    "weighting_scheme": scheme["name"], "seed": seed,
                    "timestamp_utc": timestamp_utc(), "command": command_str(),
                }
                rows.append(row)
                print(f"[weighted-cost] trace={trace_name} cap={cap} policy={name} misses={n_misses} weighted={row['weighted_total_cost_ms']:.1f}")
    append_rows(out_csv, rows, fields)
    print(f"Wrote {out_csv} ({len(rows)} new rows) -- SYNTHETIC sensitivity analysis, not a real-cost measurement")


# ---------------------------------------------------------------------------
# --pareto
# ---------------------------------------------------------------------------


def _is_pareto_efficient(points: List[Tuple[float, float]]) -> List[bool]:
    """points: (latency, misses), both minimized. Returns per-point efficiency flag."""
    eff = [True] * len(points)
    for i, (li, mi) in enumerate(points):
        for j, (lj, mj) in enumerate(points):
            if i == j:
                continue
            if lj <= li and mj <= mi and (lj < li or mj < mi):
                eff[i] = False
                break
    return eff


def mode_pareto(args: argparse.Namespace, protocol: Dict[str, object]) -> None:
    out_csv = OUT_DIR / "pareto_frontier.csv"
    fields = ["variant", "capacity", "mean_ms_per_eviction_decision", "misses", "pareto_efficient", "timestamp_utc", "command"]

    points_by_cap: Dict[int, List[Tuple[str, float, float]]] = {}

    profiler_csv = OUT_DIR / "profiler_breakdown.csv"
    if profiler_csv.exists():
        with profiler_csv.open("r", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cap = int(r["capacity"])
                lat = float(r["mean_ms_per_eviction_decision_whole"])
                misses = certified_mean_misses("evict_value_v1")
                if misses is not None:
                    points_by_cap.setdefault(cap, []).append(("evict_value_v1_canonical", lat, misses))

    baseline_overhead_ms = {"lru": 0.001, "sieve": 0.003, "fifo_reinsertion": 0.001}
    for policy, lat in baseline_overhead_ms.items():
        misses = certified_mean_misses(policy)
        if misses is None:
            continue
        for cap in args.capacities:
            points_by_cap.setdefault(cap, []).append((policy, lat, misses))

    for source_csv, variant_col, lat_col in [
        (OUT_DIR / "selective_invocation.csv", "rule", None),
        (OUT_DIR / "topk_tradeoff.csv", "k", "mean_ms_per_eviction_decision"),
        (OUT_DIR / "model_complexity_tradeoff.csv", "model_family", "mean_ms_per_eviction_decision"),
    ]:
        if not source_csv.exists():
            continue
        with source_csv.open("r", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cap = int(r["capacity"])
                if lat_col and r.get(lat_col):
                    lat = float(r[lat_col])
                    misses = float(r["n_misses"])
                    label = f"{source_csv.stem}_{variant_col}={r[variant_col]}"
                    points_by_cap.setdefault(cap, []).append((label, lat, misses))

    rows: List[Dict[str, object]] = []
    for cap, points in points_by_cap.items():
        eff = _is_pareto_efficient([(p[1], p[2]) for p in points])
        for (label, lat, misses), is_eff in zip(points, eff):
            rows.append({
                "variant": label, "capacity": cap,
                "mean_ms_per_eviction_decision": round(lat, 6), "misses": round(misses, 3),
                "pareto_efficient": is_eff,
                "timestamp_utc": timestamp_utc(), "command": command_str(),
            })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows, {sum(r['pareto_efficient'] for r in rows)} pareto-efficient)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Practical-significance ablation runner (Reviewer 1, Concern 4).")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--exact-optimizations", action="store_true")
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--top-k", action="store_true")
    ap.add_argument("--model-complexity", action="store_true")
    ap.add_argument("--break-even", action="store_true")
    ap.add_argument("--miss-cost-sweep", action="store_true")
    ap.add_argument("--weighted-cost", action="store_true")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-scale", dest="smoke_scale", action="store_true", default=True)
    ap.add_argument("--controlled-final", dest="smoke_scale", action="store_false")
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--max-requests", type=int, default=1500, help="requests per trace; smoke-scale default, use full 50000 for the final controlled campaign")
    ap.add_argument("--data-read-root", default="/home/soroush/Augmented-caching")
    ap.add_argument("--evict-value-model", default="/home/soroush/Augmented-caching/models/evict_value_wulver_v1_best_heavy_r1.pkl")
    args = ap.parse_args()
    args.capacities = [int(x) for x in args.capacities.split(",") if x.strip()]

    protocol = load_protocol()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ran_any = False
    if args.exact_optimizations or args.all:
        mode_exact_optimizations(args, protocol)
        ran_any = True
    if args.profile or args.all:
        mode_profile(args, protocol)
        ran_any = True
    if args.selective or args.all:
        mode_selective(args, protocol)
        ran_any = True
    if args.top_k or args.all:
        mode_topk(args, protocol)
        ran_any = True
    if args.model_complexity or args.all:
        mode_model_complexity(args, protocol)
        ran_any = True
    if args.miss_cost_sweep or args.all:
        mode_miss_cost_sweep(args, protocol)
        ran_any = True
    if args.break_even or args.all:
        mode_break_even(args, protocol)
        ran_any = True
    if args.weighted_cost or args.all:
        mode_weighted_cost(args, protocol)
        ran_any = True
    if args.pareto or args.all:
        mode_pareto(args, protocol)
        ran_any = True

    if not ran_any:
        print("No mode selected; pass at least one of --profile/--exact-optimizations/--selective/--top-k/"
              "--model-complexity/--break-even/--miss-cost-sweep/--weighted-cost/--pareto/--all")

    provenance = {
        "protocol_id": protocol["protocol_id"],
        "smoke_scale": args.smoke_scale,
        "max_requests_per_trace": args.max_requests,
        "capacities": args.capacities,
        "timestamp_utc": timestamp_utc(),
        "command": command_str(),
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
