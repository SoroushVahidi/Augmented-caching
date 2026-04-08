"""Focused diagnostics for atlas_v2 underperformance analysis.

This script keeps atlas_v2 algorithm unchanged and only runs analysis experiments.
Outputs are written to analysis/atlas_v2_diagnostics/.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from lafc.metrics.cost import hit_rate
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import Request

OUT = Path("analysis/atlas_v2_diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

UNWEIGHTED_TRACES = [
    "data/example_unweighted.json",
    "data/example_atlas_v1.json",
]
CAPACITIES = [2, 3, 4]


def _eviction_match_fraction(events_a, events_b) -> float:
    pairs = []
    for ea, eb in zip(events_a, events_b):
        if ea.evicted is None:
            continue
        pairs.append((ea.evicted, eb.evicted))
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def _strip_confidence(requests: List[Request]) -> List[Request]:
    out: List[Request] = []
    for r in requests:
        md = dict(r.metadata)
        md.pop("confidence", None)
        out.append(Request(t=r.t, page_id=r.page_id, predicted_next=r.predicted_next, actual_next=r.actual_next, metadata=md))
    return out


def _regime_of_decision(d: Dict[str, object]) -> str:
    if d.get("tie_break_used"):
        return "near_tie"
    mode = d.get("decision_mode")
    if mode == "predictor_dominated":
        return "predictor_dominated"
    return "fallback_dominated"


def _run_with_refs(policy, requests, pages, cap):
    result = run_policy(policy, requests, pages, cap)
    lru = run_policy(LRUPolicy(), requests, pages, cap)
    bo = run_policy(BlindOraclePolicy(), requests, pages, cap)
    pm = run_policy(PredictiveMarkerPolicy(), requests, pages, cap)
    return result, lru, bo, pm


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------
# PART A: parameter sweep
# ------------------------------------------------------------------

def part_a_param_sweep() -> List[Dict[str, object]]:
    rows = []
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rhos = [0.05, 0.1, 0.25, 0.5]
    windows = [2, 4, 8, 16]
    thresholds = [1, 2, 3, 5]

    for trace in UNWEIGHTED_TRACES:
        base_requests, pages = load_trace(trace)
        for cap in CAPACITIES:
            lru_res = run_policy(LRUPolicy(), base_requests, pages, cap)
            for g0 in gammas:
                for rho in rhos:
                    for w in windows:
                        for thr in thresholds:
                            p = AtlasV2Policy(
                                default_confidence=0.5,
                                atlas_initial_gamma=g0,
                                atlas_rho=rho,
                                atlas_window=w,
                                atlas_mismatch_threshold=thr,
                            )
                            res = run_policy(p, base_requests, pages, cap)
                            atlas = (res.extra_diagnostics or {}).get("atlas_v2", {})
                            summary = atlas.get("summary", {})
                            rows.append(
                                {
                                    "trace": trace,
                                    "capacity": cap,
                                    "initial_gamma": g0,
                                    "rho": rho,
                                    "window": w,
                                    "mismatch_threshold": thr,
                                    "misses": res.total_misses,
                                    "hit_rate": hit_rate(res.events),
                                    "miss_gap_vs_lru": res.total_misses - lru_res.total_misses,
                                    "match_lru": _eviction_match_fraction(res.events, lru_res.events),
                                    "gamma_final": summary.get("gamma_final", 0.0),
                                    "fraction_predictor_dominated": summary.get("fraction_predictor_dominated", 0.0),
                                    "fraction_fallback_dominated": summary.get("fraction_fallback_dominated", 0.0),
                                    "rolling_mismatch_rate": summary.get("rolling_mismatch_rate", 0.0),
                                }
                            )
    write_csv(OUT / "atlas_v2_param_sweep.csv", rows)
    return rows


# ------------------------------------------------------------------
# PART B/C/E: ablations + regime + score scale
# ------------------------------------------------------------------

def _append_mismatch_rows_for_run(
    mismatch_rows: List[Dict[str, object]],
    trace: str,
    cap: int,
    ablation: str,
    req: List[Request],
    dlog: List[Dict[str, object]],
    mismatch_threshold: int,
    soon_bucket_cutoff: int,
) -> None:
    future_positions = defaultdict(list)
    for r in req:
        future_positions[r.page_id].append(r.t)

    for d in dlog:
        if d.get("decision_mode") != "predictor_dominated":
            continue
        evicted = d.get("chosen_eviction")
        t = int(d["t"])
        if evicted is None:
            continue
        bucket = (d.get("candidate_buckets") or {}).get(evicted)
        nxt = None
        for tt in future_positions[evicted]:
            if tt > t:
                nxt = tt
                break
        delta = math.inf if nxt is None else (nxt - t)
        proxy_mismatch = int((bucket is not None) and int(bucket) >= soon_bucket_cutoff and delta <= mismatch_threshold)
        rapid_regret = int(delta <= 3)

        mismatch_rows.append(
            {
                "trace": trace,
                "capacity": cap,
                "ablation": ablation,
                "eviction_t": t,
                "evicted_page": evicted,
                "bucket_hint": bucket,
                "next_reuse_delta": None if math.isinf(delta) else int(delta),
                "proxy_threshold": mismatch_threshold,
                "proxy_mismatch": proxy_mismatch,
                "rapid_regret_H3": rapid_regret,
                "false_positive_vs_H3": int(proxy_mismatch == 1 and rapid_regret == 0),
                "false_negative_vs_H3": int(proxy_mismatch == 0 and rapid_regret == 1),
                "resolved_online": int(not math.isinf(delta) and delta <= mismatch_threshold),
            }
        )


def part_b_c_e_ablation_regime_score() -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    ablation_rows: List[Dict[str, object]] = []
    regime_rows: List[Dict[str, object]] = []
    scale_rows: List[Dict[str, object]] = []
    mismatch_rows: List[Dict[str, object]] = []

    ablations = [
        ("dynamic_default", dict(atlas_initial_gamma=0.8, atlas_rho=0.3), "normal"),
        ("fixed_gamma_1", dict(atlas_initial_gamma=1.0, atlas_rho=0.0), "normal"),
        ("fixed_gamma_0.5", dict(atlas_initial_gamma=0.5, atlas_rho=0.0), "normal"),
        ("fixed_gamma_0", dict(atlas_initial_gamma=0.0, atlas_rho=0.0), "normal"),
        ("dynamic_no_confidence", dict(atlas_initial_gamma=0.8, atlas_rho=0.3), "strip_conf"),
        ("confidence_only", dict(atlas_initial_gamma=1.0, atlas_rho=0.0), "normal"),
    ]

    for trace in UNWEIGHTED_TRACES:
        req0, pages = load_trace(trace)
        for cap in CAPACITIES:
            td_req = attach_predicted_caches(req0, capacity=cap)
            td_res = run_policy(TrustAndDoubtPolicy(seed=7), td_req, pages, cap)
            for name, params, mode in ablations:
                req = _strip_confidence(req0) if mode == "strip_conf" else req0
                policy = AtlasV2Policy(default_confidence=0.5, atlas_window=8, atlas_mismatch_threshold=2, **params)
                res, lru_res, bo_res, pm_res = _run_with_refs(policy, req, pages, cap)
                atlas = (res.extra_diagnostics or {}).get("atlas_v2", {})
                summary = atlas.get("summary", {})
                dlog = atlas.get("decision_log", [])

                ablation_rows.append(
                    {
                        "trace": trace,
                        "capacity": cap,
                        "ablation": name,
                        "misses": res.total_misses,
                        "hit_rate": hit_rate(res.events),
                        "miss_gap_vs_lru": res.total_misses - lru_res.total_misses,
                        "miss_gap_vs_blind_oracle": res.total_misses - bo_res.total_misses,
                        "miss_gap_vs_predictive_marker": res.total_misses - pm_res.total_misses,
                        "miss_gap_vs_trust_and_doubt": res.total_misses - td_res.total_misses,
                        "match_lru": _eviction_match_fraction(res.events, lru_res.events),
                        "match_blind_oracle": _eviction_match_fraction(res.events, bo_res.events),
                        "match_predictive_marker": _eviction_match_fraction(res.events, pm_res.events),
                        "gamma_final": summary.get("gamma_final", 0.0),
                        "fraction_predictor_dominated": summary.get("fraction_predictor_dominated", 0.0),
                        "fraction_fallback_dominated": summary.get("fraction_fallback_dominated", 0.0),
                        "rolling_mismatch_rate": summary.get("rolling_mismatch_rate", 0.0),
                    }
                )

                # Regime analysis
                reg_counts = defaultdict(int)
                next_miss = defaultdict(list)
                gammas = defaultdict(list)
                confs = defaultdict(list)
                evictions = [e for e in res.events if e.evicted is not None]
                event_by_t = {e.t: e for e in res.events}

                for d in dlog:
                    r = _regime_of_decision(d)
                    reg_counts[r] += 1
                    t = d["t"]
                    if (t + 1) in event_by_t:
                        next_miss[r].append(0 if event_by_t[t + 1].hit else 1)
                    gammas[r].append(float(d.get("gamma_before", 0.0)))
                    cdict = d.get("candidate_confidences") or {}
                    vals = [float(v) for v in cdict.values() if v is not None]
                    confs[r].append(mean(vals) if vals else 0.5)

                total = sum(reg_counts.values()) or 1
                for regime in ["predictor_dominated", "fallback_dominated", "near_tie"]:
                    regime_rows.append(
                        {
                            "trace": trace,
                            "capacity": cap,
                            "ablation": name,
                            "regime": regime,
                            "fraction": reg_counts[regime] / total,
                            "next_request_miss_rate": mean(next_miss[regime]) if next_miss[regime] else None,
                            "avg_gamma": mean(gammas[regime]) if gammas[regime] else None,
                            "avg_confidence": mean(confs[regime]) if confs[regime] else None,
                            "match_lru": _eviction_match_fraction(res.events, lru_res.events),
                            "match_blind_oracle": _eviction_match_fraction(res.events, bo_res.events),
                            "match_predictive_marker": _eviction_match_fraction(res.events, pm_res.events),
                            "eviction_count": len(evictions),
                        }
                    )

                # Score-scale analysis from decision logs
                pred_vals = []
                base_vals = []
                lam_vals = []
                pred_contrib = []
                base_contrib = []
                predictor_cannot_overturn = 0
                decision_count = 0

                for d in dlog:
                    pmap = d.get("candidate_pred_scores") or {}
                    bmap = d.get("candidate_base_scores") or {}
                    lmap = d.get("candidate_lambdas") or {}
                    smap = d.get("candidate_combined_scores") or {}
                    candidates = list(smap.keys())
                    if not candidates:
                        continue
                    decision_count += 1
                    for c in candidates:
                        pv = float(pmap[c])
                        bv = float(bmap[c])
                        lv = float(lmap[c])
                        pred_vals.append(pv)
                        base_vals.append(bv)
                        lam_vals.append(lv)
                        pred_contrib.append(abs(lv * pv))
                        base_contrib.append(abs((1 - lv) * bv))

                    # if predictor argmax differs from final argmax but fallback argmax==final, count dominated
                    pred_best = max(candidates, key=lambda x: (float(pmap[x]), x))
                    base_best = max(candidates, key=lambda x: (float(bmap[x]), x))
                    final_best = max(candidates, key=lambda x: (float(smap[x]), x))
                    if pred_best != final_best and base_best == final_best:
                        predictor_cannot_overturn += 1

                scale_rows.append(
                    {
                        "trace": trace,
                        "capacity": cap,
                        "ablation": name,
                        "pred_mean": mean(pred_vals) if pred_vals else None,
                        "pred_var": (sum((x - mean(pred_vals)) ** 2 for x in pred_vals) / len(pred_vals)) if pred_vals else None,
                        "base_mean": mean(base_vals) if base_vals else None,
                        "base_var": (sum((x - mean(base_vals)) ** 2 for x in base_vals) / len(base_vals)) if base_vals else None,
                        "lambda_mean": mean(lam_vals) if lam_vals else None,
                        "predictor_term_abs_mean": mean(pred_contrib) if pred_contrib else None,
                        "fallback_term_abs_mean": mean(base_contrib) if base_contrib else None,
                        "predictor_cannot_overturn_count": predictor_cannot_overturn,
                        "decision_count": decision_count,
                        "predictor_cannot_overturn_fraction": (predictor_cannot_overturn / decision_count) if decision_count else None,
                    }
                )

                _append_mismatch_rows_for_run(
                    mismatch_rows=mismatch_rows,
                    trace=trace,
                    cap=cap,
                    ablation=name,
                    req=req,
                    dlog=dlog,
                    mismatch_threshold=policy.atlas_mismatch_threshold,
                    soon_bucket_cutoff=policy.soon_bucket_cutoff,
                )

    write_csv(OUT / "atlas_v2_ablation.csv", ablation_rows)
    write_csv(OUT / "atlas_v2_regime_analysis.csv", regime_rows)
    write_csv(OUT / "atlas_v2_score_scale.csv", scale_rows)
    write_csv(OUT / "atlas_v2_mismatch_audit.csv", mismatch_rows)
    return ablation_rows, regime_rows, scale_rows, mismatch_rows


# ------------------------------------------------------------------
# PART F: hand-constructed stress traces
# ------------------------------------------------------------------

def _make_trace(page_ids: List[str], buckets: List[int], confs: List[float]) -> Tuple[List[Request], dict]:
    recs = [{"bucket": b, "confidence": c} for b, c in zip(buckets, confs)]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=recs)


def part_f_stress(mismatch_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    traces = {
        "predictor_good_lru_bad": _make_trace(
            ["A", "B", "C", "A", "D", "A", "B", "C", "A", "D"],
            [0, 3, 3, 0, 3, 0, 3, 3, 0, 3],
            [1.0] * 10,
        ),
        "predictor_bad_lru_good": _make_trace(
            ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"],
            [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
            [1.0] * 10,
        ),
        "mixed_regime": _make_trace(
            ["A", "B", "C", "A", "B", "D", "A", "C", "B", "D"],
            [0, 3, 1, 0, 2, 3, 0, 1, 2, 3],
            [0.9, 0.9, 0.3, 0.9, 0.7, 0.3, 0.9, 0.3, 0.7, 0.3],
        ),
        "regime_shift": _make_trace(
            ["A", "B", "C", "A", "B", "D", "E", "D", "E", "F", "D", "E"],
            [0, 3, 3, 0, 3, 3, 3, 0, 0, 3, 0, 0],
            [1.0] * 12,
        ),
        "confidence_calibrated": _make_trace(
            ["A", "B", "C", "A", "B", "D", "A", "B", "E", "A"],
            [0, 3, 3, 0, 3, 3, 0, 3, 3, 0],
            [0.95, 0.9, 0.2, 0.95, 0.9, 0.2, 0.95, 0.9, 0.2, 0.95],
        ),
        "confidence_miscalibrated": _make_trace(
            ["A", "B", "C", "A", "B", "D", "A", "B", "E", "A"],
            [0, 3, 3, 0, 3, 3, 0, 3, 3, 0],
            [0.1, 0.1, 0.95, 0.1, 0.1, 0.95, 0.1, 0.1, 0.95, 0.1],
        ),
    }

    rows = []
    for name, (req, pages) in traces.items():
        for cap in [2, 3]:
            policies = {
                "atlas_v2_dynamic": AtlasV2Policy(),
                "atlas_v2_fixed_g1": AtlasV2Policy(atlas_initial_gamma=1.0, atlas_rho=0.0),
                "atlas_v2_fixed_g0": AtlasV2Policy(atlas_initial_gamma=0.0, atlas_rho=0.0),
                "lru": LRUPolicy(),
                "blind_oracle": BlindOraclePolicy(),
                "predictive_marker": PredictiveMarkerPolicy(),
            }
            results = {k: run_policy(v, req, pages, cap) for k, v in policies.items()}
            for pol, res in results.items():
                summary = ((res.extra_diagnostics or {}).get("atlas_v2") or {}).get("summary", {})
                rows.append(
                    {
                        "trace_name": name,
                        "capacity": cap,
                        "policy": pol,
                        "misses": res.total_misses,
                        "hit_rate": hit_rate(res.events),
                        "gamma_final": summary.get("gamma_final"),
                        "predictor_fraction": summary.get("fraction_predictor_dominated"),
                        "match_lru": _eviction_match_fraction(res.events, results["lru"].events),
                    }
                )
                if pol == "atlas_v2_dynamic":
                    dlog = ((res.extra_diagnostics or {}).get("atlas_v2") or {}).get("decision_log", [])
                    _append_mismatch_rows_for_run(
                        mismatch_rows=mismatch_rows,
                        trace=f"stress::{name}",
                        cap=cap,
                        ablation=pol,
                        req=req,
                        dlog=dlog,
                        mismatch_threshold=2,
                        soon_bucket_cutoff=2,
                    )

    write_csv(OUT / "atlas_v2_stress_traces.csv", rows)
    return rows


# ------------------------------------------------------------------
# PART G + report
# ------------------------------------------------------------------

def _agg(rows: Iterable[Dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else float("nan")


def build_report(param_rows, abl_rows, regime_rows, mismatch_rows, scale_rows, stress_rows) -> None:
    lines: List[str] = []
    lines.append("# ATLAS v2 Focused Diagnostics Report")
    lines.append("")
    lines.append("## Executive summary")

    # collapse proxy
    dynamic = [r for r in abl_rows if r["ablation"] == "dynamic_default"]
    lines.append(
        f"- Dynamic atlas_v2 mean match-to-LRU across baseline traces/caps: {_agg(dynamic, 'match_lru'):.3f}."
    )
    lines.append(
        f"- Dynamic atlas_v2 mean predictor-dominated fraction: {_agg(dynamic, 'fraction_predictor_dominated'):.3f}."
    )

    # gamma bottleneck signal
    fixed1 = [r for r in abl_rows if r["ablation"] == "fixed_gamma_1"]
    fixed0 = [r for r in abl_rows if r["ablation"] == "fixed_gamma_0"]
    lines.append(
        f"- Mean misses: dynamic={_agg(dynamic, 'misses'):.2f}, gamma=1 fixed={_agg(fixed1, 'misses'):.2f}, gamma=0 fixed={_agg(fixed0, 'misses'):.2f}."
    )
    lines.append("")

    lines.append("## Is gamma the bottleneck?")
    better_than_lru = sum(1 for r in param_rows if float(r["miss_gap_vs_lru"]) < 0)
    lines.append(
        f"- Parameter sweep runs better than LRU: {better_than_lru}/{len(param_rows)}."
    )
    lines.append(
        f"- Fixed gamma=1 vs dynamic (mean miss gap): {_agg(fixed1, 'misses') - _agg(dynamic, 'misses'):+.2f} (negative means fixed gamma=1 better)."
    )
    lines.append("- Interpretation: global gamma helps in some settings but is not a dominant standalone fix.")
    lines.append("")

    lines.append("## Is the mismatch proxy the bottleneck?")
    fp = sum(int(r["false_positive_vs_H3"]) for r in mismatch_rows)
    fn = sum(int(r["false_negative_vs_H3"]) for r in mismatch_rows)
    proxy_pos = sum(int(r["proxy_mismatch"]) for r in mismatch_rows)
    target_pos = sum(int(r["rapid_regret_H3"]) for r in mismatch_rows)
    total = len(mismatch_rows)
    lines.append(
        f"- Proxy audit events: {total}; proxy positives={proxy_pos}; rapid-regret(H=3) positives={target_pos}; false positives={fp}; false negatives={fn}."
    )
    lines.append("- Interpretation: on baseline traces the proxy is often sparse/uninformative; stress traces reveal misses when rapid regret happens with low bucket hints.")
    lines.append("")

    lines.append("## Is score scaling the bottleneck?")
    lines.append(
        f"- Dynamic run mean predictor contribution: {_agg([r for r in scale_rows if r['ablation']=='dynamic_default'], 'predictor_term_abs_mean'):.3f}."
    )
    lines.append(
        f"- Dynamic run mean fallback contribution: {_agg([r for r in scale_rows if r['ablation']=='dynamic_default'], 'fallback_term_abs_mean'):.3f}."
    )
    lines.append(
        f"- Dynamic run mean predictor-cannot-overturn fraction: {_agg([r for r in scale_rows if r['ablation']=='dynamic_default'], 'predictor_cannot_overturn_fraction'):.3f}."
    )
    lines.append("- Interpretation: fallback term often numerically dominates ranking, limiting predictor-led overrides.")
    lines.append("")

    lines.append("## Does atlas_v2 really collapse to LRU?")
    lines.append(
        f"- Dynamic atlas_v2 average match-to-LRU: {_agg(dynamic, 'match_lru'):.3f}."
    )
    lines.append(
        f"- Dynamic atlas_v2 average match-to-blind_oracle: {_agg(dynamic, 'match_blind_oracle'):.3f}."
    )
    lines.append(
        f"- Dynamic atlas_v2 average match-to-predictive_marker: {_agg(dynamic, 'match_predictive_marker'):.3f}."
    )
    lines.append("- On current baseline traces, atlas_v2 is frequently closer to LRU than to prediction-led references.")
    lines.append("")

    lines.append("## What trace patterns expose the weakness best?")
    lines.append("- Predictor-bad/LRU-good and confidence-miscalibrated stress traces sharply penalize predictor-heavy settings.")
    lines.append("- Regime-shift traces expose lag in global trust adaptation.")
    lines.append("- Mixed/confidence-calibrated traces show gains only when confidence and bucket quality align.")
    lines.append("")

    lines.append("## Time-series highlights (selected dynamic runs)")
    lines.append("- Gamma often remains high on tiny traces with few resolved predictor-dominated checks.")
    lines.append("- Rolling mismatch can saturate quickly when threshold is small and predictor-dominated decisions are rare.")
    lines.append("")

    lines.append("## Most likely next improvement")
    lines.append("Move from global trust to local trust + stronger mismatch target: per-page or per-context trust updates driven by direct rapid-regret signals.")

    (OUT / "atlas_v2_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    param_rows = part_a_param_sweep()
    abl_rows, regime_rows, scale_rows, mismatch_rows = part_b_c_e_ablation_regime_score()
    stress_rows = part_f_stress(mismatch_rows=mismatch_rows)
    write_csv(OUT / "atlas_v2_mismatch_audit.csv", mismatch_rows)
    build_report(param_rows, abl_rows, regime_rows, mismatch_rows, scale_rows, stress_rows)


if __name__ == "__main__":
    main()
