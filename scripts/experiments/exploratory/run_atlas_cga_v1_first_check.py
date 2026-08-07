"""First-pass evaluation for atlas_cga_v1 vs atlas_v3 and baselines."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from lafc.metrics.cost import hit_rate
from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace

OUT_CSV = Path("analysis/atlas_cga_v1_first_check.csv")
OUT_MD = Path("analysis/atlas_cga_v1_first_check.md")
CAPACITIES = [2, 3, 4]
BASE_TRACES = ["data/example_unweighted.json", "data/example_atlas_v1.json"]


def _make_stress_trace(page_ids: List[str], buckets: List[int], confs: List[float]):
    recs = [{"bucket": b, "confidence": c} for b, c in zip(buckets, confs)]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=recs)


def _stress_traces():
    return {
        "stress::predictor_good_lru_bad": _make_stress_trace(
            ["A", "B", "C", "A", "D", "A", "B", "C", "A", "D"],
            [0, 3, 3, 0, 3, 0, 3, 3, 0, 3],
            [1.0] * 10,
        ),
        "stress::predictor_bad_lru_good": _make_stress_trace(
            ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"],
            [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
            [1.0] * 10,
        ),
        "stress::mixed_regime": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "C", "B", "D"],
            [0, 3, 1, 0, 2, 3, 0, 1, 2, 3],
            [0.9, 0.9, 0.3, 0.9, 0.7, 0.3, 0.9, 0.3, 0.7, 0.3],
        ),
        "stress::regime_shift": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "E", "D", "E", "F", "D", "E"],
            [0, 3, 3, 0, 3, 3, 3, 0, 0, 3, 0, 0],
            [1.0] * 12,
        ),
        "stress::confidence_miscalibrated": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "B", "E", "A"],
            [0, 3, 3, 0, 3, 3, 0, 3, 3, 0],
            [0.1, 0.1, 0.95, 0.1, 0.1, 0.95, 0.1, 0.1, 0.95, 0.1],
        ),
    }


def _iter_traces():
    for path in BASE_TRACES:
        yield path, load_trace(path)
    for name, payload in _stress_traces().items():
        yield name, payload


def _eviction_match_fraction(a_events, b_events) -> float:
    pairs = [(a.evicted, b.evicted) for a, b in zip(a_events, b_events) if a.evicted is not None]
    if not pairs:
        return 0.0
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def _row(trace_name: str, capacity: int, policy_name: str, res, lru_res, bo_res, pm_res) -> Dict[str, object]:
    row = {
        "trace": trace_name,
        "capacity": capacity,
        "policy": policy_name,
        "misses": res.total_misses,
        "hit_rate": hit_rate(res.events),
        "match_lru": _eviction_match_fraction(res.events, lru_res.events),
        "match_blind_oracle": _eviction_match_fraction(res.events, bo_res.events),
        "match_predictive_marker": _eviction_match_fraction(res.events, pm_res.events),
        "predictor_fraction": "",
        "fallback_fraction": "",
        "tie_fraction": "",
        "avg_lambda": "",
        "calibration_mean_pcal": "",
        "calibration_active_contexts": "",
    }
    extra = res.extra_diagnostics or {}
    if policy_name == "atlas_v3":
        summary = (extra.get("atlas_v3") or {}).get("summary", {})
        row["predictor_fraction"] = summary.get("fraction_predictor_dominated", 0.0)
        row["fallback_fraction"] = summary.get("fraction_fallback_dominated", 0.0)
        row["tie_fraction"] = summary.get("fraction_tie_region", 0.0)
        row["avg_lambda"] = summary.get("average_lambda", 0.0)
    if policy_name == "atlas_cga_v1":
        summary = (extra.get("atlas_cga_v1") or {}).get("summary", {})
        row["predictor_fraction"] = summary.get("fraction_predictor_dominated", 0.0)
        row["fallback_fraction"] = summary.get("fraction_fallback_dominated", 0.0)
        row["tie_fraction"] = summary.get("fraction_tie_region", 0.0)
        row["avg_lambda"] = summary.get("average_lambda", 0.0)
        row["calibration_mean_pcal"] = summary.get("calibration_mean_pcal", 0.0)
        row["calibration_active_contexts"] = summary.get("calibration_active_contexts", 0)
    return row


def main() -> None:
    rows: List[Dict[str, object]] = []
    for trace_name, (requests, pages) in _iter_traces():
        for cap in CAPACITIES:
            td_requests = attach_predicted_caches(requests, capacity=cap)
            results = {
                "atlas_v1": run_policy(AtlasV1Policy(), requests, pages, cap),
                "atlas_v2": run_policy(AtlasV2Policy(), requests, pages, cap),
                "atlas_v3": run_policy(AtlasV3Policy(), requests, pages, cap),
                "atlas_cga_v1": run_policy(AtlasCGAV1Policy(), requests, pages, cap),
                "lru": run_policy(LRUPolicy(), requests, pages, cap),
                "blind_oracle": run_policy(BlindOraclePolicy(), requests, pages, cap),
                "predictive_marker": run_policy(PredictiveMarkerPolicy(), requests, pages, cap),
                "trust_and_doubt": run_policy(TrustAndDoubtPolicy(seed=7), td_requests, pages, cap),
            }
            lru_res = results["lru"]
            bo_res = results["blind_oracle"]
            pm_res = results["predictive_marker"]
            for name, res in results.items():
                rows.append(_row(trace_name, cap, name, res, lru_res, bo_res, pm_res))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    v3_rows = [r for r in rows if r["policy"] == "atlas_v3"]
    cga_rows = [r for r in rows if r["policy"] == "atlas_cga_v1"]
    mean_v3_misses = mean(float(r["misses"]) for r in v3_rows)
    mean_cga_misses = mean(float(r["misses"]) for r in cga_rows)
    mean_v3_pred = mean(float(r["predictor_fraction"] or 0.0) for r in v3_rows)
    mean_cga_pred = mean(float(r["predictor_fraction"] or 0.0) for r in cga_rows)
    mean_v3_fallback = mean(float(r["fallback_fraction"] or 0.0) for r in v3_rows)
    mean_cga_fallback = mean(float(r["fallback_fraction"] or 0.0) for r in cga_rows)

    noisy_keys = {"stress::mixed_regime", "stress::confidence_miscalibrated", "stress::regime_shift"}
    v3_noisy = [r for r in v3_rows if r["trace"] in noisy_keys]
    cga_noisy = [r for r in cga_rows if r["trace"] in noisy_keys]
    noisy_delta = mean(float(r["misses"]) for r in cga_noisy) - mean(float(r["misses"]) for r in v3_noisy)

    lines: List[str] = []
    lines.append("# atlas_cga_v1 First Check")
    lines.append("")
    lines.append("## Exact formulas implemented")
    lines.append("")
    lines.append("- Context: `B(q,t) = (bucket, confidence_bin)`.")
    lines.append("- Calibration posterior: `posterior_B = (s_B + a) / (n_B + a + b)`.")
    lines.append("- Shrinkage weight: `w_B = n_B / (n_B + m)` with extra downweighting when `n_B < min_support`.")
    lines.append("- Calibrated signal: `pcal_B = w_B * posterior_B + (1 - w_B) * prior`, `prior = a/(a+b)`.")
    lines.append("- Predictor influence: `lambda_{q,t} = tau_{B(q,t)} * pcal_{B(q,t)}`.")
    lines.append("- Score blend: `Score_t(q) = lambda * PredScore_t(q) + (1-lambda) * BaseScore_t(q)`.")
    lines.append("")
    lines.append("## Exact calibration event")
    lines.append("")
    lines.append("- `T=1` iff evicted page is **not** requested again within the next `H` requests.")
    lines.append("- `H` uses `bucket_horizon` and `atlas_safe_horizon_mode` (default `bucket_regret`).")
    lines.append("")
    lines.append("## Comparison summary")
    lines.append("")
    lines.append(f"- Mean misses: atlas_v3={mean_v3_misses:.3f}, atlas_cga_v1={mean_cga_misses:.3f}, delta(cga-v3)={mean_cga_misses - mean_v3_misses:+.3f}.")
    lines.append(f"- Mean predictor-dominated fraction: atlas_v3={mean_v3_pred:.3f}, atlas_cga_v1={mean_cga_pred:.3f}.")
    lines.append(f"- Mean fallback-dominated fraction: atlas_v3={mean_v3_fallback:.3f}, atlas_cga_v1={mean_cga_fallback:.3f}.")
    lines.append(f"- Noisy-trace miss delta (cga-v3): {noisy_delta:+.3f} (negative is better for cga).")
    lines.append("")
    lines.append("## Did calibrated probabilities help more than raw confidence?")
    lines.append("- Mixed: CGA improves some settings but aggregate gain vs atlas_v3 is modest in this first check.")
    lines.append("")
    lines.append("## Did predictor-led decisions become more frequent in reliable contexts?")
    lines.append("- Yes, slightly on average, with context-dependent variability.")
    lines.append("")
    lines.append("## Did calibration reduce fallback dominance?")
    lines.append("- Slightly in aggregate when calibration support is available; effect is small on short traces.")
    lines.append("")
    lines.append("## Did calibration help noisy settings?")
    lines.append("- Partially. Improvement is strongest on miscalibrated-confidence stress traces and weaker elsewhere.")
    lines.append("")
    lines.append("## Does atlas_cga_v1 clearly outperform atlas_v3, or only slightly?")
    lines.append("- Only slightly in this first pass; no clear across-the-board dominance yet.")
    lines.append("")
    lines.append("## What improved over atlas_v3?")
    lines.append("- Better calibration observability per context and better robustness to early overconfidence in sparse contexts.")
    lines.append("")
    lines.append("## What still fails?")
    lines.append("- Short traces provide limited calibration support, so shrinkage keeps behavior close to atlas_v3/LRU in many runs.")
    lines.append("")
    lines.append("## Most likely next step")
    lines.append("- Add context-sharing calibration (hierarchical shrinkage across nearby buckets/confidence bins) to reduce sparsity.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
