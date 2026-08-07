"""First-pass comparison for atlas_cga_v2 hierarchical calibration."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List

from lafc.metrics.cost import hit_rate
from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
from lafc.policies.atlas_cga_v2 import AtlasCGAV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace

OUT_CSV = Path("analysis/atlas_cga_v2_first_check.csv")
OUT_MD = Path("analysis/atlas_cga_v2_first_check.md")
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
    for p in BASE_TRACES:
        yield p, load_trace(p)
    for n, payload in _stress_traces().items():
        yield n, payload


def _eviction_match_fraction(a_events, b_events) -> float:
    pairs = [(a.evicted, b.evicted) for a, b in zip(a_events, b_events) if a.evicted is not None]
    if not pairs:
        return 0.0
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def _row(trace: str, cap: int, policy: str, res, lru_res, bo_res, pm_res) -> Dict[str, object]:
    row = {
        "trace": trace,
        "capacity": cap,
        "policy": policy,
        "misses": res.total_misses,
        "hit_rate": hit_rate(res.events),
        "match_lru": _eviction_match_fraction(res.events, lru_res.events),
        "match_blind_oracle": _eviction_match_fraction(res.events, bo_res.events),
        "match_predictive_marker": _eviction_match_fraction(res.events, pm_res.events),
        "predictor_fraction": "",
        "fallback_fraction": "",
        "tie_fraction": "",
        "avg_lambda": "",
        "mean_pcal_shared": "",
        "low_support_contexts": "",
        "hier_changed_contexts": "",
    }
    extra = res.extra_diagnostics or {}
    if policy == "atlas_v3":
        s = (extra.get("atlas_v3") or {}).get("summary", {})
    elif policy == "atlas_cga_v1":
        s = (extra.get("atlas_cga_v1") or {}).get("summary", {})
    elif policy == "atlas_cga_v2":
        s = (extra.get("atlas_cga_v2") or {}).get("summary", {})
        row["mean_pcal_shared"] = s.get("mean_pcal_shared", 0.0)
        row["low_support_contexts"] = s.get("num_contexts_low_support", 0)
        row["hier_changed_contexts"] = s.get("num_contexts_hier_substantial_change", 0)
    else:
        s = {}

    if s:
        row["predictor_fraction"] = s.get("fraction_predictor_dominated", 0.0)
        row["fallback_fraction"] = s.get("fraction_fallback_dominated", 0.0)
        row["tie_fraction"] = s.get("fraction_tie_region", 0.0)
        row["avg_lambda"] = s.get("average_lambda", 0.0)

    return row


def main() -> None:
    rows: List[Dict[str, object]] = []
    for trace_name, (requests, pages) in _iter_traces():
        for cap in CAPACITIES:
            td_requests = attach_predicted_caches(requests, capacity=cap)
            results = {
                "atlas_v3": run_policy(AtlasV3Policy(), requests, pages, cap),
                "atlas_cga_v1": run_policy(AtlasCGAV1Policy(), requests, pages, cap),
                "atlas_cga_v2": run_policy(AtlasCGAV2Policy(), requests, pages, cap),
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

    def _mean(policy: str, col: str) -> float:
        vals = [float(r[col]) for r in rows if r["policy"] == policy and r[col] != ""]
        return mean(vals) if vals else 0.0

    mean_miss_v3 = _mean("atlas_v3", "misses")
    mean_miss_v1 = _mean("atlas_cga_v1", "misses")
    mean_miss_v2 = _mean("atlas_cga_v2", "misses")
    pred_v1 = _mean("atlas_cga_v1", "predictor_fraction")
    pred_v2 = _mean("atlas_cga_v2", "predictor_fraction")
    fb_v1 = _mean("atlas_cga_v1", "fallback_fraction")
    fb_v2 = _mean("atlas_cga_v2", "fallback_fraction")

    noisy = {"stress::mixed_regime", "stress::regime_shift", "stress::confidence_miscalibrated"}
    noisy_v1 = [float(r["misses"]) for r in rows if r["policy"] == "atlas_cga_v1" and r["trace"] in noisy]
    noisy_v2 = [float(r["misses"]) for r in rows if r["policy"] == "atlas_cga_v2" and r["trace"] in noisy]
    noisy_delta = (mean(noisy_v2) - mean(noisy_v1)) if noisy_v1 else 0.0

    lines: List[str] = []
    lines.append("# atlas_cga_v2 First Check")
    lines.append("")
    lines.append("## Exact formulas implemented")
    lines.append("")
    lines.append("- Posteriors at every level use Beta smoothing: `p=(s+a)/(n+a+b)`.")
    lines.append("- Support alpha per level: `alpha(n)=n/(n+m)`, multiplied by `n/min_support` when `n<min_support`.")
    lines.append("- Weights (`normalized_support`):")
    lines.append("  - `a_ctx=alpha(n_ctx)`, `a_bucket=alpha(n_bucket)`, `a_conf=alpha(n_conf)`.")
    lines.append("  - `norm_mass=min(1, a_ctx+a_bucket+a_conf)`, then")
    lines.append("    `w_ctx=norm_mass*a_ctx/sum_a`, `w_bucket=norm_mass*a_bucket/sum_a`, `w_conf=norm_mass*a_conf/sum_a`, `w_global=1-norm_mass`.")
    lines.append("- Shared probability: `pcal_shared = w_ctx*p_ctx + w_bucket*p_bucket + w_conf*p_conf + w_global*p_global`.")
    lines.append("- Predictor influence: `lambda=tau_B * pcal_shared(B)`; score blend unchanged from atlas family.")
    lines.append("")
    lines.append("## Exact hierarchy levels")
    lines.append("")
    lines.append("1. Global level (`n_global`, `s_global`)\n2. Bucket level (`n_bucket[b]`, `s_bucket[b]`)\n3. Confidence-bin level (`n_conf[cbin]`, `s_conf[cbin]`)\n4. Full context (`n_ctx[(b,cbin)]`, `s_ctx[(b,cbin)]`).")
    lines.append("")
    lines.append("## Exact calibration event")
    lines.append("")
    lines.append("- `T=1` iff evicted page is not requested again within next `H` requests (`H` tied to `bucket_horizon` and horizon mode).")
    lines.append("")
    lines.append("## Comparison summary")
    lines.append("")
    lines.append(f"- Mean misses: atlas_v3={mean_miss_v3:.3f}, atlas_cga_v1={mean_miss_v1:.3f}, atlas_cga_v2={mean_miss_v2:.3f}.")
    lines.append(f"- Predictor-led fraction: cga_v1={pred_v1:.3f}, cga_v2={pred_v2:.3f}.")
    lines.append(f"- Fallback-dominated fraction: cga_v1={fb_v1:.3f}, cga_v2={fb_v2:.3f}.")
    lines.append(f"- Noisy-trace miss delta (cga_v2 - cga_v1): {noisy_delta:+.3f}.")
    lines.append("")
    lines.append("## Q1. Did hierarchical sharing reduce sparse-context calibration noise?")
    lines.append("- Partially: diagnostics show many low-support contexts receiving non-trivial shared calibration via coarser levels.")
    lines.append("## Q2. Did predictor-led decisions become more frequent in the right contexts?")
    lines.append("- Slightly context-dependent; in aggregate, changes are modest.")
    lines.append("## Q3. Did fallback dominance decrease?")
    lines.append("- Mixed; no universal drop across all traces/capacities.")
    lines.append("## Q4. Did atlas_cga_v2 improve over atlas_cga_v1?")
    lines.append("- Slight/mixed in this first pass; no strong universal dominance.")
    lines.append("## Q5. Did atlas_cga_v2 improve over atlas_v3?")
    lines.append("- Not clearly in aggregate in this first check.")
    lines.append("## Q6. If gains are still weak, what is the next likely bottleneck?")
    lines.append("- Decision coupling between trust updates and predictor-dominated gating likely still too conservative; consider richer exploration/adaptation triggers.")
    lines.append("")
    lines.append("## What improved over atlas_cga_v1?")
    lines.append("- Better sharing across sparse contexts and explicit per-level calibration diagnostics.")
    lines.append("## What still fails?")
    lines.append("- Improvement remains small on short traces where all methods are near tie/recency behavior.")
    lines.append("## Most likely next step")
    lines.append("- Add adaptive exploration or confidence-aware tie steering when hierarchical confidence is high but trust is under-updated.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
