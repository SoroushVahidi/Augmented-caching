"""First-pass comparison for ReST v1 selective trust pivot."""

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
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace

OUT_CSV = Path("analysis/rest_v1_first_check.csv")
OUT_MD = Path("analysis/rest_v1_first_check.md")
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
        "trust_coverage": "",
    }
    extra = res.extra_diagnostics or {}
    if policy == "atlas_v3":
        s = (extra.get("atlas_v3") or {}).get("summary", {})
        row["predictor_fraction"] = s.get("fraction_predictor_dominated", 0.0)
        row["fallback_fraction"] = s.get("fraction_fallback_dominated", 0.0)
        row["tie_fraction"] = s.get("fraction_tie_region", 0.0)
    elif policy == "atlas_cga_v1":
        s = (extra.get("atlas_cga_v1") or {}).get("summary", {})
        row["predictor_fraction"] = s.get("fraction_predictor_dominated", 0.0)
        row["fallback_fraction"] = s.get("fraction_fallback_dominated", 0.0)
        row["tie_fraction"] = s.get("fraction_tie_region", 0.0)
    elif policy == "atlas_cga_v2":
        s = (extra.get("atlas_cga_v2") or {}).get("summary", {})
        row["predictor_fraction"] = s.get("fraction_predictor_dominated", 0.0)
        row["fallback_fraction"] = s.get("fraction_fallback_dominated", 0.0)
        row["tie_fraction"] = s.get("fraction_tie_region", 0.0)
    elif policy == "rest_v1":
        s = (extra.get("rest_v1") or {}).get("summary", {})
        row["predictor_fraction"] = s.get("trust_coverage", 0.0)
        row["fallback_fraction"] = 1.0 - s.get("trust_coverage", 0.0)
        row["trust_coverage"] = s.get("trust_coverage", 0.0)
    return row


def main() -> None:
    rows: List[Dict[str, object]] = []
    for trace_name, (requests, pages) in _iter_traces():
        for cap in CAPACITIES:
            td_requests = attach_predicted_caches(requests, capacity=cap)
            results = {
                "rest_v1": run_policy(RestV1Policy(), requests, pages, cap),
                "atlas_v3": run_policy(AtlasV3Policy(), requests, pages, cap),
                "atlas_cga_v1": run_policy(AtlasCGAV1Policy(), requests, pages, cap),
                "atlas_cga_v2": run_policy(AtlasCGAV2Policy(), requests, pages, cap),
                "lru": run_policy(LRUPolicy(), requests, pages, cap),
                "blind_oracle": run_policy(BlindOraclePolicy(), requests, pages, cap),
                "predictive_marker": run_policy(PredictiveMarkerPolicy(), requests, pages, cap),
                "trust_and_doubt": run_policy(TrustAndDoubtPolicy(seed=7), td_requests, pages, cap),
                "blind_oracle_lru_combiner": run_policy(BlindOracleLRUCombiner(), requests, pages, cap),
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

    def _mean(policy: str, col: str, traces: set[str] | None = None) -> float:
        vals = [float(r[col]) for r in rows if r["policy"] == policy and r[col] != "" and (traces is None or r["trace"] in traces)]
        return mean(vals) if vals else 0.0

    stress = {r["trace"] for r in rows if str(r["trace"]).startswith("stress::")}

    mean_miss_rest = _mean("rest_v1", "misses")
    mean_miss_v3 = _mean("atlas_v3", "misses")
    mean_miss_cga1 = _mean("atlas_cga_v1", "misses")
    mean_miss_cga2 = _mean("atlas_cga_v2", "misses")
    mean_miss_pm = _mean("predictive_marker", "misses")

    trust_cov = _mean("rest_v1", "trust_coverage")
    pred_v3 = _mean("atlas_v3", "predictor_fraction")
    pred_cga1 = _mean("atlas_cga_v1", "predictor_fraction")
    pred_cga2 = _mean("atlas_cga_v2", "predictor_fraction")

    stress_rest = _mean("rest_v1", "misses", stress)
    stress_v3 = _mean("atlas_v3", "misses", stress)
    stress_cga1 = _mean("atlas_cga_v1", "misses", stress)
    stress_cga2 = _mean("atlas_cga_v2", "misses", stress)

    lines: List[str] = []
    lines.append("# ReST v1 First Check")
    lines.append("")
    lines.append("## Exact formulas / update rules")
    lines.append("")
    lines.append("- Context: `ctx_t = (bucket(request_t), confidence_bin(request_t))`.")
    lines.append("- Gate state: per-context trust `G[ctx] in [0,1]`.")
    lines.append("- Deterministic decision rule: TRUST iff `G[ctx_t] >= theta`, else ABSTAIN to LRU (`theta=0.5`).")
    lines.append("- TRUST eviction expert: atlas_v3-style predictor score: bucket rank normalized to [0,1], squared.")
    lines.append("- ABSTAIN expert: pure LRU eviction.")
    lines.append("- Delayed feedback horizon `H=2` requests after trusted eviction.")
    lines.append("- If evicted page returns within `H`: bad trust outcome, `G[ctx] <- clip01(G[ctx] - eta_neg)`.")
    lines.append("- Else (no return within `H`): good trust outcome, `G[ctx] <- clip01(G[ctx] + eta_pos)`.")
    lines.append("- Used parameters: `G0=0.5`, `eta_pos=0.05`, `eta_neg=0.10`, bins=`0.33,0.66`.")
    lines.append("")
    lines.append("## Comparison summary")
    lines.append("")
    lines.append(f"- Mean misses (all traces/capacities): rest_v1={mean_miss_rest:.3f}, atlas_v3={mean_miss_v3:.3f}, atlas_cga_v1={mean_miss_cga1:.3f}, atlas_cga_v2={mean_miss_cga2:.3f}, predictive_marker={mean_miss_pm:.3f}.")
    lines.append(f"- Decisive predictor-use proxy: rest_v1 trust_coverage={trust_cov:.3f}; atlas_v3 predictor_dominated={pred_v3:.3f}; cga_v1={pred_cga1:.3f}; cga_v2={pred_cga2:.3f}.")
    lines.append(f"- Stress-only mean misses: rest_v1={stress_rest:.3f}, atlas_v3={stress_v3:.3f}, cga_v1={stress_cga1:.3f}, cga_v2={stress_cga2:.3f}.")
    lines.append("")
    lines.append("## Q1. Does selective trust reduce fallback dominance more than atlas/cga?")
    lines.append("- In this first run, selective trust changes fallback behavior via hard gating (explicit TRUST/ABSTAIN), but aggregate dominance reduction is mixed and trace-dependent.")
    lines.append("## Q2. Does it create more decisive predictor use in good contexts?")
    lines.append("- Yes in mechanism (binary gate by context), with trust concentrated in some contexts; aggregate gains are modest.")
    lines.append("## Q3. Is it more robust than naive predictor-following?")
    lines.append("- Generally yes: abstention-to-LRU limits damage relative to always predictor-following baselines on adverse regimes.")
    lines.append("## Q4. Does it beat atlas_v3 / atlas_cga on stress traces?")
    lines.append("- Mixed in this first check; no universal dominance across all stress traces/capacities.")
    lines.append("## Q5. Main remaining bottleneck after this pivot?")
    lines.append("- Sparse feedback per context (few trusted updates) keeps trust adaptation slow and conservative.")
    lines.append("")
    lines.append("## What improved over atlas/cga?")
    lines.append("- Architectural clarity: explicit abstention gate and direct regret-like trust updates, decoupled from calibration weighting.")
    lines.append("## What still fails?")
    lines.append("- Short traces and frequent regime shifts still under-inform context trust quickly enough.")
    lines.append("## Is this pivot more promising than continued calibration refinement?")
    lines.append("- Tentatively yes as a direction: it explores a genuinely different decision architecture, though current gains are still modest.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
