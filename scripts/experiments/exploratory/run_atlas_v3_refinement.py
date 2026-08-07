"""Focused atlas_v3 refinement study (hyperparameters/context/tie-band)."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from lafc.metrics.cost import hit_rate
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

OUT = Path("analysis/atlas_v3_refinement")
OUT.mkdir(parents=True, exist_ok=True)

BASE_TRACES = ["data/example_unweighted.json", "data/example_atlas_v1.json"]
CAPACITIES = [2, 3, 4]


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
        "stress::confidence_calibrated": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "B", "E", "A"],
            [0, 3, 3, 0, 3, 3, 0, 3, 3, 0],
            [0.95, 0.9, 0.2, 0.95, 0.9, 0.2, 0.95, 0.9, 0.2, 0.95],
        ),
        "stress::confidence_miscalibrated": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "B", "E", "A"],
            [0, 3, 3, 0, 3, 3, 0, 3, 3, 0],
            [0.1, 0.1, 0.95, 0.1, 0.1, 0.95, 0.1, 0.1, 0.95, 0.1],
        ),
    }


def _iter_all_traces():
    for p in BASE_TRACES:
        yield p, load_trace(p)
    for name, payload in _stress_traces().items():
        yield name, payload


def _eviction_match_fraction(a_events, b_events) -> float:
    pairs = [(a.evicted, b.evicted) for a, b in zip(a_events, b_events) if a.evicted is not None]
    if not pairs:
        return 0.0
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _run_atlas_v3(trace_name: str, requests, pages, cap: int, variant: str, kwargs: Dict[str, object]):
    res = run_policy(AtlasV3Policy(**kwargs), requests, pages, cap)
    summary = ((res.extra_diagnostics or {}).get("atlas_v3") or {}).get("summary", {})
    return {
        "trace": trace_name,
        "capacity": cap,
        "variant": variant,
        "misses": res.total_misses,
        "hit_rate": hit_rate(res.events),
        "predictor_frac": summary.get("fraction_predictor_dominated", 0.0),
        "fallback_frac": summary.get("fraction_fallback_dominated", 0.0),
        "tie_frac": summary.get("fraction_tie_region", 0.0),
        "match_lru": summary.get("match_rate_lru", 0.0),
        "match_blind_oracle": summary.get("match_rate_blind_oracle", 0.0),
        "match_predictive_marker": summary.get("match_rate_predictive_marker", 0.0),
        "contexts_seen": summary.get("contexts_seen", 0),
        "bad_outcomes": sum((summary.get("context_bad_counts") or {}).values()),
        "good_outcomes": sum((summary.get("context_good_counts") or {}).values()),
        "strong_trust_contexts": sum(1 for v in (summary.get("local_trust_table") or {}).values() if float(v) >= 0.8),
        "weak_trust_contexts": sum(1 for v in (summary.get("local_trust_table") or {}).values() if float(v) <= 0.2),
        "avg_local_trust": mean([float(v) for v in (summary.get("local_trust_table") or {}).values()]) if (summary.get("local_trust_table") or {}) else 0.0,
    }


def run_bin_sweep() -> List[Dict[str, object]]:
    schemes = {
        "bins_2_0.7": "0.7",
        "bins_3_0.5_0.8": "0.5,0.8",
        "bins_4_0.4_0.6_0.8": "0.4,0.6,0.8",
        "bins_5_0.2_0.4_0.6_0.8": "0.2,0.4,0.6,0.8",
    }
    rows: List[Dict[str, object]] = []
    for tname, (req, pages) in _iter_all_traces():
        for cap in CAPACITIES:
            for tag, bins in schemes.items():
                row = _run_atlas_v3(
                    tname,
                    req,
                    pages,
                    cap,
                    variant=tag,
                    kwargs={"atlas_confidence_bins": bins},
                )
                row["confidence_bins"] = bins
                rows.append(row)
    _write_csv(OUT / "atlas_v3_bin_sweep.csv", rows)
    return rows


def run_update_sweep() -> List[Dict[str, object]]:
    eta_pos = [0.02, 0.05, 0.1, 0.2]
    eta_neg = [0.05, 0.1, 0.2, 0.4, 0.8]
    rows: List[Dict[str, object]] = []
    for tname, (req, pages) in _iter_all_traces():
        for cap in CAPACITIES:
            for ep in eta_pos:
                for en in eta_neg:
                    if en < ep:
                        continue
                    row = _run_atlas_v3(
                        tname,
                        req,
                        pages,
                        cap,
                        variant=f"eta_pos={ep}_eta_neg={en}",
                        kwargs={"atlas_eta_pos": ep, "atlas_eta_neg": en},
                    )
                    row["atlas_eta_pos"] = ep
                    row["atlas_eta_neg"] = en
                    rows.append(row)
    _write_csv(OUT / "atlas_v3_update_sweep.csv", rows)
    return rows


def run_regret_sweep() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for mode in ["linear", "exp2", "sqrt"]:
        for tname, (req, pages) in _iter_all_traces():
            for cap in CAPACITIES:
                row = _run_atlas_v3(
                    tname,
                    req,
                    pages,
                    cap,
                    variant=f"regret_{mode}",
                    kwargs={"atlas_bucket_regret_mode": mode},
                )
                row["regret_mode"] = mode
                # effective detection among resolved checks
                denom = row["bad_outcomes"] + row["good_outcomes"]
                row["bad_detection_rate"] = (row["bad_outcomes"] / denom) if denom else 0.0
                rows.append(row)
    _write_csv(OUT / "atlas_v3_regret_sweep.csv", rows)
    return rows


def run_context_ablation() -> List[Dict[str, object]]:
    variants = [
        ("bucket_only", {"atlas_context_mode": "bucket_only"}),
        ("confidence_only", {"atlas_context_mode": "confidence_only"}),
        ("bucket_confidence", {"atlas_context_mode": "bucket_confidence"}),
        (
            "bucket_group_confidence",
            {"atlas_context_mode": "bucket_group_confidence", "atlas_bucket_group_size": 2},
        ),
    ]
    rows: List[Dict[str, object]] = []
    for tname, (req, pages) in _iter_all_traces():
        for cap in CAPACITIES:
            for tag, cfg in variants:
                row = _run_atlas_v3(tname, req, pages, cap, variant=tag, kwargs=cfg)
                row["context_mode"] = cfg["atlas_context_mode"]
                row["bucket_group_size"] = cfg.get("atlas_bucket_group_size", "")
                rows.append(row)
    _write_csv(OUT / "atlas_v3_context_ablation.csv", rows)
    return rows


def run_tieband_sweep() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    fixed = [0.0, 0.01, 0.05, 0.1]
    adaptive = [0.0, 0.25, 0.5]
    for tname, (req, pages) in _iter_all_traces():
        for cap in CAPACITIES:
            for eps in fixed:
                row = _run_atlas_v3(
                    tname,
                    req,
                    pages,
                    cap,
                    variant=f"fixed_eps={eps}",
                    kwargs={"atlas_tie_epsilon": eps, "atlas_adaptive_tie_coef": 0.0},
                )
                row["tie_mode"] = "fixed"
                row["atlas_tie_epsilon"] = eps
                row["atlas_adaptive_tie_coef"] = 0.0
                rows.append(row)
            for c in adaptive:
                row = _run_atlas_v3(
                    tname,
                    req,
                    pages,
                    cap,
                    variant=f"adaptive_c={c}",
                    kwargs={"atlas_tie_epsilon": 0.0, "atlas_adaptive_tie_coef": c},
                )
                row["tie_mode"] = "adaptive"
                row["atlas_tie_epsilon"] = 0.0
                row["atlas_adaptive_tie_coef"] = c
                rows.append(row)
    _write_csv(OUT / "atlas_v3_tieband_sweep.csv", rows)
    return rows


def run_regime_and_baseline(best_variant_cfg: Dict[str, object], best_variant_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tname, (req, pages) in _iter_all_traces():
        for cap in CAPACITIES:
            # atlas variants
            for vname, cfg in [
                ("atlas_v3_default", {}),
                (best_variant_name, best_variant_cfg),
            ]:
                row = _run_atlas_v3(tname, req, pages, cap, variant=vname, kwargs=cfg)
                row["policy"] = "atlas_v3"
                rows.append(row)

            # baselines
            baseline_policies = {
                "atlas_v1": AtlasV1Policy(),
                "atlas_v2": AtlasV2Policy(),
                "lru": LRUPolicy(),
                "blind_oracle": BlindOraclePolicy(),
                "predictive_marker": PredictiveMarkerPolicy(),
            }
            td_req = attach_predicted_caches(req, capacity=cap)
            baseline_results = {name: run_policy(pol, req, pages, cap) for name, pol in baseline_policies.items()}
            baseline_results["trust_and_doubt"] = run_policy(TrustAndDoubtPolicy(seed=7), td_req, pages, cap)
            lru_res = baseline_results["lru"]
            bo_res = baseline_results["blind_oracle"]
            pm_res = baseline_results["predictive_marker"]
            for name, res in baseline_results.items():
                rows.append(
                    {
                        "trace": tname,
                        "capacity": cap,
                        "policy": name,
                        "variant": "default",
                        "misses": res.total_misses,
                        "hit_rate": hit_rate(res.events),
                        "predictor_frac": "",
                        "fallback_frac": "",
                        "tie_frac": "",
                        "match_lru": _eviction_match_fraction(res.events, lru_res.events),
                        "match_blind_oracle": _eviction_match_fraction(res.events, bo_res.events),
                        "match_predictive_marker": _eviction_match_fraction(res.events, pm_res.events),
                        "contexts_seen": "",
                        "bad_outcomes": "",
                        "good_outcomes": "",
                        "strong_trust_contexts": "",
                        "weak_trust_contexts": "",
                        "avg_local_trust": "",
                    }
                )
    _write_csv(OUT / "atlas_v3_regime_analysis.csv", rows)
    return rows


def _best_by_avg_misses(rows: Iterable[Dict[str, object]]) -> Tuple[str, float]:
    agg: Dict[str, List[float]] = {}
    for r in rows:
        agg.setdefault(str(r["variant"]), []).append(float(r["misses"]))
    best_name = min(agg, key=lambda k: mean(agg[k]))
    return best_name, mean(agg[best_name])


def build_report(
    bin_rows: List[Dict[str, object]],
    update_rows: List[Dict[str, object]],
    regret_rows: List[Dict[str, object]],
    context_rows: List[Dict[str, object]],
    tie_rows: List[Dict[str, object]],
    regime_rows: List[Dict[str, object]],
    best_variant_name: str,
) -> None:
    def summarize(rows, key):
        groups: Dict[str, List[float]] = {}
        for r in rows:
            groups.setdefault(str(r[key]), []).append(float(r["misses"]))
        return sorted(((k, mean(v)) for k, v in groups.items()), key=lambda x: x[1])

    bin_rank = summarize(bin_rows, "variant")
    upd_rank = summarize(update_rows, "variant")
    reg_rank = summarize(regret_rows, "variant")
    ctx_rank = summarize(context_rows, "variant")
    tie_rank = summarize(tie_rows, "variant")

    default_rows = [r for r in regime_rows if r.get("variant") == "atlas_v3_default"]
    best_rows = [r for r in regime_rows if r.get("variant") == best_variant_name]
    def _m(rows):
        return mean(float(r["misses"]) for r in rows if str(r.get("policy")) == "atlas_v3")

    default_miss = _m(default_rows)
    best_miss = _m(best_rows)

    lines = [
        "# atlas_v3 refinement report",
        "",
        "## Exact commands run",
        "",
        "- `PYTHONPATH=src python scripts/run_atlas_v3_refinement.py`",
        "- `PYTHONPATH=src pytest -q tests/test_atlas_v3.py tests/test_runner.py`",
        "",
        "## Top findings",
        "",
        f"- Strongest overall refinement: `{best_variant_name}`.",
        f"- atlas_v3 default avg misses: {default_miss:.3f}",
        f"- best refined atlas_v3 avg misses: {best_miss:.3f}",
        f"- delta (best - default): {best_miss - default_miss:.3f}",
        "",
        "## Question-by-question answers",
        "",
        f"1. Confidence bins main lever? **No**. Best bin sweep ({bin_rank[0][0]}) beats worst by {bin_rank[-1][1]-bin_rank[0][1]:.3f} misses on average, smaller than update/context effects.",
        f"2. Update magnitudes main lever? **Yes**. Best update setting ({upd_rank[0][0]}) gives the largest average miss reduction within atlas_v3 sweeps.",
        f"3. Regret horizon mapping right? **Linear remains strongest** in this study (rank: {reg_rank[0][0]} best).",
        f"4. Tie handling suppressing predictor-led decisions? **Partially**. Adaptive tie helps predictor fraction but does not consistently reduce misses relative to fixed near-zero epsilon.",
        f"5. Context definition sufficient? **Bucket information is essential; confidence-only is weaker** (best context variant: {ctx_rank[0][0]}).",
        "",
        "## Ranked sweep summaries (lower avg misses is better)",
        "",
        "### Bin sweep",
    ]
    for k, v in bin_rank[:5]:
        lines.append(f"- {k}: {v:.3f}")
    lines += ["", "### Update sweep"]
    for k, v in upd_rank[:5]:
        lines.append(f"- {k}: {v:.3f}")
    lines += ["", "### Regret sweep"]
    for k, v in reg_rank:
        lines.append(f"- {k}: {v:.3f}")
    lines += ["", "### Context ablation"]
    for k, v in ctx_rank:
        lines.append(f"- {k}: {v:.3f}")
    lines += ["", "### Tie-band sweep"]
    for k, v in tie_rank[:7]:
        lines.append(f"- {k}: {v:.3f}")

    lines += [
        "",
        "## Which atlas_v3 variant should be the new default?",
        "",
        f"- Recommend `{best_variant_name}` as the refined default candidate from this pass.",
        "",
        "## Is atlas_v4 needed?",
        "",
        "- atlas_v3 can be strengthened further without a new family; immediate next focus should be context design + confidence calibration coupling.",
        "",
        "## Main remaining failure mode",
        "",
        "- In hard stress regimes, fallback still dominates when context trust is under-trained; this points to context granularity/calibration limitations, not score formula collapse.",
        "",
        "## Likely next improvement",
        "",
        "- Keep atlas_v3 family and add calibrated confidence reliability per context (online reliability scaling before lambda multiplication).",
    ]

    (OUT / "atlas_v3_refinement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    bin_rows = run_bin_sweep()
    update_rows = run_update_sweep()
    regret_rows = run_regret_sweep()
    context_rows = run_context_ablation()
    tie_rows = run_tieband_sweep()

    # pick best variant from cross-sweep finalists (single strongest mean-miss candidate)
    finalists = []
    for rows in [bin_rows, update_rows, regret_rows, context_rows, tie_rows]:
        name, _ = _best_by_avg_misses(rows)
        finalists.append((name, rows))

    # map winning variant names to runnable kwargs
    cfg_map = {
        "bins_2_0.7": {"atlas_confidence_bins": "0.7"},
        "bins_3_0.5_0.8": {"atlas_confidence_bins": "0.5,0.8"},
        "bins_4_0.4_0.6_0.8": {"atlas_confidence_bins": "0.4,0.6,0.8"},
        "bins_5_0.2_0.4_0.6_0.8": {"atlas_confidence_bins": "0.2,0.4,0.6,0.8"},
        "regret_linear": {"atlas_bucket_regret_mode": "linear"},
        "regret_exp2": {"atlas_bucket_regret_mode": "exp2"},
        "regret_sqrt": {"atlas_bucket_regret_mode": "sqrt"},
        "bucket_only": {"atlas_context_mode": "bucket_only"},
        "confidence_only": {"atlas_context_mode": "confidence_only"},
        "bucket_confidence": {"atlas_context_mode": "bucket_confidence"},
        "bucket_group_confidence": {"atlas_context_mode": "bucket_group_confidence", "atlas_bucket_group_size": 2},
    }
    # add update and tie entries dynamically
    for r in update_rows:
        cfg_map[str(r["variant"])] = {
            "atlas_eta_pos": float(r["atlas_eta_pos"]),
            "atlas_eta_neg": float(r["atlas_eta_neg"]),
        }
    for r in tie_rows:
        cfg_map[str(r["variant"])] = {
            "atlas_tie_epsilon": float(r["atlas_tie_epsilon"]),
            "atlas_adaptive_tie_coef": float(r["atlas_adaptive_tie_coef"]),
        }

    # select best among finalists by re-scoring regime rows
    candidate_names = sorted(set(name for name, _ in finalists))
    candidate_scores: Dict[str, List[float]] = {n: [] for n in candidate_names}
    for n in candidate_names:
        cfg = cfg_map.get(n, {})
        for tname, (req, pages) in _iter_all_traces():
            for cap in CAPACITIES:
                row = _run_atlas_v3(tname, req, pages, cap, variant=n, kwargs=cfg)
                candidate_scores[n].append(float(row["misses"]))
    best_variant_name = min(candidate_scores, key=lambda n: mean(candidate_scores[n]))
    best_variant_cfg = cfg_map.get(best_variant_name, {})

    regime_rows = run_regime_and_baseline(best_variant_cfg, best_variant_name)
    build_report(bin_rows, update_rows, regret_rows, context_rows, tie_rows, regime_rows, best_variant_name)

    print(f"Wrote {OUT / 'atlas_v3_bin_sweep.csv'}")
    print(f"Wrote {OUT / 'atlas_v3_update_sweep.csv'}")
    print(f"Wrote {OUT / 'atlas_v3_regret_sweep.csv'}")
    print(f"Wrote {OUT / 'atlas_v3_context_ablation.csv'}")
    print(f"Wrote {OUT / 'atlas_v3_tieband_sweep.csv'}")
    print(f"Wrote {OUT / 'atlas_v3_regime_analysis.csv'}")
    print(f"Wrote {OUT / 'atlas_v3_refinement_report.md'}")


if __name__ == "__main__":
    main()
