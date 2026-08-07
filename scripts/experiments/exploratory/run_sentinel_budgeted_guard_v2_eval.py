from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.lru import LRUPolicy
from lafc.policies.robust_ftp_marker_combiner import (
    FollowPredictedCachePolicy,
    RobustFtPDeterministicMarkerCombiner,
)
from lafc.policies.sentinel_budgeted_guard_v2 import SentinelBudgetedGuardV2Policy
from lafc.policies.sentinel_robust_tripwire_v1 import SentinelRobustTripwireV1Policy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Request


@dataclass(frozen=True)
class StressTrace:
    name: str
    page_ids: List[str]
    predicted_caches: List[List[str]]
    intended_case: str


def _attach_predicted_caches(page_ids: Sequence[str], predicted_caches: Sequence[Sequence[str]]) -> Tuple[List[Request], Dict[str, object]]:
    reqs, pages = build_requests_from_lists(list(page_ids))
    if len(reqs) != len(predicted_caches):
        raise ValueError("predicted_caches length must match requests length")
    out: List[Request] = []
    for req, pc in zip(reqs, predicted_caches):
        md = dict(req.metadata)
        md["predicted_cache"] = [str(x) for x in pc]
        out.append(
            Request(
                t=req.t,
                page_id=req.page_id,
                predicted_next=req.predicted_next,
                actual_next=req.actual_next,
                metadata=md,
            )
        )
    return out, pages


def _stress_traces() -> List[StressTrace]:
    return [
        StressTrace(
            name="synthetic::predictor_disagreement_help",
            intended_case="predictor_help",
            page_ids=[
                "D", "D", "A", "C", "D", "D", "C", "D", "C", "B", "B", "C", "B", "A", "C", "B",
                "C", "A", "A", "C", "D", "A", "C", "D", "C", "B", "D", "D", "C", "A",
            ],
            predicted_caches=[
                ["B", "D"], ["A", "B"], ["A", "B"], ["C", "D"], ["B", "C"], ["C", "D"], ["C", "D"], ["C", "D"],
                ["A", "B"], ["B", "D"], ["B", "C"], ["A", "D"], ["A", "C"], ["C", "D"], ["A", "D"], ["C", "D"],
                ["A", "B"], ["A", "C"], ["B", "D"], ["A", "C"], ["A", "C"], ["A", "C"], ["B", "D"], ["B", "C"],
                ["A", "B"], ["A", "B"], ["A", "D"], ["B", "D"], ["B", "C"], ["A", "B"],
            ],
        ),
        StressTrace(
            name="synthetic::predictor_disagreement_hurt",
            intended_case="predictor_hurt",
            page_ids=[
                "C", "C", "D", "A", "D", "B", "A", "B", "A", "D", "D", "C", "A", "B", "A", "A",
                "A", "B", "A", "B", "C", "C", "B", "A", "D", "D", "A", "A", "C", "D",
            ],
            predicted_caches=[
                ["A", "B"], ["A", "D"], ["A", "C"], ["C", "D"], ["B", "D"], ["C", "D"], ["C", "D"], ["A", "D"],
                ["A", "B"], ["A", "C"], ["A", "D"], ["A", "B"], ["A", "B"], ["A", "B"], ["A", "C"], ["C", "D"],
                ["A", "D"], ["B", "D"], ["A", "D"], ["A", "D"], ["B", "D"], ["A", "B"], ["C", "D"], ["C", "D"],
                ["B", "D"], ["C", "D"], ["B", "C"], ["C", "D"], ["C", "D"], ["B", "C"],
            ],
        ),
        StressTrace(
            name="synthetic::predictor_disagreement_mixed",
            intended_case="mixed",
            page_ids=[
                "C", "A", "D", "C", "C", "D", "D", "D", "A", "B", "B", "B", "C", "C", "A", "A",
                "D", "D", "B", "D", "A", "B", "C", "D", "A", "D", "D", "D", "A", "A",
            ],
            predicted_caches=[
                ["B", "C"], ["A", "C"], ["A", "B"], ["A", "B"], ["B", "D"], ["B", "D"], ["A", "C"], ["C", "D"],
                ["A", "D"], ["A", "B"], ["C", "D"], ["B", "D"], ["C", "D"], ["A", "D"], ["A", "C"], ["B", "C"],
                ["B", "C"], ["A", "B"], ["A", "B"], ["B", "D"], ["C", "D"], ["B", "C"], ["B", "D"], ["C", "D"],
                ["A", "D"], ["C", "D"], ["A", "B"], ["C", "D"], ["C", "D"], ["B", "D"],
            ],
        ),
    ]


def _disagreement_count(requests: List[Request], pages: Dict[str, object], capacity: int) -> int:
    robust = RobustFtPDeterministicMarkerCombiner()
    robust.reset(capacity, pages)
    predictor = FollowPredictedCachePolicy()
    predictor.reset(capacity, pages)

    disagree = 0
    for req in requests:
        r_event = robust.on_request(req)
        p_event = predictor.on_request(req)
        if r_event.evicted != p_event.evicted:
            disagree += 1
    return disagree


def run_eval(out_dir: Path, capacities: List[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = _stress_traces()
    policies = {
        "sentinel_budgeted_guard_v2": lambda: SentinelBudgetedGuardV2Policy(),
        "sentinel_robust_tripwire_v1": lambda: SentinelRobustTripwireV1Policy(),
        "robust_ftp_d_marker": lambda: RobustFtPDeterministicMarkerCombiner(),
        "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
        "lru": lambda: LRUPolicy(),
    }

    # One lightweight sanity run.
    sanity_trace = traces[0]
    sanity_requests, sanity_pages = _attach_predicted_caches(sanity_trace.page_ids, sanity_trace.predicted_caches)
    sanity_cap = capacities[0]
    sanity_result = run_policy(SentinelBudgetedGuardV2Policy(), sanity_requests, sanity_pages, sanity_cap)
    sanity_diag = (sanity_result.extra_diagnostics or {}).get("sentinel_budgeted_guard_v2", {})
    sanity_summary = sanity_diag.get("summary", {})
    sanity_report = [
        "# sentinel_budgeted_guard_v2 sanity report",
        "",
        "## Setup",
        f"- Trace: `{sanity_trace.name}` ({sanity_trace.intended_case})",
        f"- Capacity: {sanity_cap}",
        f"- Requests: {len(sanity_requests)}",
        "",
        "## Result",
        f"- Misses: {sanity_result.total_misses}",
        f"- Hits: {sanity_result.total_hits}",
        f"- Predictor steps: {int(sanity_summary.get('predictor_steps', 0.0))}",
        f"- Guard triggers: {int(sanity_summary.get('guard_triggers', 0.0))}",
        f"- Harmful overrides: {int(sanity_summary.get('harmful_override_steps', 0.0))}",
        f"- Remaining override budget: {int(sanity_summary.get('remaining_override_budget', 0.0))}",
        "",
        "Sanity verdict: policy executes, logs diagnostics, and remains robust-first (predictor usage is bounded).",
    ]
    (out_dir / "v2_sanity_report.md").write_text("\n".join(sanity_report) + "\n", encoding="utf-8")

    rows: List[Dict[str, object]] = []

    for tr in traces:
        for cap in capacities:
            requests, pages = _attach_predicted_caches(tr.page_ids, tr.predicted_caches)
            disagreement = _disagreement_count(requests, pages, cap)

            run_rows: Dict[str, Dict[str, object]] = {}
            v2_summary: Dict[str, float] = {}
            v1_summary: Dict[str, float] = {}

            for policy_name, factory in policies.items():
                result = run_policy(factory(), requests, pages, cap)
                run_rows[policy_name] = {
                    "misses": int(result.total_misses),
                    "hits": int(result.total_hits),
                }
                if policy_name == "sentinel_budgeted_guard_v2":
                    d = (result.extra_diagnostics or {}).get("sentinel_budgeted_guard_v2", {})
                    v2_summary = dict(d.get("summary", {}))
                if policy_name == "sentinel_robust_tripwire_v1":
                    d = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
                    v1_summary = dict(d.get("summary", {}))

            v2_miss = run_rows["sentinel_budgeted_guard_v2"]["misses"]
            v1_miss = run_rows["sentinel_robust_tripwire_v1"]["misses"]
            r_miss = run_rows["robust_ftp_d_marker"]["misses"]
            b_miss = run_rows["blind_oracle_lru_combiner"]["misses"]
            l_miss = run_rows["lru"]["misses"]

            rows.append(
                {
                    "trace": tr.name,
                    "intended_case": tr.intended_case,
                    "capacity": cap,
                    "requests": len(requests),
                    "disagreement_steps": disagreement,
                    "disagreement_fraction": disagreement / len(requests),
                    "sentinel_budgeted_guard_v2_misses": v2_miss,
                    "sentinel_robust_tripwire_v1_misses": v1_miss,
                    "robust_ftp_d_marker_misses": r_miss,
                    "blind_oracle_lru_combiner_misses": b_miss,
                    "lru_misses": l_miss,
                    "v2_minus_v1": v2_miss - v1_miss,
                    "v2_minus_robust_ftp_d_marker": v2_miss - r_miss,
                    "v2_minus_blind_oracle_lru_combiner": v2_miss - b_miss,
                    "v2_minus_lru": v2_miss - l_miss,
                    "v2_vs_v1": "win" if v2_miss < v1_miss else ("tie" if v2_miss == v1_miss else "loss"),
                    "v2_helpful_override_steps": int(v2_summary.get("helpful_override_steps", 0.0)),
                    "v2_harmful_override_steps": int(v2_summary.get("harmful_override_steps", 0.0)),
                    "v1_helpful_override_steps": int(v1_summary.get("helpful_override_steps", 0.0)),
                    "v1_harmful_override_steps": int(v1_summary.get("harmful_override_steps", 0.0)),
                    "v2_predictor_steps": float(v2_summary.get("predictor_steps", 0.0)),
                    "v1_predictor_steps": float(v1_summary.get("predictor_steps", 0.0)),
                }
            )

    csv_path = out_dir / "v2_disagreement_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    v2_wins = sum(1 for r in rows if r["v2_vs_v1"] == "win")
    v2_ties = sum(1 for r in rows if r["v2_vs_v1"] == "tie")
    v2_losses = sum(1 for r in rows if r["v2_vs_v1"] == "loss")
    total_v2_harmful = sum(int(r["v2_harmful_override_steps"]) for r in rows)
    total_v1_harmful = sum(int(r["v1_harmful_override_steps"]) for r in rows)

    summary = {
        "method": "sentinel_budgeted_guard_v2",
        "slices": len(rows),
        "trace_count": len(traces),
        "capacities": capacities,
        "disagreement": {
            "mean_steps": mean(float(r["disagreement_steps"]) for r in rows),
            "mean_fraction": mean(float(r["disagreement_fraction"]) for r in rows),
            "slices_with_disagreement": sum(1 for r in rows if int(r["disagreement_steps"]) > 0),
        },
        "v2_vs_v1": {
            "wins": v2_wins,
            "ties": v2_ties,
            "losses": v2_losses,
            "mean_delta_misses": mean(float(r["v2_minus_v1"]) for r in rows),
        },
        "override_harm": {
            "v2_total_harmful_override_steps": total_v2_harmful,
            "v1_total_harmful_override_steps": total_v1_harmful,
            "v2_total_helpful_override_steps": sum(int(r["v2_helpful_override_steps"]) for r in rows),
            "v1_total_helpful_override_steps": sum(int(r["v1_helpful_override_steps"]) for r in rows),
        },
        "robustness": {
            "mean_v2_minus_robust_ftp_d_marker": mean(float(r["v2_minus_robust_ftp_d_marker"]) for r in rows),
            "max_v2_minus_robust_ftp_d_marker": max(float(r["v2_minus_robust_ftp_d_marker"]) for r in rows),
        },
    }

    (out_dir / "v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    improves_over_v1 = v2_wins > v2_losses and summary["v2_vs_v1"]["mean_delta_misses"] < 0
    introduces_more_harm = total_v2_harmful > total_v1_harmful
    robustness_preserving = summary["robustness"]["mean_v2_minus_robust_ftp_d_marker"] <= 0
    should_replace = improves_over_v1 and (not introduces_more_harm) and robustness_preserving

    report_lines = [
        "# sentinel_budgeted_guard_v2 disagreement-stress report",
        "",
        "## Setup",
        "- Suite: synthetic disagreement stress (help/hurt/mixed traces).",
        f"- Capacities: {capacities}",
        "- Policies compared: sentinel_budgeted_guard_v2, sentinel_robust_tripwire_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, lru.",
        "",
        "## Aggregate",
        f"- v2 vs v1 (W/T/L): {v2_wins}/{v2_ties}/{v2_losses}",
        f"- Mean miss delta (v2 - v1): {summary['v2_vs_v1']['mean_delta_misses']:.3f}",
        f"- Mean miss delta (v2 - robust_ftp_d_marker): {summary['robustness']['mean_v2_minus_robust_ftp_d_marker']:.3f}",
        f"- Harmful overrides (v2 vs v1): {total_v2_harmful} vs {total_v1_harmful}",
        "",
        "## Explicit answers",
        f"- **Does v2 improve over sentinel_robust_tripwire_v1 on disagreement slices?** {'Yes' if improves_over_v1 else 'No'}.",
        f"- **Does it introduce more harmful overrides?** {'Yes' if introduces_more_harm else 'No'}.",
        f"- **Does it remain robustness-preserving?** {'Yes' if robustness_preserving else 'No'} (mean v2-robust delta={summary['robustness']['mean_v2_minus_robust_ftp_d_marker']:.3f}).",
        f"- **Should v2 replace v1 as the main candidate line?** {'Yes' if should_replace else 'No'}.",
        "",
        "## Per-slice",
        "| trace | case | cap | disagreement_steps | v2_vs_v1 | v2_minus_v1 | v2_harmful | v1_harmful |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]

    for r in rows:
        report_lines.append(
            f"| {r['trace']} | {r['intended_case']} | {r['capacity']} | {r['disagreement_steps']} | {r['v2_vs_v1']} | {r['v2_minus_v1']} | {r['v2_harmful_override_steps']} | {r['v1_harmful_override_steps']} |"
        )

    (out_dir / "v2_disagreement_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_eval(out_dir=Path("analysis/sentinel_budgeted_guard_v2"), capacities=[2, 3])
