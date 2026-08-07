from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Sequence, Tuple

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
        if robust.on_request(req).evicted != predictor.on_request(req).evicted:
            disagree += 1
    return disagree


def _v1_harmful_helpful(result) -> Tuple[int, int]:
    diag = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
    harmful = 0
    helpful = 0
    for step in diag.get("step_log", []):
        if step.get("chosen_line") != "predictor":
            continue
        if bool(step.get("predictor_hit")) and (not bool(step.get("robust_hit"))):
            helpful += 1
        if (not bool(step.get("predictor_hit"))) and bool(step.get("robust_hit")):
            harmful += 1
    return harmful, helpful


def run_ablation(out_dir: Path, capacities: List[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: Dict[str, Callable[[], object]] = {
        "v1_baseline": lambda: SentinelRobustTripwireV1Policy(),
        "v2_full": lambda: SentinelBudgetedGuardV2Policy(),
        "v2_no_override_budget": lambda: SentinelBudgetedGuardV2Policy(override_budget_total=10_000),
        "v2_no_temporary_guard": lambda: SentinelBudgetedGuardV2Policy(guard_trigger_threshold=10_000),
        "v2_no_reentry_gating": lambda: SentinelBudgetedGuardV2Policy(reentry_stable_steps=0, reentry_memory_threshold=10_000),
        "v2_budget_only": lambda: SentinelBudgetedGuardV2Policy(
            guard_trigger_threshold=10_000,
            reentry_stable_steps=0,
            reentry_memory_threshold=10_000,
        ),
        "v2_guard_only": lambda: SentinelBudgetedGuardV2Policy(
            override_budget_total=10_000,
            reentry_stable_steps=0,
            reentry_memory_threshold=10_000,
        ),
        "v2_reentry_only": lambda: SentinelBudgetedGuardV2Policy(
            override_budget_total=10_000,
            guard_trigger_threshold=10_000,
        ),
    }

    traces = _stress_traces()
    rows: List[Dict[str, object]] = []

    for tr in traces:
        for cap in capacities:
            requests, pages = _attach_predicted_caches(tr.page_ids, tr.predicted_caches)
            disagreement_steps = _disagreement_count(requests, pages, cap)

            per_variant: Dict[str, Dict[str, float]] = {}
            for name, factory in variants.items():
                result = run_policy(factory(), requests, pages, cap)
                harmful, helpful = 0, 0
                predictor_steps = 0.0
                if name == "v1_baseline":
                    harmful, helpful = _v1_harmful_helpful(result)
                    v1diag = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
                    predictor_steps = float(v1diag.get("summary", {}).get("predictor_steps", 0.0))
                else:
                    diag = (result.extra_diagnostics or {}).get("sentinel_budgeted_guard_v2", {})
                    summary = diag.get("summary", {})
                    harmful = int(summary.get("harmful_override_steps", 0.0))
                    helpful = int(summary.get("helpful_override_steps", 0.0))
                    predictor_steps = float(summary.get("predictor_steps", 0.0))
                per_variant[name] = {
                    "misses": float(result.total_misses),
                    "harmful": float(harmful),
                    "helpful": float(helpful),
                    "predictor_steps": predictor_steps,
                }

            v1_miss = per_variant["v1_baseline"]["misses"]
            v2_miss = per_variant["v2_full"]["misses"]
            for name, metrics in per_variant.items():
                rows.append(
                    {
                        "trace": tr.name,
                        "intended_case": tr.intended_case,
                        "capacity": cap,
                        "disagreement_steps": disagreement_steps,
                        "variant": name,
                        "misses": int(metrics["misses"]),
                        "miss_delta_vs_v1": float(metrics["misses"] - v1_miss),
                        "miss_delta_vs_v2_full": float(metrics["misses"] - v2_miss),
                        "harmful_override_steps": int(metrics["harmful"]),
                        "helpful_override_steps": int(metrics["helpful"]),
                        "predictor_steps": float(metrics["predictor_steps"]),
                    }
                )

    csv_path = out_dir / "v2_ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_variant: Dict[str, Dict[str, float]] = {}
    for name in variants.keys():
        vr = [r for r in rows if r["variant"] == name]
        by_variant[name] = {
            "mean_misses": mean(float(r["misses"]) for r in vr),
            "mean_delta_vs_v1": mean(float(r["miss_delta_vs_v1"]) for r in vr),
            "mean_delta_vs_v2_full": mean(float(r["miss_delta_vs_v2_full"]) for r in vr),
            "total_harmful_overrides": float(sum(int(r["harmful_override_steps"]) for r in vr)),
            "total_helpful_overrides": float(sum(int(r["helpful_override_steps"]) for r in vr)),
            "mean_predictor_steps": mean(float(r["predictor_steps"]) for r in vr),
        }

    removal_variants = {
        "override_budget": "v2_no_override_budget",
        "temporary_guard": "v2_no_temporary_guard",
        "reentry_gating": "v2_no_reentry_gating",
    }
    component_relief = {
        k: by_variant[v]["mean_delta_vs_v2_full"] for k, v in removal_variants.items()
    }
    worst_component = min(component_relief, key=component_relief.get)

    single_component_variants = ["v2_budget_only", "v2_guard_only", "v2_reentry_only"]
    best_single = min(single_component_variants, key=lambda x: by_variant[x]["mean_delta_vs_v1"])

    v15_candidates = [
        "v1_baseline",
        "v2_budget_only",
        "v2_guard_only",
        "v2_reentry_only",
        "v2_no_override_budget",
        "v2_no_temporary_guard",
        "v2_no_reentry_gating",
    ]
    best_v15 = min(v15_candidates, key=lambda x: by_variant[x]["mean_misses"])

    summary = {
        "slices": len(rows),
        "trace_count": len(traces),
        "capacities": capacities,
        "variants": by_variant,
        "component_relief_vs_v2_full": component_relief,
        "worst_component_by_removal_test": worst_component,
        "best_single_component_variant": best_single,
        "best_simple_v1_5_candidate": best_v15,
    }
    summary_path = out_dir / "v2_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    useful_single = [
        v for v in single_component_variants if by_variant[v]["mean_delta_vs_v1"] < 0
    ]

    report = [
        "# sentinel_budgeted_guard_v2 component ablation report",
        "",
        "## Setup",
        "- Same lightweight disagreement-stress suite as prior v2 check (3 synthetic traces × capacities [2,3]).",
        "- Variants: v1 baseline, v2 full, v2 without each main component, plus single-component variants.",
        "",
        "## Aggregate by variant",
        "| variant | mean_misses | mean_delta_vs_v1 | mean_delta_vs_v2_full | harmful_overrides | helpful_overrides |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, m in by_variant.items():
        report.append(
            f"| {name} | {m['mean_misses']:.3f} | {m['mean_delta_vs_v1']:.3f} | {m['mean_delta_vs_v2_full']:.3f} | {int(m['total_harmful_overrides'])} | {int(m['total_helpful_overrides'])} |"
        )

    report.extend(
        [
            "",
            "## Explicit answers",
            f"- **Which v2 component hurts most?** `{worst_component}` (removing it gives the largest miss reduction vs v2 full: {component_relief[worst_component]:.3f}).",
            f"- **Is any single v2 component actually useful?** {('Yes: ' + ', '.join(useful_single)) if useful_single else 'No: none of the single-component variants beat v1 on mean misses.'}",
            f"- **Is there a simpler v1.5 variant that keeps the best part of v2 without the harmful parts?** Candidate: `{best_v15}` (best mean misses among simple ablation variants).",
            f"- **What should be the main empirical candidate after this ablation?** `{best_v15}` (lowest mean misses in this ablation suite).",
            "",
            "## Notes",
            "- Negative delta means fewer misses (better).",
            "- Positive delta means more misses (worse).",
        ]
    )

    report_path = out_dir / "v2_ablation_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_ablation(Path("analysis/sentinel_budgeted_guard_v2"), capacities=[2, 3])
