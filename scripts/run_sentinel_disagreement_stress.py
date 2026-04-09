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


def run_stress(capacities: List[int], out_dir: Path) -> None:
    traces = _stress_traces()
    policies = {
        "sentinel_robust_tripwire_v1": lambda: SentinelRobustTripwireV1Policy(),
        "robust_ftp_d_marker": lambda: RobustFtPDeterministicMarkerCombiner(),
        "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
        "lru": lambda: LRUPolicy(),
    }

    rows: List[Dict[str, object]] = []

    for tr in traces:
        for cap in capacities:
            requests, pages = _attach_predicted_caches(tr.page_ids, tr.predicted_caches)
            disagreement = _disagreement_count(requests, pages, cap)

            run_rows: Dict[str, Dict[str, object]] = {}
            sentinel_summary: Dict[str, float] = {}
            harmful_overrides = helpful_overrides = 0

            for policy_name, factory in policies.items():
                result = run_policy(factory(), requests, pages, cap)
                run_rows[policy_name] = {
                    "misses": int(result.total_misses),
                    "hits": int(result.total_hits),
                }
                if policy_name == "sentinel_robust_tripwire_v1":
                    sdiag = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
                    sentinel_summary = dict(sdiag.get("summary", {}))
                    for step in sdiag.get("step_log", []):
                        if step.get("chosen_line") != "predictor":
                            continue
                        if bool(step.get("predictor_hit")) and (not bool(step.get("robust_hit"))):
                            helpful_overrides += 1
                        if (not bool(step.get("predictor_hit"))) and bool(step.get("robust_hit")):
                            harmful_overrides += 1

            s_miss = run_rows["sentinel_robust_tripwire_v1"]["misses"]
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
                    "sentinel_misses": s_miss,
                    "robust_ftp_d_marker_misses": r_miss,
                    "blind_oracle_lru_combiner_misses": b_miss,
                    "lru_misses": l_miss,
                    "sentinel_minus_robust_ftp_d_marker": s_miss - r_miss,
                    "sentinel_minus_blind_oracle_lru_combiner": s_miss - b_miss,
                    "sentinel_minus_lru": s_miss - l_miss,
                    "sentinel_vs_robust_ftp_d_marker": "win" if s_miss < r_miss else ("tie" if s_miss == r_miss else "loss"),
                    "sentinel_predictor_steps": float(sentinel_summary.get("predictor_steps", 0.0)),
                    "sentinel_predictor_coverage": float(sentinel_summary.get("predictor_coverage", 0.0)),
                    "sentinel_guard_triggers": float(sentinel_summary.get("guard_triggers", 0.0)),
                    "helpful_override_steps": helpful_overrides,
                    "harmful_override_steps": harmful_overrides,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "disagreement_stress_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    wins = sum(1 for r in rows if r["sentinel_vs_robust_ftp_d_marker"] == "win")
    ties = sum(1 for r in rows if r["sentinel_vs_robust_ftp_d_marker"] == "tie")
    losses = sum(1 for r in rows if r["sentinel_vs_robust_ftp_d_marker"] == "loss")

    summary = {
        "slices": len(rows),
        "trace_count": len(traces),
        "capacities": capacities,
        "disagreement": {
            "mean_steps": mean(float(r["disagreement_steps"]) for r in rows),
            "min_steps": min(int(r["disagreement_steps"]) for r in rows),
            "max_steps": max(int(r["disagreement_steps"]) for r in rows),
            "mean_fraction": mean(float(r["disagreement_fraction"]) for r in rows),
            "slices_with_disagreement": sum(1 for r in rows if int(r["disagreement_steps"]) > 0),
        },
        "sentinel_vs_robust_ftp_d_marker": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_delta_misses": mean(float(r["sentinel_minus_robust_ftp_d_marker"]) for r in rows),
        },
        "override_activity": {
            "mean_predictor_steps": mean(float(r["sentinel_predictor_steps"]) for r in rows),
            "mean_predictor_coverage": mean(float(r["sentinel_predictor_coverage"]) for r in rows),
            "total_helpful_override_steps": sum(int(r["helpful_override_steps"]) for r in rows),
            "total_harmful_override_steps": sum(int(r["harmful_override_steps"]) for r in rows),
            "slices_with_harmful_overrides": sum(1 for r in rows if int(r["harmful_override_steps"]) > 0),
            "slices_with_helpful_overrides": sum(1 for r in rows if int(r["helpful_override_steps"]) > 0),
        },
        "per_slice": rows,
    }

    summary_path = out_dir / "disagreement_stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Sentinel disagreement-stress report",
        "",
        "## Setup",
        "- Small synthetic suite purpose-built to induce disagreement between robust and predictor shadows.",
        f"- Traces: {len(traces)} (help/hurt/mixed), capacities: {capacities}, total slices: {len(rows)}.",
        "- Policies compared: sentinel_robust_tripwire_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, lru.",
        "",
        "## Aggregate results",
        f"- Sentinel vs robust_ftp_d_marker (W/T/L): {wins}/{ties}/{losses}.",
        f"- Mean miss delta (sentinel - robust_ftp_d_marker): {summary['sentinel_vs_robust_ftp_d_marker']['mean_delta_misses']:.3f}.",
        f"- Sentinel vs blind_oracle_lru_combiner mean delta: {mean(float(r['sentinel_minus_blind_oracle_lru_combiner']) for r in rows):.3f}.",
        f"- Sentinel vs lru mean delta: {mean(float(r['sentinel_minus_lru']) for r in rows):.3f}.",
        "",
        "## Explicit answers",
        f"- **Does the suite generate disagreement states?** Yes. Disagreement appears on {summary['disagreement']['slices_with_disagreement']}/{len(rows)} slices, mean {summary['disagreement']['mean_steps']:.2f} steps per slice.",
        f"- **Does sentinel produce wins over robust_ftp_d_marker here?** {('Yes' if wins > 0 else 'No')} ({wins} wins, {losses} losses).",
        f"- **Does sentinel introduce harmful overrides?** {'Yes' if summary['override_activity']['total_harmful_override_steps'] > 0 else 'No'} (harmful override steps={summary['override_activity']['total_harmful_override_steps']}, helpful={summary['override_activity']['total_helpful_override_steps']}).",
        "- **Is mechanism worth refining further?** "
        + (
            "Yes, cautiously: disagreement is now observable and override behavior can be studied under stress; continue refinement if wins appear without rising harmful overrides."
            if wins >= losses
            else "Not yet: harmful behavior currently outweighs gains; redesign/refinement needed before promotion."
        ),
        "",
        "## Per-slice table",
        "| trace | case | cap | disagreement_steps | sentinel_vs_robust_ftp_d_marker | sentinel_minus_robust | predictor_steps | helpful_overrides | harmful_overrides |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]

    for r in rows:
        report_lines.append(
            f"| {r['trace']} | {r['intended_case']} | {r['capacity']} | {r['disagreement_steps']} | {r['sentinel_vs_robust_ftp_d_marker']} | {r['sentinel_minus_robust_ftp_d_marker']} | {r['sentinel_predictor_steps']:.1f} | {r['helpful_override_steps']} | {r['harmful_override_steps']} |"
        )

    report_path = out_dir / "disagreement_stress_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    run_stress(capacities=[2, 3], out_dir=Path("analysis/sentinel_disagreement_stress"))
