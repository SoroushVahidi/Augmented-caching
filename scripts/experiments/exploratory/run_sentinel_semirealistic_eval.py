from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Sequence, Tuple

from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.lru import LRUPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.robust_ftp_marker_combiner import (
    FollowPredictedCachePolicy,
    RobustFtPDeterministicMarkerCombiner,
)
from lafc.policies.sentinel_robust_tripwire_v1 import SentinelRobustTripwireV1Policy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Request


@dataclass(frozen=True)
class SemiRealisticTrace:
    name: str
    page_ids: List[str]
    note: str


def _phase_locality_trace(seed: int, n: int = 240) -> SemiRealisticTrace:
    rng = random.Random(seed)
    pages = [chr(ord("A") + i) for i in range(14)]
    phase_hotsets = [
        pages[0:5],
        pages[3:9],
        pages[1:6],
        pages[7:12],
    ]
    out: List[str] = []
    for t in range(n):
        phase = (t // 60) % len(phase_hotsets)
        hot = phase_hotsets[phase]
        if rng.random() < 0.82:
            out.append(rng.choice(hot))
        elif rng.random() < 0.75:
            # Transitional requests from neighboring phase sets.
            out.append(rng.choice(phase_hotsets[(phase + 1) % len(phase_hotsets)]))
        else:
            out.append(rng.choice(pages))
    return SemiRealisticTrace(
        name="semi::phase_locality_drift",
        page_ids=out,
        note="Rotating hot sets with moderate transition overlap and rare off-hot requests.",
    )


def _burst_and_scan_trace(seed: int, n: int = 240) -> SemiRealisticTrace:
    rng = random.Random(seed)
    pages = [chr(ord("A") + i) for i in range(16)]
    base_hot = pages[0:6]
    out: List[str] = []
    t = 0
    while t < n:
        if (t % 80) in range(55, 67):
            # Short scan burst (semi-realistic cache pollution pattern).
            span = pages[8:16]
            out.append(span[(t - 55) % len(span)])
        else:
            if rng.random() < 0.78:
                out.append(rng.choice(base_hot))
            elif rng.random() < 0.65:
                out.append(rng.choice(pages[4:10]))
            else:
                out.append(rng.choice(pages))
        t += 1
    return SemiRealisticTrace(
        name="semi::locality_with_short_scans",
        page_ids=out,
        note="Stable locality with periodic short scans that transiently perturb the cache.",
    )


def _daypart_mix_trace(seed: int, n: int = 240) -> SemiRealisticTrace:
    rng = random.Random(seed)
    pages = [chr(ord("A") + i) for i in range(15)]
    out: List[str] = []
    for t in range(n):
        bucket = (t // 40) % 6
        if bucket in {0, 3}:
            hot = pages[0:5]
        elif bucket in {1, 4}:
            hot = pages[4:10]
        else:
            hot = pages[2:8]

        if rng.random() < 0.80:
            out.append(rng.choice(hot))
        elif rng.random() < 0.70:
            out.append(rng.choice(pages[9:15]))
        else:
            out.append(rng.choice(pages))

        # Occasional immediate repeat (bursty endpoint behavior).
        if (t % 37) == 0 and len(out) < n and rng.random() < 0.6:
            out.append(out[-1])
    return SemiRealisticTrace(
        name="semi::daypart_popularity_mix",
        page_ids=out[:n],
        note="Daypart-like popularity shifts with repeats and long-tail traffic.",
    )


def _build_semirealistic_traces() -> List[SemiRealisticTrace]:
    return [
        _phase_locality_trace(seed=17),
        _burst_and_scan_trace(seed=29),
        _daypart_mix_trace(seed=41),
    ]


def _predicted_caches_from_recent_history(page_ids: Sequence[str], capacity: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed + capacity * 11)
    universe = sorted(set(page_ids))
    recent_window = 28
    prev_cache: List[str] = []
    pred: List[List[str]] = []

    for t in range(len(page_ids)):
        start = max(0, t - recent_window)
        hist = page_ids[start:t]
        freq: Dict[str, int] = {p: 0 for p in universe}
        recency: Dict[str, int] = {p: 10_000 for p in universe}

        for i, p in enumerate(hist):
            freq[p] += 1
            recency[p] = len(hist) - i

        scored = sorted(
            universe,
            key=lambda p: (-(0.75 * freq[p] + 0.25 * (1.0 / (1.0 + recency[p]))), p),
        )
        candidate = list(scored[:capacity])

        # Lag updates occasionally (predictor staleness).
        if prev_cache and rng.random() < 0.18:
            candidate = list(prev_cache)

        # Mild corruption: swap one slot with random non-member.
        if rng.random() < 0.17 and capacity > 0:
            non_members = [p for p in universe if p not in candidate]
            if non_members:
                idx = rng.randrange(capacity)
                candidate[idx] = rng.choice(non_members)
                candidate = sorted(set(candidate))
                while len(candidate) < capacity:
                    candidate.append(rng.choice(universe))
                candidate = candidate[:capacity]

        prev_cache = list(candidate)
        pred.append(candidate)

    return pred


def _attach_predicted_caches(page_ids: Sequence[str], predicted_caches: Sequence[Sequence[str]]) -> Tuple[List[Request], Dict[str, object]]:
    reqs, pages = build_requests_from_lists(list(page_ids))
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


def run_semirealistic_eval(out_dir: Path, capacities: List[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = _build_semirealistic_traces()
    policies: Dict[str, Callable[[], object]] = {
        "sentinel_robust_tripwire_v1": lambda: SentinelRobustTripwireV1Policy(),
        "robust_ftp_d_marker": lambda: RobustFtPDeterministicMarkerCombiner(),
        "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
        "rest_v1": lambda: RestV1Policy(),
        "atlas_v3": lambda: AtlasV3Policy(),
        "lru": lambda: LRUPolicy(),
    }

    rows: List[Dict[str, object]] = []
    slice_rows: List[Dict[str, object]] = []

    for tr_i, tr in enumerate(traces):
        for cap in capacities:
            pred = _predicted_caches_from_recent_history(tr.page_ids, capacity=cap, seed=101 + tr_i)
            requests, pages = _attach_predicted_caches(tr.page_ids, pred)
            disagreement_steps = _disagreement_count(requests, pages, cap)

            per_policy: Dict[str, int] = {}
            sentinel_harmful = 0
            sentinel_helpful = 0
            sentinel_predictor_steps = 0.0

            for policy_name, factory in policies.items():
                result = run_policy(factory(), requests, pages, cap)
                misses = int(result.total_misses)
                per_policy[policy_name] = misses

                if policy_name == "sentinel_robust_tripwire_v1":
                    sdiag = (result.extra_diagnostics or {}).get("sentinel_robust_tripwire_v1", {})
                    summary = sdiag.get("summary", {})
                    sentinel_predictor_steps = float(summary.get("predictor_steps", 0.0))
                    for step in sdiag.get("step_log", []):
                        if step.get("chosen_line") != "predictor":
                            continue
                        if bool(step.get("predictor_hit")) and (not bool(step.get("robust_hit"))):
                            sentinel_helpful += 1
                        if (not bool(step.get("predictor_hit"))) and bool(step.get("robust_hit")):
                            sentinel_harmful += 1

                rows.append(
                    {
                        "trace": tr.name,
                        "trace_note": tr.note,
                        "capacity": cap,
                        "requests": len(requests),
                        "disagreement_steps": disagreement_steps,
                        "policy": policy_name,
                        "misses": misses,
                        "hits": int(result.total_hits),
                        "hit_rate": float(result.total_hits) / float(len(requests)),
                    }
                )

            s = per_policy["sentinel_robust_tripwire_v1"]
            r = per_policy["robust_ftp_d_marker"]
            b = per_policy["blind_oracle_lru_combiner"]
            a = per_policy["atlas_v3"]
            re = per_policy["rest_v1"]
            l = per_policy["lru"]
            slice_rows.append(
                {
                    "trace": tr.name,
                    "capacity": cap,
                    "disagreement_steps": disagreement_steps,
                    "sentinel_minus_robust": s - r,
                    "sentinel_minus_blind_oracle_lru_combiner": s - b,
                    "sentinel_minus_rest_v1": s - re,
                    "sentinel_minus_atlas_v3": s - a,
                    "sentinel_minus_lru": s - l,
                    "sentinel_vs_robust": "win" if s < r else ("tie" if s == r else "loss"),
                    "sentinel_helpful_override_steps": sentinel_helpful,
                    "sentinel_harmful_override_steps": sentinel_harmful,
                    "sentinel_predictor_steps": sentinel_predictor_steps,
                }
            )

    csv_path = out_dir / "semirealistic_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    wins = sum(1 for r in slice_rows if r["sentinel_vs_robust"] == "win")
    ties = sum(1 for r in slice_rows if r["sentinel_vs_robust"] == "tie")
    losses = sum(1 for r in slice_rows if r["sentinel_vs_robust"] == "loss")

    mean_by_policy: Dict[str, float] = {}
    for name in policies.keys():
        pr = [r for r in rows if r["policy"] == name]
        mean_by_policy[name] = mean(float(r["misses"]) for r in pr)

    robustness_preserving = mean(float(r["sentinel_minus_robust"]) for r in slice_rows) <= 0.0
    best_policy = min(mean_by_policy, key=mean_by_policy.get)

    summary = {
        "suite": "sentinel_semirealistic_eval",
        "trace_count": len(traces),
        "capacities": capacities,
        "slice_count": len(slice_rows),
        "disagreement": {
            "mean_steps": mean(float(r["disagreement_steps"]) for r in slice_rows),
            "min_steps": min(int(r["disagreement_steps"]) for r in slice_rows),
            "max_steps": max(int(r["disagreement_steps"]) for r in slice_rows),
            "slices_with_disagreement": sum(1 for r in slice_rows if int(r["disagreement_steps"]) > 0),
        },
        "sentinel_vs_robust": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_delta_misses": mean(float(r["sentinel_minus_robust"]) for r in slice_rows),
        },
        "sentinel_vs_others": {
            "mean_delta_vs_blind_oracle_lru_combiner": mean(float(r["sentinel_minus_blind_oracle_lru_combiner"]) for r in slice_rows),
            "mean_delta_vs_rest_v1": mean(float(r["sentinel_minus_rest_v1"]) for r in slice_rows),
            "mean_delta_vs_atlas_v3": mean(float(r["sentinel_minus_atlas_v3"]) for r in slice_rows),
            "mean_delta_vs_lru": mean(float(r["sentinel_minus_lru"]) for r in slice_rows),
        },
        "override_activity": {
            "total_helpful_overrides": sum(int(r["sentinel_helpful_override_steps"]) for r in slice_rows),
            "total_harmful_overrides": sum(int(r["sentinel_harmful_override_steps"]) for r in slice_rows),
            "mean_predictor_steps": mean(float(r["sentinel_predictor_steps"]) for r in slice_rows),
        },
        "mean_misses_by_policy": mean_by_policy,
        "best_policy_by_mean_misses": best_policy,
        "is_robustness_preserving": robustness_preserving,
    }

    summary_path = out_dir / "semirealistic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    has_real_wins = wins > 0
    still_main_candidate = best_policy == "sentinel_robust_tripwire_v1"

    report_lines = [
        "# Sentinel semi-realistic targeted evaluation",
        "",
        "## Setup",
        "- Goal: bridge ordinary lightweight suite and fully synthetic disagreement-stress suite.",
        "- Traces are generated with locality drift, short scan bursts, and daypart popularity shifts; predicted caches are recency-frequency based with lag/noise.",
        f"- Traces: {len(traces)}, capacities: {capacities}, slices: {len(slice_rows)}.",
        "- Policies: sentinel_robust_tripwire_v1, robust_ftp_d_marker, blind_oracle_lru_combiner, rest_v1, atlas_v3, lru.",
        "",
        "## Aggregate",
        f"- Sentinel vs robust_ftp_d_marker (W/T/L): {wins}/{ties}/{losses}.",
        f"- Mean delta misses (sentinel - robust_ftp_d_marker): {summary['sentinel_vs_robust']['mean_delta_misses']:.3f}.",
        f"- Mean misses by policy: {mean_by_policy}.",
        f"- Helpful/harmful overrides: {summary['override_activity']['total_helpful_overrides']}/{summary['override_activity']['total_harmful_overrides']}.",
        "",
        "## Explicit answers",
        f"- **Does v1 show any real wins in this semi-realistic setting?** {'Yes' if has_real_wins else 'No'} ({wins} wins).",
        f"- **Does it preserve robustness?** {'Yes' if robustness_preserving else 'No'} (mean sentinel-robust delta={summary['sentinel_vs_robust']['mean_delta_misses']:.3f}).",
        f"- **Is it still the best main empirical candidate in the repo?** {'Yes' if still_main_candidate else 'No'} (best mean-miss policy here: {best_policy}).",
        "",
        "## Per-slice sentinel vs robust",
        "| trace | cap | disagreement_steps | sentinel_vs_robust | sentinel_minus_robust | helpful_overrides | harmful_overrides |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in slice_rows:
        report_lines.append(
            f"| {r['trace']} | {r['capacity']} | {r['disagreement_steps']} | {r['sentinel_vs_robust']} | {r['sentinel_minus_robust']} | {r['sentinel_helpful_override_steps']} | {r['sentinel_harmful_override_steps']} |"
        )

    report_path = out_dir / "semirealistic_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_semirealistic_eval(Path("analysis/sentinel_semirealistic_eval"), capacities=[3, 4])
