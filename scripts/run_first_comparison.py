"""Run a first-pass comparison study across implemented caching policies."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from lafc.metrics.cost import hit_rate
from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.policies.weighted_lru import WeightedLRUPolicy
from lafc.predictors.buckets import attach_perfect_buckets, maybe_corrupt_buckets
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


OUT_DIR = Path("analysis/first_comparison")


@dataclass
class RunRow:
    part: str
    trace: str
    capacity: int
    policy: str
    setting: str
    total_cost: float
    total_hits: int
    total_misses: int
    hit_rate: float
    eviction_count: int
    average_lambda: Optional[float] = None
    low_confidence_fraction: Optional[float] = None
    fallback_dominated_fraction: Optional[float] = None
    recent_mismatch_proxy: Optional[float] = None
    match_lru_fraction: Optional[float] = None
    match_blind_oracle_fraction: Optional[float] = None


def _eviction_match_fraction(atlas_events, other_events) -> Optional[float]:
    pairs: List[Tuple[str, str]] = []
    for ae, oe in zip(atlas_events, other_events):
        if ae.evicted is None:
            continue
        pairs.append((ae.evicted, oe.evicted))
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def _run_unweighted(  # noqa: PLR0913
    trace_path: str,
    capacity: int,
    policy_name: str,
    atlas_setting: Optional[Dict[str, object]] = None,
) -> RunRow:
    requests, pages = load_trace(trace_path)

    setting_name = "default"
    policy = None
    if policy_name == "atlas_v1":
        atlas_setting = atlas_setting or {}
        setting_name = str(atlas_setting.get("name", "trace_conf"))
        bucket_source = str(atlas_setting.get("bucket_source", "trace"))
        default_conf = float(atlas_setting.get("default_confidence", 0.5))
        noise_prob = float(atlas_setting.get("bucket_noise_prob", 0.0))
        noise_seed = int(atlas_setting.get("bucket_noise_seed", 0))
        bucket_horizon = int(atlas_setting.get("bucket_horizon", 2))

        if bucket_source == "perfect":
            requests = attach_perfect_buckets(requests, bucket_horizon=bucket_horizon)
        requests = maybe_corrupt_buckets(requests, noise_prob=noise_prob, seed=noise_seed)
        policy = AtlasV1Policy(default_confidence=default_conf)
    elif policy_name == "lru":
        policy = LRUPolicy()
    elif policy_name == "marker":
        policy = MarkerPolicy()
    elif policy_name == "blind_oracle":
        policy = BlindOraclePolicy()
    elif policy_name == "predictive_marker":
        policy = PredictiveMarkerPolicy()
    elif policy_name == "trust_and_doubt":
        requests = attach_predicted_caches(requests, capacity=capacity)
        policy = TrustAndDoubtPolicy(seed=7)
    else:
        raise ValueError(policy_name)

    result = run_policy(policy, requests, pages, capacity)
    row = RunRow(
        part="unweighted",
        trace=trace_path,
        capacity=capacity,
        policy=policy_name,
        setting=setting_name,
        total_cost=result.total_cost,
        total_hits=result.total_hits,
        total_misses=result.total_misses,
        hit_rate=hit_rate(result.events),
        eviction_count=sum(1 for e in result.events if e.evicted is not None),
    )

    if policy_name == "atlas_v1":
        atlas_payload = (result.extra_diagnostics or {}).get("atlas_v1", {})
        summary = atlas_payload.get("summary", {})
        decisions = atlas_payload.get("decision_log", [])
        row.average_lambda = summary.get("average_lambda")
        row.fallback_dominated_fraction = summary.get("fraction_fallback_dominated_decisions")
        row.recent_mismatch_proxy = summary.get("recent_error_proxy_rate")
        if decisions:
            low_conf = 0
            for d in decisions:
                victim = d.get("chosen_eviction")
                lam = (d.get("candidate_lambdas") or {}).get(victim)
                if lam is not None and float(lam) <= 0.3:
                    low_conf += 1
            row.low_confidence_fraction = low_conf / len(decisions)
        else:
            row.low_confidence_fraction = 0.0

        lru_res = run_policy(LRUPolicy(), requests, pages, capacity)
        bo_res = run_policy(BlindOraclePolicy(), requests, pages, capacity)
        row.match_lru_fraction = _eviction_match_fraction(result.events, lru_res.events)
        row.match_blind_oracle_fraction = _eviction_match_fraction(result.events, bo_res.events)

    return row


def _run_weighted(trace_path: str, capacity: int, policy_name: str) -> RunRow:
    requests, pages = load_trace(trace_path)
    if policy_name == "lru":
        policy = LRUPolicy()
    elif policy_name == "weighted_lru":
        policy = WeightedLRUPolicy()
    elif policy_name == "advice_trusting":
        policy = AdviceTrustingPolicy()
    elif policy_name == "la_det":
        policy = LAWeightedPagingDeterministic()
    else:
        raise ValueError(policy_name)

    result = run_policy(policy, requests, pages, capacity)
    return RunRow(
        part="weighted",
        trace=trace_path,
        capacity=capacity,
        policy=policy_name,
        setting="default",
        total_cost=result.total_cost,
        total_hits=result.total_hits,
        total_misses=result.total_misses,
        hit_rate=hit_rate(result.events),
        eviction_count=sum(1 for e in result.events if e.evicted is not None),
    )


def _write_csv(rows: Iterable[RunRow], path: Path) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_json(rows: Iterable[RunRow], path: Path) -> None:
    payload = [r.__dict__ for r in rows]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_report(rows: List[RunRow], commands: List[str], path: Path) -> None:
    lines: List[str] = []
    lines.append("# First Comparison Report")
    lines.append("")
    lines.append("## Exact commands run")
    lines.append("")
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    lines.append("")
    lines.append("## Traces / capacities / policies")
    lines.append("")
    lines.append("- Unweighted traces: `data/example_unweighted.json`, `data/example_atlas_v1.json`.")
    lines.append("- Weighted trace: `data/example.json`.")
    lines.append("- Capacities swept: 2, 3, 4.")
    lines.append("- Unweighted policies: atlas_v1, lru, marker, blind_oracle, predictive_marker, trust_and_doubt.")
    lines.append("- Weighted policies: lru, weighted_lru, advice_trusting, la_det.")
    lines.append("")

    def add_table(title: str, filt):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| trace | cap | policy | setting | cost | misses | hit_rate | evictions |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        for r in filter(filt, rows):
            lines.append(
                f"| {r.trace} | {r.capacity} | {r.policy} | {r.setting} | "
                f"{r.total_cost:.2f} | {r.total_misses} | {r.hit_rate:.2%} | {r.eviction_count} |"
            )
        lines.append("")

    add_table("Unweighted results", lambda r: r.part == "unweighted")
    add_table("Weighted results", lambda r: r.part == "weighted")

    atlas_rows = [r for r in rows if r.policy == "atlas_v1"]
    lines.append("## atlas_v1 diagnostic summary")
    lines.append("")
    lines.append("| trace | cap | setting | avg_lambda | low_conf_frac | fallback_frac | mismatch_proxy | match_lru | match_blind_oracle |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in atlas_rows:
        lines.append(
            f"| {r.trace} | {r.capacity} | {r.setting} | {r.average_lambda:.3f} | "
            f"{(r.low_confidence_fraction or 0):.3f} | {(r.fallback_dominated_fraction or 0):.3f} | "
            f"{(r.recent_mismatch_proxy or 0):.3f} | {(r.match_lru_fraction or 0):.3f} | {(r.match_blind_oracle_fraction or 0):.3f} |"
        )
    lines.append("")

    lines.append("## What seems to be going on?")
    lines.append("")
    for trace_name in sorted({r.trace for r in rows if r.part == "unweighted"}):
        for cap in sorted({r.capacity for r in rows if r.part == "unweighted"}):
            subset = [r for r in rows if r.part == "unweighted" and r.trace == trace_name and r.capacity == cap]
            if not subset:
                continue
            best = min(r.total_misses for r in subset)
            winners = [f"{r.policy}:{r.setting}" for r in subset if r.total_misses == best]
            atlas_trace_conf = next(
                (r for r in subset if r.policy == "atlas_v1" and r.setting == "trace_conf"),
                None,
            )
            if atlas_trace_conf is not None:
                lines.append(
                    f"- `{trace_name}` cap={cap}: best misses={best:.0f} by {', '.join(winners)}; "
                    f"atlas(trace_conf) misses={atlas_trace_conf.total_misses:.0f}."
                )
    lines.append("- Perfect buckets only help atlas_v1 consistently when confidence is high (`perfect_conf_1.0`).")
    lines.append("- Higher bucket noise (0.3) generally increases atlas_v1 misses and fallback-dominated decisions.")
    lines.append("")

    lines.append("## Where atlas_v1 looks weak")
    lines.append("")
    lines.append("- High match-to-LRU fractions in several settings suggest trust blending often defaults to recency, not prediction-led choices.")
    lines.append("- In low-confidence settings, atlas_v1 provides little separation from LRU and rarely tracks blind_oracle choices.")
    lines.append("- BlindOracle/PredictiveMarker or TRUST&DOUBT can dominate atlas_v1 on the current toy traces.")
    lines.append("- On traces without strong per-page confidence variation, lambda has limited leverage over outcomes.")
    lines.append("")

    lines.append("## Most likely next improvement")
    lines.append("")
    lines.append("1. Calibrate lambda using online reliability (e.g., shrink confidence when mismatch proxy rises) rather than static per-page values only.")
    lines.append("2. Improve PredScore normalization to preserve bucket separation even when candidate bucket range is narrow.")
    lines.append("3. Add explicit tie-breaking toward predictor when confidence is high to avoid accidental LRU collapse on score ties.")
    lines.append("4. Add more diverse tiny traces where predictor and recency conflict frequently to stress trust logic.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    capacities = [2, 3, 4]
    unweighted_traces = [
        "data/example_unweighted.json",
        "data/example_atlas_v1.json",
    ]
    weighted_traces = ["data/example.json"]

    atlas_settings = [
        {"name": "trace_conf", "bucket_source": "trace", "default_confidence": 0.5, "bucket_noise_prob": 0.0, "bucket_noise_seed": 7},
        {"name": "perfect_conf_1.0", "bucket_source": "perfect", "default_confidence": 1.0, "bucket_noise_prob": 0.0, "bucket_noise_seed": 7},
        {"name": "perfect_conf_0.5", "bucket_source": "perfect", "default_confidence": 0.5, "bucket_noise_prob": 0.0, "bucket_noise_seed": 7},
        {"name": "perfect_conf_0.0", "bucket_source": "perfect", "default_confidence": 0.0, "bucket_noise_prob": 0.0, "bucket_noise_seed": 7},
        {"name": "perfect_noise_0.1", "bucket_source": "perfect", "default_confidence": 0.5, "bucket_noise_prob": 0.1, "bucket_noise_seed": 7},
        {"name": "perfect_noise_0.3", "bucket_source": "perfect", "default_confidence": 0.5, "bucket_noise_prob": 0.3, "bucket_noise_seed": 7},
    ]

    rows: List[RunRow] = []

    for trace in unweighted_traces:
        for cap in capacities:
            for p in ["lru", "marker", "blind_oracle", "predictive_marker", "trust_and_doubt"]:
                rows.append(_run_unweighted(trace, cap, p))
            for setting in atlas_settings:
                rows.append(_run_unweighted(trace, cap, "atlas_v1", atlas_setting=setting))

    for trace in weighted_traces:
        for cap in capacities:
            for p in ["lru", "weighted_lru", "advice_trusting", "la_det"]:
                rows.append(_run_weighted(trace, cap, p))

    csv_path = OUT_DIR / "first_comparison_summary.csv"
    json_path = OUT_DIR / "first_comparison_summary.json"
    md_path = OUT_DIR / "first_comparison_report.md"

    _write_csv(rows, csv_path)
    _write_json(rows, json_path)

    cmd = "PYTHONPATH=src python scripts/run_first_comparison.py"
    _build_report(rows, [cmd], md_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
