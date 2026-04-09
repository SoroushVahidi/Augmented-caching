from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Tuple

from lafc.metrics.cost import hit_rate
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import Request


OUT_DIR = Path("analysis/lightweight_comparison")


@dataclass(frozen=True)
class TraceSpec:
    name: str
    kind: str
    loader: Callable[[int], Tuple[List[Request], Dict[str, object]]]


@dataclass
class RunRow:
    trace: str
    trace_kind: str
    requests: int
    capacity: int
    policy: str
    misses: int
    hits: int
    hit_rate: float
    scorer_mode: str | None = None
    rel_improvement_vs_lru: float | None = None
    rank: float | None = None
    regret_vs_best: float | None = None
    normalized_gap_to_best: float | None = None


def _compute_actual_next(page_ids: List[str]) -> List[float]:
    last_seen: Dict[str, int] = {}
    out: List[float] = [math.inf] * len(page_ids)
    for t in range(len(page_ids) - 1, -1, -1):
        pid = page_ids[t]
        if pid in last_seen:
            out[t] = float(last_seen[pid])
        last_seen[pid] = t
    return out


def _synthetic_prediction_records(page_ids: List[str], noisy_actual_next: List[float]) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for t, nxt in enumerate(noisy_actual_next):
        dist = 1000 if math.isinf(nxt) else max(0, int(nxt - t))
        if dist <= 2:
            bucket = 0
        elif dist <= 5:
            bucket = 1
        elif dist <= 12:
            bucket = 2
        else:
            bucket = 3
        confidence = max(0.05, min(0.99, 0.95 - 0.08 * abs((t % 5) - 2)))
        records.append({"bucket": bucket, "confidence": confidence})
    return records


def _inject_light_noise(actual_next: List[float]) -> List[float]:
    noisy: List[float] = []
    for t, val in enumerate(actual_next):
        if math.isinf(val):
            noisy.append(float(len(actual_next) + 64 + (t % 7)))
            continue
        jitter = ((t * 3) % 5) - 2
        pred = max(float(t + 1), float(val + jitter))
        noisy.append(pred)
    return noisy


def _enrich_sequence(page_ids: List[str]) -> Tuple[List[Request], Dict[str, object]]:
    actual_next = _compute_actual_next(page_ids)
    predictions = _inject_light_noise(actual_next)
    records = _synthetic_prediction_records(page_ids, predictions)
    return build_requests_from_lists(
        page_ids=page_ids,
        predictions=predictions,
        prediction_records=records,
    )


def _load_json_trace_with_enrichment(path: str, max_requests: int) -> Tuple[List[Request], Dict[str, object]]:
    requests, pages = load_trace(path)
    page_ids = [req.page_id for req in requests[:max_requests]]
    return _enrich_sequence(page_ids)


def _build_trace_specs() -> List[TraceSpec]:
    def make_from_sequence(name: str, page_ids: List[str]) -> TraceSpec:
        return TraceSpec(
            name=name,
            kind="synthetic",
            loader=lambda max_req, seq=page_ids: _enrich_sequence(seq[:max_req]),
        )

    hot_loop = ["A", "B", "C", "D", "E", "A", "B", "F", "A", "C", "B", "A"] * 18
    bursty_scan = (
        ["A", "B", "C", "D", "E", "F", "G", "H"] * 5
        + [f"S{i}" for i in range(1, 41)]
        + ["A", "B", "C", "D", "E", "F", "G", "H"] * 5
    )
    phase_shift = (
        ["P1", "P2", "P3", "P4", "P1", "P2"] * 8
        + ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"] * 8
        + ["P1", "Q1", "P2", "Q2", "P3", "Q3", "R1", "R2"] * 6
    )
    mixed_locality = [
        "H1", "H2", "H3", "H1", "H2", "W1", "H1", "H2", "W2", "H3", "W3", "H1", "W4", "H2", "W5",
    ] * 12

    return [
        TraceSpec(
            name="file::example_unweighted",
            kind="repo_example",
            loader=lambda max_req: _load_json_trace_with_enrichment("data/example_unweighted.json", max_req),
        ),
        TraceSpec(
            name="file::example_atlas_v1",
            kind="repo_example",
            loader=lambda max_req: _load_json_trace_with_enrichment("data/example_atlas_v1.json", max_req),
        ),
        make_from_sequence("synthetic::hot_loop", hot_loop),
        make_from_sequence("synthetic::bursty_scan", bursty_scan),
        make_from_sequence("synthetic::phase_shift", phase_shift),
        make_from_sequence("synthetic::mixed_locality", mixed_locality),
    ]


def _policy_factories() -> Dict[str, Callable[[], object]]:
    return {
        "lru": lambda: LRUPolicy(),
        "blind_oracle": lambda: BlindOraclePolicy(),
        "predictive_marker": lambda: PredictiveMarkerPolicy(),
        "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
        "trust_and_doubt": lambda: TrustAndDoubtPolicy(seed=7),
        "rest_v1": lambda: RestV1Policy(),
        # Force artifact-free local mode so this script remains binary-checkpoint free.
        "evict_value_v1": lambda: EvictValueV1Policy(scorer_mode="lightweight"),
    }


def _dense_rank(values: Dict[str, int]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    rank_map: Dict[str, float] = {}
    current_rank = 1.0
    prev_val: int | None = None
    for idx, (name, val) in enumerate(ordered):
        if prev_val is not None and val != prev_val:
            current_rank = float(idx + 1)
        rank_map[name] = current_rank
        prev_val = val
    return rank_map


def _compute_pairwise_wtl(a: Iterable[int], b: Iterable[int]) -> Dict[str, int]:
    wins = ties = losses = 0
    for x, y in zip(a, b):
        if x < y:
            wins += 1
        elif x > y:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def run_lightweight_comparison(capacities: List[int], max_requests: int, selected_policies: List[str]) -> Dict[str, object]:
    traces = _build_trace_specs()
    policy_factories = _policy_factories()

    rows: List[RunRow] = []
    skipped_policies: Dict[str, str] = {}
    included_policies: List[str] = []

    for trace_spec in traces:
        requests, pages = trace_spec.loader(max_requests)
        for capacity in capacities:
            combo_results: Dict[str, RunRow] = {}
            for policy_name in selected_policies:
                if policy_name in skipped_policies:
                    continue
                try:
                    run_reqs = requests
                    if policy_name == "trust_and_doubt":
                        run_reqs = attach_predicted_caches(requests, capacity=capacity)
                    result = run_policy(policy_factories[policy_name](), run_reqs, pages, capacity)
                    combo_results[policy_name] = RunRow(
                        trace=trace_spec.name,
                        trace_kind=trace_spec.kind,
                        requests=len(run_reqs),
                        capacity=capacity,
                        policy=policy_name,
                        misses=result.total_misses,
                        hits=result.total_hits,
                        hit_rate=hit_rate(result.events),
                        scorer_mode=((result.extra_diagnostics or {}).get("evict_value_v1", {}).get("summary", {}).get("scorer_mode")),
                    )
                    if policy_name not in included_policies:
                        included_policies.append(policy_name)
                except Exception as exc:  # pragma: no cover - runtime-dependent availability.
                    skipped_policies[policy_name] = f"{type(exc).__name__}: {exc}"

            if "lru" not in combo_results:
                continue

            lru_misses = combo_results["lru"].misses
            best_misses = min(r.misses for r in combo_results.values())
            ranks = _dense_rank({name: row.misses for name, row in combo_results.items()})

            for policy_name, row in combo_results.items():
                if lru_misses > 0:
                    row.rel_improvement_vs_lru = (lru_misses - row.misses) / lru_misses
                else:
                    row.rel_improvement_vs_lru = 0.0 if row.misses == 0 else None
                row.rank = ranks[policy_name]
                row.regret_vs_best = float(row.misses - best_misses)
                row.normalized_gap_to_best = 0.0 if best_misses == 0 else (row.misses - best_misses) / best_misses
                rows.append(row)

    by_policy: Dict[str, List[RunRow]] = defaultdict(list)
    for row in rows:
        by_policy[row.policy].append(row)

    summary_policy: Dict[str, Dict[str, object]] = {}
    lru_index: Dict[Tuple[str, int], int] = {
        (r.trace, r.capacity): r.misses for r in rows if r.policy == "lru"
    }

    for policy, policy_rows in sorted(by_policy.items()):
        policy_rows_sorted = sorted(policy_rows, key=lambda r: (r.trace, r.capacity))
        lru_series = [lru_index[(r.trace, r.capacity)] for r in policy_rows_sorted if (r.trace, r.capacity) in lru_index]
        pol_series = [r.misses for r in policy_rows_sorted if (r.trace, r.capacity) in lru_index]
        wtl = _compute_pairwise_wtl(pol_series, lru_series)
        summary_policy[policy] = {
            "runs": len(policy_rows_sorted),
            "mean_misses": mean(r.misses for r in policy_rows_sorted),
            "mean_hit_rate": mean(r.hit_rate for r in policy_rows_sorted),
            "mean_rel_improvement_vs_lru": mean((r.rel_improvement_vs_lru or 0.0) for r in policy_rows_sorted),
            "average_rank": mean((r.rank or 0.0) for r in policy_rows_sorted),
            "worst_rank": max((r.rank or 0.0) for r in policy_rows_sorted),
            "mean_regret_vs_best": mean((r.regret_vs_best or 0.0) for r in policy_rows_sorted),
            "mean_normalized_gap_to_best": mean((r.normalized_gap_to_best or 0.0) for r in policy_rows_sorted),
            "vs_lru": wtl,
        }
        mode_counts: Dict[str, int] = defaultdict(int)
        for row in policy_rows_sorted:
            if row.scorer_mode:
                mode_counts[str(row.scorer_mode)] += 1
        if mode_counts:
            summary_policy[policy]["scorer_mode_counts"] = dict(mode_counts)

    wins_vs: Dict[str, Dict[str, Dict[str, int]]] = {}
    comparable = sorted(by_policy.keys())
    for p in comparable:
        wins_vs[p] = {}
        p_rows = {(r.trace, r.capacity): r.misses for r in by_policy[p]}
        for q in comparable:
            if p == q:
                continue
            q_rows = {(r.trace, r.capacity): r.misses for r in by_policy[q]}
            keys = sorted(set(p_rows).intersection(q_rows))
            wins_vs[p][q] = _compute_pairwise_wtl([p_rows[k] for k in keys], [q_rows[k] for k in keys])

    per_trace_winners: Dict[str, List[str]] = {}
    for trace in sorted({r.trace for r in rows}):
        trace_rows = [r for r in rows if r.trace == trace]
        if not trace_rows:
            continue
        miss_by_policy: Dict[str, float] = {}
        for policy in sorted({r.policy for r in trace_rows}):
            vals = [r.misses for r in trace_rows if r.policy == policy]
            if vals:
                miss_by_policy[policy] = mean(vals)
        best = min(miss_by_policy.values())
        per_trace_winners[trace] = sorted([p for p, v in miss_by_policy.items() if v == best])

    return {
        "rows": [r.__dict__ for r in rows],
        "included_policies": sorted(included_policies),
        "skipped_policies": skipped_policies,
        "summary_by_policy": summary_policy,
        "wins_vs_each_policy": wins_vs,
        "per_trace_winners": per_trace_winners,
        "capacities": capacities,
        "max_requests_per_trace": max_requests,
        "traces": [t.name for t in traces],
    }


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "trace",
        "trace_kind",
        "requests",
        "capacity",
        "policy",
        "misses",
        "hits",
        "hit_rate",
        "scorer_mode",
        "rel_improvement_vs_lru",
        "rank",
        "regret_vs_best",
        "normalized_gap_to_best",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_report(payload: Dict[str, object], out_path: Path, command: str) -> None:
    lines: List[str] = []
    lines.append("# Lightweight baseline comparison (Codex-web scale)")
    lines.append("")
    lines.append("## Scope and framing")
    lines.append("- This is a lightweight local comparison intended as a sanity-but-nontrivial check.")
    lines.append("- Results are preliminary and are **not** a final paper-grade benchmark.")
    lines.append("- A larger multi-family evaluation on stronger compute should still be run later.")
    lines.append("")
    lines.append("## Exact command run")
    lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Policies requested")
    for policy in payload["included_policies"]:
        lines.append(f"- `{policy}`")
    if payload["skipped_policies"]:
        lines.append("")
        lines.append("## Skipped policies and reasons")
        for policy, reason in sorted(payload["skipped_policies"].items()):
            lines.append(f"- `{policy}` skipped: `{reason}`")

    lines.append("")
    lines.append("## Traces and settings")
    lines.append(f"- Traces used ({len(payload['traces'])}): " + ", ".join(f"`{t}`" for t in payload["traces"]))
    lines.append(f"- Capacities: {', '.join(str(c) for c in payload['capacities'])}")
    lines.append(f"- Max requests per trace: {payload['max_requests_per_trace']}")
    lines.append("")

    lines.append("## Aggregate policy summary")
    lines.append("| policy | mean_misses | mean_hit_rate | mean_rel_impr_vs_lru | avg_rank | worst_rank | mean_regret_vs_best | mean_norm_gap | W/T/L vs LRU |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for policy, stats in sorted(payload["summary_by_policy"].items()):
        wtl = stats["vs_lru"]
        lines.append(
            f"| {policy} | {stats['mean_misses']:.3f} | {stats['mean_hit_rate']:.3%} | "
            f"{stats['mean_rel_improvement_vs_lru']:.3%} | {stats['average_rank']:.2f} | {stats['worst_rank']:.0f} | "
            f"{stats['mean_regret_vs_best']:.3f} | {stats['mean_normalized_gap_to_best']:.3f} | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
        )
    evict_stats = payload["summary_by_policy"].get("evict_value_v1")
    if evict_stats:
        lines.append("")
        lines.append("## evict_value_v1 scorer mode usage")
        mode_counts = evict_stats.get("scorer_mode_counts", {})
        if mode_counts:
            for mode, count in sorted(mode_counts.items()):
                lines.append(f"- `{mode}`: {count} run(s)")
        else:
            lines.append("- No scorer mode data captured.")

    lines.append("")
    lines.append("## Per-trace winners (lower mean misses across capacities)")
    for trace, winners in sorted(payload["per_trace_winners"].items()):
        lines.append(f"- `{trace}`: {', '.join(f'`{w}`' for w in winners)}")

    lines.append("")
    lines.append("## Main caveats")
    lines.append("- Synthetic traces are included to provide lightweight diversity because repository examples are very short.")
    lines.append("- Prediction metadata is generated heuristically from each trace sequence; this is suitable for quick local checks, not definitive benchmarking.")
    if payload["skipped_policies"]:
        lines.append("- Policies requiring unavailable binary model artifacts are skipped rather than retrained, by design, to keep the run light and text-only.")
    else:
        lines.append("- `evict_value_v1` is included via a text-only lightweight surrogate scorer in this run; interpret this as local-surrogate behavior, not trained-checkpoint performance.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight baseline comparison on small traces.")
    parser.add_argument("--capacities", default="3,5,8", help="Comma-separated capacities.")
    parser.add_argument("--max-requests", type=int, default=200, help="Max requests per trace.")
    parser.add_argument(
        "--policies",
        default=(
            "lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,"
            "trust_and_doubt,rest_v1,evict_value_v1"
        ),
        help="Comma-separated policy names.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory for lightweight comparison artifacts.",
    )
    args = parser.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    selected_policies = [x.strip() for x in args.policies.split(",") if x.strip()]

    payload = run_lightweight_comparison(
        capacities=capacities,
        max_requests=args.max_requests,
        selected_policies=selected_policies,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "lightweight_results.csv"
    summary_json = out_dir / "lightweight_summary.json"
    report_md = out_dir / "lightweight_report.md"

    _write_csv(payload["rows"], results_csv)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    command = (
        "python scripts/run_lightweight_baseline_comparison.py "
        f"--capacities {args.capacities} --max-requests {args.max_requests} --policies {args.policies} --out-dir {args.out_dir}"
    )
    _build_report(payload, report_md, command)

    print(f"Wrote {results_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
