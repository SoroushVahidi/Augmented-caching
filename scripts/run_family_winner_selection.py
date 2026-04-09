from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Tuple

from lafc.learned_gate.features_v2 import ML_GATE_V2_FEATURE_COLUMNS
from lafc.learned_gate.lightweight_estimator import LinearProbabilityEstimator
from lafc.learned_gate.model_v2 import LearnedGateV2Model
from lafc.metrics.cost import hit_rate
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.ml_gate_v2 import MLGateV2Policy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.robust_ftp_marker_combiner import RobustFtPDeterministicMarkerCombiner
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import Request


@dataclass(frozen=True)
class TraceSpec:
    name: str
    loader: Callable[[int], List[str]]


@dataclass
class RunRow:
    trace: str
    regime: str
    requests: int
    capacity: int
    policy: str
    misses: int
    hits: int
    hit_rate: float
    rank: float | None = None
    rel_improvement_vs_lru: float | None = None


def _compute_actual_next(page_ids: List[str]) -> List[float]:
    last_seen: Dict[str, int] = {}
    out: List[float] = [math.inf] * len(page_ids)
    for t in range(len(page_ids) - 1, -1, -1):
        pid = page_ids[t]
        if pid in last_seen:
            out[t] = float(last_seen[pid])
        last_seen[pid] = t
    return out


def _bucket_from_distance(dist: int) -> int:
    if dist <= 2:
        return 0
    if dist <= 5:
        return 1
    if dist <= 12:
        return 2
    return 3


def _prediction_records(page_ids: List[str], predictions: List[float], regime: str) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for t, nxt in enumerate(predictions):
        dist = 1000 if math.isinf(nxt) else max(0, int(nxt - t))
        bucket = _bucket_from_distance(dist)
        if regime == "clean":
            conf = 0.95
        elif regime == "noisy":
            conf = max(0.05, min(0.99, 0.92 - 0.10 * abs((t % 5) - 2)))
        else:
            raise ValueError(f"unknown regime={regime}")
        records.append({"bucket": bucket, "confidence": conf})
    return records


def _inject_light_noise(actual_next: List[float]) -> List[float]:
    noisy: List[float] = []
    n = len(actual_next)
    for t, val in enumerate(actual_next):
        if math.isinf(val):
            noisy.append(float(n + 64 + (t % 7)))
            continue
        jitter = ((t * 3) % 5) - 2
        noisy.append(max(float(t + 1), float(val + jitter)))
    return noisy


def _build_requests(page_ids: List[str], regime: str) -> Tuple[List[Request], Dict[str, object]]:
    actual_next = _compute_actual_next(page_ids)
    predictions = actual_next if regime == "clean" else _inject_light_noise(actual_next)
    prediction_records = _prediction_records(page_ids, predictions, regime)
    return build_requests_from_lists(
        page_ids=page_ids,
        predictions=predictions,
        prediction_records=prediction_records,
    )


def _base_traces() -> List[TraceSpec]:
    def from_file(name: str, path: str) -> TraceSpec:
        return TraceSpec(
            name=name,
            loader=lambda max_requests, p=path: [str(r.page_id) for r in load_trace(p)[0][:max_requests]],
        )

    stress_good = ["A", "B", "C", "A", "D", "A", "B", "C", "A", "D"]
    stress_bad = ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"]
    stress_mixed = ["A", "B", "C", "A", "B", "D", "A", "C", "B", "D"]

    def from_seq(name: str, seq: List[str]) -> TraceSpec:
        return TraceSpec(name=name, loader=lambda max_requests, s=seq: s[:max_requests])

    return [
        from_file("file::example_unweighted", "data/example_unweighted.json"),
        from_file("file::example_atlas_v1", "data/example_atlas_v1.json"),
        from_seq("stress::predictor_good_lru_bad", stress_good),
        from_seq("stress::predictor_bad_lru_good", stress_bad),
        from_seq("stress::mixed_regime", stress_mixed),
    ]


def _read_ml_gate_rows(path: Path, horizon: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["horizon"]) != horizon:
                continue
            out = {c: float(row[c]) for c in ML_GATE_V2_FEATURE_COLUMNS}
            out["y_cls"] = float(int(row["y_cls"]))
            rows.append(out)
    return rows


def _ensure_ml_gate_v2_model(model_path: Path, horizon: int) -> Path:
    if model_path.exists():
        return model_path

    train_rows = _read_ml_gate_rows(Path("data/derived/ml_gate_v2_train.csv"), horizon=horizon)
    if not train_rows:
        raise ValueError(f"no rows available for horizon={horizon} in data/derived/ml_gate_v2_train.csv")

    pos = [r for r in train_rows if int(r["y_cls"]) == 1]
    neg = [r for r in train_rows if int(r["y_cls"]) == 0]
    pos_mean = {
        c: (sum(float(r[c]) for r in pos) / len(pos)) if pos else 0.0
        for c in ML_GATE_V2_FEATURE_COLUMNS
    }
    neg_mean = {
        c: (sum(float(r[c]) for r in neg) / len(neg)) if neg else 0.0
        for c in ML_GATE_V2_FEATURE_COLUMNS
    }
    feature_weights = {c: float(pos_mean[c] - neg_mean[c]) for c in ML_GATE_V2_FEATURE_COLUMNS}
    intercept = float(
        -0.5
        * sum(
            feature_weights[c] * (pos_mean[c] + neg_mean[c])
            for c in ML_GATE_V2_FEATURE_COLUMNS
        )
    )
    estimator = LinearProbabilityEstimator(
        feature_columns=list(ML_GATE_V2_FEATURE_COLUMNS),
        feature_weights=feature_weights,
        intercept=intercept,
    )

    model = LearnedGateV2Model(
        model_name="logistic_regression_lightweight",
        estimator=estimator,
        feature_columns=list(ML_GATE_V2_FEATURE_COLUMNS),
        threshold=0.5,
    )
    model.save(model_path)
    return model_path


def _policy_factories(ml_gate_model_path: str) -> Dict[str, Callable[[], object]]:
    return {
        "evict_value_v1": lambda: EvictValueV1Policy(scorer_mode="lightweight"),
        "rest_v1": lambda: RestV1Policy(),
        "atlas_v3": lambda: AtlasV3Policy(),
        "ml_gate_v2": lambda: MLGateV2Policy(model_path=ml_gate_model_path),
        "trust_and_doubt": lambda: TrustAndDoubtPolicy(seed=7),
        "robust_ftp_d_marker": lambda: RobustFtPDeterministicMarkerCombiner(),
        "blind_oracle_lru_combiner": lambda: BlindOracleLRUCombiner(),
        "lru": lambda: LRUPolicy(),
    }


def _dense_rank(values: Dict[str, int]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    ranks: Dict[str, float] = {}
    current = 1.0
    prev: int | None = None
    for idx, (name, val) in enumerate(ordered):
        if prev is not None and val != prev:
            current = float(idx + 1)
        ranks[name] = current
        prev = val
    return ranks


def _pairwise_wtl(a: Iterable[int], b: Iterable[int]) -> Dict[str, int]:
    wins = ties = losses = 0
    for x, y in zip(a, b):
        if x < y:
            wins += 1
        elif x > y:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def run_experiment(capacities: List[int], max_requests: int, regimes: List[str], ml_gate_model_path: str) -> Dict[str, object]:
    traces = _base_traces()
    factories = _policy_factories(ml_gate_model_path)

    rows: List[RunRow] = []

    for trace in traces:
        page_ids = trace.loader(max_requests)
        for regime in regimes:
            requests, pages = _build_requests(page_ids, regime)
            for cap in capacities:
                results: Dict[str, RunRow] = {}
                for policy_name, factory in factories.items():
                    run_requests = requests
                    if policy_name in {"trust_and_doubt", "robust_ftp_d_marker"}:
                        run_requests = attach_predicted_caches(requests, capacity=cap)
                    result = run_policy(factory(), run_requests, pages, cap)
                    results[policy_name] = RunRow(
                        trace=trace.name,
                        regime=regime,
                        requests=len(run_requests),
                        capacity=cap,
                        policy=policy_name,
                        misses=result.total_misses,
                        hits=result.total_hits,
                        hit_rate=hit_rate(result.events),
                    )

                lru_misses = results["lru"].misses
                ranks = _dense_rank({p: r.misses for p, r in results.items()})
                for policy_name, row in results.items():
                    row.rank = ranks[policy_name]
                    if lru_misses > 0:
                        row.rel_improvement_vs_lru = (lru_misses - row.misses) / lru_misses
                    else:
                        row.rel_improvement_vs_lru = 0.0 if row.misses == 0 else None
                    rows.append(row)

    grouped: Dict[str, List[RunRow]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)

    summary_by_policy: Dict[str, Dict[str, object]] = {}
    lru_index = {(r.trace, r.regime, r.capacity): r.misses for r in rows if r.policy == "lru"}
    for policy, policy_rows in sorted(grouped.items()):
        policy_rows = sorted(policy_rows, key=lambda r: (r.trace, r.regime, r.capacity))
        lru_series = [lru_index[(r.trace, r.regime, r.capacity)] for r in policy_rows]
        pol_series = [r.misses for r in policy_rows]
        wtl = _pairwise_wtl(pol_series, lru_series)
        summary_by_policy[policy] = {
            "runs": len(policy_rows),
            "mean_misses": mean(r.misses for r in policy_rows),
            "mean_hit_rate": mean(r.hit_rate for r in policy_rows),
            "average_rank": mean((r.rank or 0.0) for r in policy_rows),
            "mean_rel_improvement_vs_lru": mean((r.rel_improvement_vs_lru or 0.0) for r in policy_rows),
            "vs_lru": wtl,
        }

    strongest = min(summary_by_policy.items(), key=lambda kv: (kv[1]["average_rank"], kv[1]["mean_misses"]))[0]
    unpromising = max(summary_by_policy.items(), key=lambda kv: (kv[1]["average_rank"], kv[1]["mean_misses"]))[0]

    return {
        "rows": [r.__dict__ for r in rows],
        "summary_by_policy": summary_by_policy,
        "traces": [t.name for t in traces],
        "capacities": capacities,
        "regimes": regimes,
        "strongest_family_candidate": strongest,
        "unpromising_family_candidate": unpromising,
        "ml_gate_model_path": ml_gate_model_path,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace",
        "regime",
        "requests",
        "capacity",
        "policy",
        "misses",
        "hits",
        "hit_rate",
        "rank",
        "rel_improvement_vs_lru",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_report(payload: Dict[str, object], out_path: Path, command: str) -> None:
    lines: List[str] = []
    lines.append("# Method-family winner selection (lightweight)")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Lightweight cross-family comparison on repo examples + tiny stress traces.")
    lines.append("- Uses two prediction-quality regimes: `clean` and `noisy`.")
    lines.append("- Intended for ranking guidance, not paper-grade benchmarking.")
    lines.append("")
    lines.append("## Command")
    lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Aggregate ranking")
    lines.append("| policy | runs | mean_misses | mean_hit_rate | avg_rank | mean_rel_impr_vs_lru | W/T/L vs LRU |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for policy, stats in sorted(payload["summary_by_policy"].items(), key=lambda kv: (kv[1]["average_rank"], kv[1]["mean_misses"])):
        wtl = stats["vs_lru"]
        lines.append(
            f"| {policy} | {stats['runs']} | {stats['mean_misses']:.3f} | {stats['mean_hit_rate']:.3%} | "
            f"{stats['average_rank']:.2f} | {stats['mean_rel_improvement_vs_lru']:.3%} | {wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- Strongest family candidate in this lightweight run: **{payload['strongest_family_candidate']}**.")
    lines.append(f"- Unpromising family candidate in this lightweight run: **{payload['unpromising_family_candidate']}**.")
    lines.append("- Treat this as directional only; verify on larger traces before making durable claims.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Winner-selection experiment across method families (lightweight)")
    parser.add_argument("--capacities", default="2,3,4,5")
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--regimes", default="clean,noisy")
    parser.add_argument("--ml-gate-horizon", type=int, default=8)
    parser.add_argument("--ml-gate-model-path", default="analysis/family_winner_selection/models/ml_gate_v2_lightweight.pkl")
    parser.add_argument("--out-dir", default="analysis/family_winner_selection")
    args = parser.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]

    model_path = _ensure_ml_gate_v2_model(Path(args.ml_gate_model_path), horizon=args.ml_gate_horizon)

    payload = run_experiment(
        capacities=capacities,
        max_requests=args.max_requests,
        regimes=regimes,
        ml_gate_model_path=str(model_path),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "winner_selection_results.csv"
    summary_json = out_dir / "winner_selection_summary.json"
    report_md = out_dir / "winner_selection_report.md"

    _write_csv(results_csv, payload["rows"])
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    command = (
        "python scripts/run_family_winner_selection.py "
        f"--capacities {args.capacities} --max-requests {args.max_requests} --regimes {args.regimes} "
        f"--ml-gate-horizon {args.ml_gate_horizon} --ml-gate-model-path {args.ml_gate_model_path} --out-dir {args.out_dir}"
    )
    _build_report(payload, report_md, command)

    print(f"Wrote {results_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
