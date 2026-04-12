from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_v2_rollout import EvictValueV2RolloutConfig, build_rollout_candidate_rows_v2
from lafc.simulator.request_trace import load_trace
from lafc.types import PageId, Request


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _decision_groups(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    g: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        g.setdefault(str(r["decision_id"]), []).append(r)
    return g


def _split_train_test(rows: Sequence[Dict[str, object]], seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train, test = [], []
    for did, items in _decision_groups(rows).items():
        digest = hashlib.sha256(f"{did}|{seed}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10
        (test if bucket < 2 else train).extend(items)
    return train, test


def _fit_regressor(rows: Sequence[Dict[str, object]], seed: int) -> HistGradientBoostingRegressor:
    x = np.asarray([[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.asarray([float(r["rollout_regret_h"]) for r in rows], dtype=float)
    model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=seed)
    model.fit(x, y)
    return model


def _choose_by_model(items: Sequence[Dict[str, object]], model: HistGradientBoostingRegressor) -> Dict[str, object]:
    x = np.asarray([[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in items], dtype=float)
    pred = np.asarray(model.predict(x), dtype=float)
    idx = min(range(len(items)), key=lambda i: (float(pred[i]), str(items[i]["candidate_page_id"])))
    return items[idx]


def _offline_eval(rows: Sequence[Dict[str, object]], model: HistGradientBoostingRegressor) -> Dict[str, float]:
    decisions = _decision_groups(rows)
    regrets: List[float] = []
    top1 = 0
    for items in decisions.values():
        chosen = _choose_by_model(items, model)
        best = min(items, key=lambda r: (float(r["rollout_regret_h"]), str(r["candidate_page_id"])))
        regrets.append(float(chosen["rollout_regret_h"]))
        top1 += int(str(chosen["candidate_page_id"]) == str(best["candidate_page_id"]))
    denom = max(len(regrets), 1)
    return {
        "decision_count": float(len(regrets)),
        "top1_eviction_match": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(regrets) if regrets else 0.0),
    }


def _rate(hist: Sequence[PageId], pid: PageId) -> float:
    if not hist:
        return 0.0
    return float(sum(1 for x in hist if x == pid) / len(hist))


def _replay_misses(
    *,
    requests: Sequence[Request],
    capacity: int,
    model: HistGradientBoostingRegressor,
    history_window: int,
) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: collections.deque[PageId] = collections.deque(maxlen=history_window)
    recent_hit_hist: collections.deque[PageId] = collections.deque(maxlen=history_window)
    misses = 0

    for req in requests:
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue

        misses += 1
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))

        feat = []
        for c in candidates:
            row = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=c,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=_rate(recent_req_hist, c),
                recent_hit_rate=_rate(recent_hit_hist, c),
            ).as_dict()
            feat.append([float(row[k]) for k in EVICT_VALUE_V1_FEATURE_COLUMNS])

        pred = np.asarray(model.predict(np.asarray(feat, dtype=float)), dtype=float)
        victim_idx = min(range(len(candidates)), key=lambda i: (float(pred[i]), str(candidates[i])))
        victim = candidates[victim_idx]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses


def _replay_lru_misses(requests: Sequence[Request], capacity: int) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    misses = 0
    for req in requests:
        pid = req.page_id
        if pid in order:
            order.move_to_end(pid)
            continue
        misses += 1
        if len(order) < capacity:
            order[pid] = None
            continue
        victim = next(iter(order.keys()))
        order.pop(victim)
        order[pid] = None
    return misses


def _build_rows(
    trace_paths: Sequence[str],
    capacities: Sequence[int],
    horizon: int,
    max_requests_per_trace: int,
    reference_policy: str,
) -> List[Dict[str, object]]:
    cfg = EvictValueV2RolloutConfig(horizons=(horizon,), reference_policy=reference_policy)
    rows: List[Dict[str, object]] = []
    for trace in trace_paths:
        reqs, _pages = load_trace(trace)
        reqs = reqs[:max_requests_per_trace] if max_requests_per_trace > 0 else reqs
        fam = Path(trace).stem
        for cap in capacities:
            rows.extend(build_rollout_candidate_rows_v2(requests=reqs, capacity=cap, trace_name=trace, trace_family=fam, cfg=cfg))
    return rows


def _label_agreement(rows_by_policy: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    aligned: Dict[str, Dict[str, Dict[str, float]]] = {}
    for policy, rows in rows_by_policy.items():
        for r in rows:
            key = str(r["decision_id"])
            cand = str(r["candidate_page_id"])
            aligned.setdefault(key, {}).setdefault(cand, {})[policy] = float(r["rollout_regret_h"])

    policies = sorted(rows_by_policy.keys())
    out: List[Dict[str, object]] = []
    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            a, b = policies[i], policies[j]
            same_top1 = 0
            common_decisions = 0
            abs_diffs: List[float] = []
            for did, cand_map in aligned.items():
                regrets_a = {c: vals[a] for c, vals in cand_map.items() if a in vals and b in vals}
                regrets_b = {c: vals[b] for c, vals in cand_map.items() if a in vals and b in vals}
                if len(regrets_a) < 2:
                    continue
                common_decisions += 1
                best_a = min(regrets_a.items(), key=lambda kv: (kv[1], kv[0]))[0]
                best_b = min(regrets_b.items(), key=lambda kv: (kv[1], kv[0]))[0]
                same_top1 += int(best_a == best_b)
                abs_diffs.extend(abs(regrets_a[c] - regrets_b[c]) for c in regrets_a.keys())
            out.append(
                {
                    "policy_a": a,
                    "policy_b": b,
                    "common_decisions": common_decisions,
                    "top1_label_agreement": float(same_top1 / common_decisions) if common_decisions else 0.0,
                    "mean_abs_regret_diff": float(np.mean(abs_diffs) if abs_diffs else 0.0),
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight continuation-policy ablation for rollout labels")
    ap.add_argument("--trace-glob", default="data/example*.json")
    ap.add_argument("--capacities", default="2,3")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-requests-per-trace", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", default="analysis/continuation_policy_light")
    args = ap.parse_args()

    traces = sorted(str(p) for p in Path(".").glob(args.trace_glob))
    if not traces:
        raise ValueError(f"No traces matched: {args.trace_glob}")

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    protocols = ["lru", "blind_oracle", "fifo"]
    rows_by_policy: Dict[str, List[Dict[str, object]]] = {}
    summary_rows: List[Dict[str, object]] = []
    replay_rows: List[Dict[str, object]] = []

    for protocol in protocols:
        rows = _build_rows(
            trace_paths=traces,
            capacities=capacities,
            horizon=args.horizon,
            max_requests_per_trace=args.max_requests_per_trace,
            reference_policy=protocol,
        )
        rows_by_policy[protocol] = rows
        train_rows, test_rows = _split_train_test(rows, args.seed)
        model = _fit_regressor(train_rows, args.seed)
        metrics = _offline_eval(test_rows, model)
        summary_rows.append(
            {
                "protocol": protocol,
                "horizon": args.horizon,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                **metrics,
            }
        )

        for trace in traces:
            reqs, _ = load_trace(trace)
            reqs = reqs[: args.max_requests_per_trace] if args.max_requests_per_trace > 0 else reqs
            for cap in capacities:
                learned_misses = _replay_misses(requests=reqs, capacity=cap, model=model, history_window=64)
                lru_misses = _replay_lru_misses(reqs, cap)
                replay_rows.append(
                    {
                        "protocol": protocol,
                        "trace": trace,
                        "capacity": cap,
                        "learned_misses": learned_misses,
                        "lru_misses": lru_misses,
                        "delta_vs_lru": learned_misses - lru_misses,
                    }
                )

    agreement_rows = _label_agreement(rows_by_policy)

    _write_csv(output / "summary.csv", summary_rows)
    _write_csv(output / "downstream_replay.csv", replay_rows)
    _write_csv(output / "label_agreement.csv", agreement_rows)

    replay_means: Dict[str, float] = {}
    for p in protocols:
        vals = [float(r["delta_vs_lru"]) for r in replay_rows if str(r["protocol"]) == p]
        replay_means[p] = float(mean(vals) if vals else 0.0)

    payload = {
        "experiment": "continuation_policy_light",
        "horizon": args.horizon,
        "capacities": capacities,
        "trace_count": len(traces),
        "max_requests_per_trace": args.max_requests_per_trace,
        "summary": summary_rows,
        "downstream_replay_mean_delta_vs_lru": replay_means,
        "label_agreement": agreement_rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = min(summary_rows, key=lambda r: (float(r["mean_chosen_regret"]), -float(r["top1_eviction_match"])))
    lines = [
        "# Continuation-policy lightweight ablation",
        "",
        f"- Traces: {len(traces)} matched by `{args.trace_glob}` (capped at {args.max_requests_per_trace} requests/trace).",
        f"- Capacities: {capacities}; horizon: {args.horizon}.",
        "- Protocols compared: lru, blind_oracle, fifo.",
        "",
        "## Offline label-quality proxy (held-out decisions)",
    ]
    for r in summary_rows:
        lines.append(
            f"- {r['protocol']}: top1={float(r['top1_eviction_match']):.3f}, mean_chosen_regret={float(r['mean_chosen_regret']):.3f}, decisions={int(float(r['decision_count']))}."
        )
    lines.extend([
        "",
        "## Downstream replay proxy (model trained per protocol)",
    ])
    for p in protocols:
        lines.append(f"- {p}: mean delta vs LRU = {replay_means[p]:+.3f} misses (negative is better).")
    lines.extend([
        "",
        "## Label agreement",
    ])
    for a in agreement_rows:
        lines.append(
            f"- {a['policy_a']} vs {a['policy_b']}: top1 label agreement={float(a['top1_label_agreement']):.3f}, mean abs regret diff={float(a['mean_abs_regret_diff']):.3f} over {int(a['common_decisions'])} decisions."
        )
    lines.extend([
        "",
        f"Best mean chosen-regret protocol in this run: **{best['protocol']}**.",
        "",
        "Exploratory only: this script intentionally does not touch heavy_r1 Slurm/manuscript artifacts.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
