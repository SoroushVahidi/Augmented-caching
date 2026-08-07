from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_v2_rollout import EvictValueV2RolloutConfig, build_rollout_candidate_rows_v2
from lafc.offline.trace_inputs import load_trace_with_sizes
from lafc.offline_teacher_supervision import OfflineTeacherLabelConfig, build_offline_teacher_candidate_rows
from lafc.simulator.request_trace import load_trace
from lafc.types import PageId, Request


def _resolve_trace_paths(trace_glob: str) -> List[str]:
    out: List[str] = []
    for pattern in [p.strip() for p in trace_glob.split(",") if p.strip()]:
        out.extend(sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(pattern)])
    paths = sorted({str(p) for p in out if p.exists()})
    if not paths:
        raise ValueError(f"No traces matched --trace-glob={trace_glob}")
    return paths


def _trace_family(path: str) -> str:
    parent = Path(path).parent
    return parent.name if parent.name else "unknown"


def _split_for_seed(trace: str, seed: int) -> str:
    h = int(hashlib.md5(f"{trace}|{seed}".encode("utf-8")).hexdigest(), 16) % 10
    if h <= 5:
        return "train"
    if h <= 7:
        return "val"
    return "test"


def _as_common_rows(label_source: str, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        r = dict(row)
        if label_source == "heuristic":
            r["target_regret"] = float(r["rollout_regret_h"])
            r["target_loss"] = float(r["rollout_loss_h"])
            r["is_best"] = float(r["candidate_is_rollout_optimal"])
            r.setdefault("teacher_type", "heuristic_rollout")
        else:
            r["target_regret"] = float(r["teacher_regret"])
            r["target_loss"] = float(r["teacher_cost"])
            r["is_best"] = float(r["teacher_best"])
        t_val = int(r.get("request_t", r.get("t", 0)))
        cap_val = int(float(r["capacity"]))
        r["decision_key"] = f"{r['trace']}|c{cap_val}|t{t_val}"
        out.append(r)
    return out


def _build_rows_for_source(
    *,
    label_source: str,
    trace_paths: Sequence[str],
    capacities: Sequence[int],
    horizon: int,
    max_requests_per_trace: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace_path in trace_paths:
        family = _trace_family(trace_path)
        requests, pages = load_trace(trace_path)
        if max_requests_per_trace > 0:
            requests = requests[:max_requests_per_trace]

        for cap in capacities:
            if label_source == "heuristic":
                cfg = EvictValueV2RolloutConfig(horizons=(horizon,), reference_policy="lru")
                raw = build_rollout_candidate_rows_v2(
                    requests=requests,
                    capacity=cap,
                    trace_name=trace_path,
                    trace_family=family,
                    cfg=cfg,
                )
            else:
                try:
                    req2, pages2, sizes = load_trace_with_sizes(trace_path)
                    if max_requests_per_trace > 0:
                        req2 = req2[:max_requests_per_trace]
                except Exception:
                    req2, pages2 = load_trace(trace_path)
                    sizes = {pid: 1.0 for pid in pages2}
                cfg = OfflineTeacherLabelConfig(horizon=horizon)
                raw = build_offline_teacher_candidate_rows(
                    requests=req2,
                    pages=pages2,
                    page_sizes=sizes,
                    capacity=float(cap),
                    trace_name=trace_path,
                    trace_family=family,
                    cfg=cfg,
                )
            rows.extend(_as_common_rows(label_source, raw))
    return rows


def _xy(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x: List[List[float]] = []
    y: List[float] = []
    for r in rows:
        feat = [float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
        feat += [float(r["capacity"]), float(r["horizon"])]
        x.append(feat)
        y.append(float(r["target_regret"]))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _reg_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)) if len(y) else 0.0,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))) if len(y) else 0.0,
    }


def _decision_metrics(rows: List[Dict[str, object]], pred: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = defaultdict(list)
    for r, p in zip(rows, pred):
        grouped[str(r["decision_key"])].append((r, float(p)))

    top1 = 0
    chosen_regret: List[float] = []
    regret_gap: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        oracle = min(items, key=lambda x: (float(x[0]["target_regret"]), str(x[0]["candidate_page_id"])))
        top1 += int(str(chosen[0]["candidate_page_id"]) == str(oracle[0]["candidate_page_id"]))
        best_regret = min(float(x[0]["target_regret"]) for x in items)
        chosen_regret.append(float(chosen[0]["target_regret"]))
        regret_gap.append(float(chosen[0]["target_regret"]) - best_regret)

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regret) if chosen_regret else 0.0),
        "mean_regret_vs_best": float(np.mean(regret_gap) if regret_gap else 0.0),
    }


def _split_rows_for_seed(rows: List[Dict[str, object]], seed: int) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[_split_for_seed(str(r["trace"]), seed)].append(r)
    return out


def _run_lru_misses(requests: Sequence[Request], capacity: int) -> Tuple[int, int]:
    order: OrderedDict[PageId, None] = OrderedDict()
    misses = 0
    hits = 0
    for req in requests:
        pid = req.page_id
        if pid in order:
            order.move_to_end(pid)
            hits += 1
            continue
        misses += 1
        if len(order) >= capacity:
            order.popitem(last=False)
        order[pid] = None
    return misses, hits


def _run_model_policy_misses(requests: Sequence[Request], capacity: int, model: object, horizon: int) -> Tuple[int, int]:
    order: OrderedDict[PageId, None] = OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = deque(maxlen=64)
    recent_hit_hist: Deque[PageId] = deque(maxlen=64)
    misses = 0
    hits = 0

    for req in requests:
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            hits += 1
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

        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            req_rate = (sum(1 for x in recent_req_hist if x == cand) / len(recent_req_hist)) if recent_req_hist else 0.0
            hit_rate = (sum(1 for x in recent_hit_hist if x == cand) / len(recent_hit_hist)) if recent_hit_hist else 0.0
            feat = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=cand,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()
            x = np.asarray([[float(feat[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] + [float(capacity), float(horizon)]], dtype=float)
            score = float(model.predict(x)[0])
            scored.append((cand, score))

        victim = min(scored, key=lambda x: (x[1], str(x[0])))[0]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses, hits


def _disagreement_flags(
    heuristic_rows: List[Dict[str, object]],
    teacher_rows: List[Dict[str, object]],
) -> Tuple[Dict[str, bool], List[Dict[str, object]], List[Dict[str, object]]]:
    by_h: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_t: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in heuristic_rows:
        by_h[str(r["decision_key"])].append(r)
    for r in teacher_rows:
        by_t[str(r["decision_key"])].append(r)

    flags: Dict[str, bool] = {}
    by_family: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "disagree": 0})
    by_capacity: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "disagree": 0})

    for d in sorted(set(by_h).intersection(set(by_t))):
        hs = by_h[d]
        ts = by_t[d]
        h_best = {str(r["candidate_page_id"]) for r in hs if float(r["is_best"]) == 1.0}
        t_best = {str(r["candidate_page_id"]) for r in ts if float(r["is_best"]) == 1.0}
        disagree = h_best != t_best
        flags[d] = disagree
        fam = str(ts[0].get("family", "unknown"))
        cap = int(float(ts[0]["capacity"]))
        by_family[fam]["total"] += 1
        by_family[fam]["disagree"] += int(disagree)
        by_capacity[cap]["total"] += 1
        by_capacity[cap]["disagree"] += int(disagree)

    family_rows = [
        {
            "family": fam,
            "decisions": v["total"],
            "disagree": v["disagree"],
            "disagree_rate": (v["disagree"] / v["total"]) if v["total"] else 0.0,
        }
        for fam, v in sorted(by_family.items())
    ]
    cap_rows = [
        {
            "capacity": cap,
            "decisions": v["total"],
            "disagree": v["disagree"],
            "disagree_rate": (v["disagree"] / v["total"]) if v["total"] else 0.0,
        }
        for cap, v in sorted(by_capacity.items())
    ]

    return flags, family_rows, cap_rows


def _chosen_regret_by_decision(rows: List[Dict[str, object]], model: object) -> Dict[str, float]:
    x, _y = _xy(rows)
    pred = model.predict(x)
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = defaultdict(list)
    for r, p in zip(rows, pred):
        grouped[str(r["decision_key"])].append((r, float(p)))

    out: Dict[str, float] = {}
    for decision_key, items in grouped.items():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        out[decision_key] = float(chosen[0]["target_regret"])
    return out


def _build_model(model_family: str, seed: int) -> object:
    if model_family == "ridge":
        return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if model_family == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, min_samples_leaf=2, n_jobs=-1)
    raise ValueError(f"Unsupported --model-family={model_family}")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    *,
    trace_glob: str,
    capacities: List[int],
    horizon: int,
    max_requests_per_trace: int,
    output_dir: str,
    seeds: List[int],
    model_family: str,
) -> Dict[str, object]:
    trace_paths = _resolve_trace_paths(trace_glob)

    rows_heur = _build_rows_for_source(
        label_source="heuristic",
        trace_paths=trace_paths,
        capacities=capacities,
        horizon=horizon,
        max_requests_per_trace=max_requests_per_trace,
    )
    rows_teacher = _build_rows_for_source(
        label_source="offline_teacher",
        trace_paths=trace_paths,
        capacities=capacities,
        horizon=horizon,
        max_requests_per_trace=max_requests_per_trace,
    )

    flags, disagreement_family_rows, disagreement_capacity_rows = _disagreement_flags(rows_heur, rows_teacher)

    candidate_rows: List[Dict[str, object]] = []
    downstream_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    model_compare_rows: List[Dict[str, object]] = []
    gain_rows: List[Dict[str, object]] = []

    for seed in seeds:
        split_h = _split_rows_for_seed(rows_heur, seed)
        split_t = _split_rows_for_seed(rows_teacher, seed)

        trained: Dict[str, object] = {}
        for label_source, split_rows in [("heuristic", split_h), ("offline_teacher", split_t)]:
            x_train, y_train = _xy(split_rows["train"])
            if len(split_rows["train"]) == 0:
                continue
            model = _build_model(model_family, seed)
            model.fit(x_train, y_train)
            trained[label_source] = model
            for split in ["train", "val", "test"]:
                x, y = _xy(split_rows[split])
                pred = model.predict(x) if len(split_rows[split]) else np.asarray([], dtype=float)
                candidate_rows.append(
                    {
                        "seed": seed,
                        "model_family": model_family,
                        "label_source": label_source,
                        "split": split,
                        "rows": len(split_rows[split]),
                        **_reg_metrics(y, pred),
                        **_decision_metrics(split_rows[split], pred),
                    }
                )

        if set(trained) != {"heuristic", "offline_teacher"}:
            continue

        teacher_test_rows = split_t["test"]
        regrets_h = _chosen_regret_by_decision(teacher_test_rows, trained["heuristic"]) if teacher_test_rows else {}
        regrets_t = _chosen_regret_by_decision(teacher_test_rows, trained["offline_teacher"]) if teacher_test_rows else {}
        for d, treg in regrets_t.items():
            hreg = regrets_h.get(d)
            if hreg is None:
                continue
            row = next((r for r in teacher_test_rows if str(r["decision_key"]) == d), None)
            if row is None:
                continue
            gain_rows.append(
                {
                    "seed": seed,
                    "decision_key": d,
                    "family": str(row.get("family", "unknown")),
                    "capacity": int(float(row["capacity"])),
                    "is_disagreement": int(flags.get(d, False)),
                    "gain_regret_heur_minus_teacher": float(hreg - treg),
                }
            )

        workload_rows: List[Dict[str, object]] = []
        for trace_path in trace_paths:
            if _split_for_seed(trace_path, seed) != "test":
                continue
            requests, _pages = load_trace(trace_path)
            if max_requests_per_trace > 0:
                requests = requests[:max_requests_per_trace]
            for cap in capacities:
                lru_misses, lru_hits = _run_lru_misses(requests, cap)
                per_source: Dict[str, Tuple[int, int]] = {}
                for label_source in ["heuristic", "offline_teacher"]:
                    misses, hits = _run_model_policy_misses(requests, cap, trained[label_source], horizon)
                    per_source[label_source] = (misses, hits)
                    rec = {
                        "seed": seed,
                        "model_family": model_family,
                        "trace": trace_path,
                        "family": _trace_family(trace_path),
                        "capacity": cap,
                        "label_source": label_source,
                        "requests": len(requests),
                        "policy_misses": misses,
                        "policy_hit_rate": (hits / len(requests)) if requests else 0.0,
                        "lru_misses": lru_misses,
                        "lru_hit_rate": (lru_hits / len(requests)) if requests else 0.0,
                        "delta_misses_vs_lru": misses - lru_misses,
                    }
                    downstream_rows.append(rec)
                    workload_rows.append(rec)

                model_compare_rows.append(
                    {
                        "seed": seed,
                        "model_family": model_family,
                        "trace": trace_path,
                        "family": _trace_family(trace_path),
                        "capacity": cap,
                        "heuristic_policy_misses": per_source["heuristic"][0],
                        "offline_teacher_policy_misses": per_source["offline_teacher"][0],
                        "delta_misses_teacher_minus_heuristic": per_source["offline_teacher"][0] - per_source["heuristic"][0],
                        "winner": "offline_teacher"
                        if per_source["offline_teacher"][0] < per_source["heuristic"][0]
                        else "heuristic"
                        if per_source["offline_teacher"][0] > per_source["heuristic"][0]
                        else "tie",
                    }
                )

        workload_keys = sorted({(r["trace"], r["capacity"]) for r in workload_rows})
        for trace, cap in workload_keys:
            rows = [r for r in workload_rows if r["trace"] == trace and r["capacity"] == cap]
            if len(rows) < 2:
                continue
            ordered = sorted(rows, key=lambda r: (int(r["policy_misses"]), str(r["label_source"])))
            rank_map = {str(row["label_source"]): idx + 1 for idx, row in enumerate(ordered)}
            for src in ["heuristic", "offline_teacher"]:
                row = next(r for r in rows if r["label_source"] == src)
                summary_rows.append(
                    {
                        "seed": seed,
                        "model_family": model_family,
                        "label_source": src,
                        "trace": trace,
                        "family": row["family"],
                        "capacity": cap,
                        "rank": rank_map[src],
                        "policy_misses": row["policy_misses"],
                        "delta_misses_vs_lru": row["delta_misses_vs_lru"],
                    }
                )

    aggregate_downstream: List[Dict[str, object]] = []
    for label_source in ["heuristic", "offline_teacher"]:
        rows = [r for r in downstream_rows if str(r["label_source"]) == label_source]
        if not rows:
            continue
        total_requests = sum(int(r["requests"]) for r in rows)
        total_misses = sum(int(r["policy_misses"]) for r in rows)
        total_lru_misses = sum(int(r["lru_misses"]) for r in rows)
        aggregate_downstream.append(
            {
                "label_source": label_source,
                "workloads": len(rows),
                "total_requests": total_requests,
                "total_misses": total_misses,
                "hit_rate": 1.0 - (total_misses / total_requests if total_requests else 0.0),
                "delta_misses_vs_lru": total_misses - total_lru_misses,
                "avg_delta_misses_vs_lru_per_workload": float(mean(float(r["delta_misses_vs_lru"]) for r in rows)),
                "avg_rank": float(mean(float(r["rank"]) for r in summary_rows if str(r["label_source"]) == label_source)),
            }
        )

    wins = sum(1 for r in model_compare_rows if r["winner"] == "offline_teacher")
    losses = sum(1 for r in model_compare_rows if r["winner"] == "heuristic")
    ties = sum(1 for r in model_compare_rows if r["winner"] == "tie")

    disagreement_gain_rows = [r for r in gain_rows if int(r["is_disagreement"]) == 1]
    agreement_gain_rows = [r for r in gain_rows if int(r["is_disagreement"]) == 0]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "results.csv", candidate_rows)
    _write_csv(output / "downstream_results.csv", downstream_rows)
    _write_csv(output / "disagreement_by_family.csv", disagreement_family_rows)
    _write_csv(output / "disagreement_by_capacity.csv", disagreement_capacity_rows)
    _write_csv(output / "workload_rankings.csv", summary_rows)
    _write_csv(output / "model_comparison.csv", model_compare_rows)
    _write_csv(output / "gain_by_decision.csv", gain_rows)

    summary = {
        "trace_glob": trace_glob,
        "trace_count": len(trace_paths),
        "capacities": capacities,
        "horizon": horizon,
        "max_requests_per_trace": max_requests_per_trace,
        "seeds": seeds,
        "model_family": model_family,
        "rows_heuristic": len(rows_heur),
        "rows_offline_teacher": len(rows_teacher),
        "disagreement": {
            "decisions_common": len(flags),
            "disagree_count": int(sum(1 for v in flags.values() if v)),
            "disagree_rate": (sum(1 for v in flags.values() if v) / len(flags)) if flags else 0.0,
        },
        "aggregate_downstream": aggregate_downstream,
        "teacher_vs_heuristic": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "workloads_compared": len(model_compare_rows),
        },
        "gain_concentration": {
            "mean_gain_disagreement": float(mean(float(r["gain_regret_heur_minus_teacher"]) for r in disagreement_gain_rows)) if disagreement_gain_rows else 0.0,
            "mean_gain_agreement": float(mean(float(r["gain_regret_heur_minus_teacher"]) for r in agreement_gain_rows)) if agreement_gain_rows else 0.0,
            "count_disagreement": len(disagreement_gain_rows),
            "count_agreement": len(agreement_gain_rows),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    agg_by_src = {r["label_source"]: r for r in aggregate_downstream}
    heur = agg_by_src.get("heuristic", {})
    teach = agg_by_src.get("offline_teacher", {})

    report = [
        "# Offline-teacher vs heuristic supervision (medium-scale local evaluation)",
        "",
        "## Configuration",
        f"- model family: `{model_family}`",
        f"- traces: `{trace_glob}` ({len(trace_paths)} resolved traces)",
        f"- capacities: `{capacities}`",
        f"- horizon: `{horizon}`",
        f"- seeds: `{seeds}`",
        "",
        "## Disagreement",
        f"- decision disagreement rate: **{summary['disagreement']['disagree_rate']:.4f}** ({summary['disagreement']['disagree_count']} / {summary['disagreement']['decisions_common']})",
        f"- mean gain on disagreement decisions: **{summary['gain_concentration']['mean_gain_disagreement']:.4f}**",
        f"- mean gain on agreement decisions: **{summary['gain_concentration']['mean_gain_agreement']:.4f}**",
        "",
        "## Downstream aggregate",
        "| label_source | workloads | total_misses | hit_rate | delta_misses_vs_lru | avg_rank |",
        "|---|---:|---:|---:|---:|---:|",
        f"| heuristic | {heur.get('workloads', 0)} | {heur.get('total_misses', 0)} | {heur.get('hit_rate', 0.0):.4f} | {heur.get('delta_misses_vs_lru', 0)} | {heur.get('avg_rank', 0.0):.4f} |",
        f"| offline_teacher | {teach.get('workloads', 0)} | {teach.get('total_misses', 0)} | {teach.get('hit_rate', 0.0):.4f} | {teach.get('delta_misses_vs_lru', 0)} | {teach.get('avg_rank', 0.0):.4f} |",
        "",
        "## Wins / ties / losses",
        f"- offline_teacher wins: **{wins}**",
        f"- ties: **{ties}**",
        f"- offline_teacher losses: **{losses}**",
        "",
        "## Recommendation",
    ]

    if wins > losses and float(teach.get("delta_misses_vs_lru", 0)) <= float(heur.get("delta_misses_vs_lru", 0)):
        report.append("- Evidence is strong enough to prefer offline_teacher supervision for evict_value_v2 follow-ups.")
    elif wins == losses:
        report.append("- Evidence is mixed; carry both supervision sources forward while increasing workload diversity.")
    else:
        report.append("- Evidence does not yet justify replacing heuristic labels; keep heuristic default and continue targeted offline_teacher experiments.")

    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Medium-scale controlled offline_teacher vs heuristic evaluation")
    parser.add_argument("--trace-glob", default="data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json")
    parser.add_argument("--capacities", default="2,3,4,5")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-requests-per-trace", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4,5")
    parser.add_argument("--model-family", choices=["ridge", "random_forest"], default="ridge")
    parser.add_argument("--output-dir", default="analysis/offline_teacher_vs_heuristic_mediumscale")
    args = parser.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    summary = run_experiment(
        trace_glob=args.trace_glob,
        capacities=capacities,
        horizon=args.horizon,
        max_requests_per_trace=args.max_requests_per_trace,
        output_dir=args.output_dir,
        seeds=seeds,
        model_family=args.model_family,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
