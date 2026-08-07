from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict, defaultdict, deque
from glob import glob
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import numpy as np
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
    patterns = [p.strip() for p in trace_glob.split(",") if p.strip()]
    out: List[str] = []
    for pattern in patterns:
        out.extend(sorted(glob(pattern)))
    out = sorted(set(out))
    if not out:
        raise ValueError(f"No traces matched --trace-glob={trace_glob}")
    return out


def _trace_split(trace: str) -> str:
    h = int(hashlib.md5(trace.encode("utf-8")).hexdigest(), 16) % 10
    if h <= 5:
        return "train"
    if h <= 7:
        return "val"
    return "test"


def _trace_family(path: str) -> str:
    p = Path(path)
    return p.parent.name if p.parent.name else "unknown"


def _as_common_rows(label_source: str, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        row = dict(r)
        if label_source == "heuristic":
            row["target_regret"] = float(row["rollout_regret_h"])
            row["target_loss"] = float(row["rollout_loss_h"])
            row["is_best"] = float(row["candidate_is_rollout_optimal"])
            row.setdefault("teacher_type", "heuristic_rollout")
        else:
            row["target_regret"] = float(row["teacher_regret"])
            row["target_loss"] = float(row["teacher_cost"])
            row["is_best"] = float(row["teacher_best"])
        t_val = int(row.get("request_t", row.get("t", 0)))
        cap_val = int(float(row["capacity"]))
        row["decision_key"] = f"{row['trace']}|c{cap_val}|t{t_val}"
        out.append(row)
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


def _xy(rows: List[Dict[str, object]], include_extra: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    x: List[List[float]] = []
    y: List[float] = []
    for r in rows:
        feat = [float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
        if include_extra:
            feat += [float(r["capacity"]), float(r["horizon"])]
        x.append(feat)
        y.append(float(r["target_regret"]))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _decision_metrics(rows: List[Dict[str, object]], pred: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = defaultdict(list)
    for r, p in zip(rows, pred):
        grouped[str(r["decision_key"])].append((r, float(p)))

    top1 = 0
    chosen_regret: List[float] = []
    chosen_gap: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        oracle = min(items, key=lambda x: (float(x[0]["target_regret"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == oracle[0]["candidate_page_id"])
        best_regret = min(float(x[0]["target_regret"]) for x in items)
        chosen_regret.append(float(chosen[0]["target_regret"]))
        chosen_gap.append(float(chosen[0]["target_regret"]) - best_regret)

    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_candidate_accuracy": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regret) if chosen_regret else 0.0),
        "mean_regret_vs_best": float(np.mean(chosen_gap) if chosen_gap else 0.0),
    }


def _reg_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)) if len(y) else 0.0,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))) if len(y) else 0.0,
    }


def _split_rows(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[_trace_split(str(r["trace"]))].append(r)
    return out


def _disagreement_stats(
    heuristic_rows: List[Dict[str, object]],
    teacher_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    by_decision_h: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_decision_t: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in heuristic_rows:
        by_decision_h[str(r["decision_key"])].append(r)
    for r in teacher_rows:
        by_decision_t[str(r["decision_key"])].append(r)

    common = sorted(set(by_decision_h).intersection(set(by_decision_t)))
    disagree = 0
    by_family: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "disagree": 0})
    flags: Dict[str, bool] = {}
    for d in common:
        hs = by_decision_h[d]
        ts = by_decision_t[d]
        h_best = {str(r["candidate_page_id"]) for r in hs if float(r["is_best"]) == 1.0}
        t_best = {str(r["candidate_page_id"]) for r in ts if float(r["is_best"]) == 1.0}
        is_disagree = h_best != t_best
        flags[d] = is_disagree
        disagree += int(is_disagree)
        fam = str(ts[0].get("family", "unknown"))
        by_family[fam]["total"] += 1
        by_family[fam]["disagree"] += int(is_disagree)

    family_rows = []
    for fam, v in sorted(by_family.items()):
        family_rows.append(
            {
                "family": fam,
                "decisions": v["total"],
                "disagree": v["disagree"],
                "disagree_rate": (v["disagree"] / v["total"]) if v["total"] else 0.0,
            }
        )

    return {
        "decisions_common": len(common),
        "disagree_count": disagree,
        "disagree_rate": (disagree / len(common)) if common else 0.0,
        "by_family": family_rows,
        "decision_flags": flags,
    }


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


def _run_model_policy_misses(
    requests: Sequence[Request],
    capacity: int,
    model: Pipeline,
    horizon: int,
) -> Tuple[int, int]:
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
            s = float(model.predict(x)[0])
            scored.append((cand, s))

        victim = min(scored, key=lambda x: (x[1], str(x[0])))[0]
        order.pop(victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses, hits


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

    split_heur = _split_rows(rows_heur)
    split_teacher = _split_rows(rows_teacher)

    model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])

    results_rows: List[Dict[str, object]] = []
    trained_models: Dict[str, Pipeline] = {}
    for label_source, split_rows in [("heuristic", split_heur), ("offline_teacher", split_teacher)]:
        x_train, y_train = _xy(split_rows["train"])
        clf = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        clf.fit(x_train, y_train)
        trained_models[label_source] = clf

        for split in ["train", "val", "test"]:
            x, y = _xy(split_rows[split])
            pred = clf.predict(x) if len(split_rows[split]) else np.asarray([], dtype=float)
            row: Dict[str, object] = {
                "label_source": label_source,
                "split": split,
                "rows": len(split_rows[split]),
                **_reg_metrics(y, pred),
                **_decision_metrics(split_rows[split], pred),
            }
            results_rows.append(row)

    # disagreement analysis between label sources (common decision IDs)
    disagreement = _disagreement_stats(rows_heur, rows_teacher)

    # downstream lightweight policy evaluation on held-out traces
    downstream_rows: List[Dict[str, object]] = []
    for trace_path in trace_paths:
        if _trace_split(trace_path) != "test":
            continue
        requests, _pages = load_trace(trace_path)
        if max_requests_per_trace > 0:
            requests = requests[:max_requests_per_trace]
        for cap in capacities:
            lru_m, lru_h = _run_lru_misses(requests, cap)
            for label_source in ["heuristic", "offline_teacher"]:
                misses, hits = _run_model_policy_misses(
                    requests=requests,
                    capacity=cap,
                    model=trained_models[label_source],
                    horizon=horizon,
                )
                downstream_rows.append(
                    {
                        "trace": trace_path,
                        "family": _trace_family(trace_path),
                        "capacity": cap,
                        "label_source": label_source,
                        "policy_misses": misses,
                        "policy_hit_rate": (hits / len(requests)) if requests else 0.0,
                        "lru_misses": lru_m,
                        "delta_misses_vs_lru": misses - lru_m,
                    }
                )

    # concentration analysis: gains on disagreement decisions
    def _chosen_regret_by_decision(rows: List[Dict[str, object]], model_: Pipeline) -> Dict[str, float]:
        x, _ = _xy(rows)
        pred = model_.predict(x)
        grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = defaultdict(list)
        for r, p in zip(rows, pred):
            grouped[str(r["decision_key"])].append((r, float(p)))
        out: Dict[str, float] = {}
        for d, items in grouped.items():
            chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
            out[d] = float(chosen[0]["target_regret"])
        return out

    test_teacher_rows = split_teacher["test"]
    by_decision_h = _chosen_regret_by_decision(test_teacher_rows, trained_models["heuristic"]) if test_teacher_rows else {}
    by_decision_t = _chosen_regret_by_decision(test_teacher_rows, trained_models["offline_teacher"]) if test_teacher_rows else {}

    agree_gains: List[float] = []
    disagree_gains: List[float] = []
    for d, t_reg in by_decision_t.items():
        h_reg = by_decision_h.get(d)
        if h_reg is None:
            continue
        gain = h_reg - t_reg
        if disagreement["decision_flags"].get(d, False):
            disagree_gains.append(gain)
        else:
            agree_gains.append(gain)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "results.csv", results_rows)
    _write_csv(output / "downstream_results.csv", downstream_rows)
    _write_csv(output / "disagreement_by_family.csv", disagreement["by_family"])

    summary = {
        "trace_glob": trace_glob,
        "capacities": capacities,
        "horizon": horizon,
        "rows_heuristic": len(rows_heur),
        "rows_offline_teacher": len(rows_teacher),
        "results": results_rows,
        "downstream_results": downstream_rows,
        "disagreement": {
            "decisions_common": disagreement["decisions_common"],
            "disagree_count": disagreement["disagree_count"],
            "disagree_rate": disagreement["disagree_rate"],
        },
        "gain_concentration": {
            "mean_gain_disagreement": float(mean(disagree_gains) if disagree_gains else 0.0),
            "mean_gain_agreement": float(mean(agree_gains) if agree_gains else 0.0),
            "count_disagreement": len(disagree_gains),
            "count_agreement": len(agree_gains),
        },
        "model_family": "ridge_regression_with_standard_scaler",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Offline-teacher vs heuristic supervision experiment",
        "",
        f"- Model family: `{summary['model_family']}`",
        f"- Traces: `{trace_glob}`",
        f"- Capacities: `{capacities}`",
        f"- Horizon: `{horizon}`",
        "",
        "## Key findings",
        f"- Decision disagreement rate (teacher-best vs heuristic-best): **{summary['disagreement']['disagree_rate']:.4f}**.",
        f"- Mean gain on disagreement decisions (heuristic regret - teacher regret): **{summary['gain_concentration']['mean_gain_disagreement']:.4f}**.",
        f"- Mean gain on agreement decisions: **{summary['gain_concentration']['mean_gain_agreement']:.4f}**.",
        "",
        "## Candidate quality metrics (rows from results.csv)",
        "",
        "| Label source | Split | Top-1 accuracy | Mean chosen regret | MAE | RMSE | Rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results_rows:
        lines.append(
            f"| {r['label_source']} | {r['split']} | {r['top1_candidate_accuracy']:.4f} | {r['mean_chosen_regret']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['rows']} |"
        )
    lines += [
        "",
        "## Downstream lightweight replay",
        "",
        "`downstream_results.csv` reports misses/hit-rate and delta vs LRU on held-out traces.",
        "",
        "## Interpretation",
        "",
        "Offline-teacher labels materially change supervision where disagreement is non-trivial.",
        "Any gains should be read together with disagreement concentration: if gains are mostly on disagreement regions, the teacher is adding novel signal.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare heuristic vs offline-teacher supervision")
    ap.add_argument("--trace-glob", default="data/example_*.json,data/example_general_caching.json")
    ap.add_argument("--capacities", default="2,3")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--output-dir", default="analysis/offline_teacher_vs_heuristic")
    args = ap.parse_args()

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    summary = run_experiment(
        trace_glob=args.trace_glob,
        capacities=capacities,
        horizon=args.horizon,
        max_requests_per_trace=args.max_requests_per_trace,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
