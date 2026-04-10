from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_pairwise_model_v1 import EvictValuePairwiseV1Model
from lafc.evict_value_v2_rollout import EvictValueV2RolloutConfig, build_pairwise_rows_from_candidate_rows, build_rollout_candidate_rows_v2
from lafc.metrics.cost import hit_rate
from lafc.policies.evict_value_pairwise_v1 import EvictValuePairwiseV1Policy
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


def _resolve_trace_paths(trace_glob: str) -> List[str]:
    out: List[str] = []
    for pat in [p.strip() for p in trace_glob.split(",") if p.strip()]:
        out.extend(sorted(str(p) for p in Path().glob(pat) if p.exists()))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError(f"No traces matched --trace-glob={trace_glob}")
    return uniq


def _trace_family(path: str) -> str:
    name = Path(path).name.lower()
    if "atlas" in name:
        return "atlas"
    if "general" in name:
        return "general"
    if "unweighted" in name:
        return "legacy"
    return "misc"


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty integer list")
    return vals


def _split_by_trace(trace_paths: List[str]) -> Dict[str, str]:
    traces = sorted(trace_paths)
    split_by_trace: Dict[str, str] = {}
    if len(traces) >= 3:
        order = ["train", "val", "test"]
        for i, t in enumerate(traces):
            split_by_trace[t] = order[i % 3]
        return split_by_trace

    # deterministic fallback for tiny trace count: hashed split, then force non-empty val/test when possible
    for t in traces:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % 10
        split_by_trace[t] = "train" if h < 6 else ("val" if h < 8 else "test")
    if traces and not any(v == "val" for v in split_by_trace.values()):
        split_by_trace[traces[0]] = "val"
    if len(traces) > 1 and not any(v == "test" for v in split_by_trace.values()):
        split_by_trace[traces[-1]] = "test"
    return split_by_trace


def _build_candidate_rows(trace_paths: Sequence[str], capacities: Sequence[int], horizons: Sequence[int], max_requests_per_trace: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trace in trace_paths:
        reqs, _pages = load_trace(trace)
        if max_requests_per_trace > 0:
            reqs = reqs[:max_requests_per_trace]
        fam = _trace_family(trace)
        for cap in capacities:
            cfg = EvictValueV2RolloutConfig(horizons=tuple(horizons), reference_policy="lru")
            out = build_rollout_candidate_rows_v2(requests=reqs, capacity=cap, trace_name=trace, trace_family=fam, cfg=cfg)
            rows.extend(out)
    return rows


def _rows_split(rows: List[Dict[str, object]], split_by_trace: Dict[str, str]) -> Dict[str, List[Dict[str, object]]]:
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[split_by_trace.get(str(r["trace"]), "train")].append(r)
    return out


def _xy_pointwise(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.asarray([float(r["rollout_regret_h"]) for r in rows], dtype=float)
    return x, y


def _xy_pairwise(rows: List[Dict[str, object]], delta_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    return x, y


def _positive_prob(clf: object, x: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(x)
    if probs.shape[1] == 1:
        cls = int(getattr(clf, "classes_", [0])[0])
        return np.asarray([float(cls)] * len(x), dtype=float)
    return probs[:, 1]


def _pointwise_offline(rows: List[Dict[str, object]], model: EvictValueV1Model) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[str(r["decision_id"])].append(r)
    top1 = 0
    regrets: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda r: (model.predict_loss_one({c: float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS}), str(r["candidate_page_id"])))
        best = min(items, key=lambda r: (float(r["rollout_regret_h"]), str(r["candidate_page_id"])))
        top1 += int(str(chosen["candidate_page_id"]) == str(best["candidate_page_id"]))
        regrets.append(float(chosen["rollout_regret_h"]))
    denom = max(len(grouped), 1)
    return {"decision_count": float(len(grouped)), "top1": float(top1 / denom), "pairwise_accuracy": 0.0, "mean_regret": float(np.mean(regrets) if regrets else 0.0)}


def _pairwise_offline(rows: List[Dict[str, object]], model: EvictValuePairwiseV1Model) -> Dict[str, float]:
    decisions: Dict[str, Dict[str, Dict[str, float]]] = {}
    correct = 0
    for r in rows:
        d = decisions.setdefault(str(r["decision_id"]), {"wins": {}, "regret": {}})
        ai = str(r["candidate_i_page_id"])
        bj = str(r["candidate_j_page_id"])
        af = {k.replace("i_", "", 1): float(v) for k, v in r.items() if k.startswith("i_")}
        bf = {k.replace("j_", "", 1): float(v) for k, v in r.items() if k.startswith("j_")}
        p = model.predict_a_beats_b_proba(af, bf)
        correct += int(int(p >= 0.5) == int(float(r["label_i_better"])))
        d["wins"][ai] = float(d["wins"].get(ai, 0.0) + p)
        d["wins"][bj] = float(d["wins"].get(bj, 0.0) + (1.0 - p))
        d["regret"][ai] = float(r["rollout_regret_i"])
        d["regret"][bj] = float(r["rollout_regret_j"])
    top1 = 0
    regrets: List[float] = []
    for d in decisions.values():
        chosen = max(d["wins"].keys(), key=lambda c: (d["wins"][c], c))
        best = min(d["regret"].keys(), key=lambda c: (d["regret"][c], c))
        top1 += int(chosen == best)
        regrets.append(float(d["regret"][chosen]))
    denom = max(len(decisions), 1)
    return {"decision_count": float(len(decisions)), "top1": float(top1 / denom), "pairwise_accuracy": float(correct / max(len(rows), 1)), "mean_regret": float(np.mean(regrets) if regrets else 0.0)}


def _hard_slice_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lost = any(int(float(row.get(k, 0))) > 0 for k in row if k.startswith("evict_value_v1_diff_vs_"))
            if lost:
                out.add(f"{row.get('trace_name','')}|c{int(float(row.get('capacity',0)))}|t{int(float(row.get('t',0)))}")
    return out


def _decision_key(decision_id: str) -> str:
    return decision_id.split("|h", 1)[0]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _group_mean(rows: List[Dict[str, object]], key: str) -> Dict[str, float]:
    out: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        out[str(r[key])].append(float(r["misses"]))
    return {k: float(mean(v)) for k, v in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-trace pairwise vs pointwise eviction evaluation")
    ap.add_argument("--trace-glob", default="data/example*.json")
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--horizons", default="4,8")
    ap.add_argument("--max-requests-per-trace", type=int, default=500)
    ap.add_argument("--output-dir", default="analysis/evict_value_pairwise_multitrace_eval")
    ap.add_argument("--summary-md", default="analysis/evict_value_pairwise_multitrace_eval.md")
    args = ap.parse_args()

    trace_paths = _resolve_trace_paths(args.trace_glob)
    capacities = _parse_int_list(args.capacities)
    horizons = _parse_int_list(args.horizons)
    split_by_trace = _split_by_trace(trace_paths)

    candidate_rows = _build_candidate_rows(trace_paths, capacities, horizons, args.max_requests_per_trace)
    pairwise_rows = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=False)
    delta_cols = sorted(c for c in pairwise_rows[0].keys() if c.startswith("delta_")) if pairwise_rows else []

    csplit = _rows_split(candidate_rows, split_by_trace)
    psplit = _rows_split(pairwise_rows, split_by_trace)

    # pointwise model selection
    point_models = {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "rf": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=7),
    }
    x_ptrain, y_ptrain = _xy_pointwise(csplit["train"])
    model_rows: List[Dict[str, object]] = []
    best_p_name = "ridge"
    best_p_model = None
    best_p_val = float("inf")
    for name, est in point_models.items():
        est.fit(x_ptrain, y_ptrain)
        for split in ["train", "val", "test"]:
            if not csplit[split]:
                continue
            x, y = _xy_pointwise(csplit[split])
            pred = est.predict(x)
            mae = float(np.mean(np.abs(pred - y))) if len(y) else 0.0
            model_rows.append({"family": "pointwise", "model": name, "split": split, "metric": "mae", "value": mae})
            if split == "val" and mae < best_p_val:
                best_p_val = mae
                best_p_name = name
                best_p_model = est
    if best_p_model is None:
        best_p_model = next(iter(point_models.values()))
        best_p_model.fit(x_ptrain, y_ptrain)
    point_model = EvictValueV1Model(model_name=f"multitrace_{best_p_name}", estimator=best_p_model, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS))
    point_model_path = Path("models/evict_value_v1_multitrace_best.pkl")
    point_model.save(point_model_path)

    # pairwise model selection
    pair_models: Dict[str, object] = {"logistic": LogisticRegression(max_iter=700, random_state=7)}
    x_qtrain, y_qtrain = _xy_pairwise(psplit["train"], delta_cols)
    best_q_name = "logistic"
    best_q_model = None
    best_q_val = -1.0
    for name, clf in pair_models.items():
        if len(set(int(v) for v in y_qtrain.tolist())) < 2:
            clf = DummyClassifier(strategy="constant", constant=int(y_qtrain[0]) if len(y_qtrain) else 0)
        clf.fit(x_qtrain, y_qtrain)
        for split in ["train", "val", "test"]:
            if not psplit[split]:
                continue
            x, y = _xy_pairwise(psplit[split], delta_cols)
            pred = clf.predict(x)
            acc = float(np.mean(pred == y)) if len(y) else 0.0
            model_rows.append({"family": "pairwise", "model": name, "split": split, "metric": "pairwise_accuracy", "value": acc})
            if split == "val" and acc > best_q_val:
                best_q_val = acc
                best_q_name = name
                best_q_model = clf
    if best_q_model is None:
        best_q_model = next(iter(pair_models.values()))
        if len(set(int(v) for v in y_qtrain.tolist())) < 2:
            best_q_model = DummyClassifier(strategy="constant", constant=int(y_qtrain[0]) if len(y_qtrain) else 0)
        best_q_model.fit(x_qtrain, y_qtrain)
    pair_model = EvictValuePairwiseV1Model(model_name=f"multitrace_{best_q_name}", estimator=best_q_model, delta_feature_columns=delta_cols)
    pair_model_path = Path("models/evict_value_pairwise_v1_multitrace_best.pkl")
    pair_model.save(pair_model_path)

    # offline metrics on test split
    offline_rows: List[Dict[str, object]] = []
    offline_rows.append({"policy": "evict_value_v1", **_pointwise_offline(csplit["test"], point_model)})
    offline_rows.append({"policy": "evict_value_pairwise_v1", **_pairwise_offline(psplit["test"], pair_model)})

    hard_ids = _hard_slice_ids(Path("analysis/evict_value_failure_slice_audit.csv"))
    hard_rows: List[Dict[str, object]] = []
    if hard_ids:
        hard_c = [r for r in csplit["test"] if _decision_key(str(r["decision_id"])) in hard_ids]
        hard_p = [r for r in psplit["test"] if _decision_key(str(r["decision_id"])) in hard_ids]
        if hard_c:
            hard_rows.append({"policy": "evict_value_v1", **_pointwise_offline(hard_c, point_model)})
        if hard_p:
            hard_rows.append({"policy": "evict_value_pairwise_v1", **_pairwise_offline(hard_p, pair_model)})

    # online eval uses test traces only for stable holdout
    test_traces = [t for t, s in split_by_trace.items() if s == "test"]
    policy_rows: List[Dict[str, object]] = []
    skipped_notes: List[str] = []
    for trace in test_traces:
        reqs, pages = load_trace(trace)
        if args.max_requests_per_trace > 0:
            reqs = reqs[: args.max_requests_per_trace]
        fam = _trace_family(trace)
        for cap in capacities:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            try:
                results = {
                    "evict_value_v1": run_policy(EvictValueV1Policy(model_path=str(point_model_path), scorer_mode="artifact"), reqs, pages, cap),
                    "evict_value_pairwise_v1": run_policy(EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"), reqs, pages, cap),
                    "predictive_marker": run_policy(PredictiveMarkerPolicy(), reqs, pages, cap),
                    "trust_and_doubt": run_policy(TrustAndDoubtPolicy(seed=7), td_reqs, pages, cap),
                    "rest_v1": run_policy(RestV1Policy(), reqs, pages, cap),
                    "lru": run_policy(LRUPolicy(), reqs, pages, cap),
                }
            except Exception as exc:  # graceful skip
                skipped_notes.append(f"skip trace={trace} cap={cap}: {exc}")
                continue
            for pol, res in results.items():
                policy_rows.append({"trace": trace, "family": fam, "capacity": cap, "policy": pol, "misses": res.total_misses, "hit_rate": hit_rate(res.events)})

    _write_csv(Path(args.output_dir) / "policy_comparison.csv", policy_rows)
    _write_csv(Path(args.output_dir) / "offline_metrics.csv", offline_rows)
    _write_csv(Path(args.output_dir) / "model_selection.csv", model_rows)

    split_counts = {
        s: {
            "candidate_rows": len(csplit[s]),
            "candidate_decisions": len({str(r['decision_id']) for r in csplit[s]}),
            "pairwise_rows": len(psplit[s]),
            "pairwise_decisions": len({str(r['decision_id']) for r in psplit[s]}),
            "trace_count": len([t for t, ss in split_by_trace.items() if ss == s]),
        }
        for s in ["train", "val", "test"]
    }

    overall_mean = _group_mean(policy_rows, "policy") if policy_rows else {}
    by_family: Dict[str, Dict[str, float]] = {}
    by_cap: Dict[str, Dict[str, float]] = {}
    by_horizon: Dict[str, Dict[str, float]] = {}
    for fam in sorted({str(r["family"]) for r in policy_rows}):
        by_family[fam] = _group_mean([r for r in policy_rows if str(r["family"]) == fam], "policy")
    for cap in sorted({int(r["capacity"]) for r in policy_rows}):
        by_cap[str(cap)] = _group_mean([r for r in policy_rows if int(r["capacity"]) == cap], "policy")

    # per-horizon from offline test rows
    for h in sorted({int(r["horizon"]) for r in csplit["test"]}):
        ch = [r for r in csplit["test"] if int(r["horizon"]) == h]
        ph = [r for r in psplit["test"] if int(r["horizon"]) == h]
        by_horizon[str(h)] = {
            "pointwise_top1": _pointwise_offline(ch, point_model)["top1"] if ch else 0.0,
            "pairwise_top1": _pairwise_offline(ph, pair_model)["top1"] if ph else 0.0,
        }

    summary = {
        "trace_glob": args.trace_glob,
        "resolved_trace_count": len(trace_paths),
        "capacities": capacities,
        "horizons": horizons,
        "split_by_trace": split_by_trace,
        "split_counts": split_counts,
        "model_winners": {"pointwise": best_p_name, "pairwise": best_q_name},
        "overall_online_mean_misses": overall_mean,
        "offline_test": offline_rows,
        "hard_slice_test": hard_rows,
        "per_family_online_mean_misses": by_family,
        "per_capacity_online_mean_misses": by_cap,
        "per_horizon_offline_top1": by_horizon,
        "skipped_notes": skipped_notes,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = ["# evict_value_pairwise multitrace eval", ""]
    lines.append("## A. Overall online mean misses")
    for p in ["evict_value_v1", "evict_value_pairwise_v1", "predictive_marker", "trust_and_doubt", "rest_v1", "lru"]:
        if p in overall_mean:
            lines.append(f"- {p}: {overall_mean[p]:.4f}")
    lines.append("")
    lines.append("## B. Offline test metrics (pointwise vs pairwise)")
    for r in offline_rows:
        lines.append(f"- {r['policy']}: top1={r['top1']:.4f}, pairwise_accuracy={r['pairwise_accuracy']:.4f}, mean_regret={r['mean_regret']:.4f}")
    lines.append("")
    lines.append("## C. Per-family breakdown")
    for fam, vals in by_family.items():
        lines.append(f"- {fam}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))
    lines.append("")
    lines.append("## D. Per-capacity breakdown")
    for cap, vals in by_cap.items():
        lines.append(f"- cap={cap}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))
    lines.append("")
    lines.append("## E. Per-horizon breakdown (offline top1)")
    for h, vals in by_horizon.items():
        lines.append(f"- h={h}: pointwise_top1={vals['pointwise_top1']:.4f}, pairwise_top1={vals['pairwise_top1']:.4f}")
    lines.append("")
    lines.append("## F. Hard-slice breakdown")
    if not hard_rows:
        lines.append("- Hard-slice unavailable or no overlap with test split.")
    else:
        for r in hard_rows:
            lines.append(f"- {r['policy']}: top1={r['top1']:.4f}, pairwise_accuracy={r['pairwise_accuracy']:.4f}, mean_regret={r['mean_regret']:.4f}")
    lines.append("")
    lines.append("## G. Direct answers")
    mm = overall_mean
    lines.append(f"- pairwise beats pointwise overall: {'yes' if mm.get('evict_value_pairwise_v1',1e9) < mm.get('evict_value_v1',1e9) else 'no'}")
    if hard_rows:
        hr = {r['policy']: r for r in hard_rows}
        lines.append(f"- pairwise beats pointwise on hard slices: {'yes' if hr.get('evict_value_pairwise_v1',{}).get('top1',0) >= hr.get('evict_value_v1',{}).get('top1',0) else 'no'}")
    else:
        lines.append("- pairwise beats pointwise on hard slices: unclear")
    lines.append(f"- pairwise beats rest_v1 online: {'yes' if mm.get('evict_value_pairwise_v1',1e9) < mm.get('rest_v1',1e9) else 'no'}")
    if 'predictive_marker' in mm and 'trust_and_doubt' in mm and 'evict_value_pairwise_v1' in mm:
        lines.append(f"- pairwise gap to predictive_marker (misses): {mm['evict_value_pairwise_v1'] - mm['predictive_marker']:.4f}")
        lines.append(f"- pairwise gap to trust_and_doubt (misses): {mm['evict_value_pairwise_v1'] - mm['trust_and_doubt']:.4f}")
    lines.append("- remaining weakness signal: if pairwise wins offline but not vs strongest baselines online, likely feature weakness and deployment-rule gap under current data scale.")

    if skipped_notes:
        lines.append("")
        lines.append("## Skipped notes")
        lines.extend(f"- {n}" for n in skipped_notes)

    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
