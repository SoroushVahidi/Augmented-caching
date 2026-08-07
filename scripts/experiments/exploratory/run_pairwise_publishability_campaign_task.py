from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lafc.evict_value_pairwise_model_v1 import EvictValuePairwiseV1Model
from lafc.evict_value_v2_rollout import EvictValueV2RolloutConfig, build_pairwise_rows_from_candidate_rows, build_rollout_candidate_rows_v2
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.metrics.cost import hit_rate
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.evict_value_pairwise_v1 import EvictValuePairwiseV1Policy
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.guard_wrapper import EvictValueV1GuardedPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.ml_gate_v2 import MLGateV2Policy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.robust_ftp_marker_combiner import RobustFtPDeterministicMarkerCombiner
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import Request, load_trace


def _read_manifest(manifest: Path, start: int, count: int) -> List[Dict[str, str]]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if count <= 0:
        return rows[start:]
    return rows[start : start + count]


def _split(trace_name: str) -> str:
    bucket = int(hashlib.md5(trace_name.encode("utf-8")).hexdigest(), 16) % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "val"
    return "test"


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _pair_xy(rows: List[Dict[str, object]], delta_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    w = np.asarray([float(r.get("sample_weight", 1.0)) for r in rows], dtype=float)
    return x, y, w


def _positive_prob(estimator: object, x: np.ndarray) -> np.ndarray:
    probs = estimator.predict_proba(x)
    if probs.shape[1] == 1:
        cls = int(getattr(estimator, "classes_", [0])[0])
        return np.asarray([float(cls)] * len(x), dtype=float)
    return probs[:, 1]


def _decision_offline(rows: List[Dict[str, object]], p_i_better: np.ndarray) -> Dict[str, float]:
    decisions: Dict[str, Dict[str, Dict[str, float]]] = {}
    pair_acc_num = 0
    for row, p in zip(rows, p_i_better):
        d = decisions.setdefault(str(row["decision_id"]), {"wins": {}, "regret": {}})
        ai = str(row["candidate_i_page_id"])
        bi = str(row["candidate_j_page_id"])
        pair_acc_num += int(int(p >= 0.5) == int(float(row["label_i_better"])))
        d["wins"][ai] = float(d["wins"].get(ai, 0.0) + p)
        d["wins"][bi] = float(d["wins"].get(bi, 0.0) + (1.0 - p))
        d["regret"][ai] = float(row["rollout_regret_i"])
        d["regret"][bi] = float(row["rollout_regret_j"])
    top1 = 0
    chosen_regrets: List[float] = []
    best_regrets: List[float] = []
    for d in decisions.values():
        chosen = max(d["wins"].keys(), key=lambda c: (d["wins"][c], c))
        best = min(d["regret"].keys(), key=lambda c: (d["regret"][c], c))
        top1 += int(chosen == best)
        chosen_regrets.append(float(d["regret"][chosen]))
        best_regrets.append(float(d["regret"][best]))
    denom = max(len(decisions), 1)
    return {
        "decision_count": float(len(decisions)),
        "pairwise_accuracy": float(pair_acc_num / max(len(rows), 1)),
        "top1_reconstruction": float(top1 / denom),
        "mean_chosen_regret": float(np.mean(chosen_regrets) if chosen_regrets else 0.0),
        "mean_regret_vs_best": float(np.mean([c - b for c, b in zip(chosen_regrets, best_regrets)]) if chosen_regrets else 0.0),
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _load_trace_any(path: str) -> Tuple[List[Request], Dict[str, object]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        req_objs, pages, _src = load_trace_from_any(path)
        return req_objs, pages
    reqs, pages = load_trace(path)
    return reqs, pages


def _fit_model(model_family: str, x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> object:
    if len(y) == 0:
        clf = DummyClassifier(strategy="constant", constant=0)
        clf.fit(np.zeros((1, max(1, x.shape[1] if x.ndim == 2 else 1))), np.asarray([0]))
        return clf
    if len(set(int(v) for v in y.tolist())) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
        clf.fit(x, y)
        return clf
    if model_family == "logistic":
        clf = LogisticRegression(max_iter=900, random_state=seed)
        clf.fit(x, y, sample_weight=w)
        return clf
    if model_family == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1)
        clf.fit(x, y, sample_weight=w)
        return clf
    if model_family == "histgb":
        clf = HistGradientBoostingClassifier(max_depth=7, max_iter=300, random_state=seed)
        clf.fit(x, y, sample_weight=w)
        return clf
    raise ValueError(f"unknown model family: {model_family}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Worker task for pairwise publishability campaign.")
    ap.add_argument("--manifest", type=Path, default=Path("analysis/wulver_trace_manifest_full.csv"))
    ap.add_argument("--trace-start", type=int, default=0)
    ap.add_argument("--trace-count", type=int, default=3)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--horizons", default="4,8,16")
    ap.add_argument("--max-requests-per-trace", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-family", choices=["logistic", "rf", "histgb"], default="logistic")
    ap.add_argument("--label-variant", choices=["head_pair", "regret_diff"], default="head_pair")
    ap.add_argument("--label-noise", type=float, default=0.0, help="Probability of flipping pairwise labels in train split.")
    ap.add_argument("--job-label", default="task0")
    ap.add_argument("--out-root", type=Path, default=Path("analysis/pairwise_publishability_campaign/jobs"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = _read_manifest(args.manifest, args.trace_start, args.trace_count)
    caps = _parse_ints(args.capacities)
    horizons = _parse_ints(args.horizons)

    candidate_rows: List[Dict[str, object]] = []
    eval_traces: List[Tuple[str, str]] = []
    for row in rows:
        path = row["path"].strip()
        trace_name = row.get("trace_name", "").strip() or path
        trace_family = row.get("trace_family", "").strip() or "unknown"
        reqs, _pages = _load_trace_any(path)
        reqs = reqs[: args.max_requests_per_trace] if args.max_requests_per_trace > 0 else reqs
        eval_traces.append((path, trace_family))
        for c in caps:
            cfg = EvictValueV2RolloutConfig(horizons=tuple(horizons), reference_policy="lru")
            candidate_rows.extend(build_rollout_candidate_rows_v2(requests=reqs, capacity=c, trace_name=trace_name, trace_family=trace_family, cfg=cfg))

    pair_rows = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=False)
    if not pair_rows:
        raise ValueError("No pairwise rows generated for task")
    delta_cols = sorted(c for c in pair_rows[0].keys() if c.startswith("delta_"))

    # Label variants + optional noise/degradation.
    for r in pair_rows:
        if args.label_variant == "head_pair":
            lbl = int(float(r["rollout_regret_i"]) <= float(r["rollout_regret_j"]))
            r["label_i_better"] = lbl
            r["sample_weight"] = 1.0
        else:
            lbl = int(float(r["rollout_regret_i"]) <= float(r["rollout_regret_j"]))
            r["label_i_better"] = lbl
            r["sample_weight"] = max(1e-6, abs(float(r["rollout_regret_i"]) - float(r["rollout_regret_j"])))

    split_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for r in pair_rows:
        split_rows[_split(str(r["trace"]))].append(r)

    if args.label_noise > 0:
        for r in split_rows["train"]:
            if rng.random() < args.label_noise:
                r["label_i_better"] = 1 - int(float(r["label_i_better"]))

    train_rows = split_rows["train"]
    if not train_rows:
        train_rows = split_rows["val"] if split_rows["val"] else split_rows["test"]
    x_train, y_train, w_train = _pair_xy(train_rows, delta_cols)
    clf = _fit_model(args.model_family, x_train, y_train, w_train, args.seed)

    out_dir = args.out_root / args.job_label
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "pairwise_model.pkl"
    EvictValuePairwiseV1Model(model_name=f"{args.model_family}_{args.label_variant}", estimator=clf, delta_feature_columns=delta_cols).save(model_path)

    offline_rows: List[Dict[str, object]] = []
    for split in ["train", "val", "test"]:
        rows_split = split_rows[split]
        if not rows_split:
            continue
        x, _y, _w = _pair_xy(rows_split, delta_cols)
        prob = _positive_prob(clf, x)
        m = _decision_offline(rows_split, prob)
        offline_rows.append({"split": split, "model_family": args.model_family, "label_variant": args.label_variant, "label_noise": args.label_noise, **m})

    # per-family/horizon on test
    family_rows: List[Dict[str, object]] = []
    horizon_rows: List[Dict[str, object]] = []
    test_rows = split_rows["test"]
    if test_rows:
        for fam in sorted({str(r["trace_family"]) for r in test_rows}):
            sub = [r for r in test_rows if str(r["trace_family"]) == fam]
            x, _y, _w = _pair_xy(sub, delta_cols)
            family_rows.append({"family": fam, **_decision_offline(sub, _positive_prob(clf, x))})
        for h in sorted({int(r["horizon"]) for r in test_rows}):
            sub = [r for r in test_rows if int(r["horizon"]) == h]
            x, _y, _w = _pair_xy(sub, delta_cols)
            horizon_rows.append({"horizon": h, **_decision_offline(sub, _positive_prob(clf, x))})

    # online evaluation
    online_rows: List[Dict[str, object]] = []
    baseline_wins: Dict[str, Dict[str, int]] = defaultdict(lambda: {"W": 0, "T": 0, "L": 0})
    for path, family in eval_traces:
        reqs, pages = _load_trace_any(path)
        reqs = reqs[: args.max_requests_per_trace] if args.max_requests_per_trace > 0 else reqs
        for cap in caps:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            policy_specs = {
                "evict_value_pairwise_v1": (EvictValuePairwiseV1Policy(model_path=str(model_path), scorer_mode="artifact"), reqs),
                "lru": (LRUPolicy(), reqs),
                "blind_oracle": (BlindOraclePolicy(), reqs),
                "predictive_marker": (PredictiveMarkerPolicy(), reqs),
                "trust_and_doubt": (TrustAndDoubtPolicy(seed=7), td_reqs),
                "robust_ftp_d_marker": (RobustFtPDeterministicMarkerCombiner(), td_reqs),
                "ml_gate_v2": (MLGateV2Policy(), reqs),
                "evict_value_v1": (EvictValueV1Policy(), reqs),
                "evict_value_v1_guarded": (EvictValueV1GuardedPolicy(), reqs),
                "atlas_v3": (AtlasV3Policy(), reqs),
                "rest_v1": (RestV1Policy(), reqs),
            }
            results: Dict[str, object] = {}
            for pol, (policy_obj, req_stream) in policy_specs.items():
                try:
                    results[pol] = run_policy(policy_obj, req_stream, pages, cap)
                except Exception:
                    continue
            if "evict_value_pairwise_v1" not in results:
                continue
            pair_misses = float(results["evict_value_pairwise_v1"].total_misses)
            for pol, res in results.items():
                online_rows.append(
                    {
                        "trace": path,
                        "family": family,
                        "capacity": cap,
                        "policy": pol,
                        "misses": float(res.total_misses),
                        "hit_rate": float(hit_rate(res.events)),
                    }
                )
                if pol == "evict_value_pairwise_v1":
                    continue
                other = float(res.total_misses)
                if pair_misses < other:
                    baseline_wins[pol]["W"] += 1
                elif pair_misses > other:
                    baseline_wins[pol]["L"] += 1
                else:
                    baseline_wins[pol]["T"] += 1

    _write_csv(out_dir / "offline_metrics.csv", offline_rows)
    _write_csv(out_dir / "offline_per_family.csv", family_rows)
    _write_csv(out_dir / "offline_per_horizon.csv", horizon_rows)
    _write_csv(out_dir / "online_metrics.csv", online_rows)
    _write_csv(out_dir / "wtl_vs_baselines.csv", [{"baseline": b, **v} for b, v in sorted(baseline_wins.items())])

    # recommendation heuristic per task.
    off_test = next((r for r in offline_rows if r["split"] == "test"), None)
    online_by_pol: Dict[str, float] = {}
    if online_rows:
        by_pol: Dict[str, List[float]] = defaultdict(list)
        for r in online_rows:
            by_pol[str(r["policy"])].append(float(r["misses"]))
        online_by_pol = {k: float(mean(v)) for k, v in sorted(by_pol.items())}
    rec = "not_promising_enough"
    if off_test and online_by_pol:
        pair_off = float(off_test["mean_regret_vs_best"])
        pair_online = online_by_pol.get("evict_value_pairwise_v1", 1e18)
        ev1 = online_by_pol.get("evict_value_v1", 1e18)
        rest = online_by_pol.get("rest_v1", 1e18)
        if pair_off <= 0.25 and pair_online <= min(ev1, rest):
            rec = "promising_mainline"
        elif pair_off <= 0.5 or pair_online <= min(ev1, rest) * 1.03:
            rec = "promising_appendix_ablation"

    summary = {
        "job_label": args.job_label,
        "trace_start": args.trace_start,
        "trace_count": args.trace_count,
        "capacities": caps,
        "horizons": horizons,
        "model_family": args.model_family,
        "label_variant": args.label_variant,
        "label_noise": args.label_noise,
        "rows": {"candidate": len(candidate_rows), "pairwise": len(pair_rows)},
        "offline_test": off_test,
        "online_mean_misses": online_by_pol,
        "recommendation": rec,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
