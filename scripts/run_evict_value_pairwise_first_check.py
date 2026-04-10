from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_pairwise_model_v1 import EvictValuePairwiseV1Model
from lafc.metrics.cost import hit_rate
from lafc.policies.evict_value_pairwise_v1 import EvictValuePairwiseV1Policy
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    out: List[Dict[str, object]] = []
    for row in raw:
        parsed: Dict[str, object] = dict(row)
        for k, v in list(parsed.items()):
            if k in {"request_t", "t", "capacity", "horizon", "candidate_count"}:
                parsed[k] = int(float(v))
            elif k.startswith(("delta_", "i_", "j_")) or k.startswith("rollout_") or k in {
                "label_i_better",
                "candidate_is_rollout_optimal",
            }:
                parsed[k] = float(v)
        out.append(parsed)
    return out


def _pointwise_offline_metrics(candidate_rows: List[Dict[str, object]], model_path: str) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)

    model = EvictValueV1Model.load(model_path) if Path(model_path).exists() else None
    top1 = 0
    regrets: List[float] = []
    for items in grouped.values():
        if model is not None:
            scored = [(it, model.predict_loss_one({c: float(it[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS})) for it in items]
            chosen = min(scored, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))[0]
        else:
            chosen = min(items, key=lambda x: (float(x.get("candidate_lru_score", 0.0)), str(x["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x["rollout_regret_h"]), str(x["candidate_page_id"])))
        top1 += int(str(chosen["candidate_page_id"]) == str(best["candidate_page_id"]))
        regrets.append(float(chosen["rollout_regret_h"]))
    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1": float(top1 / denom),
        "pairwise_accuracy": 0.0,
        "pairwise_acc": 0.0,
        "mean_regret": float(np.mean(regrets) if regrets else 0.0),
    }


def _fit_or_load_pairwise_model(pairwise_rows: List[Dict[str, object]], model_path: str) -> Tuple[EvictValuePairwiseV1Model, str]:
    if Path(model_path).exists():
        return EvictValuePairwiseV1Model.load(model_path), "artifact"

    delta_cols = sorted(c for c in pairwise_rows[0].keys() if c.startswith("delta_"))
    x = np.asarray([[float(r[c]) for c in delta_cols] for r in pairwise_rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in pairwise_rows], dtype=int)
    if len(set(int(v) for v in y.tolist())) < 2:
        only = int(y[0]) if len(y) else 0
        clf = DummyClassifier(strategy="constant", constant=only)
    else:
        clf = LogisticRegression(max_iter=600, random_state=7)
    clf.fit(x, y)
    return EvictValuePairwiseV1Model(model_name="logistic_regression_inline", estimator=clf, delta_feature_columns=delta_cols), "inline_fit"


def _pairwise_offline_metrics(
    pairwise_rows: List[Dict[str, object]],
    pairwise_model: EvictValuePairwiseV1Model,
) -> Dict[str, float]:
    decisions: Dict[str, Dict[str, Dict[str, float]]] = {}
    pair_correct = 0
    for row in pairwise_rows:
        did = str(row["decision_id"])
        d = decisions.setdefault(did, {"wins": {}, "regret": {}})
        ai = str(row["candidate_i_page_id"])
        bi = str(row["candidate_j_page_id"])
        af = {c.replace("i_", "", 1): float(row[c]) for c in row.keys() if c.startswith("i_")}
        bf = {c.replace("j_", "", 1): float(row[c]) for c in row.keys() if c.startswith("j_")}
        p = pairwise_model.predict_a_beats_b_proba(af, bf)
        pred = int(p >= 0.5)
        pair_correct += int(pred == int(float(row["label_i_better"])))
        d["wins"][ai] = float(d["wins"].get(ai, 0.0) + p)
        d["wins"][bi] = float(d["wins"].get(bi, 0.0) + (1.0 - p))
        d["regret"][ai] = float(row["rollout_regret_i"])
        d["regret"][bi] = float(row["rollout_regret_j"])

    top1 = 0
    regrets: List[float] = []
    for d in decisions.values():
        chosen = max(d["wins"].keys(), key=lambda c: (d["wins"][c], c))
        best = min(d["regret"].keys(), key=lambda c: (d["regret"][c], c))
        top1 += int(chosen == best)
        regrets.append(float(d["regret"][chosen]))
    denom = max(len(decisions), 1)
    return {
        "decision_count": float(len(decisions)),
        "top1": float(top1 / denom),
        "pairwise_accuracy": float(pair_correct / max(len(pairwise_rows), 1)),
        "pairwise_acc": float(pair_correct / max(len(pairwise_rows), 1)),
        "mean_regret": float(np.mean(regrets) if regrets else 0.0),
    }


def _make_stress_trace(page_ids: List[str], buckets: List[int], confs: List[float]):
    recs = [{"bucket": b, "confidence": c} for b, c in zip(buckets, confs)]
    return build_requests_from_lists(page_ids=page_ids, prediction_records=recs)


def _stress_traces():
    return {
        "stress::predictor_good_lru_bad": _make_stress_trace(
            ["A", "B", "C", "A", "D", "A", "B", "C", "A", "D"],
            [0, 3, 3, 0, 3, 0, 3, 3, 0, 3],
            [1.0] * 10,
        ),
        "stress::predictor_bad_lru_good": _make_stress_trace(
            ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"],
            [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
            [1.0] * 10,
        ),
    }


def _iter_traces():
    for p in ["data/example_unweighted.json", "data/example_atlas_v1.json"]:
        yield p, load_trace(p)
    for n, payload in _stress_traces().items():
        yield n, payload


def _hard_slice_decision_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lost = any(int(float(row.get(k, 0))) > 0 for k in row.keys() if k.startswith("evict_value_v1_diff_vs_"))
            if not lost:
                continue
            trace = str(row.get("trace_name", ""))
            t = int(float(row.get("t", 0)))
            cap = int(float(row.get("capacity", 0)))
            out.add(f"{trace}|c{cap}|t{t}")
    return out


def _decision_key_from_id(decision_id: str) -> str:
    left = decision_id.split("|h", 1)[0]
    return left


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare pointwise vs pairwise candidate-centric eviction learning")
    ap.add_argument("--candidate-csv", default="data/derived/evict_value_decision_aligned/candidate_rows.csv")
    ap.add_argument("--pairwise-csv", default="data/derived/evict_value_pairwise/pairwise_rows.csv")
    ap.add_argument("--output-dir", default="analysis/evict_value_pairwise_first_check")
    ap.add_argument("--summary-md", default="analysis/evict_value_pairwise_first_check.md")
    ap.add_argument("--pointwise-model", default="models/evict_value_v1_hist_gb.pkl")
    ap.add_argument("--pairwise-model", default="models/evict_value_pairwise_v1_best.pkl")
    args = ap.parse_args()

    candidate_rows = _read_csv(Path(args.candidate_csv)) if Path(args.candidate_csv).exists() else []
    pairwise_rows = _read_csv(Path(args.pairwise_csv)) if Path(args.pairwise_csv).exists() else []

    pairwise_model = None
    pairwise_model_source = "missing"
    if pairwise_rows:
        pairwise_model, pairwise_model_source = _fit_or_load_pairwise_model(pairwise_rows, args.pairwise_model)

    offline_rows: List[Dict[str, object]] = []
    if candidate_rows:
        offline_rows.append({"policy": "evict_value_v1", **_pointwise_offline_metrics(candidate_rows, args.pointwise_model)})
    if pairwise_rows and pairwise_model is not None:
        offline_rows.append({"policy": "evict_value_pairwise_v1", **_pairwise_offline_metrics(pairwise_rows, pairwise_model)})

    hard_ids = _hard_slice_decision_ids(Path("analysis/evict_value_failure_slice_audit.csv"))
    hard_slice_rows: List[Dict[str, object]] = []
    if hard_ids and candidate_rows:
        cand_subset = [r for r in candidate_rows if _decision_key_from_id(str(r["decision_id"])) in hard_ids]
        if cand_subset:
            hard_slice_rows.append({"policy": "evict_value_v1", **_pointwise_offline_metrics(cand_subset, args.pointwise_model)})
    if hard_ids and pairwise_rows and pairwise_model is not None:
        pair_subset = [r for r in pairwise_rows if _decision_key_from_id(str(r["decision_id"])) in hard_ids]
        if pair_subset:
            hard_slice_rows.append({"policy": "evict_value_pairwise_v1", **_pairwise_offline_metrics(pair_subset, pairwise_model)})

    capacities = [2, 3, 4]
    online_rows: List[Dict[str, object]] = []
    for trace_name, (reqs, pages) in _iter_traces():
        for cap in capacities:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            policies = {
                "evict_value_v1": run_policy(EvictValueV1Policy(model_path=args.pointwise_model), reqs, pages, cap),
                "predictive_marker": run_policy(PredictiveMarkerPolicy(), reqs, pages, cap),
                "trust_and_doubt": run_policy(TrustAndDoubtPolicy(seed=7), td_reqs, pages, cap),
                "rest_v1": run_policy(RestV1Policy(), reqs, pages, cap),
                "lru": run_policy(LRUPolicy(), reqs, pages, cap),
                "evict_value_pairwise_v1": run_policy(
                    EvictValuePairwiseV1Policy(model_path=args.pairwise_model if pairwise_model is not None else "missing.pkl"),
                    reqs,
                    pages,
                    cap,
                ),
            }
            for policy_name, result in policies.items():
                online_rows.append(
                    {
                        "trace": trace_name,
                        "capacity": cap,
                        "policy": policy_name,
                        "misses": result.total_misses,
                        "hit_rate": hit_rate(result.events),
                    }
                )

    by_policy = sorted({str(r["policy"]) for r in online_rows})
    mean_misses = {
        p: float(mean([float(r["misses"]) for r in online_rows if r["policy"] == p]))
        for p in by_policy
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write(out_dir / "policy_comparison.csv", online_rows)
    _write(out_dir / "offline_metrics.csv", offline_rows)
    _write(out_dir / "hard_slice_metrics.csv", hard_slice_rows)
    _write(out_dir / "metrics.csv", offline_rows)

    summary = {
        "pairwise_model_source": pairwise_model_source,
        "mean_misses": mean_misses,
        "offline": offline_rows,
        "hard_slice": hard_slice_rows,
        "hard_slice_available": bool(hard_ids),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# evict_value pairwise first check")
    lines.append("")
    lines.append("## Online policy comparison (mean misses)")
    for p in ["evict_value_v1", "evict_value_pairwise_v1", "predictive_marker", "trust_and_doubt", "rest_v1", "lru"]:
        if p in mean_misses:
            lines.append(f"- {p}: {mean_misses[p]:.3f}")
    lines.append("")
    lines.append("## Offline decision quality")
    for row in offline_rows:
        lines.append(
            f"- {row['policy']}: top1={float(row['top1']):.4f}, pairwise_acc={float(row['pairwise_acc']):.4f}, mean_regret={float(row['mean_regret']):.4f}"
        )
    lines.append("")
    lines.append("## Hard loss slice (evict_value_v1 losses in failure-slice audit)")
    if not hard_ids:
        lines.append("- Skipped: analysis/evict_value_failure_slice_audit.csv missing or empty.")
    elif not hard_slice_rows:
        lines.append("- Slice present but no overlapping decision rows in current offline dataset.")
    else:
        for row in hard_slice_rows:
            lines.append(
                f"- {row['policy']}: top1={float(row['top1']):.4f}, pairwise_acc={float(row['pairwise_acc']):.4f}, mean_regret={float(row['mean_regret']):.4f}"
            )
    lines.append("")
    lines.append("## Bottleneck read")
    lines.append("- If pairwise improves hard-slice top1/regret but not online misses, bottleneck likely feature coverage or horizon mismatch.")
    lines.append("- If pairwise does not improve offline hard slices either, objective mismatch is likely not the only blocker.")

    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
