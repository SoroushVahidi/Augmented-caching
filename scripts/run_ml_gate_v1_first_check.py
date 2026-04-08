from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from lafc.metrics.cost import hit_rate
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.lru import LRUPolicy
from lafc.policies.ml_gate_v1 import MLGateV1Policy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace

OUT_CSV = Path("analysis/ml_gate_v1_first_check.csv")
OUT_MD = Path("analysis/ml_gate_v1_first_check.md")
METRICS_JSON = Path("analysis/ml_gate_v1_metrics.json")
CAPACITIES = [2, 3, 4]
BASE_TRACES = ["data/example_unweighted.json", "data/example_atlas_v1.json"]


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
    for p in BASE_TRACES:
        yield p, load_trace(p)
    for n, payload in _stress_traces().items():
        yield n, payload


def _row(trace: str, cap: int, policy: str, res) -> Dict[str, object]:
    row = {
        "trace": trace,
        "capacity": cap,
        "policy": policy,
        "misses": res.total_misses,
        "hit_rate": hit_rate(res.events),
        "trust_coverage": "",
        "mean_gate_probability": "",
    }
    extra = res.extra_diagnostics or {}
    if policy == "ml_gate_v1":
        s = (extra.get("ml_gate_v1") or {}).get("summary", {})
        row["trust_coverage"] = s.get("trust_coverage", 0.0)
        row["mean_gate_probability"] = s.get("mean_gate_probability", 0.0)
    return row


def main() -> None:
    rows: List[Dict[str, object]] = []
    for trace_name, (requests, pages) in _iter_traces():
        for cap in CAPACITIES:
            td_requests = attach_predicted_caches(requests, capacity=cap)
            results = {
                "ml_gate_v1": run_policy(MLGateV1Policy(), requests, pages, cap),
                "lru": run_policy(LRUPolicy(), requests, pages, cap),
                "blind_oracle": run_policy(BlindOraclePolicy(), requests, pages, cap),
                "predictive_marker": run_policy(PredictiveMarkerPolicy(), requests, pages, cap),
                "trust_and_doubt": run_policy(TrustAndDoubtPolicy(seed=7), td_requests, pages, cap),
                "blind_oracle_lru_combiner": run_policy(BlindOracleLRUCombiner(), requests, pages, cap),
                "atlas_v3": run_policy(AtlasV3Policy(), requests, pages, cap),
                "rest_v1": run_policy(RestV1Policy(), requests, pages, cap),
            }
            for name, res in results.items():
                rows.append(_row(trace_name, cap, name, res))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}

    def _mean(policy: str) -> float:
        vals = [float(r["misses"]) for r in rows if r["policy"] == policy]
        return mean(vals) if vals else 0.0

    md = []
    md.append("# ml_gate_v1 first check")
    md.append("")
    md.append("Experimental, proof-of-concept, lightweight learned gate.")
    md.append("")
    md.append("## Training data")
    md.append("- Traces: data/example_unweighted.json + data/example_atlas_v1.json (sample-first mode by default).")
    md.append("- Capacities: 2/3/4.")
    md.append("- Label horizon H=4.")
    md.append("")
    md.append("## Label definition")
    md.append("- y=1 iff predictor-victim penalty < LRU-victim penalty, where penalty=1 if the evicted page returns within H requests, else 0.")
    md.append("- ties map to y=0 (abstain/fallback preference).")
    md.append("")
    md.append("## Feature set")
    md.append("- request bucket/confidence")
    md.append("- predictor and LRU scores")
    md.append("- predictor-LRU disagreement and score gap")
    md.append("- predictor/LRU recency ranks")
    md.append("- bucket and confidence gaps")
    md.append("- cache bucket diversity/mean/std")
    md.append("- recent regret/disagreement rates")
    md.append("")
    md.append("## Model")
    md.append("- LogisticRegression(class_weight='balanced') + StandardScaler.")
    md.append("- Metrics JSON: analysis/ml_gate_v1_metrics.json.")
    md.append("")
    if metrics:
        md.append("## Train/val/test metrics")
        for split in ["train", "val", "test"]:
            m = metrics.get(split, {})
            md.append(f"- {split}: accuracy={m.get('accuracy', 0):.3f} precision={m.get('precision', 0):.3f} recall={m.get('recall', 0):.3f} f1={m.get('f1', 0):.3f} auc={m.get('roc_auc', 0):.3f} class_balance={m.get('class_balance', 0):.3f}")
        md.append("")
    md.append("## Baseline comparison (mean misses over first-check traces)")
    for p in ["ml_gate_v1", "rest_v1", "atlas_v3", "lru", "blind_oracle", "predictive_marker", "trust_and_doubt", "blind_oracle_lru_combiner"]:
        md.append(f"- {p}: {_mean(p):.3f}")
    md.append("")
    md.append("## Honesty check")
    md.append("- This is a small-sample POC; if gains are small or negative, treat this as plumbing validation rather than final evidence.")
    md.append("- Full-scale trace-family training/evaluation should be run on Wulver.")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
