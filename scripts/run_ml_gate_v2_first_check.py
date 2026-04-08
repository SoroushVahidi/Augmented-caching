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
from lafc.policies.ml_gate_v2 import MLGateV2Policy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace

OUT_CSV = Path("analysis/ml_gate_v2_first_check.csv")
OUT_MD = Path("analysis/ml_gate_v2_first_check.md")
METRICS_JSON = Path("analysis/ml_gate_v2_metrics.json")
CMP_CSV = Path("analysis/ml_gate_v2_model_comparison.csv")
CAPS = [2, 3, 4]


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
        "stress::mixed_regime": _make_stress_trace(
            ["A", "B", "C", "A", "B", "D", "A", "C", "B", "D"],
            [0, 3, 1, 0, 2, 3, 0, 1, 2, 3],
            [0.9, 0.9, 0.3, 0.9, 0.7, 0.3, 0.9, 0.3, 0.7, 0.3],
        ),
    }


def _iter_traces():
    for p in ["data/example_unweighted.json", "data/example_atlas_v1.json"]:
        yield p, load_trace(p)
    for n, payload in _stress_traces().items():
        yield n, payload


def main() -> None:
    rows: List[Dict[str, object]] = []
    for trace_name, (reqs, pages) in _iter_traces():
        for cap in CAPS:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            results = {
                "ml_gate_v1": run_policy(MLGateV1Policy(), reqs, pages, cap),
                "ml_gate_v2": run_policy(MLGateV2Policy(), reqs, pages, cap),
                "rest_v1": run_policy(RestV1Policy(), reqs, pages, cap),
                "atlas_v3": run_policy(AtlasV3Policy(), reqs, pages, cap),
                "lru": run_policy(LRUPolicy(), reqs, pages, cap),
                "blind_oracle": run_policy(BlindOraclePolicy(), reqs, pages, cap),
                "predictive_marker": run_policy(PredictiveMarkerPolicy(), reqs, pages, cap),
                "trust_and_doubt": run_policy(TrustAndDoubtPolicy(seed=7), td_reqs, pages, cap),
                "blind_oracle_lru_combiner": run_policy(BlindOracleLRUCombiner(), reqs, pages, cap),
            }
            for name, res in results.items():
                row = {
                    "trace": trace_name,
                    "capacity": cap,
                    "policy": name,
                    "misses": res.total_misses,
                    "hit_rate": hit_rate(res.events),
                }
                extra = res.extra_diagnostics or {}
                if name == "ml_gate_v2":
                    row["trust_coverage"] = (extra.get("ml_gate_v2") or {}).get("summary", {}).get("trust_coverage", 0.0)
                rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    def mm(policy: str) -> float:
        vals = [float(r["misses"]) for r in rows if r["policy"] == policy]
        return mean(vals) if vals else 0.0

    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}
    cmp = list(csv.DictReader(CMP_CSV.open("r", encoding="utf-8"))) if CMP_CSV.exists() else []

    lines: List[str] = []
    lines.append("# ml_gate_v2 first check")
    lines.append("")
    lines.append("## Label definition")
    lines.append("- Counterfactual local replay label: y_reg = loss_lru(H) - loss_pred(H), where each loss is miss count after forcing the eviction choice and replaying next H requests with LRU transitions.")
    lines.append("- Binary target: y_cls = 1 if y_reg > margin else 0.")
    lines.append("")
    lines.append("## Horizons and split")
    lines.append("- Dataset built with horizons {4,8,16}; training sweep reported for selected horizon from metrics.json.")
    lines.append("- Split method: hash(trace|capacity) into train/val/test buckets; fallback deterministic row split only if split buckets are empty.")
    lines.append("")
    lines.append("## Model families")
    if cmp:
        for r in cmp:
            lines.append(f"- {r['model']}: val_f1={float(r['val_f1']):.3f}, test_f1={float(r['test_f1']):.3f}, test_auc={float(r['test_auc']):.3f}")
    winner = metrics.get("winner_by_val_f1", "unknown")
    lines.append(f"- Winner by val F1: {winner}")
    lines.append("")
    lines.append("## Policy comparison (mean misses)")
    for p in ["ml_gate_v2", "ml_gate_v1", "rest_v1", "atlas_v3", "lru", "predictive_marker", "trust_and_doubt"]:
        lines.append(f"- {p}: {mm(p):.3f}")
    lines.append("")
    lines.append("## Honesty")
    lines.append("- If ml_gate_v2 does not beat rest_v1 consistently, bottleneck likely remains data scale/trace diversity and label noise under short local rollouts.")
    lines.append("- This environment is adequate for first-check iteration; Wulver is justified for large-scale trace-family experiments.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
