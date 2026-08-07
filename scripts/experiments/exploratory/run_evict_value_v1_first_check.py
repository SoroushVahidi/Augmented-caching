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
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.ml_gate_v1 import MLGateV1Policy
from lafc.policies.ml_gate_v2 import MLGateV2Policy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace

OUT_CSV = Path("analysis/evict_value_v1_first_check.csv")
OUT_MD = Path("analysis/evict_value_v1_first_check.md")
METRICS_JSON = Path("analysis/evict_value_v1_metrics.json")
CMP_CSV = Path("analysis/evict_value_v1_model_comparison.csv")
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
                "evict_value_v1": run_policy(EvictValueV1Policy(), reqs, pages, cap),
                "rest_v1": run_policy(RestV1Policy(), reqs, pages, cap),
                "ml_gate_v2": run_policy(MLGateV2Policy(), reqs, pages, cap),
                "ml_gate_v1": run_policy(MLGateV1Policy(), reqs, pages, cap),
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
    winner = metrics.get("winner_by_val_regret", "unknown")

    lines: List[str] = []
    lines.append("# evict_value_v1 first check")
    lines.append("")
    lines.append("## Core target and policy")
    lines.append("- Candidate-centric regression target: y_loss(q,t;H) = misses over next H requests after forcing eviction of candidate q and replaying LRU transitions.")
    lines.append("- Online policy evicts candidate with minimum predicted loss.")
    lines.append("")
    lines.append("## Model sweep summary")
    lines.append(f"- Winner by validation mean regret: {winner}")
    for r in cmp:
        lines.append(
            f"- {r['model']}: val_mae={float(r['val_mae']):.3f}, test_mae={float(r['test_mae']):.3f}, "
            f"val_top1={float(r['val_top1_eviction_match']):.3f}, test_top1={float(r['test_top1_eviction_match']):.3f}"
        )
    lines.append("")
    lines.append("## Policy comparison (mean misses)")
    for p in [
        "evict_value_v1",
        "rest_v1",
        "ml_gate_v2",
        "ml_gate_v1",
        "atlas_v3",
        "lru",
        "blind_oracle",
        "predictive_marker",
        "trust_and_doubt",
        "blind_oracle_lru_combiner",
    ]:
        lines.append(f"- {p}: {mm(p):.3f}")
    lines.append("")
    lines.append("## Answers")
    lines.append(f"1. Direct candidate scoring better than trust-gating? {'yes' if mm('evict_value_v1') < mm('ml_gate_v2') else 'no'}")
    lines.append(f"2. Candidate-centric supervision better than gate refinements? {'yes' if mm('evict_value_v1') < mm('ml_gate_v1') else 'no'}")
    best_h = metrics.get("horizon", "unknown")
    lines.append(f"3. Which horizon works best? this run trained horizon={best_h}; run the training sweep across horizons to confirm global best.")
    lines.append(f"4. Which model family works best? {winner}")
    lines.append(f"5. Does evict_value_v1 beat rest_v1? {'yes' if mm('evict_value_v1') < mm('rest_v1') else 'no'}")
    lines.append("6. If not, clearest bottleneck: small data/trace diversity and limited features around long-range reuse interactions.")
    lines.append("")
    lines.append("## Honesty")
    lines.append("- This is a first proof-of-concept. If rest_v1 still wins, direct candidate prediction is not yet sufficient at this data scale.")
    lines.append("- Environment is sufficient for smoke/first-check; broader claims should be re-run on larger Wulver-scale traces.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
