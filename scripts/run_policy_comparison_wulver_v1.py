from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

from lafc.evict_value_wulver_v1 import load_trace_from_any
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


POLICIES = {
    "evict_value_v1": lambda m: EvictValueV1Policy(model_path=m),
    "rest_v1": lambda _: RestV1Policy(),
    "ml_gate_v2": lambda _: MLGateV2Policy(),
    "ml_gate_v1": lambda _: MLGateV1Policy(),
    "atlas_v3": lambda _: AtlasV3Policy(),
    "lru": lambda _: LRUPolicy(),
    "blind_oracle": lambda _: BlindOraclePolicy(),
    "predictive_marker": lambda _: PredictiveMarkerPolicy(),
    "trust_and_doubt": lambda _: TrustAndDoubtPolicy(seed=7),
    "blind_oracle_lru_combiner": lambda _: BlindOracleLRUCombiner(),
}


def _ml_gate_models_present() -> bool:
    # Paths must match MLGateV1Policy / MLGateV2Policy defaults (see lafc.policies.ml_gate_*).
    return Path("models/ml_gate_v1.pkl").exists() and Path("models/ml_gate_v2_random_forest.pkl").exists()


def _read_manifest_paths(manifest_csv: Path, max_traces: int | None) -> List[Tuple[str, str, str]]:
    rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
    out: List[Tuple[str, str, str]] = []
    for r in rows:
        out.append((r["path"].strip(), r.get("trace_name", "").strip() or r["path"], r.get("trace_family", "").strip() or "unknown"))
    if max_traces is not None:
        out = out[:max_traces]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-trace policy comparison for Wulver manifest.")
    ap.add_argument("--trace-manifest", type=Path, default=Path("analysis/wulver_trace_manifest_full.csv"))
    ap.add_argument("--max-traces", type=int, default=None, help="Use only the first N rows of the manifest CSV.")
    ap.add_argument("--capacities", default="64", help="Comma-separated capacities (same pool as dataset phase).")
    ap.add_argument("--max-requests-per-trace", type=int, default=None)
    ap.add_argument("--evict-value-model", type=Path, default=Path("models/evict_value_wulver_v1_best.pkl"))
    ap.add_argument("--out-csv", type=Path, default=Path("analysis/evict_value_wulver_v1_policy_comparison.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("analysis/evict_value_wulver_v1_policy_comparison.md"))
    args = ap.parse_args()

    caps = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    traces = _read_manifest_paths(args.trace_manifest, args.max_traces)
    rows_out: List[Dict[str, object]] = []
    policies = dict(POLICIES)
    if not _ml_gate_models_present():
        policies.pop("ml_gate_v1", None)
        policies.pop("ml_gate_v2", None)

    for path, trace_name, family in traces:
        reqs, pages, _src = load_trace_from_any(path)
        if args.max_requests_per_trace:
            reqs = reqs[: args.max_requests_per_trace]
        for cap in caps:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            for pname, fac in policies.items():
                model_arg = str(args.evict_value_model) if pname == "evict_value_v1" else ""
                pol = fac(model_arg)
                res = run_policy(pol, reqs if pname != "trust_and_doubt" else td_reqs, pages, cap)
                rows_out.append(
                    {
                        "trace_name": trace_name,
                        "trace_family": family,
                        "path": path,
                        "capacity": cap,
                        "policy": pname,
                        "misses": res.total_misses,
                        "hit_rate": hit_rate(res.events),
                    }
                )

    if not rows_out:
        raise SystemExit("No policy comparison rows produced (empty manifest or trace load failure).")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            w.writeheader()
            w.writerows(rows_out)

    # Aggregate: mean misses by policy; per family; vs LRU and rest_v1
    def mean_misses(pred: str) -> float:
        vals = [float(r["misses"]) for r in rows_out if r["policy"] == pred]
        return mean(vals) if vals else 0.0

    lru_m = mean_misses("lru")
    rest_m = mean_misses("rest_v1")
    ev_m = mean_misses("evict_value_v1")

    lines: List[str] = []
    lines.append("# evict_value_v1 Wulver policy comparison")
    lines.append("")
    lines.append("## Aggregate mean misses (all traces × capacities in run)")
    pol_list = sorted({r["policy"] for r in rows_out})
    for p in pol_list:
        lines.append(f"- **{p}:** {mean_misses(p):.4f}")
    lines.append("")
    lines.append("## Relative vs LRU (lower misses is better; positive % = fewer misses than LRU)")
    for p in pol_list:
        pm = mean_misses(p)
        imp = ((lru_m - pm) / lru_m * 100.0) if lru_m > 0 else 0.0
        lines.append(f"- {p}: {imp:.2f}% vs LRU")
    lines.append("")
    lines.append("## Relative vs rest_v1")
    for p in pol_list:
        pm = mean_misses(p)
        imp = ((rest_m - pm) / rest_m * 100.0) if rest_m > 0 else 0.0
        lines.append(f"- {p}: {imp:.2f}% vs rest_v1")
    lines.append("")
    lines.append("## Per-family mean misses (evict_value_v1 vs LRU vs rest_v1)")
    fams = sorted({r["trace_family"] for r in rows_out})
    for fam in fams:
        def fm(pred: str) -> float:
            vals = [float(r["misses"]) for r in rows_out if r["trace_family"] == fam and r["policy"] == pred]
            return mean(vals) if vals else 0.0

        lru_f, rest_f, ev_f = fm("lru"), fm("rest_v1"), fm("evict_value_v1")
        w_ev = "win" if ev_f < min(lru_f, rest_f) else ("tie" if ev_f <= min(lru_f, rest_f) + 1e-9 else "loss")
        lines.append(f"- **{fam}:** evict_value_v1={ev_f:.2f}, lru={lru_f:.2f}, rest_v1={rest_f:.2f} ({w_ev} vs best baseline here)")
    lines.append("")
    lines.append(f"- evict_value_v1 model: `{args.evict_value_model}`")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv} and {args.out_md}")


if __name__ == "__main__":
    main()
