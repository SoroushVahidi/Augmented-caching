from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.dummy import DummyClassifier
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
from lafc.policies.guard_wrapper import GuardWrapperPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace


class PairwiseLRUCombinerPolicy(GuardWrapperPolicy):
    name: str = "pairwise_lru_blackbox_combiner"

    def __init__(self, *, pairwise_policy: EvictValuePairwiseV1Policy, tie_breaker: str = "pairwise") -> None:
        tie = str(tie_breaker).strip().lower()
        if tie not in {"pairwise", "lru"}:
            raise ValueError("tie_breaker must be pairwise or lru")
        self.tie_breaker = tie
        super().__init__(
            base_policy=pairwise_policy,
            fallback_policy=LRUPolicy(),
            early_return_window=1,
            trigger_threshold=10**9,
            trigger_window=1,
            guard_duration=1,
            wrapper_name=self.name,
        )

    def reset(self, capacity: int, pages: Dict[str, object]) -> None:
        super().reset(capacity, pages)
        self._pairwise_misses = 0
        self._lru_misses = 0
        self._pairwise_steps = 0
        self._lru_steps = 0
        self._chosen_pairwise = 0
        self._chosen_lru = 0

    def on_request(self, request):
        pair_event = self._base.on_request(request)
        lru_event = self._fallback.on_request(request)

        if pair_event.hit:
            self._record_hit()
        else:
            self._record_miss(1.0)
            self._pairwise_misses += 1
        if not lru_event.hit:
            self._lru_misses += 1

        choose_pairwise = self._pairwise_misses < self._lru_misses
        if self._pairwise_misses == self._lru_misses:
            choose_pairwise = self.tie_breaker == "pairwise"

        chosen = pair_event if choose_pairwise else lru_event
        if choose_pairwise:
            self._chosen_pairwise += 1
        else:
            self._chosen_lru += 1

        self._sync_visible_cache("base" if choose_pairwise else "fallback")

        return chosen

    def diagnostics_summary(self):
        return {
            "chosen_pairwise_steps": float(self._chosen_pairwise),
            "chosen_lru_steps": float(self._chosen_lru),
            "pairwise_shadow_misses": float(self._pairwise_misses),
            "lru_shadow_misses": float(self._lru_misses),
            "tie_breaker": self.tie_breaker,
        }


class PairwiseShortlistPolicy(EvictValuePairwiseV1Policy):
    name: str = "pairwise_shortlist_lru"

    def __init__(self, *, shortlist_m: Optional[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.shortlist_m = shortlist_m
        self.name = f"pairwise_shortlist_lru_m{shortlist_m if shortlist_m is not None else 'full'}"

    def _choose_victim(self, request):
        if self.shortlist_m is None or self.shortlist_m <= 0:
            return super()._choose_victim(request)
        candidates = list(self._order.keys())
        shortlist = candidates[: min(self.shortlist_m, len(candidates))]
        if len(shortlist) <= 1:
            return shortlist[0]

        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))

        feat_by_candidate = {}
        for candidate in shortlist:
            req_rate = (sum(1 for x in self._recent_req_hist if x == candidate) / len(self._recent_req_hist)) if self._recent_req_hist else 0.0
            hit_rate_val = (sum(1 for x in self._recent_hit_hist if x == candidate) / len(self._recent_hit_hist)) if self._recent_hit_hist else 0.0
            feat_by_candidate[candidate] = self._feature_row(
                candidates=shortlist,
                candidate=candidate,
                req_bucket=req_bucket,
                req_conf=req_conf,
                req_rate=req_rate,
                hit_rate=hit_rate_val,
            )

        wins = {c: 0.0 for c in shortlist}
        for i, a in enumerate(shortlist):
            for b in shortlist[i + 1 :]:
                p_a = self._scorer.predict_a_beats_b(feat_by_candidate[a], feat_by_candidate[b])
                wins[a] += p_a
                wins[b] += 1.0 - p_a
        return max(shortlist, key=lambda c: (wins[c], str(c)))

    def _feature_row(self, *, candidates, candidate, req_bucket, req_conf, req_rate, hit_rate):
        from lafc.evict_value_features_v1 import compute_candidate_features_v1

        return compute_candidate_features_v1(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=list(candidates),
            candidate=candidate,
            bucket_by_page=self._bucket_by_page,
            confidence_by_page=self._confidence_by_page,
            recent_request_rate=req_rate,
            recent_hit_rate=hit_rate,
        ).as_dict()


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
            rows.extend(build_rollout_candidate_rows_v2(requests=reqs, capacity=cap, trace_name=trace, trace_family=fam, cfg=cfg))
    return rows


def _rows_split(rows: List[Dict[str, object]], split_by_trace: Dict[str, str]) -> Dict[str, List[Dict[str, object]]]:
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out[split_by_trace.get(str(r["trace"]), "train")].append(r)
    return out


def _xy_pointwise(rows: List[Dict[str, object]]):
    import numpy as np

    x = np.asarray([[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.asarray([float(r["rollout_regret_h"]) for r in rows], dtype=float)
    return x, y


def _xy_pairwise(rows: List[Dict[str, object]], delta_cols: List[str]):
    import numpy as np

    x = np.asarray([[float(r[c]) for c in delta_cols] for r in rows], dtype=float)
    y = np.asarray([int(float(r["label_i_better"])) for r in rows], dtype=int)
    return x, y


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
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        grouped[str(r[key])].append(float(r["misses"]))
    return {k: float(mean(v)) for k, v in sorted(grouped.items())}


def _pairwise_inversion_summary(rows: List[Dict[str, object]], model: EvictValuePairwiseV1Model) -> Dict[str, object]:
    by_trace: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pairs": 0.0, "mistakes": 0.0})
    total_pairs = 0
    total_mistakes = 0
    for r in rows:
        trace = str(r["trace"])
        af = {k.replace("i_", "", 1): float(v) for k, v in r.items() if k.startswith("i_")}
        bf = {k.replace("j_", "", 1): float(v) for k, v in r.items() if k.startswith("j_")}
        pred_i = model.predict_a_beats_b_proba(af, bf) >= 0.5
        truth_i = int(float(r["label_i_better"])) == 1
        mistake = int(pred_i != truth_i)
        total_pairs += 1
        total_mistakes += mistake
        by_trace[trace]["pairs"] += 1.0
        by_trace[trace]["mistakes"] += float(mistake)
    out_rows = []
    for tr, vals in sorted(by_trace.items()):
        rate = vals["mistakes"] / vals["pairs"] if vals["pairs"] else 0.0
        out_rows.append({"trace": tr, "pair_count": int(vals["pairs"]), "mistake_count": int(vals["mistakes"]), "mistake_rate": float(rate)})
    return {
        "total_pair_count": total_pairs,
        "total_mistake_count": total_mistakes,
        "overall_mistake_rate": float(total_mistakes / max(total_pairs, 1)),
        "per_trace": out_rows,
        "definition": "For each labeled pair (i,j), count a mistake when predicted preference p(i better than j)>=0.5 disagrees with rollout label_i_better.",
    }


def _disagreement_rate(policy_rows: List[Dict[str, object]], left: str, right: str) -> float:
    lmap = {(r["trace"], int(r["capacity"]), int(r["t"])): str(r["evicted"]) for r in policy_rows if str(r["policy"]) == left and r.get("evicted") not in (None, "")}
    rmap = {(r["trace"], int(r["capacity"]), int(r["t"])): str(r["evicted"]) for r in policy_rows if str(r["policy"]) == right and r.get("evicted") not in (None, "")}
    keys = sorted(set(lmap.keys()) & set(rmap.keys()))
    if not keys:
        return 0.0
    disagree = sum(1 for k in keys if lmap[k] != rmap[k])
    return float(disagree / len(keys))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise deployment-rule ablation")
    ap.add_argument("--trace-glob", default="data/example*.json")
    ap.add_argument("--capacities", default="2,3,4,8")
    ap.add_argument("--horizons", default="4,8")
    ap.add_argument("--max-requests-per-trace", type=int, default=500)
    ap.add_argument("--output-dir", default="analysis/pairwise_deployment_rule_ablation")
    ap.add_argument("--summary-md", default="analysis/pairwise_deployment_rule_ablation.md")
    ap.add_argument("--shortlist-ms", default="2,4,8")
    args = ap.parse_args()

    trace_paths = _resolve_trace_paths(args.trace_glob)
    capacities = _parse_int_list(args.capacities)
    horizons = _parse_int_list(args.horizons)
    shortlist_ms = _parse_int_list(args.shortlist_ms)
    split_by_trace = _split_by_trace(trace_paths)

    candidate_rows = _build_candidate_rows(trace_paths, capacities, horizons, args.max_requests_per_trace)
    pairwise_rows = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=False)
    delta_cols = sorted(c for c in pairwise_rows[0].keys() if c.startswith("delta_")) if pairwise_rows else []

    csplit = _rows_split(candidate_rows, split_by_trace)
    psplit = _rows_split(pairwise_rows, split_by_trace)

    x_ptrain, y_ptrain = _xy_pointwise(csplit["train"])
    p_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    p_model.fit(x_ptrain, y_ptrain)
    point_model = EvictValueV1Model(model_name="pairwise_ablation_pointwise", estimator=p_model, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS))

    x_qtrain, y_qtrain = _xy_pairwise(psplit["train"], delta_cols)
    q_model = LogisticRegression(max_iter=700, random_state=7)
    if len(set(int(v) for v in y_qtrain.tolist())) < 2:
        q_model = DummyClassifier(strategy="constant", constant=int(y_qtrain[0]) if len(y_qtrain) else 0)
    q_model.fit(x_qtrain, y_qtrain)
    pair_model = EvictValuePairwiseV1Model(model_name="pairwise_ablation_pairwise", estimator=q_model, delta_feature_columns=delta_cols)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_model_path = out_dir / "evict_value_v1_ablation.pkl"
    pair_model_path = out_dir / "evict_value_pairwise_v1_ablation.pkl"
    point_model.save(point_model_path)
    pair_model.save(pair_model_path)

    test_traces = [t for t, s in split_by_trace.items() if s == "test"]
    results_rows: List[Dict[str, object]] = []
    evict_rows: List[Dict[str, object]] = []
    skipped_notes: List[str] = []

    for trace in test_traces:
        reqs, pages = load_trace(trace)
        if args.max_requests_per_trace > 0:
            reqs = reqs[: args.max_requests_per_trace]
        fam = _trace_family(trace)
        for cap in capacities:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            policies = {
                "evict_value_v1": EvictValueV1Policy(model_path=str(point_model_path), scorer_mode="artifact"),
                "evict_value_pairwise_v1": EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"),
                "pairwise_unguarded": EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"),
                "pairwise_lru_blackbox_combiner": PairwiseLRUCombinerPolicy(pairwise_policy=EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"), tie_breaker="pairwise"),
                "predictive_marker": PredictiveMarkerPolicy(),
                "trust_and_doubt": TrustAndDoubtPolicy(seed=7),
                "rest_v1": RestV1Policy(),
                "lru": LRUPolicy(),
            }
            for m in shortlist_ms:
                policies[f"pairwise_shortlist_lru_m{m}"] = PairwiseShortlistPolicy(shortlist_m=m, model_path=str(pair_model_path), scorer_mode="artifact")
            policies["pairwise_shortlist_lru_mfull"] = PairwiseShortlistPolicy(shortlist_m=cap, model_path=str(pair_model_path), scorer_mode="artifact")

            # existing robust fallback wrapper with pairwise expert
            try:
                policies["pairwise_guard_wrapper_lru"] = GuardWrapperPolicy(
                    base_policy=EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"),
                    fallback_policy=LRUPolicy(),
                    early_return_window=2,
                    trigger_threshold=2,
                    trigger_window=16,
                    guard_duration=8,
                    wrapper_name="pairwise_guard_wrapper_lru",
                )
            except Exception as exc:
                skipped_notes.append(f"guarded_variant_skipped trace={trace} cap={cap}: {exc}")

            for name, policy in policies.items():
                try:
                    use_reqs = td_reqs if name == "trust_and_doubt" else reqs
                    res = run_policy(policy, use_reqs, pages, cap)
                except Exception as exc:
                    skipped_notes.append(f"skip trace={trace} cap={cap} policy={name}: {exc}")
                    continue
                results_rows.append(
                    {
                        "trace": trace,
                        "family": fam,
                        "capacity": cap,
                        "policy": name,
                        "misses": res.total_misses,
                        "hit_rate": hit_rate(res.events),
                    }
                )
                for ev in res.events:
                    if ev.evicted is None:
                        continue
                    evict_rows.append(
                        {
                            "trace": trace,
                            "family": fam,
                            "capacity": cap,
                            "policy": name,
                            "t": int(ev.t),
                            "evicted": str(ev.evicted),
                        }
                    )

    _write_csv(out_dir / "policy_comparison.csv", results_rows)

    rule_rows: List[Dict[str, object]] = []
    for fam in sorted({str(r["family"]) for r in results_rows}):
        fam_rows = [r for r in results_rows if str(r["family"]) == fam]
        mean_by_policy = _group_mean(fam_rows, "policy")
        for p, v in mean_by_policy.items():
            rule_rows.append({"slice_type": "family", "slice": fam, "policy": p, "mean_misses": v})
    for cap in sorted({int(r["capacity"]) for r in results_rows}):
        cap_rows = [r for r in results_rows if int(r["capacity"]) == cap]
        mean_by_policy = _group_mean(cap_rows, "policy")
        for p, v in mean_by_policy.items():
            rule_rows.append({"slice_type": "capacity", "slice": str(cap), "policy": p, "mean_misses": v})
    _write_csv(out_dir / "deployment_rule_breakdown.csv", rule_rows)

    hard_ids = _hard_slice_ids(Path("analysis/evict_value_failure_slice_audit.csv"))
    hard_online: Dict[str, List[float]] = defaultdict(list)
    if hard_ids:
        for r in results_rows:
            key = f"{r['trace']}|c{int(r['capacity'])}"
            if any(h.startswith(key + "|") for h in hard_ids):
                hard_online[str(r["policy"])].append(float(r["misses"]))

    overall_mean = _group_mean(results_rows, "policy") if results_rows else {}
    by_family: Dict[str, Dict[str, float]] = {}
    by_cap: Dict[str, Dict[str, float]] = {}
    for fam in sorted({str(r["family"]) for r in results_rows}):
        by_family[fam] = _group_mean([r for r in results_rows if str(r["family"]) == fam], "policy")
    for cap in sorted({int(r["capacity"]) for r in results_rows}):
        by_cap[str(cap)] = _group_mean([r for r in results_rows if int(r["capacity"]) == cap], "policy")

    inversion = _pairwise_inversion_summary(psplit["test"], pair_model)
    disagreement = {
        "pairwise_vs_lru": _disagreement_rate(evict_rows, "pairwise_unguarded", "lru"),
        "pairwise_vs_predictive_marker": _disagreement_rate(evict_rows, "pairwise_unguarded", "predictive_marker"),
    }

    def _best(xs: Iterable[Tuple[str, float]]) -> Tuple[str, float]:
        arr = sorted(xs, key=lambda z: (z[1], z[0]))
        return arr[0] if arr else ("n/a", float("inf"))

    shortlist_perf = {p: v for p, v in overall_mean.items() if p.startswith("pairwise_shortlist_lru_m")}
    best_short = _best(shortlist_perf.items())

    summary = {
        "trace_glob": args.trace_glob,
        "resolved_trace_count": len(trace_paths),
        "test_trace_count": len(test_traces),
        "capacities": capacities,
        "horizons": horizons,
        "shortlist_ms": shortlist_ms,
        "split_by_trace": split_by_trace,
        "overall_online_mean_misses": overall_mean,
        "per_family_online_mean_misses": by_family,
        "per_capacity_online_mean_misses": by_cap,
        "hard_slice_online_mean_misses": {k: float(mean(v)) for k, v in hard_online.items()} if hard_online else {},
        "pairwise_disagreement_rate": disagreement,
        "pairwise_inversion_summary": inversion,
        "best_shortlist_variant": {"policy": best_short[0], "mean_misses": best_short[1]},
        "skipped_notes": skipped_notes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = ["# Pairwise deployment-rule ablation", ""]
    lines.append("## Setup")
    lines.append("- Pairwise scorer fixed to `evict_value_pairwise_v1` artifact trained on train split candidate-pair labels.")
    lines.append("- Conditions: unguarded pairwise, pairwise+LRU black-box combiner, pairwise+LRU shortlist pre-filter (m sweep), pairwise+guard wrapper fallback.")
    lines.append("")
    lines.append("## Overall online mean misses (lower is better)")
    for p, v in sorted(overall_mean.items(), key=lambda z: (z[1], z[0])):
        lines.append(f"- {p}: {v:.4f}")

    lines.append("")
    lines.append("## Required direct answers")
    unguarded = overall_mean.get("pairwise_unguarded", float("inf"))
    guarded = overall_mean.get("pairwise_guard_wrapper_lru", float("inf"))
    comb = overall_mean.get("pairwise_lru_blackbox_combiner", float("inf"))
    robust_best = min(overall_mean.get("predictive_marker", float("inf")), overall_mean.get("trust_and_doubt", float("inf")), overall_mean.get("rest_v1", float("inf")))
    lines.append(f"1. Unguarded pairwise collapse vs guarded? {'yes' if unguarded > guarded else 'no'} (unguarded={unguarded:.4f}, guarded={guarded:.4f}).")
    lines.append(f"2. Simple combiner recovers most robust-baseline gap? {'yes' if comb <= robust_best + 0.02 * max(1.0, robust_best) else 'no'} (combiner={comb:.4f}, robust_best={robust_best:.4f}).")
    shortlist_material = 'unclear'
    if shortlist_perf:
        shortlist_material = 'yes' if (max(shortlist_perf.values()) - min(shortlist_perf.values()) > 0.1) else 'no'
    lines.append(f"3. Heuristic shortlist size materially affects performance? {shortlist_material} (best={best_short[0]}:{best_short[1]:.4f}).")
    lines.append(
        "4. Remaining weakness likely deployment-rule or scorer quality? "
        + ("deployment-rule dominated" if min(guarded, comb, best_short[1]) + 1e-9 < unguarded else "scorer-quality dominated")
        + "."
    )
    default_candidate = _best([(k, v) for k, v in overall_mean.items() if k.startswith("pairwise_")])[0]
    lines.append(f"5. Strongest next default deployment rule: {default_candidate}.")

    lines.append("")
    lines.append("## Diagnostics")
    lines.append(f"- Pairwise disagreement rate vs LRU: {disagreement['pairwise_vs_lru']:.4f}")
    lines.append(f"- Pairwise disagreement rate vs predictive_marker: {disagreement['pairwise_vs_predictive_marker']:.4f}")
    lines.append(
        f"- Pairwise inversion-style mistake rate: {inversion['overall_mistake_rate']:.4f} "
        f"({inversion['total_mistake_count']}/{inversion['total_pair_count']} pair comparisons)."
    )
    lines.append(f"- Inversion definition: {inversion['definition']}")

    lines.append("")
    lines.append("## Per-family mean misses")
    for fam, vals in by_family.items():
        lines.append(f"- {fam}: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(vals.items(), key=lambda z: (z[1], z[0]))))

    lines.append("")
    lines.append("## Per-capacity mean misses")
    for cap, vals in by_cap.items():
        lines.append(f"- cap={cap}: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(vals.items(), key=lambda z: (z[1], z[0]))))

    if skipped_notes:
        lines.append("")
        lines.append("## Skipped / caveats")
        lines.extend(f"- {n}" for n in skipped_notes)

    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
