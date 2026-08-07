from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

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
        self._chosen_pairwise = 0
        self._chosen_lru = 0
        self._corrected_events = 0
        self._step_rows: List[Dict[str, object]] = []

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
            if (not lru_event.hit) and pair_event.hit:
                self._corrected_events += 1

        self._sync_visible_cache("base" if choose_pairwise else "fallback")
        self._step_rows.append(
            {
                "t": int(request.t),
                "chosen_side": "pairwise" if choose_pairwise else "lru",
                "pairwise_hit": bool(pair_event.hit),
                "lru_hit": bool(lru_event.hit),
                "pairwise_shadow_misses_so_far": int(self._pairwise_misses),
                "lru_shadow_misses_so_far": int(self._lru_misses),
                "selected_hit": bool(chosen.hit),
            }
        )
        return chosen

    def diagnostics_summary(self):
        total = max(1, self._chosen_pairwise + self._chosen_lru)
        return {
            "chosen_pairwise_steps": float(self._chosen_pairwise),
            "chosen_lru_steps": float(self._chosen_lru),
            "fraction_choose_pairwise": float(self._chosen_pairwise / total),
            "fraction_choose_lru": float(self._chosen_lru / total),
            "pairwise_shadow_misses": float(self._pairwise_misses),
            "lru_shadow_misses": float(self._lru_misses),
            "lru_correction_events": float(self._corrected_events),
            "tie_breaker": self.tie_breaker,
        }

    def step_rows(self) -> List[Dict[str, object]]:
        return list(self._step_rows)


class PairwiseShortlistPolicy(EvictValuePairwiseV1Policy):
    def __init__(self, *, shortlist_m: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shortlist_m = int(shortlist_m)
        self.name = f"pairwise_shortlist_lru_m{shortlist_m}"

    def _choose_victim(self, request):
        candidates = list(self._order.keys())
        shortlist = candidates[: min(self.shortlist_m, len(candidates))]
        if len(shortlist) <= 1:
            return shortlist[0]
        req_bucket = int(request.metadata.get("bucket", 0))
        req_conf = float(request.metadata.get("confidence", 0.5))

        from lafc.evict_value_features_v1 import compute_candidate_features_v1

        feat_by_candidate = {}
        for candidate in shortlist:
            req_rate = (sum(1 for x in self._recent_req_hist if x == candidate) / len(self._recent_req_hist)) if self._recent_req_hist else 0.0
            hit_rate_val = (sum(1 for x in self._recent_hit_hist if x == candidate) / len(self._recent_hit_hist)) if self._recent_hit_hist else 0.0
            feat_by_candidate[candidate] = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=shortlist,
                candidate=candidate,
                bucket_by_page=self._bucket_by_page,
                confidence_by_page=self._confidence_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate_val,
            ).as_dict()

        wins = {c: 0.0 for c in shortlist}
        for i, a in enumerate(shortlist):
            for b in shortlist[i + 1 :]:
                p_a = self._scorer.predict_a_beats_b(feat_by_candidate[a], feat_by_candidate[b])
                wins[a] += p_a
                wins[b] += 1.0 - p_a
        return max(shortlist, key=lambda c: (wins[c], str(c)))


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


def _split_by_trace_family_aware(trace_paths: List[str]) -> Dict[str, str]:
    by_family: Dict[str, List[str]] = defaultdict(list)
    for t in sorted(trace_paths):
        by_family[_trace_family(t)].append(t)
    out: Dict[str, str] = {}
    for _fam, items in sorted(by_family.items()):
        for i, t in enumerate(sorted(items)):
            out[t] = ["train", "val", "test"][i % 3]
    # ensure non-degenerate where possible
    traces = sorted(trace_paths)
    for split in ["train", "val", "test"]:
        if not any(v == split for v in out.values()) and traces:
            hvals = sorted((int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16), t) for t in traces)
            out[hvals[{"train": 0, "val": len(hvals)//2, "test": -1}[split]][1]] = split
    return out


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


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _group_breakdown(rows: List[Dict[str, object]], *, by_key: str) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        grouped[(str(r[by_key]), str(r["policy"]))].append(float(r["misses"]))
    out: List[Dict[str, object]] = []
    for (slice_name, pol), vals in sorted(grouped.items()):
        out.append({by_key: slice_name, "policy": pol, "mean_misses": float(mean(vals)), "count": len(vals)})
    return out


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


def _inversion_rate(rows: List[Dict[str, object]], model: EvictValuePairwiseV1Model) -> float:
    total = 0
    mistakes = 0
    for r in rows:
        af = {k.replace("i_", "", 1): float(v) for k, v in r.items() if k.startswith("i_")}
        bf = {k.replace("j_", "", 1): float(v) for k, v in r.items() if k.startswith("j_")}
        pred_i = model.predict_a_beats_b_proba(af, bf) >= 0.5
        true_i = int(float(r["label_i_better"])) == 1
        total += 1
        mistakes += int(pred_i != true_i)
    return float(mistakes / max(total, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress evaluation for pairwise combiner deployment rule")
    ap.add_argument("--trace-glob", default="data/example*.json")
    ap.add_argument("--capacities", default="2,3,4,5,6")
    ap.add_argument("--horizons", default="4,8,12")
    ap.add_argument("--max-requests-per-trace", type=int, default=0)
    ap.add_argument("--output-dir", default="analysis/pairwise_combiner_stress_eval")
    ap.add_argument("--summary-md", default="analysis/pairwise_combiner_stress_eval.md")
    args = ap.parse_args()

    trace_paths = _resolve_trace_paths(args.trace_glob)
    capacities = _parse_int_list(args.capacities)
    horizons = _parse_int_list(args.horizons)
    split_by_trace = _split_by_trace_family_aware(trace_paths)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = _build_candidate_rows(trace_paths, capacities, horizons, args.max_requests_per_trace)
    pairwise_rows = build_pairwise_rows_from_candidate_rows(candidate_rows, include_ties=False)
    delta_cols = sorted(c for c in pairwise_rows[0].keys() if c.startswith("delta_")) if pairwise_rows else []

    csplit = _rows_split(candidate_rows, split_by_trace)
    psplit = _rows_split(pairwise_rows, split_by_trace)

    x_ptrain, y_ptrain = _xy_pointwise(csplit["train"])
    p_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    p_model.fit(x_ptrain, y_ptrain)
    point_model = EvictValueV1Model(model_name="pairwise_stress_pointwise", estimator=p_model, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS))

    x_qtrain, y_qtrain = _xy_pairwise(psplit["train"], delta_cols)
    q_model = LogisticRegression(max_iter=700, random_state=7)
    if len(set(int(v) for v in y_qtrain.tolist())) < 2:
        q_model = DummyClassifier(strategy="constant", constant=int(y_qtrain[0]) if len(y_qtrain) else 0)
    q_model.fit(x_qtrain, y_qtrain)
    pair_model = EvictValuePairwiseV1Model(model_name="pairwise_stress_pairwise", estimator=q_model, delta_feature_columns=delta_cols)

    point_model_path = out_dir / "evict_value_v1_stress.pkl"
    pair_model_path = out_dir / "evict_value_pairwise_v1_stress.pkl"
    point_model.save(point_model_path)
    pair_model.save(pair_model_path)

    test_traces = [t for t, s in split_by_trace.items() if s == "test"]
    results_rows: List[Dict[str, object]] = []
    combiner_rows: List[Dict[str, object]] = []
    missing_inputs: List[str] = []

    for trace in sorted(test_traces):
        reqs, pages = load_trace(trace)
        if args.max_requests_per_trace > 0:
            reqs = reqs[: args.max_requests_per_trace]
        fam = _trace_family(trace)
        for cap in capacities:
            td_reqs = attach_predicted_caches(reqs, capacity=cap)
            combiner = PairwiseLRUCombinerPolicy(pairwise_policy=EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"), tie_breaker="pairwise")
            policies = {
                "pairwise_unguarded": EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"),
                "pairwise_lru_blackbox_combiner": combiner,
                "pairwise_guard_wrapper_lru": GuardWrapperPolicy(
                    base_policy=EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"),
                    fallback_policy=LRUPolicy(),
                    early_return_window=2,
                    trigger_threshold=2,
                    trigger_window=16,
                    guard_duration=8,
                    wrapper_name="pairwise_guard_wrapper_lru",
                ),
                "pairwise_shortlist_lru_m4": PairwiseShortlistPolicy(shortlist_m=4, model_path=str(pair_model_path), scorer_mode="artifact"),
                "pairwise_shortlist_lru_mfull": PairwiseShortlistPolicy(shortlist_m=max(cap, 1), model_path=str(pair_model_path), scorer_mode="artifact"),
                "evict_value_v1": EvictValueV1Policy(model_path=str(point_model_path), scorer_mode="artifact"),
                "evict_value_pairwise_v1": EvictValuePairwiseV1Policy(model_path=str(pair_model_path), scorer_mode="artifact"),
                "predictive_marker": PredictiveMarkerPolicy(),
                "trust_and_doubt": TrustAndDoubtPolicy(seed=7),
                "rest_v1": RestV1Policy(),
                "lru": LRUPolicy(),
            }

            for pol_name, policy in policies.items():
                use_reqs = td_reqs if pol_name == "trust_and_doubt" else reqs
                res = run_policy(policy, use_reqs, pages, cap)
                results_rows.append(
                    {
                        "trace": trace,
                        "family": fam,
                        "capacity": cap,
                        "policy": pol_name,
                        "misses": int(res.total_misses),
                        "hit_rate": float(hit_rate(res.events)),
                    }
                )

            for row in combiner.step_rows():
                combiner_rows.append(
                    {
                        "trace": trace,
                        "family": fam,
                        "capacity": cap,
                        **row,
                    }
                )

    # remove temporary artifacts from output folder after run
    for p in [point_model_path, pair_model_path]:
        if p.exists():
            p.unlink()

    _write_csv(out_dir / "policy_comparison.csv", results_rows)
    family_rows = _group_breakdown(results_rows, by_key="family")
    cap_rows = _group_breakdown(results_rows, by_key="capacity")
    _write_csv(out_dir / "per_family_breakdown.csv", family_rows)
    _write_csv(out_dir / "per_capacity_breakdown.csv", cap_rows)
    _write_csv(out_dir / "combiner_behavior.csv", combiner_rows)

    overall: Dict[str, float] = {}
    by_pol: Dict[str, List[float]] = defaultdict(list)
    for r in results_rows:
        by_pol[str(r["policy"])].append(float(r["misses"]))
    for p, vals in by_pol.items():
        overall[p] = float(mean(vals))

    hard_slice_path = Path("analysis/evict_value_failure_slice_audit.csv")
    if not hard_slice_path.exists():
        missing_inputs.append(str(hard_slice_path))
    hard_ids = _hard_slice_ids(hard_slice_path)
    hard_overlap: Dict[str, List[float]] = defaultdict(list)
    if hard_ids:
        for r in results_rows:
            base = f"{r['trace']}|c{int(r['capacity'])}|"
            if any(h.startswith(base) for h in hard_ids):
                hard_overlap[str(r["policy"])].append(float(r["misses"]))

    comb_choose_pairwise = sum(1 for r in combiner_rows if str(r["chosen_side"]) == "pairwise")
    comb_choose_lru = sum(1 for r in combiner_rows if str(r["chosen_side"]) == "lru")
    comb_total = max(1, comb_choose_pairwise + comb_choose_lru)
    comb_lru_wins = sum(1 for r in combiner_rows if str(r["chosen_side"]) == "lru" and bool(r["pairwise_hit"]) and not bool(r["lru_hit"]))

    summary = {
        "trace_glob": args.trace_glob,
        "resolved_traces": trace_paths,
        "split_by_trace": split_by_trace,
        "split_counts": {s: sum(1 for v in split_by_trace.values() if v == s) for s in ["train", "val", "test"]},
        "capacities": capacities,
        "horizons": horizons,
        "max_requests_per_trace": args.max_requests_per_trace,
        "overall_online_mean_misses": overall,
        "hard_slice_overlap_mean_misses": {k: float(mean(v)) for k, v in hard_overlap.items()} if hard_overlap else {},
        "combiner_behavior_summary": {
            "choose_pairwise_steps": comb_choose_pairwise,
            "choose_lru_steps": comb_choose_lru,
            "fraction_choose_pairwise": float(comb_choose_pairwise / comb_total),
            "fraction_choose_lru": float(comb_choose_lru / comb_total),
            "lru_choice_when_pairwise_would_hit": comb_lru_wins,
        },
        "pairwise_inversion_rate_test": _inversion_rate(psplit["test"], pair_model),
        "missing_inputs": missing_inputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    comb = overall.get("pairwise_lru_blackbox_combiner", float("inf"))
    unguard = overall.get("pairwise_unguarded", float("inf"))
    robust_best = min(overall.get("predictive_marker", float("inf")), overall.get("trust_and_doubt", float("inf")), overall.get("rest_v1", float("inf")))

    family_gain_rows = [r for r in family_rows if r["policy"] in {"pairwise_lru_blackbox_combiner", "pairwise_unguarded"}]
    fam_gain: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in family_gain_rows:
        fam_gain[str(r["family"])][str(r["policy"])] = float(r["mean_misses"])
    gain_broad = sum(1 for fam, vals in fam_gain.items() if vals.get("pairwise_lru_blackbox_combiner", 1e9) < vals.get("pairwise_unguarded", 1e9))

    lines = ["# Pairwise combiner stress evaluation", ""]
    lines.append("## Stress setup")
    lines.append(f"- Traces used: {len(trace_paths)} ({', '.join(Path(t).name for t in trace_paths)}).")
    lines.append(f"- Split counts (trace-level): {summary['split_counts']}.")
    lines.append(f"- Capacities: {capacities}; horizons: {horizons}; max_requests_per_trace={args.max_requests_per_trace} (0 means full).")
    if missing_inputs:
        lines.append(f"- Missing optional inputs: {', '.join(missing_inputs)}")
    lines.append("")
    lines.append("## Overall online mean misses")
    for p, v in sorted(overall.items(), key=lambda z: (z[1], z[0])):
        lines.append(f"- {p}: {v:.4f}")
    lines.append("")
    lines.append("## Required direct answers")
    lines.append(f"1. Combiner still beats pairwise_unguarded? {'yes' if comb < unguard else ('tie' if comb == unguard else 'no')} (combiner={comb:.4f}, unguarded={unguard:.4f}).")
    lines.append(f"2. Combiner beats/ties strong robust baselines? {'yes' if comb <= robust_best else 'no'} (combiner={comb:.4f}, best_robust={robust_best:.4f}).")
    lines.append(f"3. Gain broad across families or concentrated? {'broad' if gain_broad >= max(1, len(fam_gain)//2) else 'concentrated'} ({gain_broad}/{len(fam_gain)} families where combiner < unguarded).")
    lines.append(
        "4. Combiner help mode: "
        + ("catastrophic-correction dominated" if comb_choose_lru / comb_total < 0.2 else "many-close-decisions / frequent-side-switching")
        + f" (choose_lru_fraction={comb_choose_lru/comb_total:.4f})."
    )
    lines.append(
        "5. Strongest next bottleneck: "
        + ("data scale" if len(trace_paths) <= 6 else "feature quality / horizon construction")
        + "."
    )

    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
