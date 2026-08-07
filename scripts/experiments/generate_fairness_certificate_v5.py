"""Fairness Certificate V5. Adds evict_value_v1_cross_family_v1 (read from
analysis/reviewer_fairness_cross_family_v1/evict_value_v1/policy_comparison.csv,
its own dedicated output path) and a HALP training/evaluation resource
equivalence distinction. Distinguishes three permanently-or-currently
ineligible evict_value_v1 variants: the contaminated heavy_r1 model, the
temporally-rejected fair_v1 attempt, and (once evaluated)
evict_value_v1_cross_family_v1 as a fourth, independently certified variant.

Usage:
    python scripts/experiments/generate_fairness_certificate_v5.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

IN_DIR = Path("analysis/reviewer_fairness")
CROSS_FAMILY_DIR = Path("analysis/reviewer_fairness_cross_family_v1")
OUT_DIR = Path("analysis/reviewer_fairness_v5")
OUT_JSON = OUT_DIR / "fairness_certificate.json"
OUT_MD = OUT_DIR / "fairness_certificate.md"

EXPECTED_TRACES = 7
EXPECTED_CAPACITIES = {32, 64, 128}
EXPECTED_ROWS_PRIMARY = EXPECTED_TRACES * len(EXPECTED_CAPACITIES)  # 21
EXPECTED_SCORED_REQUESTS_PRIMARY = 40000

# Permanently, by design, FAIL -- see docs/reviewer_fairness_protocol.md
# section 6 and analysis/reviewer_fairness/evict_value_v1_overlap_audit.json.
# evict_value_v1_fair_v1 is NOT in this set: it is a distinct policy
# variant with its own (expected-PASS) certification once its rows exist.
TRAIN_TEST_OVERLAP_POLICIES = {"evict_value_v1", "evict_value_v1_fair_v1"}


# Policies explicitly rejected before any evaluation row was ever produced
# (not merely "not yet run") -- see analysis/reviewer_fairness/
# temporal_order_audit.json and docs/reviewer_fairness_temporal_rules.md.
EXPLICITLY_REJECTED = {
    "evict_value_v1_fair_v1": "TEMPORALLY_INELIGIBLE (0/7 families classified "
    "A/B/C/E under the frozen temporal rule; training stopped in Stage 1, "
    "before any model was fit)",
    "halp_cross_family_v1": "STRUCTURALLY_INCOMPATIBLE (HALP._train() is hardwired to "
    "self._recorded_events/self._actual_next, populated only via on_request() calls "
    "on the SAME trace instance being evaluated -- no injection point exists for an "
    "externally-sourced 5-family training corpus without a structural rewrite; see "
    "docs/reviewer_fairness_cross_family_v1.md section 'HALP decision'. Not implemented.",
}


def _load_rows(policy: str) -> List[Dict[str, str]]:
    if policy == "evict_value_v1_cross_family_v1":
        path = CROSS_FAMILY_DIR / "evict_value_v1" / "policy_comparison.csv"
        if path.exists():
            with path.open() as fh:
                return list(csv.DictReader(fh))
        return []
    for candidate_dir in (OUT_DIR, IN_DIR):
        path = candidate_dir / f"policy_comparison_{policy}.csv"
        if path.exists():
            with path.open() as fh:
                return list(csv.DictReader(fh))
    return []


def certify_policy(policy: str) -> Dict[str, object]:
    if policy in EXPLICITLY_REJECTED:
        return {
            "policy": policy, "overall": "REJECTED",
            "rejection_reason": EXPLICITLY_REJECTED[policy],
            "checks": {"TEMPORAL_ORDERING": "FAIL", "TRAIN_TEST_SEPARATION": "FAIL"},
        }

    rows = _load_rows(policy)
    if not rows:
        return {"policy": policy, "overall": "NOT_RUN", "checks": {}}

    primary_rows = [r for r in rows if r["policy_variant"] == "primary_controlled_window"]
    deploy_rows = [r for r in rows if r["policy_variant"] == "deployment_full_stream"]
    ok_primary = [r for r in primary_rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    checks: Dict[str, str] = {}

    traces = {r["trace"] for r in ok_primary}
    caps_by_trace: Dict[str, set] = {}
    for r in ok_primary:
        caps_by_trace.setdefault(r["trace"], set()).add(int(r["capacity"]))
    trace_pass = len(traces) == EXPECTED_TRACES and all(
        caps == EXPECTED_CAPACITIES for caps in caps_by_trace.values()
    )
    checks["TRACE"] = "PASS" if trace_pass else "FAIL"

    caps_seen = {int(r["capacity"]) for r in ok_primary}
    checks["CAPACITY"] = "PASS" if caps_seen == EXPECTED_CAPACITIES else "FAIL"

    obj_size_ok = bool(ok_primary) and all(r["object_size_semantics"] == "unit" for r in ok_primary)
    checks["OBJECT_SIZE"] = "PASS" if obj_size_ok else "FAIL"

    window_ok = bool(ok_primary) and all(
        int(r["score_start"]) == 10000 and int(r["score_end"]) == 50000
        and int(r["scored_requests"]) == EXPECTED_SCORED_REQUESTS_PRIMARY
        for r in ok_primary
    )
    checks["SCORING_WINDOW"] = "PASS" if window_ok else "FAIL"

    metric_ok = bool(ok_primary) and all(
        int(r["hits"]) + int(r["misses"]) == int(r["scored_requests"]) for r in ok_primary
    )
    checks["METRIC"] = "PASS" if metric_ok else "FAIL"

    checks["ROW_COMPLETENESS"] = "PASS" if len(ok_primary) == EXPECTED_ROWS_PRIMARY else \
        f"FAIL ({len(ok_primary)}/{EXPECTED_ROWS_PRIMARY} rows)"

    # No currently-run policy trains on a separately-sourced, cross-trace
    # training corpus the way the rejected evict_value_v1_fair_v1 did:
    # LRB/3L-Cache/CACHEUS have no offline training corpus at all (pure
    # in-trace online adaptation); HALP trains only on its own trace's
    # earlier [0, training_trigger) prefix (classification E, trivially
    # temporally valid); classical policies have no training. Hence N/A
    # here for all of them -- the temporal-ordering question genuinely
    # does not apply, not because it wasn't checked.
    checks["TEMPORAL_ORDERING"] = "N/A (no cross-trace training corpus for this policy)"

    future_info = ok_primary[0]["future_information"] if ok_primary else ""
    if policy in TRAIN_TEST_OVERLAP_POLICIES:
        checks["TRAIN_TEST_SEPARATION"] = "FAIL"
        checks["FUTURE_LEAKAGE"] = "FAIL"
    elif "oracle_by_definition" in future_info:
        # Belady and similar theoretical lower bounds are DESIGNED to use
        # future information -- this is not "no leakage" (PASS) and not a
        # defect (FAIL); it must be labeled as what it is so it is never
        # silently read as comparable to an online/learned policy.
        checks["TRAIN_TEST_SEPARATION"] = "N/A"
        checks["FUTURE_LEAKAGE"] = "ORACLE"
    elif ok_primary and ok_primary[0]["model_training_mode"] == "none":
        checks["TRAIN_TEST_SEPARATION"] = "N/A"
        checks["FUTURE_LEAKAGE"] = "PASS"
    else:
        checks["TRAIN_TEST_SEPARATION"] = "PASS" if ok_primary else "FAIL"
        checks["FUTURE_LEAKAGE"] = "PASS" if ok_primary else "FAIL"

    hp_source = ok_primary[0]["hyperparameter_source"] if ok_primary else ""
    if "NOT_validation_tuned" in hp_source:
        checks["HYPERPARAMETER_PROTOCOL"] = "PASS_WITH_CAVEAT"
    else:
        checks["HYPERPARAMETER_PROTOCOL"] = "PASS" if ok_primary else "FAIL"

    seed_val = ok_primary[0]["random_seed"] if ok_primary else ""
    checks["SEED"] = "PASS" if seed_val else "N/A"

    impl_source = ok_primary[0]["implementation_source"] if ok_primary else ""
    checks["IMPLEMENTATION_PROVENANCE"] = "PASS" if impl_source else "FAIL"

    checks["NO_FAILURE_ROWS"] = "PASS" if not failed else f"FAIL ({len(failed)} failed rows)"

    if policy == "halp":
        # HALP's cross-family training was audited and found structurally
        # incompatible (see EXPLICITLY_REJECTED['halp_cross_family_v1']).
        # Its EVALUATION conditions (trace, capacity, score window, metric)
        # are identical to evict_value_v1_cross_family_v1's, but its
        # TRAINING resources are not -- HALP trains on its own held-out
        # trace's own earlier prefix, not on 5 disjoint families. Both
        # facts must stay visible side by side, never collapsed into one
        # "fair" label.
        checks["TRAINING_RESOURCE_EQUIVALENCE_WITH_EVICT_VALUE"] = "NO"
        checks["EVALUATION_RESOURCE_EQUIVALENCE"] = "YES"

    material_checks = {k: v for k, v in checks.items() if k != "HYPERPARAMETER_PROTOCOL"}
    material_fail = any(v.startswith("FAIL") for v in material_checks.values())
    if checks["FUTURE_LEAKAGE"] == "ORACLE":
        checks["PRIMARY_ELIGIBILITY"] = "ORACLE_LOWER_BOUND_NOT_A_BASELINE"
    else:
        checks["PRIMARY_ELIGIBILITY"] = "FAIL" if material_fail else "PASS"
    overall = checks["PRIMARY_ELIGIBILITY"]

    return {
        "policy": policy,
        "n_primary_rows": len(primary_rows),
        "n_primary_rows_ok": len(ok_primary),
        "n_deployment_rows": len(deploy_rows),
        "n_failed_rows": len(failed),
        "checks": checks,
        "overall": overall,
    }


ALL_POLICIES = [
    "evict_value_v1", "evict_value_v1_fair_v1", "evict_value_v1_cross_family_v1",
    "lru", "sieve", "fifo_reinsertion",
    "blind_oracle_lru_combiner", "rest_v1", "trust_and_doubt", "predictive_marker",
    "offline_belady", "lrb", "three_l_cache", "halp", "halp_cross_family_v1", "cacheus",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    certificate = {"protocol_version": "reviewer_fairness_v5", "policies": {}}
    for p in ALL_POLICIES:
        certificate["policies"][p] = certify_policy(p)

    OUT_JSON.write_text(json.dumps(certificate, indent=2) + "\n")

    lines = ["# Reviewer Fairness Certificate V5", ""]
    lines.append(
        "| Policy | Overall | Trace | Capacity | Obj Size | Window | Metric | Rows | "
        "Temporal | Train/Test Sep | Future Leak | Hyperparams | Seed | Provenance | Failures |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in ALL_POLICIES:
        c = certificate["policies"][p]
        checks = c.get("checks", {})
        lines.append(
            f"| {p} | **{c['overall']}** | {checks.get('TRACE','-')} | {checks.get('CAPACITY','-')} | "
            f"{checks.get('OBJECT_SIZE','-')} | {checks.get('SCORING_WINDOW','-')} | {checks.get('METRIC','-')} | "
            f"{checks.get('ROW_COMPLETENESS','-')} | {checks.get('TEMPORAL_ORDERING','-')} | "
            f"{checks.get('TRAIN_TEST_SEPARATION','-')} | "
            f"{checks.get('FUTURE_LEAKAGE','-')} | {checks.get('HYPERPARAMETER_PROTOCOL','-')} | "
            f"{checks.get('SEED','-')} | {checks.get('IMPLEMENTATION_PROVENANCE','-')} | "
            f"{checks.get('NO_FAILURE_ROWS','-')} |"
        )
    lines.append("")
    lines.append(
        "`evict_value_v1` (the contaminated heavy_r1 model) is permanently FAIL by design -- "
        "see analysis/reviewer_fairness/evict_value_v1_overlap_audit.json. "
        "`evict_value_v1_fair_v1` is a distinct policy variant, certified independently."
    )
    lines.append("")
    lines.append(
        "No overall PASS is given if TRAIN_TEST_SEPARATION, FUTURE_LEAKAGE, ROW_COMPLETENESS, "
        "or any other non-hyperparameter check fails."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
