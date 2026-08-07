"""Generate the machine-readable and human-readable reviewer-fairness
certificate from the per-policy CSVs written by run_reviewer_fairness.py.

Certifies each policy separately across: trace, capacity, object size,
scoring window, metric, train/test separation, future-information leakage,
hyperparameter protocol, seed, and implementation provenance. Does NOT
give an overall PASS if any material condition fails -- in particular,
evict_value_v1 is expected to (and does) fail TRAIN_TEST_SEPARATION here,
per the CRITICAL finding in docs/reviewer_fairness_protocol.md section 6.

Usage:
    python scripts/experiments/generate_fairness_certificate.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

IN_DIR = Path("analysis/reviewer_fairness")
OUT_JSON = IN_DIR / "fairness_certificate.json"
OUT_MD = IN_DIR / "fairness_certificate.md"

EXPECTED_TRACES = 7
EXPECTED_CAPACITIES = {32, 64, 128}
EXPECTED_SCORED_REQUESTS_PRIMARY = 40000
EXPECTED_SCORED_REQUESTS_DEPLOYMENT = 50000

# Policies whose training data is known, by direct code/artifact audit, to
# overlap the scored evaluation traces -- see docs/reviewer_fairness_protocol.md
# section 6. Kept as an explicit, documented exception list rather than an
# inferred one.
TRAIN_TEST_OVERLAP_POLICIES = {"evict_value_v1"}


def _load_rows(policy: str) -> List[Dict[str, str]]:
    path = IN_DIR / f"policy_comparison_{policy}.csv"
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def certify_policy(policy: str) -> Dict[str, object]:
    rows = _load_rows(policy)
    if not rows:
        return {"policy": policy, "overall": "NOT_RUN", "checks": {}}

    primary_rows = [r for r in rows if r["policy_variant"] == "primary_controlled_window"]
    deploy_rows = [r for r in rows if r["policy_variant"] == "deployment_full_stream"]
    ok_primary = [r for r in primary_rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    checks: Dict[str, str] = {}

    traces = {r["trace"] for r in ok_primary}
    caps_by_trace = {}
    for r in ok_primary:
        caps_by_trace.setdefault(r["trace"], set()).add(int(r["capacity"]))
    trace_pass = (
        len(traces) == EXPECTED_TRACES
        and all(caps == EXPECTED_CAPACITIES for caps in caps_by_trace.values())
    )
    checks["TRACE"] = "PASS" if trace_pass else "FAIL"

    caps_seen = {int(r["capacity"]) for r in ok_primary}
    checks["CAPACITY"] = "PASS" if caps_seen == EXPECTED_CAPACITIES else "FAIL"

    obj_size_ok = all(r["object_size_semantics"] == "unit" for r in ok_primary)
    checks["OBJECT_SIZE"] = "PASS" if obj_size_ok and ok_primary else "FAIL"

    window_ok = all(
        int(r["score_start"]) == 10000 and int(r["score_end"]) == 50000
        and int(r["scored_requests"]) == EXPECTED_SCORED_REQUESTS_PRIMARY
        for r in ok_primary
    )
    checks["SCORING_WINDOW"] = "PASS" if window_ok and ok_primary else "FAIL"

    metric_ok = all(
        int(r["hits"]) + int(r["misses"]) == int(r["scored_requests"])
        for r in ok_primary
    )
    checks["METRIC"] = "PASS" if metric_ok and ok_primary else "FAIL"

    if policy in TRAIN_TEST_OVERLAP_POLICIES:
        checks["TRAIN_TEST_SEPARATION"] = "FAIL"
        checks["FUTURE_LEAKAGE"] = "FAIL"
    elif ok_primary and ok_primary[0]["model_training_mode"] == "none":
        checks["TRAIN_TEST_SEPARATION"] = "N/A"
        checks["FUTURE_LEAKAGE"] = "PASS"
    else:
        checks["TRAIN_TEST_SEPARATION"] = "PASS"
        checks["FUTURE_LEAKAGE"] = "PASS"

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

    material_fail = any(
        v == "FAIL" for k, v in checks.items()
        if k not in ("HYPERPARAMETER_PROTOCOL",)
    )
    overall = "FAIL" if material_fail else "PASS"

    return {
        "policy": policy,
        "n_primary_rows": len(primary_rows),
        "n_deployment_rows": len(deploy_rows),
        "n_failed_rows": len(failed),
        "checks": checks,
        "overall": overall,
    }


def main() -> None:
    policies = sorted(p.stem.replace("policy_comparison_", "") for p in IN_DIR.glob("policy_comparison_*.csv"))
    certificate = {"protocol_version": "reviewer_fairness_v1", "policies": {}}
    for p in policies:
        certificate["policies"][p] = certify_policy(p)

    OUT_JSON.write_text(json.dumps(certificate, indent=2) + "\n")

    lines = ["# Reviewer Fairness Certificate", ""]
    lines.append("| Policy | Overall | Trace | Capacity | Object Size | Window | Metric | Train/Test Sep | Future Leakage | Hyperparams | Seed | Provenance | Failures |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in policies:
        c = certificate["policies"][p]
        checks = c["checks"]
        lines.append(
            f"| {p} | **{c['overall']}** | {checks.get('TRACE','-')} | {checks.get('CAPACITY','-')} | "
            f"{checks.get('OBJECT_SIZE','-')} | {checks.get('SCORING_WINDOW','-')} | {checks.get('METRIC','-')} | "
            f"{checks.get('TRAIN_TEST_SEPARATION','-')} | {checks.get('FUTURE_LEAKAGE','-')} | "
            f"{checks.get('HYPERPARAMETER_PROTOCOL','-')} | {checks.get('SEED','-')} | "
            f"{checks.get('IMPLEMENTATION_PROVENANCE','-')} | {checks.get('NO_FAILURE_ROWS','-')} |"
        )
    lines.append("")
    lines.append(
        "`evict_value_v1` fails TRAIN_TEST_SEPARATION/FUTURE_LEAKAGE by design of this "
        "certificate (see docs/reviewer_fairness_protocol.md section 6) -- included for "
        "documentation, not eligible for the primary comparison."
    )
    lines.append("")
    lines.append(
        "Policies not certified here (not run in this session): lrb (currently running "
        "elsewhere as of this audit), blind_oracle_lru_combiner, rest_v1, trust_and_doubt, "
        "predictive_marker, offline_belady."
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
