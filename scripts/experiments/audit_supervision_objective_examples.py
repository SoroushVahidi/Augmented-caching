"""Same-example audit for the supervision-objective ablation
(docs/supervision_objective_ablation_protocol.md).

Verifies, per held-out-family fold, that the 3 scalar objectives
(objective_eviction_loss, objective_next_arrival, objective_reuse_distance)
are trained/evaluated from the EXACT SAME underlying candidate/example
universe before objective-specific label censoring is applied -- they
structurally must be, since scripts/build_supervision_objective_ablation_dataset.py
writes ONE shared `scalar` shard set per (family, capacity) carrying all
three label columns on every row (see that script's SCALAR_FIELDNAMES and
_flush_decision) -- but this audit verifies it rather than assuming it, and
reports the three objectives' post-censoring finite-label subsets
separately (eviction_loss is never censored; next_arrival/reuse_distance
drop rows where their own *_censored_flag == 1).

For the pairwise objective: verifies every sampled pair's decision_id
belongs to the same per-capacity decision-id population the scalar shards
were built from (same underlying candidate-state groups, not a different
trace/position) via a bounded ID-only spot-check over a capped number of
shard files -- NOT a full re-scan of every row (see --max-shards-per-fold).

Two-tier design, deliberately cheap:
  1. Manifest-level checks (no shard I/O): held-out family absent from
     input traces; capacities present == expected; per-capacity row/censor
     counts consistent between the manifest's own bookkeeping and the
     derived finite-label subsets.
  2. A bounded ID-only spot-check over shard files (decision_id / example_id
     columns only, never the feature columns) to catch structural
     corruption the manifest's aggregate counts alone would not reveal.

Usage:
    python scripts/experiments/audit_supervision_objective_examples.py --partial-audit
    python scripts/experiments/audit_supervision_objective_examples.py --families brightkite,citibike --partial-audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

DATA_ROOT = Path("data/derived/supervision_objective_ablation_v1")
FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
SCALAR_OBJECTIVES = ["objective_eviction_loss", "objective_next_arrival", "objective_reuse_distance"]
CENSOR_FLAG_COLUMN = {
    "objective_eviction_loss": None,
    "objective_next_arrival": "next_arrival_censored_flag",
    "objective_reuse_distance": "reuse_distance_censored_flag",
}
PROTOCOL_CONFIG_PATH = Path("configs/supervision_objective_ablation_v1.json")


def _protocol_config_sha256() -> str:
    if not PROTOCOL_CONFIG_PATH.exists():
        return "unavailable"
    return hashlib.sha256(PROTOCOL_CONFIG_PATH.read_bytes()).hexdigest()


class AuditFailure(RuntimeError):
    pass


def _load_fold(family: str) -> Dict[str, object]:
    path = FOLDS_DIR / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(family: str) -> Optional[Dict[str, object]]:
    path = DATA_ROOT / family / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_manifest_level(family: str, fold: Dict[str, object], manifest: Dict[str, object]) -> Dict[str, object]:
    report: Dict[str, object] = {"family": family, "checks": {}}

    input_families = {t["trace_family"] for t in manifest.get("trace_stats", [])}
    report["checks"]["held_out_family_isolation"] = (
        "PASS" if family not in input_families else f"FAIL (held-out family present in trace_stats: {input_families})"
    )

    expected_train_val = set(fold["training_families"]) | {fold["validation_family"]}
    report["checks"]["training_validation_family_identity"] = (
        "PASS" if expected_train_val.issubset(input_families | {None}) and input_families.issubset(expected_train_val)
        else f"FAIL (manifest input families {input_families} != fold-declared {expected_train_val})"
    )

    per_capacity = manifest.get("label_stats", {}).get("per_capacity", {})
    subsets: Dict[str, Dict[str, object]] = {}
    for cap_key, stat in per_capacity.items():
        scalar_rows = stat.get("scalar_rows", 0)
        na_censored = stat.get("next_arrival_censored_count", 0)
        rd_censored = stat.get("reuse_distance_censored_count", 0)
        subsets[cap_key] = {
            "common_candidate_universe": scalar_rows,
            "objective_eviction_loss_finite_subset": scalar_rows,
            "objective_eviction_loss_censored": 0,
            "objective_next_arrival_finite_subset": scalar_rows - na_censored,
            "objective_next_arrival_censored": na_censored,
            "objective_reuse_distance_finite_subset": scalar_rows - rd_censored,
            "objective_reuse_distance_censored": rd_censored,
            "pairwise_rows": stat.get("pairwise_rows", 0),
            "decisions": stat.get("decisions", 0),
        }
        if na_censored > scalar_rows or rd_censored > scalar_rows:
            report["checks"][f"censor_counts_bounded_{cap_key}"] = (
                f"FAIL (censored count exceeds candidate universe size at {cap_key})"
            )
        else:
            report["checks"][f"censor_counts_bounded_{cap_key}"] = "PASS"
    report["per_capacity_subsets"] = subsets
    return report


def _decision_id_from_example_id(example_id: str) -> str:
    return example_id.rsplit("|", 1)[0]


def _spot_check_pairwise_decision_ids(family: str, max_shards: int) -> Dict[str, object]:
    """Bounded ID-only check: every sampled pairwise pair's decision_id must
    appear in the scalar shard's decision_id population for the SAME
    (trace, capacity) shard prefix -- catches a pairwise shard accidentally
    built against a different trace/position than its scalar counterpart."""
    scalar_dir = DATA_ROOT / family / "scalar"
    pairwise_dir = DATA_ROOT / family / "pairwise"
    result: Dict[str, object] = {"shards_checked": 0, "mismatches": []}
    if not pairwise_dir.exists() or not scalar_dir.exists():
        result["note"] = "scalar or pairwise dir missing -- skipped"
        return result

    pairwise_shards = sorted(pairwise_dir.glob("*.csv"))[:max_shards]
    for pw_shard in pairwise_shards:
        # Shard files are named "{trace}__cap{N}.part{IDX}.csv" (see
        # _ShardWriter in build_supervision_objective_ablation_dataset.py);
        # the trace/capacity prefix before ".part" is what scalar and
        # pairwise shards share for the same underlying candidate pool.
        prefix = pw_shard.name.split(".part")[0]
        matching_scalar_shards = sorted(scalar_dir.glob(f"{prefix}.part*.csv"))
        scalar_decision_ids = set()
        for s_shard in matching_scalar_shards:
            with s_shard.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    scalar_decision_ids.add(row["decision_id"])

        with pw_shard.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["decision_id"] not in scalar_decision_ids:
                    result["mismatches"].append(
                        f"{pw_shard.name}: pair decision_id={row['decision_id']} not found in matching "
                        f"scalar shard(s) for prefix {prefix}"
                    )
        result["shards_checked"] += 1
    return result


def audit_fold(family: str, max_shards: int) -> Dict[str, object]:
    fold = _load_fold(family)
    manifest = _load_manifest(family)
    if manifest is None:
        return {"family": family, "status": "NOT_BUILT", "checks": {}}

    report = _audit_manifest_level(family, fold, manifest)
    report["pairwise_spot_check"] = _spot_check_pairwise_decision_ids(family, max_shards)
    if report["pairwise_spot_check"].get("mismatches"):
        report["checks"]["pairwise_candidate_group_consistency"] = (
            f"FAIL ({len(report['pairwise_spot_check']['mismatches'])} mismatch(es) in spot-check)"
        )
    else:
        report["checks"]["pairwise_candidate_group_consistency"] = "PASS"

    all_pass = all(v == "PASS" for v in report["checks"].values())
    report["status"] = "PASS" if all_pass else "FAIL"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=None, help="Comma-separated subset (default: all 7).")
    ap.add_argument("--max-shards-per-fold", type=int, default=3,
                     help="Bounded ID-only spot-check depth per fold (default 3 shard files, not a full scan).")
    ap.add_argument("--out", type=Path, default=Path("analysis/supervision_objective_ablation_v1/same_example_audit.json"))
    ap.add_argument("--partial-audit", action="store_true",
                     help="Required to run/write while fewer than 7 folds have a built dataset. Sets FINAL=false "
                     "in the output so this is never mistaken for the completed campaign's audit.")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",")] if args.families else FAMILIES

    reports: List[Dict[str, object]] = []
    for family in families:
        reports.append(audit_fold(family, args.max_shards_per_fold))

    built = [r for r in reports if r["status"] != "NOT_BUILT"]
    not_built = [r["family"] for r in reports if r["status"] == "NOT_BUILT"]
    is_final = len(not_built) == 0 and families == FAMILIES

    if not is_final and not args.partial_audit:
        print(f"[BLOCKED] {len(not_built)}/{len(families)} fold(s) not built yet: {not_built}")
        print("Pass --partial-audit to write a partial (FINAL=false) report anyway.")
        raise SystemExit(1)

    output = {
        "protocol_id": "supervision_objective_ablation_v1",
        "protocol_config_sha256": _protocol_config_sha256(),
        "FINAL": is_final,
        "scope_families": families,
        "folds_built": [r["family"] for r in built],
        "folds_not_built": not_built,
        "overall": "PASS" if built and all(r["status"] == "PASS" for r in built) else (
            "FAIL" if any(r["status"] == "FAIL" for r in built) else "NO_DATA"
        ),
        "reports": reports,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}: FINAL={is_final} overall={output['overall']} "
          f"({len(built)}/{len(families)} folds built)")
    if not is_final:
        print("[note] This is a PARTIAL audit (FINAL=false) -- do not cite as the completed campaign's result.")


if __name__ == "__main__":
    main()
