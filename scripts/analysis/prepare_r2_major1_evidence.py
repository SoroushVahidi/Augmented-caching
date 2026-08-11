#!/usr/bin/env python3
"""Prepare compact Reviewer #2 Major Comment 1 evidence summaries.

This script intentionally does not run experiments. It audits the existing
modern-baseline CSVs, writes small provenance/status artifacts, and, when a
verified corrected evict_value_v1 CSV is supplied, builds the matched primary
controlled-window comparison table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


POLICIES = {
    "lrb": {
        "label": "LRB",
        "csv": "policy_comparison_lrb.csv",
        "provenance": "provenance_lrb.json",
        "expected_sha256": "1972f359c2419d5470fe9c897ba74cd7a17ac8b48574734257e259e1c76410fa",
        "classification": "LOCAL_EXACT_PROTOCOL_VALIDATED",
        "caveat": "Independent repository reimplementation; unit-size/object-slot adaptation.",
    },
    "three_l_cache": {
        "label": "3L-Cache",
        "csv": "policy_comparison_three_l_cache.csv",
        "provenance": "provenance_three_l_cache.json",
        "expected_sha256": "9f666aadc036eac8abc690a3028f63bc889c7892705bb713a0c1846c8a2f46a6",
        "classification": "LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT",
        "caveat": "Independent repository reimplementation; fixed batch_size=4096 default, certificate PASS_WITH_CAVEAT.",
    },
    "cacheus": {
        "label": "CACHEUS",
        "csv": "policy_comparison_cacheus.csv",
        "provenance": "provenance_cacheus.json",
        "expected_sha256": "dc0c501002713a1ed1005e764b28b72f207c02a9cc40fe7b2d36db0c6fc50645",
        "classification": "LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT",
        "caveat": "Protocol exact; official-source clone/provenance is not currently live-verifiable in this worktree.",
    },
}

EXPECTED_TREATMENT_SHA256 = "982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296"
EXPECTED_CAPACITIES = [32, 64, 128]
EXPECTED_VARIANTS = ["deployment_full_stream", "primary_controlled_window"]
PRIMARY = "primary_controlled_window"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def finite_csv_values(rows: list[dict[str, str]]) -> int:
    bad = 0
    numeric_columns = {
        "capacity",
        "history_start",
        "history_end",
        "score_start",
        "score_end",
        "history_requests",
        "scored_requests",
        "hits",
        "misses",
        "miss_ratio",
        "runtime_seconds",
        "n_history_events",
        "n_scored_events",
    }
    for row in rows:
        for key, value in row.items():
            if key not in numeric_columns or value in {"", None}:
                continue
            try:
                if not math.isfinite(float(value)):
                    bad += 1
            except ValueError:
                continue
    return bad


def audit_baseline(policy: str, csv_path: Path, expected_sha: str) -> dict[str, Any]:
    rows = read_csv(csv_path)
    primary_rows = [r for r in rows if r.get("policy_variant") == PRIMARY]
    duplicate_keys = len(rows) - len(
        {
            (
                r.get("policy"),
                r.get("policy_variant"),
                r.get("trace"),
                r.get("capacity"),
            )
            for r in rows
        }
    )
    statuses = Counter(r.get("status", "") for r in rows)
    variants = sorted({r.get("policy_variant", "") for r in rows})
    families = sorted({r.get("trace", "") for r in rows})
    capacities = sorted({int(r["capacity"]) for r in rows})
    windows = sorted(
        {
            (
                r.get("policy_variant"),
                int(r["history_start"]),
                int(r["history_end"]),
                int(r["score_start"]),
                int(r["score_end"]),
                int(r["history_requests"]),
                int(r["scored_requests"]),
                r.get("capacity_semantics"),
                r.get("object_size_semantics"),
            )
            for r in rows
        }
    )
    sha = sha256_file(csv_path)
    primary_miss_ratios = [float(r["miss_ratio"]) for r in primary_rows]
    primary_misses = [int(float(r["misses"])) for r in primary_rows]
    checks = {
        "sha256_matches_expected": sha == expected_sha,
        "rows_42": len(rows) == 42,
        "primary_rows_21": len(primary_rows) == 21,
        "seven_families": len(families) == 7,
        "capacities_32_64_128": capacities == EXPECTED_CAPACITIES,
        "both_expected_variants": variants == EXPECTED_VARIANTS,
        "all_status_ok": statuses == Counter({"ok": 42}),
        "duplicate_keys_zero": duplicate_keys == 0,
        "nan_inf_zero": finite_csv_values(rows) == 0,
        "primary_window_exact": all(
            (
                r["policy_variant"] != PRIMARY
                or (
                    int(r["history_start"]) == 0
                    and int(r["history_end"]) == 10000
                    and int(r["score_start"]) == 10000
                    and int(r["score_end"]) == 50000
                    and int(r["history_requests"]) == 10000
                    and int(r["scored_requests"]) == 40000
                    and r["capacity_semantics"] == "object_slots"
                    and r["object_size_semantics"] == "unit"
                )
            )
            for r in rows
        ),
        "deployment_window_exact": all(
            (
                r["policy_variant"] != "deployment_full_stream"
                or (
                    int(r["history_start"]) == 0
                    and int(r["history_end"]) == 10000
                    and int(r["score_start"]) == 0
                    and int(r["score_end"]) == 50000
                    and int(r["history_requests"]) == 0
                    and int(r["scored_requests"]) == 50000
                    and r["capacity_semantics"] == "object_slots"
                    and r["object_size_semantics"] == "unit"
                )
            )
            for r in rows
        ),
        "hits_plus_misses_equals_scored": all(
            int(float(r["hits"])) + int(float(r["misses"])) == int(r["scored_requests"])
            for r in rows
        ),
    }
    return {
        "policy": policy,
        "path": str(csv_path),
        "sha256": sha,
        "expected_sha256": expected_sha,
        "row_count": len(rows),
        "primary_row_count": len(primary_rows),
        "families": families,
        "capacities": capacities,
        "variants": variants,
        "status_distribution": dict(statuses),
        "duplicate_key_count": duplicate_keys,
        "nan_inf_count": finite_csv_values(rows),
        "windows": [list(w) for w in windows],
        "checks": checks,
        "integrity": "PASS" if all(checks.values()) else "FAIL",
        "primary_mean_miss_ratio": mean(primary_miss_ratios),
        "primary_median_miss_ratio": median(primary_miss_ratios),
        "primary_total_misses": sum(primary_misses),
    }


def build_trace_manifest(repo: Path, data_root: Path) -> dict[str, Any]:
    manifest_csv = repo / "analysis" / "wulver_trace_manifest_full.csv"
    provenance = json.loads((repo / "analysis/reviewer_fairness/provenance_lrb.json").read_text())
    expected_hashes = provenance["trace_hashes_sha256"]
    traces = []
    with manifest_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            path = data_root / row["path"]
            entry: dict[str, Any] = {
                "logical_family": row["trace_family"],
                "trace_name": row["trace_name"],
                "manifest_relative_path": row["path"],
                "data_root": str(data_root),
                "exists": path.exists(),
            }
            if path.exists():
                request_count = 0
                digest = hashlib.sha256()
                with path.open("rb") as trace_fh:
                    for line in trace_fh:
                        digest.update(line)
                        request_count += 1
                sha = digest.hexdigest()
                entry.update(
                    {
                        "file_size_bytes": path.stat().st_size,
                        "request_count": request_count,
                        "sha256": sha,
                        "matches_baseline_provenance": sha == expected_hashes[row["trace_name"]],
                    }
                )
            traces.append(entry)
    return {
        "source_manifest": str(manifest_csv),
        "temporary_manifest_gap_repaired_by": "actual local trace files plus stored baseline trace_hashes_sha256",
        "preprocessing_identifier": "processed 50k JSONL traces listed in analysis/wulver_trace_manifest_full.csv",
        "traces": traces,
        "all_available": all(t["exists"] for t in traces),
        "all_hashes_match_baseline_provenance": all(
            t.get("matches_baseline_provenance") for t in traces if t["exists"]
        ),
    }


def audit_fairness_certificate(repo: Path) -> dict[str, Any]:
    path = repo / "analysis/reviewer_fairness/fairness_certificate.json"
    certificate = json.loads(path.read_text())
    policies = certificate.get("policies", {})
    covered = {}
    for policy in POLICIES:
        entry = policies.get(policy)
        if entry is None:
            covered[policy] = {
                "covered_by_certificate": False,
                "overall": "NOT_PRESENT",
                "caveat": "Not in this certificate; covered by direct CSV/provenance audit instead.",
            }
            continue
        covered[policy] = {
            "covered_by_certificate": True,
            "overall": entry.get("overall"),
            "checks": entry.get("checks", {}),
            "n_primary_rows": entry.get("n_primary_rows"),
            "n_deployment_rows": entry.get("n_deployment_rows"),
            "n_failed_rows": entry.get("n_failed_rows"),
            "caveat": (
                "3L-Cache carries PASS_WITH_CAVEAT on hyperparameter protocol."
                if policy == "three_l_cache"
                else "Certificate covers this policy."
            ),
        }
    return {
        "certificate_path": str(path),
        "protocol_version": certificate.get("protocol_version"),
        "can_one_certificate_cover_all_three": False,
        "reason": "The strongest local certificate covers CACHEUS and 3L-Cache, but not LRB; LRB is covered by the direct exact-row audit and provenance JSON.",
        "policy_status": covered,
    }


def keyed_primary(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (r["trace"], int(r["capacity"])): r
        for r in rows
        if r.get("policy_variant") == PRIMARY and r.get("status") == "ok"
    }


def compare_treatment(treatment_csv: Path, baseline_csvs: dict[str, Path]) -> list[dict[str, Any]]:
    treatment = keyed_primary(read_csv(treatment_csv))
    comparisons = []
    for policy, path in baseline_csvs.items():
        baseline = keyed_primary(read_csv(path))
        common_keys = sorted(set(treatment) & set(baseline))
        wins = Counter()
        by_family: dict[str, Counter[str]] = defaultdict(Counter)
        by_capacity: dict[int, Counter[str]] = defaultdict(Counter)
        diffs = []
        for key in common_keys:
            t = float(treatment[key]["miss_ratio"])
            b = float(baseline[key]["miss_ratio"])
            diff = t - b
            diffs.append(diff)
            family, capacity = key
            if abs(diff) <= 1e-12:
                outcome = "tie"
            elif diff < 0:
                outcome = "evict_value_v1_wins"
            else:
                outcome = "baseline_wins"
            wins[outcome] += 1
            by_family[family][outcome] += 1
            by_capacity[capacity][outcome] += 1
        b_mr = [float(r["miss_ratio"]) for r in baseline.values()]
        t_mr = [float(r["miss_ratio"]) for r in treatment.values()]
        comparisons.append(
            {
                "baseline": policy,
                "cells": len(common_keys),
                "baseline_mean_miss_ratio": mean(b_mr),
                "baseline_median_miss_ratio": median(b_mr),
                "baseline_total_misses": sum(int(float(r["misses"])) for r in baseline.values()),
                "evict_value_v1_mean_miss_ratio": mean(t_mr),
                "absolute_mean_difference_evict_minus_baseline": mean(diffs),
                "relative_mean_difference_evict_minus_baseline": mean(diffs) / mean(b_mr),
                "baseline_wins": wins["baseline_wins"],
                "evict_value_v1_wins": wins["evict_value_v1_wins"],
                "ties": wins["tie"],
                "family_summary": {k: dict(v) for k, v in sorted(by_family.items())},
                "capacity_summary": {str(k): dict(v) for k, v in sorted(by_capacity.items())},
            }
        )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("/home/soroush/Augmented-caching"))
    parser.add_argument("--baselines-dir", type=Path, default=Path("analysis/reviewer_fairness"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/kbs_r2_major1_evidence_prep_20260811"))
    parser.add_argument("--treatment-csv", type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    baselines_dir = (repo / args.baselines_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_integrity = {}
    baseline_provenance = {}
    baseline_summary_rows = []
    protocol_rows = []
    baseline_csvs = {}
    for policy, meta in POLICIES.items():
        csv_path = baselines_dir / meta["csv"]
        provenance_path = baselines_dir / meta["provenance"]
        baseline_csvs[policy] = csv_path
        audit = audit_baseline(policy, csv_path, meta["expected_sha256"])
        baseline_integrity[policy] = audit
        provenance = json.loads(provenance_path.read_text())
        sample_row = next(r for r in read_csv(csv_path) if r["policy_variant"] == PRIMARY)
        baseline_provenance[policy] = {
            "label": meta["label"],
            "csv_path": str(csv_path),
            "provenance_path": str(provenance_path),
            "repository_commit": provenance.get("repository_commit"),
            "repository_branch": provenance.get("repository_branch"),
            "producing_runner": "scripts/experiments/run_reviewer_fairness.py",
            "implementation_source": sample_row.get("implementation_source"),
            "implementation_commit": sample_row.get("implementation_commit"),
            "model_training_mode": sample_row.get("model_training_mode"),
            "model_training_data": sample_row.get("model_training_data"),
            "model_frozen_during_test": sample_row.get("model_frozen_during_test"),
            "online_adaptation_during_test": sample_row.get("online_adaptation_during_test"),
            "hyperparameter_source": sample_row.get("hyperparameter_source"),
            "random_seed": sample_row.get("random_seed"),
            "future_information": sample_row.get("future_information"),
            "batch_size_or_equivalent": sample_row.get("batch_size_or_equivalent"),
            "classification": meta["classification"],
            "caveat": meta["caveat"],
        }
        baseline_summary_rows.append(
            {
                "baseline": meta["label"],
                "primary_mean_miss_ratio": f"{audit['primary_mean_miss_ratio']:.12f}",
                "primary_median_miss_ratio": f"{audit['primary_median_miss_ratio']:.12f}",
                "primary_total_misses": audit["primary_total_misses"],
                "integrity": audit["integrity"],
                "classification": meta["classification"],
                "sha256": audit["sha256"],
            }
        )
        protocol_rows.append(
            {
                "baseline": meta["label"],
                "row_count": audit["row_count"],
                "primary_rows": audit["primary_row_count"],
                "families": len(audit["families"]),
                "capacities": "32;64;128",
                "variants": ";".join(audit["variants"]),
                "history_window": "[0,10000)",
                "score_window": "[10000,50000)",
                "scored_requests": 40000,
                "capacity_semantics": "object_slots",
                "object_size_semantics": "unit",
                "object_miss_semantics": "hits + misses == scored_requests; miss_ratio = misses / scored_requests",
                "classification": meta["classification"],
                "caveat": meta["caveat"],
            }
        )

    trace_manifest = build_trace_manifest(repo, args.data_root)
    fairness_certificate_status = audit_fairness_certificate(repo)
    write_json(out_dir / "baseline_integrity.json", baseline_integrity)
    write_json(out_dir / "baseline_provenance.json", baseline_provenance)
    write_json(out_dir / "trace_manifest.json", trace_manifest)
    write_json(out_dir / "fairness_certificate_status.json", fairness_certificate_status)
    write_csv(
        out_dir / "baseline_summary.csv",
        baseline_summary_rows,
        ["baseline", "primary_mean_miss_ratio", "primary_median_miss_ratio", "primary_total_misses", "integrity", "classification", "sha256"],
    )
    write_csv(
        out_dir / "baseline_protocol_comparison.csv",
        protocol_rows,
        ["baseline", "row_count", "primary_rows", "families", "capacities", "variants", "history_window", "score_window", "scored_requests", "capacity_semantics", "object_size_semantics", "object_miss_semantics", "classification", "caveat"],
    )

    treatment_status = {
        "expected_path": "analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv",
        "expected_sha256": EXPECTED_TREATMENT_SHA256,
        "found_locally": False,
        "classification": "WULVER_ONLY_VALIDATED",
        "comparison_status": "FINAL_COMPARISON_PENDING",
        "old_contaminated_local_file_excluded": "analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv",
    }
    if args.treatment_csv:
        treatment_csv = args.treatment_csv if args.treatment_csv.is_absolute() else repo / args.treatment_csv
        treatment_status["candidate_path"] = str(treatment_csv)
        if treatment_csv.exists():
            treatment_sha = sha256_file(treatment_csv)
            treatment_status.update(
                {
                    "found_locally": treatment_sha == EXPECTED_TREATMENT_SHA256,
                    "candidate_sha256": treatment_sha,
                    "sha256_matches_expected": treatment_sha == EXPECTED_TREATMENT_SHA256,
                }
            )
            if treatment_sha == EXPECTED_TREATMENT_SHA256:
                comparisons = compare_treatment(treatment_csv, baseline_csvs)
                write_json(out_dir / "reviewer_ready_comparison.json", comparisons)
                flat_rows = [
                    {
                        "baseline": row["baseline"],
                        "cells": row["cells"],
                        "baseline_mean_miss_ratio": f"{row['baseline_mean_miss_ratio']:.12f}",
                        "baseline_median_miss_ratio": f"{row['baseline_median_miss_ratio']:.12f}",
                        "baseline_total_misses": row["baseline_total_misses"],
                        "evict_value_v1_mean_miss_ratio": f"{row['evict_value_v1_mean_miss_ratio']:.12f}",
                        "absolute_mean_difference_evict_minus_baseline": f"{row['absolute_mean_difference_evict_minus_baseline']:.12f}",
                        "relative_mean_difference_evict_minus_baseline": f"{row['relative_mean_difference_evict_minus_baseline']:.12f}",
                        "baseline_wins": row["baseline_wins"],
                        "evict_value_v1_wins": row["evict_value_v1_wins"],
                        "ties": row["ties"],
                    }
                    for row in comparisons
                ]
                write_csv(
                    out_dir / "reviewer_ready_comparison.csv",
                    flat_rows,
                    ["baseline", "cells", "baseline_mean_miss_ratio", "baseline_median_miss_ratio", "baseline_total_misses", "evict_value_v1_mean_miss_ratio", "absolute_mean_difference_evict_minus_baseline", "relative_mean_difference_evict_minus_baseline", "baseline_wins", "evict_value_v1_wins", "ties"],
                )
                treatment_status["classification"] = "LOCAL_AND_WULVER_VALIDATED"
                treatment_status["comparison_status"] = "COMPLETE"
    write_json(out_dir / "treatment_status.json", treatment_status)

    (out_dir / "fairness_statement.md").write_text(
        """# Publication-facing fairness statement

The modern learned/adaptive baselines are evaluated under the same
controlled-window replay protocol as the corrected treatment evaluation:
seven trace families, capacities 32/64/128, object-slot capacity, unit
objects, and object misses over the primary scored suffix [10000,50000)
after each policy processes the common history prefix [0,10000). LRB and
3L-Cache are online repository reimplementations that adapt only from their
own in-trace stream; CACHEUS is represented as an official-source wrapper
with a separate provenance caveat because the external clone is not
vendored and is not currently live-verifiable in this worktree. The
offline evict_value_v1 treatment is conceptually different: its corrected
version uses leave-one-family-out training and a frozen model before the
held-out-family replay.
""".lstrip()
    )
    (out_dir / "FINAL_COMPARISON_PENDING.md").write_text(
        f"""# Final matched comparison status

The corrected treatment CSV was not found or not supplied with the expected
SHA-256 `{EXPECTED_TREATMENT_SHA256}` during this preparation pass.

Do not use `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`;
that file is the old contaminated/ineligible treatment-side result.

Once the verified CSV is synchronized, run:

```bash
python3 scripts/analysis/prepare_r2_major1_evidence.py \\
  --treatment-csv analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv
```

The script will write `reviewer_ready_comparison.csv` and
`reviewer_ready_comparison.json` in this directory.
""".lstrip()
    )
    (out_dir / "reviewer_response_version_a.md").write_text(
        """# Reviewer #2 Major Comment 1 response draft - internal pending version

We have completed the baseline-side audit for the modern learned/adaptive
comparisons. LRB, 3L-Cache, and CACHEUS each have 42-row local controlled
window result files with 21 primary rows over the seven families and
capacities 32/64/128. The baseline rows use the common primary window
[10000,50000) after an identical history prefix [0,10000), object-slot
capacity, unit objects, and object-miss metrics. LRB and 3L-Cache are
online/adaptive reimplementations that learn only from their own in-trace
stream; CACHEUS uses the official-source wrapper, with provenance caveats
recorded separately.

The corrected evict_value_v1 treatment artifact remains
WULVER_ONLY_VALIDATED locally, so the final matched numerical synthesis is
pending synchronization of the verified CSV. No additional baseline
experiment is scientifically required for this comment; the pending Wulver
baseline jobs would serve as independent replication/provenance
strengthening.
""".lstrip()
    )
    (out_dir / "reviewer_response_version_b.md").write_text(
        """# Reviewer #2 Major Comment 1 response draft - manuscript template

To address the request for stronger learned/adaptive baselines, we added a
matched controlled-window comparison against LRB, 3L-Cache, and CACHEUS.
The proposed evict_value_v1 model is evaluated as an offline learned policy:
for each held-out trace family, model training and selection exclude that
family, and the resulting model is frozen before replay. In contrast, LRB,
3L-Cache, and CACHEUS are online/adaptive baselines; they are not trained
with leave-one-family-out offline corpora, but learn only from their own
observed in-trace streams during replay.

All methods are compared on the same seven trace families, capacities
32/64/128, object-slot capacity, unit objects, and object-miss metric over
the primary scored suffix [10000,50000) after processing the common history
prefix [0,10000). Under this matched protocol, evict_value_v1 obtains mean
miss ratio [EV_MEAN], compared with LRB [LRB_MEAN], 3L-Cache [3L_MEAN], and
CACHEUS [CACHEUS_MEAN]. Across the 21 matched family-capacity cells,
evict_value_v1 wins/ties/loses against LRB [LRB_WTL], against 3L-Cache
[3L_WTL], and against CACHEUS [CACHEUS_WTL]. These comparisons separate
the offline learned treatment setting from online adaptive baselines while
using identical replay windows and metric semantics.
""".lstrip()
    )
    (out_dir / "README.md").write_text(
        """# R2 Major 1 evidence preparation package

This directory contains compact, non-duplicative evidence summaries for
Reviewer #2 Major Comment 1. The original generated CSVs are not copied;
they are referenced by path and SHA-256 in the JSON/CSV manifests here.

- `baseline_integrity.json`: row counts, hashes, protocol checks, and
  primary baseline summaries.
- `baseline_provenance.json`: runner, commit, branch, learning mode, seed,
  and caveats for LRB, 3L-Cache, and CACHEUS.
- `baseline_protocol_comparison.csv`: compact protocol-equivalence table.
- `baseline_summary.csv`: primary controlled-window baseline metrics.
- `trace_manifest.json`: durable reconstruction of the seven trace
  identities from local files and stored provenance hashes.
- `fairness_statement.md`: publication-facing fairness wording.
- `fairness_certificate_status.json`: certificate coverage and caveats.
- `treatment_status.json`: corrected evict_value_v1 local availability.
- `FINAL_COMPARISON_PENDING.md`: deterministic procedure for final synthesis
  once the verified corrected treatment CSV is synchronized.
- `reviewer_response_version_a.md` and `reviewer_response_version_b.md`:
  pending/internal and manuscript-ready response drafts.
""".lstrip()
    )


if __name__ == "__main__":
    main()
