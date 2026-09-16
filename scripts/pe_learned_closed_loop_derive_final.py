#!/usr/bin/env python3
"""Deterministic, replay-free post-processing of the already-completed
pe_publication_learned_closed_loop_20260916 campaign outputs. Reads only
existing result.json / campaign_summary.json / random_seed_metrics.csv /
per_cell_metrics.csv; performs no simulation, no retraining, no re-scoring.
"""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

SCRATCH = Path("/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916")
PROJECT = Path("/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916")
FINAL = PROJECT / "final"
FINAL.mkdir(parents=True, exist_ok=True)

MODEL_SHA = "8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6"
PROTOCOL_SHA = "897b8d42ebfcb7e401a5bbac5245186acba4608a7d3b4af23694d6ca1aeff4b1"

# ---- load per_cell_metrics.csv (deterministic policies) ----
per_cell = {}
with open(PROJECT / "per_cell_metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["family"], row["capacity"], row["policy"])
        per_cell[key] = row

families = ["alibaba-block", "metacdn", "metakv", "twemcache", "wiki2018"]
capacities = ["32", "128"]
cells = [(fam, cap) for fam in families for cap in capacities]
det_policies = ["learned", "lru", "mru", "sieve", "lfu"]

for fam, cap in cells:
    for pol in det_policies:
        assert (fam, cap, pol) in per_cell, f"missing {fam} {cap} {pol}"
        mr = float(per_cell[(fam, cap, pol)]["miss_ratio"])
        assert 0.0 <= mr <= 1.0, f"out of range {fam} {cap} {pol} {mr}"

# ---- load random_seed_metrics.csv (no header) ----
random_rows = defaultdict(list)
with open(PROJECT / "random_seed_metrics.csv") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        fam, cap, seed, mr = parts
        mr = float(mr)
        assert 0.0 <= mr <= 1.0, f"out of range random {fam} {cap} {seed} {mr}"
        random_rows[(fam, cap)].append(mr)

for fam, cap in cells:
    assert len(random_rows[(fam, cap)]) == 20, f"expected 20 seeds for {fam} {cap}"

random_stats = {}
for (fam, cap), vals in random_rows.items():
    random_stats[(fam, cap)] = {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
    }

TOL = 1e-12  # exact-equality tolerance for tie classification


def mr(fam, cap, pol):
    return float(per_cell[(fam, cap, pol)]["miss_ratio"])


def wins_ties_losses(comparator):
    w = t = l = 0
    rows = []
    for fam, cap in cells:
        lv = mr(fam, cap, "learned")
        if comparator == "random_mean":
            cv = random_stats[(fam, cap)]["mean"]
        else:
            cv = mr(fam, cap, comparator)
        diff = lv - cv
        if abs(diff) <= TOL:
            outcome = "tie"
            t += 1
        elif diff < 0:
            outcome = "win"
            w += 1
        else:
            outcome = "loss"
            l += 1
        rows.append({"family": fam, "capacity": cap, "learned": lv, "comparator": cv,
                      "diff": diff, "outcome": outcome})
    return w, t, l, rows


# ---- learned_vs_baselines.csv ----
comparators = ["lru", "mru", "sieve", "lfu", "random_mean"]
lvb_rows = []
summary_wtl = {}
for comp in comparators:
    w, t, l, rows = wins_ties_losses(comp)
    summary_wtl[comp] = {"wins": w, "ties": t, "losses": l}
    for r in rows:
        lvb_rows.append({
            "comparator": comp, "family": r["family"], "capacity": r["capacity"],
            "learned_miss_ratio": r["learned"], "comparator_miss_ratio": r["comparator"],
            "diff_learned_minus_comparator": r["diff"], "outcome": r["outcome"],
        })

with open(FINAL / "learned_vs_baselines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["comparator", "family", "capacity",
                                       "learned_miss_ratio", "comparator_miss_ratio",
                                       "diff_learned_minus_comparator", "outcome"])
    w.writeheader()
    for r in lvb_rows:
        w.writerow(r)

# ---- macro / micro summaries (all 10 cells and excluding wiki2018) ----
def macro_micro(policy, exclude_wiki=False):
    use_cells = [c for c in cells if not (exclude_wiki and c[0] == "wiki2018")]
    vals = [mr(fam, cap, policy) for fam, cap in use_cells]
    macro = statistics.mean(vals)
    total_miss = sum(int(per_cell[(fam, cap, policy)]["misses"]) for fam, cap in use_cells)
    total_req = sum(int(per_cell[(fam, cap, policy)]["scored_requests"]) for fam, cap in use_cells)
    micro = total_miss / total_req
    return macro, micro

overall_summary = {}
for pol in det_policies:
    macro, micro = macro_micro(pol)
    macro_ex, micro_ex = macro_micro(pol, exclude_wiki=True)
    overall_summary[pol] = {
        "macro_miss_ratio_10cell": macro,
        "micro_miss_ratio_10cell": micro,
        "macro_miss_ratio_excl_wiki2018_8cell": macro_ex,
        "micro_miss_ratio_excl_wiki2018_8cell": micro_ex,
    }
# random (use per-seed mean-of-cell-means for macro; micro not well-defined without per-seed counts, use mean-of-means as approximation labeled clearly)
random_macro_10 = statistics.mean(random_stats[c]["mean"] for c in cells)
random_macro_8 = statistics.mean(random_stats[c]["mean"] for c in cells if c[0] != "wiki2018")
overall_summary["random_mean"] = {
    "macro_miss_ratio_10cell": random_macro_10,
    "macro_miss_ratio_excl_wiki2018_8cell": random_macro_8,
    "note": "random micro/request-weighted omitted: per-cell CSV stores only mean miss_ratio per seed, not per-seed miss/scored counts",
}

# ---- by_family_summary.csv ----
with open(FINAL / "by_family_summary.csv", "w", newline="") as f:
    fieldnames = ["family"] + det_policies + ["random_mean"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for fam in families:
        row = {"family": fam}
        for pol in det_policies:
            vals = [mr(fam, cap, pol) for cap in capacities]
            row[pol] = round(statistics.mean(vals), 6)
        row["random_mean"] = round(statistics.mean(random_stats[(fam, cap)]["mean"] for cap in capacities), 6)
        w.writerow(row)

# ---- by_capacity_summary.csv ----
with open(FINAL / "by_capacity_summary.csv", "w", newline="") as f:
    fieldnames = ["capacity"] + det_policies + ["random_mean"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for cap in capacities:
        row = {"capacity": cap}
        for pol in det_policies:
            vals = [mr(fam, cap, pol) for fam in families]
            row[pol] = round(statistics.mean(vals), 6)
        row["random_mean"] = round(statistics.mean(random_stats[(fam, cap)]["mean"] for fam in families), 6)
        w.writerow(row)

# ---- learned_vs_random_distribution.csv ----
with open(FINAL / "learned_vs_random_distribution.csv", "w", newline="") as f:
    fieldnames = ["family", "capacity", "learned_miss_ratio", "random_mean", "random_median",
                  "random_std", "random_min", "random_max", "learned_minus_random_mean",
                  "seeds_better_than_learned", "seeds_equal_to_learned", "seeds_worse_than_learned"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for fam, cap in cells:
        lv = mr(fam, cap, "learned")
        vals = random_rows[(fam, cap)]
        better = sum(1 for v in vals if v < lv - TOL)   # lower miss ratio = better
        equal = sum(1 for v in vals if abs(v - lv) <= TOL)
        worse = sum(1 for v in vals if v > lv + TOL)
        s = random_stats[(fam, cap)]
        w.writerow({
            "family": fam, "capacity": cap, "learned_miss_ratio": lv,
            "random_mean": s["mean"], "random_median": s["median"], "random_std": s["std"],
            "random_min": s["min"], "random_max": s["max"],
            "learned_minus_random_mean": lv - s["mean"],
            "seeds_better_than_learned": better, "seeds_equal_to_learned": equal,
            "seeds_worse_than_learned": worse,
        })

# ---- FINAL_SCIENTIFIC_SUMMARY.json ----
summary = {
    "campaign": "pe_publication_learned_closed_loop_20260916",
    "model_sha256": MODEL_SHA,
    "protocol_sha256": PROTOCOL_SHA,
    "cells": 10,
    "families": families,
    "capacities": [int(c) for c in capacities],
    "wins_ties_losses": summary_wtl,
    "overall_summary": overall_summary,
    "tie_tolerance": TOL,
    "tie_tolerance_note": "Exact floating-point equality within 1e-12; observed ties are all wiki2018 cells where every policy scores miss_ratio=1.0 exactly (degenerate all-miss trace tail), not near-ties.",
    "wiki2018_degenerate": True,
    "provenance": {
        "embedded_manifest_runner_commit": "62ccfa252733b5d20d413af2cb74590d0af2f235",
        "actual_submission_checkout_commit": "9684b4dac6e107289738e8d3cdb801b04dd4ef78",
        "slurm_jobs": {"production": "1291048", "validation": "1291049", "aggregation": "1291050"},
    },
}
(FINAL / "FINAL_SCIENTIFIC_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

# ---- checksums.sha256 over all raw + compact artifacts ----
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

checksum_lines = []
result_files = sorted(SCRATCH.glob("*/*/*/result.json"))
assert len(result_files) == 250, len(result_files)
for p in result_files:
    checksum_lines.append(f"{sha256_of(p)}  {p.relative_to(SCRATCH)}")
for name in ["validation_report.json"]:
    p = SCRATCH / name
    checksum_lines.append(f"{sha256_of(p)}  {name}")
for name in ["campaign_summary.json", "per_cell_metrics.csv", "random_seed_metrics.csv", "provenance.json"]:
    p = PROJECT / name
    checksum_lines.append(f"{sha256_of(p)}  {name}")
for name in ["learned_vs_baselines.csv", "by_family_summary.csv", "by_capacity_summary.csv",
             "learned_vs_random_distribution.csv", "FINAL_SCIENTIFIC_SUMMARY.json"]:
    p = FINAL / name
    checksum_lines.append(f"{sha256_of(p)}  final/{name}")

(PROJECT / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")

print(json.dumps({
    "wins_ties_losses": summary_wtl,
    "learned_macro_10cell": overall_summary["learned"]["macro_miss_ratio_10cell"],
    "learned_micro_10cell": overall_summary["learned"]["micro_miss_ratio_10cell"],
    "checksum_lines": len(checksum_lines),
}, indent=2))
