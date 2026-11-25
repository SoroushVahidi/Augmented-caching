"""Summarize reproducible Phase 2B results into an allowlisted evidence package."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import sklearn
import torch

from diagnostics import WORK
from replay_experiment import CONFIG, GRID, STUDENT_GRID
from verify import HERE, OUT, SEED, write_csv, write_json
from sources import PROTOCOL, NOVEMBER, digest

REQUIRED = (
    "EXPERIMENT_PROTOCOL.md", "EXACT_LRU_LABELER_PROOF.md", "LABELER_CORRECTNESS_TESTS.json",
    "ARCHIVED_JOIN_CORRECTNESS.json", "LABELER_BENCHMARK.csv", "SHORTLIST_COVERAGE.csv",
    "SAMPLE_ADEQUACY.csv", "HORIZON_SENSITIVITY.csv", "FEATURE_ABLATION_RESULTS.csv",
    "LABEL_GENERATION_RESOURCES.csv", "MODEL_CONFIGURATIONS.json", "VALIDATION_RESULTS.csv",
    "CLOSED_LOOP_RESULTS.csv", "FAMILY_LEVEL_RESULTS.csv", "CAPACITY_LEVEL_RESULTS.csv",
    "OVERRIDE_DIAGNOSTICS.csv", "RESOURCE_PROFILE.csv", "POLICY_SUMMARY.csv", "EXPERIMENT_CONCLUSIONS.md",
)


def read(name):
    with (OUT / name).open(newline="") as source:
        return list(csv.DictReader(source))


def git(path, *commands):
    return subprocess.check_output(["git", "-C", str(path), *commands], text=True).strip()


def build(partial=False):
    summary = read("POLICY_SUMMARY.csv")
    selections = {}
    checkpoint_provenance = {}
    stages = ("students",) if partial else ("students", "neural")
    for family in PROTOCOL:
        selections[family] = {stage: json.loads((WORK / f"{family}-{stage}-selection.json").read_text())
                              for stage in stages}
        root = NOVEMBER / "runs/campaign/h8" / family
        candidates = json.loads((root / "offline.json").read_text())["models"]
        checkpoint_provenance[family] = [dict(seed=m["seed"], sha256=m["sha256"], bytes=m["size_bytes"])
                                         for m in candidates if m["method"] == "deepsets" and m["seed"] in (7,19)]
    write_json("MODEL_CONFIGURATIONS.json", dict(
        source_sha=git(HERE,"rev-parse","HEAD"), implementation_branch=git(HERE,"branch","--show-current"),
        phase2a_source_sha="206defb993cc074ed69401fc062233816384de37",
        november_source_sha=git(NOVEMBER,"rev-parse","HEAD"), seed=SEED,
        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
        protocol_sha256=digest(OUT/"EXPERIMENT_PROTOCOL.md"),
        joins_sha256=digest(OUT/"ARCHIVED_JOIN_CORRECTNESS.json"),
        protocol=PROTOCOL, student_estimators=CONFIG, tuning_grid=GRID, student_grid=STUDENT_GRID,
        frozen_source_checkpoints=checkpoint_provenance,
        selections=selections,
        commands=[
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B experiments/cache_supervision_phase2b/verify.py test",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B experiments/cache_supervision_phase2b/verify.py benchmark",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python3 -B experiments/cache_supervision_phase2b/diagnostics.py",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python3 -B experiments/cache_supervision_phase2b/replay_experiment.py --stage students --workers 7",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python3 -B experiments/cache_supervision_phase2b/replay_experiment.py --stage neural --workers 7",
        ]))
    adequacy = read("SAMPLE_ADEQUACY.csv")
    primary = [r for r in adequacy if r["budget"] == "1176" and r["candidate_set"] == "low_frequency_k8"]
    train = [int(r["resolved_positive"]) for r in primary if r["role"] == "train"]
    val = [int(r["resolved_positive"]) for r in primary if r["role"] == "validation"]
    assert min(train) < 30 or min(val) < 10
    write_csv("FEATURE_ABLATION_RESULTS.csv", [dict(status="NOT_EXECUTED_DATA_LIMITED",
        minimum_required_positive_training=30, minimum_required_positive_validation=10,
        observed_minimum_training=min(train), observed_minimum_validation=min(val),
        observed_maximum_training=max(train), observed_maximum_validation=max(val))])
    k8 = {r["mechanism"]: [] for r in read("SHORTLIST_COVERAGE.csv") if r["k"] == "8"}
    for r in read("SHORTLIST_COVERAGE.csv"):
        if r["k"] == "8":
            k8[r["mechanism"]].append(int(r["covered_opportunities"]))
    positive = defaultdict(int)
    for r in adequacy:
        if r["stratum"] and r["candidate_set"] == "low_frequency_k8":
            positive[r["stratum"]] += int(r["resolved_positive"])
    horizons = defaultdict(lambda: [0]*4)
    for r in read("HORIZON_SENSITIVITY.csv"):
        if r["candidate_set"] == "all" and r["kind"] in ("fixed_requests", "resolved"):
            measurements = horizons[r["kind"], float(r["horizon_spec"])]
            for i,k in enumerate(("all_tie","all_tie_resolved_nonconstant","positive_decision","value_disagreement")):
                measurements[i] += int(r[k])
    benchmark = defaultdict(dict)
    for r in read("LABELER_BENCHMARK.csv"):
        benchmark[int(r["capacity"]),int(r["horizon"])][r["method"]] = float(r["seconds_per_decision"])
    speed = {}
    for k in ("november_style_exhaustive_compiled","phase2a_coalescence_compiled","repository_simulator"):
        ratios = [x[k]/x["shared_ordered_dict"] for x in benchmark.values() if k in x]
        speed[k] = median(ratios)
    rows = []
    for r in summary:
        rows.append(f"| {r['policy']} | {r['seed']} | {int(r['misses']):,} | "
                    f"{float(r['relative_miss_reduction']):+.3%} | "
                    f"{float(r['macro_family_relative_miss_reduction']):+.3%} | "
                    f"{r['worst_family']} {max(0.,float(r['worst_family_regression'])):.3%} | "
                    f"{float(r['cpu_vs_lru']):.2f}× |")
    fixed = {r["policy"]: r for r in summary}
    students = []
    for r in summary:
        if "student_" not in r["policy"] or not r["policy"].endswith("_resolved"):
            continue
        matched = fixed[r["policy"].removesuffix("_resolved")+"_h8"]
        if (float(r["relative_miss_reduction"]) >= .005
                and int(r["misses"]) < int(matched["misses"])
                and float(r["macro_family_relative_miss_reduction"]) > 0
                and float(r["leave_one_family_out_min"]) > 0
                and float(r["worst_family_regression"]) <= .02
                and float(r["cpu_vs_lru"]) <= 5.):
            students.append(r["policy"])
    screening = "The matched student passed screening." if students else "No resolved student passed screening."
    stage_note = ("The checkpoint fallback campaign is in progress. The table covers completed matched students, "
                  "classical controls and oracles." if partial else
                  "All seven folds completed the classical, matched student and frozen-checkpoint policy replays.")
    uniform = [sum(v) for m,v in k8.items() if m.startswith("uniform")]
    f8 = horizons["fixed_requests",8.]
    resolved = horizons["resolved",0.]
    label_resources = read("LABEL_GENERATION_RESOURCES.csv")
    label_seconds = sum(float(r["label_seconds"]) for r in label_resources)
    prep_seconds = sum(float(r["cpu_seconds"]) for r in label_resources)
    text = f"""# Phase 2B experiment conclusions

{stage_note} {screening} The primary comparison is **DATA_LIMITED** under the recorded fold criterion. Training folds contained {min(train)}–{max(train)} resolved-positive shortlist decisions and validation folds contained {min(val)}–{max(val)}. Feature ablation did not run.

The exact labeler matched the repository comparisons on 26,530 exhaustive states, 10,000 randomized cases and 44 structured cases, plus all 1,176 primary and 504 enriched archived decision groups. The established Phase 2A comparator has a horizon-zero edge case. The general repository simulator and compiled exhaustive comparator agreed there. The archive uses positive horizons.

Median speedups over measured configurations were {speed['november_style_exhaustive_compiled']:.1f}× over the compiled exhaustive replay, {speed['phase2a_coalescence_compiled']:.1f}× over Phase 2A paired replay and {speed['repository_simulator']:.1f}× over the Python repository simulator. The compiled baselines were faster in some small cells. The paired implementation also allocates an (H+1) by C reference matrix. Python allocation tracing excludes some native compiler and runtime memory.

Low-frequency K=8 covered {sum(k8['low_frequency'])}/64 profitable decision states and captured 37 of the 74 total hindsight miss savings across those common states. LRU-tail covered {sum(k8['lru_tail'])}, ghost-informed {sum(k8['ghost_informed'])}, SIEVE-derived {sum(k8['sieve_derived'])} and uniform seeds covered {min(uniform)}–{max(uniform)}. Uniform coverage was diagnostic and did not change the frozen shortlist. These gains describe one forced action on a common LRU state.

The nested 1,176 / 5,000 / 20,000 samples gave {positive['temporal_stratified']} / {positive['temporal_stratified']+positive['additional_5k']} / {sum(positive.values())} resolved-positive K=8 decisions. Across 21 cells, label evaluation took {label_seconds:.2f} wall seconds within {prep_seconds:.1f} CPU seconds for the preparation pipeline. Supplemental labels were generated and counted in memory. The stored local training archive contains the original 1,176 groups. A compact raw representation of two one-byte labels per candidate costs about 2.99 MB at 20,000 groups and an average of 74.7 candidates. Retaining 26 float64 features for all candidates would add about 310 MB before compression, plus metadata.

Across full candidate sets, H=8 gave {f8[0]}/1,176 all-tie states and {f8[2]} positive states. Of the H=8 all-tie states, {f8[1]} split under resolution. Fully resolved advantages retained {resolved[0]} all-tie decisions and exposed {resolved[2]} positive states. H=512 recovered the same 64 profitable states with five remaining action-value disagreements. This diagnostic establishes a target difference. It does not establish improved learnability.

| Policy | Seed | Misses | Relative reduction | Macro-family relative reduction | Worst family regression | CPU vs LRU |
|---|---:|---:|---:|---:|---|---:|
{chr(10).join(rows)}

The software CPU measurements include replay and feature work and use a bounded exact-feature memoization cache for each checkpoint and capacity. They exclude compilation. Process RSS includes the Python, Torch and numerical runtimes and prior allocations in the worker. A validation selection of LRU uses the LRU implementation and does not call the model. Random intervention counts vary around the target intervention rate.

Keep independent confirmation sealed. The next controlled comparison should use the existing 20,000-state budget to fit matched students after ensuring enough positive validation examples in each family. Compare causal shortlist coverage on development states separately. For these bounded steps local CPU suffices. A later Wulver pilot, if local replay justifies it, could start with 8 CPU tasks, 16 GB RAM, 1 hour and zero GPUs for a bounded 100,000-group profiling run, then scale from measured preparation and replay costs.
"""
    (OUT/"EXPERIMENT_CONCLUSIONS.md").write_text(text)
    names = [*REQUIRED]
    if not partial:
        names.append("CHECKPOINT_FALLBACK_RESULTS.csv")
    for name in names:
        assert (OUT/name).is_file(), name
    (OUT/"MANIFEST.sha256").write_text("".join(f"{digest(OUT/name)}  {name}\n" for name in sorted(names)))
    if not partial:
        archive = WORK / (OUT.name + ".zip")
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
            for name in sorted([*names, "MANIFEST.sha256"]):
                z.write(OUT/name, name)
        with ZipFile(archive) as z:
            assert z.testzip() is None
        print(f"Archive {archive} SHA256 {digest(archive)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    build(parser.parse_args().partial)
