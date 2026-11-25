"""Record exact deployment inputs and measured pilot resource estimates."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import joblib
import numba
import numpy as np

from campaign import (ALPHAS, BASE, BUDGETS, CAPS, DIAGNOSTIC, FEATURE_GROUPS, FIXED, KS, MECHANISMS,
                      PROTOCOL, ROOT, SCHEMA, SEED, UNIFORM, dump, horizon_keys, sha)


def run(out, source, branch, tar, trace_root):
    outcomes = json.loads((out/"LABELER_PLATFORM_PARITY.json").read_text())
    if not outcomes["zero_unexplained_mismatches"]:
        raise ValueError("Platform parity did not pass")
    pilot=json.loads((out/"PILOT_RESOURCES.json").read_text())
    primary = [r for r in pilot if r["capacity"] in CAPS]
    plan=json.loads((out/"PLAN.json").read_text())
    elements = set((r["family"],r["capacity"]) for r in plan)
    assert elements == {(f,c) for f in PROTOCOL for c in CAPS+DIAGNOSTIC}
    env=Path(os.environ["PHASE2CA_RUNTIME"])
    paths=("experiments/cache_supervision_phase2b/exact_lru.py",
           "experiments/cache_supervision_phase2ca/campaign.py",
           "experiments/cache_supervision_phase2ca/reduce.py",
           "experiments/neural_eviction_20261121/data.py",
           "experiments/recurrent_neural_eviction/features.py",
           "experiments/neural_eviction_20261121/protocol.json")
    n=sum(x["groups"] for x in primary)
    p=[dict(family=r["family"], capacity=r["capacity"], groups=r["groups"],
            wall_seconds=r["wall_seconds"], cpu_seconds=r["cpu_seconds"], feature_seconds=r["feature_seconds"],
            label_seconds=r["label_seconds"], peak_rss_kib=r["peak_rss_kib"],
            compressed_bytes=sum(p["bytes"] for p in r["partitions"]),
            raw_bytes=sum(p["raw_bytes"] for p in r["partitions"])) for r in pilot]
    source_repo=trace_root.parents[1]
    revisions={}
    for repo in (source_repo, ROOT):
        revisions[str(repo)] = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    payload=dict(base_phase2b_revision=BASE, production_revision=source, implementation_branch=branch,
                 source_archive_sha256=sha(tar), immutable_source_sha256={name:sha(ROOT/name) for name in paths},
                 pilot_source_archive_sha256=sha(out/"inputs/pilot-source.tar"),
                 private_alignment_fixture_sha256=sha(out/"inputs/alignment.npz"),
                 trace_source_repository_revisions=revisions, trace_root=str(trace_root), runtime=str(env),
                 environment=dict(python=platform.python_version(),numpy=np.__version__,numba=numba.__version__,
                                  joblib=joblib.__version__,joblib_overlay=joblib.__file__,
                                  conditions="single-threaded numerical libraries; zero accelerators"),
                 folds={f:{k:m[k] for k in ("trace","sha256","train","validation")} for f,m in PROTOCOL.items()},
                 feature_schema=SCHEMA, groups=FEATURE_GROUPS, seed=SEED, uniform_seeds=UNIFORM, shortlists=MECHANISMS,
                 shortlist_k=KS, primary_capacities=CAPS, diagnostic_capacities=DIAGNOSTIC,
                 budgets=BUDGETS, fixed_horizons=FIXED, scaled_horizon_alphas=ALPHAS,
                 horizon_slots={str(c):horizon_keys(c)[0] for c in CAPS+DIAGNOSTIC},
                 census_primary_full_miss_groups=sum(r["eligible"] for r in plan if not r["diagnostic"]),
                 diagnostic_groups=sum(r["quotas"][-1] for r in plan if r["diagnostic"]),
                 pilot=dict(cells=p,groups=n,
                            primary_label_groups_per_second=n/sum(r["label_seconds"] for r in primary),
                            primary_feature_groups_per_second=n/sum(r["feature_seconds"] for r in primary),
                            primary_end_to_end_groups_per_second=n/sum(r["wall_seconds"] for r in primary),
                            peak_rss_kib=max(r["peak_rss_kib"] for r in pilot)),
                 allocation_basis="Measured pilot used one core. Producer requests 2 GiB, reducer 4 GiB. "
                                  "Provision 2 GiB of compressed output storage including a margin. "
                                  "Raw uncompressed all-candidate array estimate is 4.8 GiB.")
    dump(out/"evidence"/"SOURCE_PROVENANCE.json",payload)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("source")
    p.add_argument("branch")
    p.add_argument("archive",type=Path)
    p.add_argument("trace_root",type=Path)
    a=p.parse_args()
    run(a.out,a.source,a.branch,a.archive,a.trace_root)
