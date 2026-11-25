"""Freeze the measured resource decision and static experiment metadata."""
import json
import os
from pathlib import Path
from shutil import copyfile
import sys

import joblib
import numba
import numpy as np
import sklearn

from assets import FEATURE_NAMES, EXPECTED_CA, EXPECTED_MANIFEST, dump, sha
from learning import GRID,EPSILONS,RISK_GRID
from run import ALL_BUDGETS,ARCHITECTURES,RANDOM_SEEDS,UNIFORM_SEEDS


def run(out,source,branch):
    smoke=json.loads((out/"smoke/receipts/A-brightkite.json").read_text())
    alignment=json.loads((out/"smoke/evidence/OPTIMIZED_INFERENCE_VALIDATION.json").read_text())
    verified=json.loads((out/"smoke/evidence/DATASET_VERIFICATION.json").read_text())
    assert smoke["status"]=="COMPLETED" and alignment["all_checks_match"] and verified["status"]=="VERIFIED"
    assert verified["phase2ca_manifest_sha256"]==EXPECTED_MANIFEST
    copyfile(out/"smoke/evidence/DATASET_VERIFICATION.json",out/"evidence/DATASET_VERIFICATION.json")
    replays=smoke["replays"]
    payload=dict(source=source,implementation_branch=branch,source_archive_sha256=sha(out/"inputs/production-source.tar"),
        phase2ca_source=EXPECTED_CA,phase2ca_manifest=EXPECTED_MANIFEST,
        archived_phase2b_inputs_sha256=sha(out/"inputs/phase2b-inputs.tar"),
        archived_phase2b_baseline_sha256=sha(out/"inputs/CLOSED_LOOP_RESULTS.csv"),
        environment=dict(python=sys.version.split()[0],numpy=np.__version__,numba=numba.__version__,
                         sklearn=sklearn.__version__,joblib=joblib.__version__,lightgbm_available=False),
        planned=dict(model_components=231,validation_and_outer_replays=3234,additional_reference_parity_replays=14,
                     capacities=[32,64,128],budget_groups=ALL_BUDGETS),
        configurations=dict(architectures=ARCHITECTURES,feature_names=FEATURE_NAMES,gate_grid=GRID,
                            approximate_tie_grid=EPSILONS,risk_grid=RISK_GRID,random_control_seeds=RANDOM_SEEDS,
                            uniform_shortlist_seeds=UNIFORM_SEEDS),
        smoke=dict(status=smoke["status"],actual_train_groups=[m["train_groups"] for m in smoke["fit"]],
                   fits=[dict(spec=m["spec"],cpu_seconds=m["training_cpu_seconds"],wall_seconds=m["training_wall_seconds"])
                         for m in smoke["fit"]],replays=len(replays),
                   min_cpu_us_per_request=min(r["cpu_us_per_request"] for r in replays),
                   max_cpu_us_per_request=max(r["cpu_us_per_request"] for r in replays),
                   fit_replay_peak_rss_kib=smoke["resource"]["rss_kib"],
                   shortened_closed_loop_scored_requests=replays[0]["requests"],
                   chronological_optimized_reference=[dict(model=r["model"],
                     optimized_us=r["optimized"]["cpu_us_per_request"],reference_us=r["reference"]["cpu_us_per_request"])
                       for r in alignment["chronological_parity"]]))
    dump(out/"evidence/MODEL_CONFIGURATIONS.json",payload)


if __name__=="__main__":
    run(Path(sys.argv[1]),*sys.argv[2:])
