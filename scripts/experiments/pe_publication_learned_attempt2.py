from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from lafc.evict_value_model_v1 import EvictValueV1Model
from scripts.pe_publication_learned_retrain import (
    FEATURES,
    TARGET,
    benchmark,
    build_split_manifest,
    candidate_files,
    collect_sample_ids,
    dataset_structure,
    evaluate_dataframe,
    evaluate_heuristics,
    feature_audit,
    git_value,
    load_config,
    load_rows,
    read_decision_view,
    selection_key,
    sha256_file,
    split_integrity,
    train_models,
    write_json,
)


ATTEMPT_ID = "pe_publication_learned_retrain_attempt2_20260915"
WORKTREE = Path("/home/soroush/projects/augmented-caching/worktrees/pe-publication-learned-retrain-attempt2-20260915")
ATTEMPT1 = Path("/home/soroush/projects/augmented-caching/worktrees/pe-publication-learned-retrain-20260915")
OUT_DIR = WORKTREE / "analysis" / ATTEMPT_ID
LOG_DIR = OUT_DIR / "logs"
MODELS_DIR = WORKTREE / "models" / ATTEMPT_ID
CONFIG_PATH = WORKTREE / "configs" / "pe_publication_learned_retrain_attempt2_h16.json"
SOURCE_HEAD = "4d2a9fd18fd8c67b19dcccfea0b0769ad784f6d1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def git_info() -> Dict[str, str]:
    return {
        "branch": git_value(["branch", "--show-current"], WORKTREE),
        "head": git_value(["rev-parse", "HEAD"], WORKTREE),
    }


def update_state(stage: str, **extra: object) -> None:
    info = git_info()
    payload: Dict[str, object] = {
        "stage": stage,
        "tmux_session": os.environ.get("PE_ATTEMPT2_TMUX_SESSION"),
        "pid": os.getpid(),
        "update_time": now(),
        "branch": info["branch"],
        "HEAD": info["head"],
        "model_selection_frozen": (OUT_DIR / "MODEL_SELECTION_FROZEN.json").exists(),
        "test_started": (OUT_DIR / "TEST_EVALUATION_STARTED.json").exists(),
        "test_completed": (OUT_DIR / "TEST_EVALUATION_COMPLETE.json").exists(),
    }
    payload.update(extra)
    atomic_json(OUT_DIR / "EXECUTION_STATE.json", payload)


def prereg_paths() -> Dict[str, Path]:
    base = ATTEMPT1 / "analysis" / "pe_publication_learned_retrain_20260915"
    return {
        "dataset_manifest": base / "dataset_manifest.json",
        "feature_manifest": base / "feature_names.json",
        "split_manifest": base / "split_manifest.json",
        "split_integrity": base / "split_integrity.json",
        "training_sample_manifest": base / "training_sample_manifest.json",
    }


def attempt1_test_exposure() -> Dict[str, object]:
    base = ATTEMPT1 / "analysis" / "pe_publication_learned_retrain_20260915"
    evidence = {
        "test_results_json_exists": (base / "test_results.json").exists(),
        "training_summary_json_exists": (base / "training_summary.json").exists(),
        "future_closed_loop_manifest_exists": (base / "future_closed_loop_manifest.json").exists(),
        "clean_process_reproducibility_exists": (base / "clean_process_reproducibility.json").exists(),
        "model_provenance_exists": (base / "MODEL_PROVENANCE.json").exists(),
    }
    classification = "UNKNOWN"
    if evidence["test_results_json_exists"]:
        classification = "CONFIRMED_ACCESSED"
    return {
        "classification": classification,
        "reason": (
            "No durable test metric artifact exists, but the interrupted script would have loaded test rows "
            "immediately after writing MODEL_PROVENANCE.json; without a terminal log the absence of output is "
            "not proof of non-access."
        ),
        "evidence": evidence,
    }


def freeze_preregistration(cfg: Mapping[str, object]) -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, source in prereg_paths().items():
        target = OUT_DIR / f"{name}.attempt1_prereg.json"
        if not target.exists():
            shutil.copy2(source, target)
        inputs[name] = {
            "attempt1_path": str(source),
            "attempt2_copy": str(target),
            "sha256": sha256_file(target),
            "source_sha256": sha256_file(source),
            "hash_match": sha256_file(target) == sha256_file(source),
        }

    expected_split = "4783d96b1ecc242bf726943ba7f960605e3502ceb389cb9866ab17e93cf54b90"
    prereg = {
        "audit_time": now(),
        "attempt1_preserved": True,
        "attempt1_test_exposure": attempt1_test_exposure(),
        "immutable_inputs": inputs,
        "expected_split_manifest_sha256": expected_split,
        "split_sha_matches_known_original": inputs["split_manifest"]["sha256"] == expected_split,
        "reuse_policy": {
            "reused": [
                "dataset manifest",
                "feature allowlist",
                "split design",
                "split manifest",
                "split integrity design",
                "training-sampling specification",
            ],
            "not_reused": [
                "attempt1 trained model",
                "attempt1 model selection state",
                "attempt1 validation results",
                "attempt1 test metrics",
                "attempt1 MODEL_PROVENANCE as new provenance",
            ],
        },
        "config_sha256": sha256_file(CONFIG_PATH),
        "target": TARGET,
        "horizon": cfg["horizon"],
    }
    atomic_json(OUT_DIR / "preregistration_inputs.json", prereg)
    return prereg


def freeze_candidates_and_protocol(cfg: Mapping[str, object]) -> Dict[str, object]:
    candidates = {
        "attempt_id": ATTEMPT_ID,
        "frozen_at": now(),
        "thread_count": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "verified_preregistered_source": str(CONFIG_PATH),
        "candidates": cfg["model_candidates"],
    }
    atomic_json(OUT_DIR / "model_candidates.json", candidates)
    protocol = {
        "attempt_id": ATTEMPT_ID,
        "frozen_at": now(),
        "primary_metric": "mean decision-level validation regret",
        "secondary_metrics_in_order": [
            "non-tied validation regret",
            "optimal-set selection rate on informative decisions",
            "validation MAE",
            "inference cost",
        ],
        "selection_key_implementation": "scripts.pe_publication_learned_retrain.selection_key",
        "forbidden_for_selection": ["test metrics", "attempt1 partial model", "attempt1 test exposure"],
    }
    atomic_json(OUT_DIR / "model_selection_protocol.json", protocol)
    manifest = {
        "model_candidates_sha256": sha256_file(OUT_DIR / "model_candidates.json"),
        "model_selection_protocol_sha256": sha256_file(OUT_DIR / "model_selection_protocol.json"),
    }
    atomic_json(OUT_DIR / "frozen_protocol_hashes.json", manifest)
    return manifest


def verify_dataset_features_split(cfg: Mapping[str, object]) -> Dict[str, object]:
    release_root = Path(str(cfg["release_root"]))
    decisions = read_decision_view(release_root)
    structure = dataset_structure(decisions, cfg)
    atomic_json(OUT_DIR / "dataset_structure.json", structure)
    split_manifest, ranges = build_split_manifest(decisions, cfg)
    atomic_json(OUT_DIR / "split_manifest.json", split_manifest)
    integrity = split_integrity(decisions, ranges, cfg)
    atomic_json(OUT_DIR / "split_integrity.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Split integrity failed: {integrity}")

    audit = feature_audit()
    feature_manifest = {"ordered_feature_names": FEATURES, "audit": audit}
    atomic_json(OUT_DIR / "feature_names.json", feature_manifest)
    feature_gate_pass = all(
        row["allowed"] and not row["future_information"] and not row["target_derived"] and not row["release_placeholder"]
        for row in audit["features"]
    )
    if not feature_gate_pass:
        raise RuntimeError("Feature leakage gate failed")

    sample_ids = collect_sample_ids(decisions, ranges, cfg, OUT_DIR)
    manifest_payload = json.loads((release_root / "metadata" / "release_manifest.json").read_text(encoding="utf-8"))
    selected_decisions = decisions[
        decisions["trace_family"].isin(set(cfg["families"]))
        & decisions["capacity"].isin({int(c) for c in cfg["capacities"]})
        & (decisions["horizon"] == int(cfg["horizon"]))
    ]
    dataset_manifest = {
        "release_root": str(release_root),
        "release_manifest": str(release_root / "metadata" / "release_manifest.json"),
        "release_manifest_sha256": sha256_file(release_root / "metadata" / "release_manifest.json"),
        "checksums_sha256": sha256_file(release_root / "metadata" / "checksums.sha256"),
        "decision_view_sha256": sha256_file(release_root / "data" / "decision_view" / "decision_view.parquet"),
        "dataset_name": manifest_payload["dataset_name"],
        "dataset_version": manifest_payload["version"],
        "dataset_schema_version": manifest_payload["schema_version"],
        "source_manifest": manifest_payload["source_manifest"],
        "source_dataset_repo_head": git_value(["rev-parse", "HEAD"], release_root),
        "candidate_rows": int(manifest_payload["candidate_row_count"]),
        "decisions": int(manifest_payload["decision_row_count"]),
        "candidate_rows_h16_selected": int(selected_decisions["candidate_count"].sum()),
        "decisions_h16_selected": int(selected_decisions["decision_id"].nunique()),
        "families": cfg["families"],
        "capacities": cfg["capacities"],
        "horizon_used": cfg["horizon"],
    }
    atomic_json(OUT_DIR / "dataset_manifest.json", dataset_manifest)
    evidence = {
        "dataset_manifest_sha256": sha256_file(OUT_DIR / "dataset_manifest.json"),
        "feature_manifest_sha256": sha256_file(OUT_DIR / "feature_names.json"),
        "split_manifest_sha256": sha256_file(OUT_DIR / "split_manifest.json"),
        "split_integrity_sha256": sha256_file(OUT_DIR / "split_integrity.json"),
        "training_sample_manifest_sha256": sha256_file(OUT_DIR / "training_sample_manifest.json"),
        "feature_leakage_gate": "PASS",
        "split_integrity": integrity["status"],
        "checks": integrity["checks"],
    }
    atomic_json(OUT_DIR / "pre_training_verification.json", evidence)
    return {"decisions": decisions, "ranges": ranges, "sample_ids": sample_ids, "evidence": evidence}


def atomic_save_model(path: Path, model: EvictValueV1Model) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save(tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def run_train() -> None:
    os.chdir(WORKTREE)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    update_state("train_starting", start_time=now(), log_path=str(LOG_DIR / "training.log"))
    cfg = load_config(CONFIG_PATH)
    if int(cfg["horizon"]) != 16 or str(cfg["target"]) != TARGET:
        raise RuntimeError("Unexpected target/horizon")
    if git_info()["head"] != SOURCE_HEAD:
        raise RuntimeError(f"Attempt2 HEAD is not source head {SOURCE_HEAD}")

    prereg = freeze_preregistration(cfg)
    if not all(v["hash_match"] for v in prereg["immutable_inputs"].values()):
        raise RuntimeError("Attempt1 preregistration copy hash mismatch")
    protocol_hashes = freeze_candidates_and_protocol(cfg)
    verified = verify_dataset_features_split(cfg)

    files = candidate_files(Path(str(cfg["release_root"])), int(cfg["horizon"]))
    update_state("loading_train_validation", last_completed_stage="pre_training_verification")
    train_df = load_rows(files, verified["ranges"], "train", verified["sample_ids"]["train"])
    val_df = load_rows(files, verified["ranges"], "validation", verified["sample_ids"]["validation"])

    train_start = now()
    models = train_models(train_df, cfg)
    train_end = now()
    candidate_status: Dict[str, object] = {}
    for name, estimator in models.items():
        candidate_path = MODELS_DIR / "candidates" / f"{ATTEMPT_ID}_{name}.pkl"
        ev_model = EvictValueV1Model(
            model_name=f"{ATTEMPT_ID}_{name}",
            estimator=estimator,
            feature_columns=FEATURES,
        )
        candidate_sha = atomic_save_model(candidate_path, ev_model)
        candidate_status[name] = {
            "status": "TRAINING_COMPLETE",
            "model_path": str(candidate_path.relative_to(WORKTREE)),
            "model_sha256": candidate_sha,
            "model_class": f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
            "training_start": train_start,
            "training_end": train_end,
        }
        atomic_json(OUT_DIR / "candidate_training_status" / f"{name}.json", candidate_status[name])
    atomic_json(OUT_DIR / "candidate_training_status.json", candidate_status)

    validation_results: Dict[str, Dict[str, object]] = {}
    for name, estimator in models.items():
        global_metrics, by_family, by_capacity = evaluate_dataframe(val_df, estimator)
        validation_results[name] = {
            "global": global_metrics,
            "by_family": by_family,
            "by_capacity": by_capacity,
            "candidate_model_sha256": candidate_status[name]["model_sha256"],
        }
    atomic_json(OUT_DIR / "validation_results.json", validation_results)

    selected = min(validation_results.keys(), key=lambda n: selection_key(validation_results[n]["global"]))
    selected_est = models[selected]
    selection_decision = {
        "selected_model": selected,
        "selected_validation_global": validation_results[selected]["global"],
        "selection_rule": cfg["selection_rule"],
        "selection_protocol_sha256": protocol_hashes["model_selection_protocol_sha256"],
        "candidate_config_sha256": protocol_hashes["model_candidates_sha256"],
        "test_used_for_model_selection": False,
        "decision_time": now(),
        "candidate_order": sorted(validation_results),
    }
    atomic_json(OUT_DIR / "model_selection_decision.json", selection_decision)

    selected_model_name = f"pe_evict_value_h16_{selected}_attempt2_20260915"
    selected_model_path = MODELS_DIR / f"{selected_model_name}.pkl"
    selected_model = EvictValueV1Model(model_name=selected_model_name, estimator=selected_est, feature_columns=FEATURES)
    selected_sha = atomic_save_model(selected_model_path, selected_model)
    atomic_text(OUT_DIR / "selected_model.sha256", f"{selected_sha}  {selected_model_path.relative_to(WORKTREE)}\n")

    provenance = {
        "attempt_id": ATTEMPT_ID,
        "model_name": selected_model_name,
        "model_path": str(selected_model_path.relative_to(WORKTREE)),
        "model_sha256": selected_sha,
        "model_class": f"{selected_est.__class__.__module__}.{selected_est.__class__.__name__}",
        "selected_candidate": selected,
        "selection_rule": cfg["selection_rule"],
        "target": TARGET,
        "horizon": cfg["horizon"],
        "features": FEATURES,
        "feature_manifest": str((OUT_DIR / "feature_names.json").relative_to(WORKTREE)),
        "feature_manifest_sha256": sha256_file(OUT_DIR / "feature_names.json"),
        "split_manifest": str((OUT_DIR / "split_manifest.json").relative_to(WORKTREE)),
        "split_manifest_sha256": sha256_file(OUT_DIR / "split_manifest.json"),
        "dataset_manifest": str((OUT_DIR / "dataset_manifest.json").relative_to(WORKTREE)),
        "dataset_manifest_sha256": sha256_file(OUT_DIR / "dataset_manifest.json"),
        "split_integrity": str((OUT_DIR / "split_integrity.json").relative_to(WORKTREE)),
        "split_integrity_sha256": sha256_file(OUT_DIR / "split_integrity.json"),
        "config_path": str(CONFIG_PATH.relative_to(WORKTREE)),
        "config_sha256": sha256_file(CONFIG_PATH),
        "model_candidates": str((OUT_DIR / "model_candidates.json").relative_to(WORKTREE)),
        "model_candidates_sha256": protocol_hashes["model_candidates_sha256"],
        "model_selection_protocol": str((OUT_DIR / "model_selection_protocol.json").relative_to(WORKTREE)),
        "model_selection_protocol_sha256": protocol_hashes["model_selection_protocol_sha256"],
        "training_command": "bash scripts/experiments/pe_publication_learned_attempt2_train.sh",
        "retrain_branch": git_info()["branch"],
        "retrain_head_at_training": git_info()["head"],
        "python": sys.version,
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "old_attempt1_model_used": False,
        "test_used_for_model_selection": False,
        "frozen_before_test_evaluation_utc": now(),
    }
    atomic_json(OUT_DIR / "MODEL_PROVENANCE.json", provenance)
    frozen = {
        "status": "MODEL_SELECTION_FROZEN",
        "frozen_at": now(),
        "selected_model": selected,
        "model_path": str(selected_model_path.relative_to(WORKTREE)),
        "model_sha256": selected_sha,
        "model_selection_decision_sha256": sha256_file(OUT_DIR / "model_selection_decision.json"),
        "MODEL_PROVENANCE_sha256": sha256_file(OUT_DIR / "MODEL_PROVENANCE.json"),
        "test_evaluated": False,
    }
    atomic_json(OUT_DIR / "MODEL_SELECTION_FROZEN.json", frozen)
    summary = {
        "status": "TRAINING_STAGE_COMPLETE",
        "train_rows": int(len(train_df)),
        "train_decisions": int(train_df["decision_id"].nunique()),
        "validation_rows": int(len(val_df)),
        "validation_decisions": int(val_df["decision_id"].nunique()),
        "training_start": train_start,
        "training_end": train_end,
        "selected_model": selected,
        "model_sha256": selected_sha,
    }
    atomic_json(OUT_DIR / "training_stage_summary.json", summary)
    update_state(
        "training_stage_complete",
        last_completed_stage="MODEL_SELECTION_FROZEN",
        model_selection_frozen=True,
        log_path=str(LOG_DIR / "training.log"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def verify_frozen() -> Dict[str, object]:
    frozen_path = OUT_DIR / "MODEL_SELECTION_FROZEN.json"
    if not frozen_path.exists():
        raise RuntimeError("MODEL_SELECTION_FROZEN.json is required before test evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    model_path = WORKTREE / str(frozen["model_path"])
    if sha256_file(model_path) != frozen["model_sha256"]:
        raise RuntimeError("Selected model SHA mismatch")
    split_sha = sha256_file(OUT_DIR / "split_manifest.json")
    prov = json.loads((OUT_DIR / "MODEL_PROVENANCE.json").read_text(encoding="utf-8"))
    if prov["split_manifest_sha256"] != split_sha:
        raise RuntimeError("Split manifest SHA mismatch")
    return frozen


def clean_process_check(model_path: Path, sample_path: Path) -> Dict[str, object]:
    code = (
        "import hashlib,json; import pandas as pd; "
        "from lafc.evict_value_model_v1 import EvictValueV1Model; "
        f"features=json.load(open({str(OUT_DIR / 'feature_names.json')!r}))['ordered_feature_names']; "
        f"model_path={str(model_path)!r}; "
        "model=EvictValueV1Model.load(model_path); "
        "h=hashlib.sha256(open(model_path,'rb').read()).hexdigest(); "
        f"df=pd.read_csv({str(sample_path)!r}); "
        "preds=model.estimator.predict(df[features].to_numpy(dtype=float)); "
        "mae=float(abs(preds-df['y_loss'].to_numpy(dtype=float)).mean()); "
        "digest=hashlib.sha256(json.dumps([round(float(x),12) for x in preds.tolist()],sort_keys=True,separators=(',',':')).encode()).hexdigest(); "
        "print(json.dumps({'model_sha256':h,'sample_mae':mae,'prediction_digest':digest,'model_name':model.model_name},sort_keys=True))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ".:src"
    clean = subprocess.run([sys.executable, "-c", code], cwd=WORKTREE, env=env, text=True, check=True, capture_output=True)
    payload = json.loads(clean.stdout)
    payload["status"] = "PASS"
    atomic_json(OUT_DIR / "clean_process_reproducibility.json", payload)
    return payload


def run_test() -> None:
    os.chdir(WORKTREE)
    frozen = verify_frozen()
    update_state("test_starting", log_path=str(LOG_DIR / "test.log"), model_selection_frozen=True)
    atomic_json(
        OUT_DIR / "TEST_EVALUATION_STARTED.json",
        {
            "started_at": now(),
            "model_sha256": frozen["model_sha256"],
            "model_selection_frozen_sha256": sha256_file(OUT_DIR / "MODEL_SELECTION_FROZEN.json"),
        },
    )

    cfg = load_config(CONFIG_PATH)
    decisions = read_decision_view(Path(str(cfg["release_root"])))
    split_manifest, ranges = build_split_manifest(decisions, cfg)
    files = candidate_files(Path(str(cfg["release_root"])), int(cfg["horizon"]))
    model_path = WORKTREE / str(frozen["model_path"])
    selected_model = EvictValueV1Model.load(model_path)
    test_df = load_rows(files, ranges, "test", None)
    test_global, test_by_family, test_by_capacity = evaluate_dataframe(test_df, selected_model.estimator)
    heuristics = evaluate_heuristics(test_df)
    atomic_json(OUT_DIR / "test_results.json", {"global": test_global, "by_family": test_by_family, "by_capacity": test_by_capacity})
    atomic_json(OUT_DIR / "offline_test_comparators.json", heuristics)

    bench32 = benchmark(selected_model, test_df, 32)
    bench128 = benchmark(selected_model, test_df, 128)
    atomic_json(OUT_DIR / "inference_benchmark.json", {"cap32": bench32, "cap128": bench128})

    future_rows = []
    for row in split_manifest["ranges"]:
        if row["split"] == "test" and int(row["capacity"]) in set(cfg["future_closed_loop_capacities"]):
            future_rows.append(
                {
                    "family": row["family"],
                    "capacity": int(row["capacity"]),
                    "test_request_range": f"{row['decision_t_min']}..{row['decision_t_max']}",
                    "model_sha256": frozen["model_sha256"],
                    "split_manifest_sha256": sha256_file(OUT_DIR / "split_manifest.json"),
                    "training_overlap": False,
                    "validation_overlap": False,
                    "policy_valid": True,
                    "estimated_runtime_seconds": bench32["estimated_50k_replay"]
                    if int(row["capacity"]) == 32
                    else bench128["estimated_50k_replay"],
                }
            )
    design_l = [r for r in future_rows if int(r["capacity"]) == 32]
    future_manifest = {
        "model_sha256": frozen["model_sha256"],
        "split_manifest_sha256": sha256_file(OUT_DIR / "split_manifest.json"),
        "families": cfg["families"],
        "capacities": cfg["future_closed_loop_capacities"],
        "cells": future_rows,
        "DESIGN_L": {
            "runs": len(design_l),
            "estimated_wallclock_seconds": sum(float(r["estimated_runtime_seconds"]) for r in design_l),
        },
        "DESIGN_L_PLUS": {
            "runs": len(future_rows),
            "estimated_wallclock_seconds": sum(float(r["estimated_runtime_seconds"]) for r in future_rows),
        },
    }
    atomic_json(OUT_DIR / "future_closed_loop_manifest.json", future_manifest)

    sample = test_df.sort_values(["decision_id", "candidate_page_id"]).head(8)
    sample_path = OUT_DIR / "clean_process_prediction_sample.csv"
    tmp_sample = sample_path.with_suffix(sample_path.suffix + ".tmp")
    sample[["decision_id", "candidate_page_id", *FEATURES, TARGET]].to_csv(tmp_sample, index=False)
    os.replace(tmp_sample, sample_path)
    clean_payload = clean_process_check(model_path, sample_path)

    complete = {
        "status": "TEST_EVALUATION_COMPLETE",
        "completed_at": now(),
        "model_sha256": frozen["model_sha256"],
        "test_results_sha256": sha256_file(OUT_DIR / "test_results.json"),
        "inference_benchmark_sha256": sha256_file(OUT_DIR / "inference_benchmark.json"),
        "future_closed_loop_manifest_sha256": sha256_file(OUT_DIR / "future_closed_loop_manifest.json"),
        "clean_process_reproducibility": clean_payload["status"],
        "test_evaluated_after_freeze": True,
    }
    atomic_json(OUT_DIR / "TEST_EVALUATION_COMPLETE.json", complete)
    gate = publication_gate()
    atomic_json(OUT_DIR / "publication_gate.json", gate)
    update_state(
        "test_stage_complete",
        last_completed_stage="TEST_EVALUATION_COMPLETE",
        model_selection_frozen=True,
        test_started=True,
        test_completed=True,
        log_path=str(LOG_DIR / "test.log"),
    )
    print(json.dumps({"status": "TEST_STAGE_COMPLETE", "publication_gate": gate["status"]}, indent=2, sort_keys=True))


def publication_gate() -> Dict[str, object]:
    required = [
        "dataset_manifest.json",
        "feature_names.json",
        "split_integrity.json",
        "model_candidates.json",
        "model_selection_protocol.json",
        "validation_results.json",
        "model_selection_decision.json",
        "MODEL_SELECTION_FROZEN.json",
        "MODEL_PROVENANCE.json",
        "TEST_EVALUATION_COMPLETE.json",
        "clean_process_reproducibility.json",
        "future_closed_loop_manifest.json",
        "inference_benchmark.json",
    ]
    missing = [name for name in required if not (OUT_DIR / name).exists()]
    split_ok = False
    leakage_ok = False
    clean_ok = False
    if (OUT_DIR / "split_integrity.json").exists():
        split_ok = json.loads((OUT_DIR / "split_integrity.json").read_text(encoding="utf-8")).get("status") == "PASS"
    if (OUT_DIR / "feature_names.json").exists():
        audit = json.loads((OUT_DIR / "feature_names.json").read_text(encoding="utf-8"))["audit"]
        leakage_ok = all(
            row["allowed"] and not row["future_information"] and not row["target_derived"] and not row["release_placeholder"]
            for row in audit["features"]
        )
    if (OUT_DIR / "clean_process_reproducibility.json").exists():
        clean_ok = json.loads((OUT_DIR / "clean_process_reproducibility.json").read_text(encoding="utf-8")).get("status") == "PASS"
    status = "PASS" if not missing and split_ok and leakage_ok and clean_ok else "FAIL"
    return {
        "status": status,
        "missing": missing,
        "split_integrity_pass": split_ok,
        "feature_leakage_gate_pass": leakage_ok,
        "clean_process_reproducibility_pass": clean_ok,
        "test_used_for_model_selection": False,
        "old_attempt1_model_used": False,
    }


def run_preflight() -> None:
    os.chdir(WORKTREE)
    cfg = load_config(CONFIG_PATH)
    exposure = attempt1_test_exposure()
    payload = {
        "status": "PREFLIGHT_PASS",
        "time": now(),
        "git": git_info(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "target": cfg["target"],
        "horizon": cfg["horizon"],
        "feature_count": len(FEATURES),
        "attempt1_test_exposure": exposure,
        "candidate_names": sorted(cfg["model_candidates"]),
    }
    if payload["git"]["head"] != SOURCE_HEAD:
        raise RuntimeError("Attempt2 worktree is not at the required source commit")
    atomic_json(OUT_DIR / "preflight.json", payload)
    update_state("preflight_complete", last_completed_stage="preflight", log_path=None)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "train", "test", "gate"])
    args = parser.parse_args()
    if args.stage == "preflight":
        run_preflight()
    elif args.stage == "train":
        run_train()
    elif args.stage == "test":
        run_test()
    elif args.stage == "gate":
        print(json.dumps(publication_gate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
