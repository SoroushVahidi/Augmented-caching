"""Resumable, repository-controlled runner for the 7-fold evict_value_v1
cross-family fairness pipeline (protocol: configs/reviewer_fairness_cross_family_v1.json).

Replaces the ad-hoc /tmp/run_cross_family_pipeline.sh, whose only failure
mode on 2026-08-06 was that a shell process death (an OOM kill during Stage 2
training for fold=brightkite -- see docs/evict_cross_family_oom_diagnosis.md)
left no record of what had already completed, so a naive restart would have
rebuilt brightkite's already-valid Stage-1 dataset from scratch.

This runner re-derives fold/stage completion by inspecting the actual
filesystem state before doing any work (never trusts a stale flag), and
records what it found/did in a JSON checkpoint file purely for provenance --
resumability itself comes from those live validity checks, so the checkpoint
file can be deleted and regenerated without losing resumability.

Frozen scientific config (unchanged by this runner): the 7 fold assignments,
5 training families, 1 validation family, family weighting via pooling,
feature schema, label definition, model families/hyperparameters, seed=0,
horizon=4, capacities 32/64/128 -- see configs/fair_cross_family_v1/folds/.
The only new behavior this script adds is --memory-bounded / --memory-guard-gb
on Stage 2 (an implementation-only change, see the diagnosis doc) and
resumability itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]

DATA_DIR = REPO_ROOT / "data/derived/evict_value_v1_cross_family_v1"
STAGING_MODELS_DIR = REPO_ROOT / "models/cross_family_v1_staging"
FINAL_MODELS_DIR = REPO_ROOT / "models"
ANALYSIS_DIR = REPO_ROOT / "analysis/reviewer_fairness_cross_family_v1"
FOLDS_CONFIG_DIR = REPO_ROOT / "configs/fair_cross_family_v1/folds"
DEFAULT_STATE_PATH = ANALYSIS_DIR / "build_state.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"folds": {}}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def stage1_valid(fam: str) -> bool:
    """True iff fold `fam`'s dataset manifest exists, is parseable, has at
    least one shard, every shard file it references exists on disk, and the
    held-out family itself never appears among the shard traces (the
    train/test-separation invariant)."""
    manifest_path = DATA_DIR / fam / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    shards = manifest.get("shards", [])
    if not shards:
        return False
    for shard in shards:
        p = Path(shard["path"])
        if not p.exists():
            return False
        if p.name.split("_")[0] == fam:
            return False  # held-out family must never contribute training shards
    return True


def stage2_valid(fam: str) -> bool:
    staging = STAGING_MODELS_DIR / fam
    best_pkl = staging / "evict_value_wulver_v1_best.pkl"
    best_config = ANALYSIS_DIR / fam / "best_config.json"
    comparison_csv = ANALYSIS_DIR / fam / "model_comparison.csv"
    metrics_json = ANALYSIS_DIR / fam / "train_metrics.json"
    if not (best_pkl.exists() and best_config.exists() and comparison_csv.exists() and metrics_json.exists()):
        return False
    try:
        json.loads(best_config.read_text(encoding="utf-8"))
        json.loads(metrics_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return best_pkl.stat().st_size > 0


def stage3_valid(fam: str) -> bool:
    dst = FINAL_MODELS_DIR / f"evict_value_v1_cross_family_v1_{fam}.pkl"
    return dst.exists() and dst.stat().st_size > 0


def _abs_manifest_for(fam: str, data_read_root: Path) -> Path:
    """Materialize an absolute-path trace manifest for fold `fam` from the
    frozen repo-committed relative-path one, resolved against
    `data_read_root` (this worktree's data/processed/ is intentionally empty
    -- see run_distribution_shift_ablation.py for the same convention).
    Written under data/derived/ (repo-relative, not /tmp) so it isn't lost on
    reboot and resumability doesn't depend on ephemeral state."""
    rel_manifest = FOLDS_CONFIG_DIR / f"{fam}_train_manifest.csv"
    lines = rel_manifest.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    out_lines = [header]
    for line in rows:
        if not line.strip():
            continue
        rel_path, rest = line.split(",", 1)
        abs_path = str((data_read_root / rel_path).resolve())
        out_lines.append(f"{abs_path},{rest}")
    out_path = DATA_DIR / "_manifests" / f"{fam}_train_manifest_abs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_path


def run_stage1(fam: str, python_bin: str, data_read_root: Path, dry_run: bool) -> None:
    trace_manifest = _abs_manifest_for(fam, data_read_root)
    split_map_path = FOLDS_CONFIG_DIR / f"{fam}_family_split_map.json"
    out_dir = DATA_DIR / fam
    cmd = [
        python_bin, "-u", "scripts/build_evict_value_dataset_wulver_v1.py",
        "--trace-manifest", str(trace_manifest),
        "--capacities", "32,64,128",
        "--horizons", "4",
        "--split-mode", "family_map",
        "--family-split-map-json", split_map_path.read_text(encoding="utf-8"),
        "--out-dir", str(out_dir),
    ]
    print(f"[stage1] fold={fam} cmd={' '.join(cmd[:3])} ...")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_stage2(fam: str, python_bin: str, memory_bounded: bool, memory_guard_gb: float | None, dry_run: bool) -> None:
    staging = STAGING_MODELS_DIR / fam
    staging.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / fam).mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin, "-u", "scripts/train_evict_value_wulver_v1.py",
        "--manifest", str(DATA_DIR / fam / "manifest.json"),
        "--horizons", "4",
        "--seed", "0",
        "--models-dir", str(staging),
        "--metrics-json", str(ANALYSIS_DIR / fam / "train_metrics.json"),
        "--comparison-csv", str(ANALYSIS_DIR / fam / "model_comparison.csv"),
        "--best-config-json", str(ANALYSIS_DIR / fam / "best_config.json"),
    ]
    if memory_bounded:
        cmd.append("--memory-bounded")
        if memory_guard_gb is not None:
            cmd += ["--memory-guard-gb", str(memory_guard_gb)]
    print(f"[stage2] fold={fam} memory_bounded={memory_bounded} memory_guard_gb={memory_guard_gb}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_stage3(fam: str, dry_run: bool) -> str | None:
    src = STAGING_MODELS_DIR / fam / "evict_value_wulver_v1_best.pkl"
    dst = FINAL_MODELS_DIR / f"evict_value_v1_cross_family_v1_{fam}.pkl"
    print(f"[stage3] fold={fam} {src} -> {dst}")
    if dry_run:
        return None
    dst.write_bytes(src.read_bytes())
    digest = sha256_file(dst)
    print(f"[stage3] fold={fam} sha256={digest}")
    return digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--python-bin", default=str(REPO_ROOT / ".venv_fairness/bin/python"))
    ap.add_argument("--data-read-root", type=Path, default=Path("/home/soroush/Augmented-caching"))
    ap.add_argument("--memory-bounded", dest="memory_bounded", action="store_true", default=True)
    ap.add_argument("--no-memory-bounded", dest="memory_bounded", action="store_false")
    ap.add_argument("--memory-guard-gb", type=float, default=45.0,
                     help="Stage-2 soft memory guard; abort cleanly before the kernel OOM-killer would. "
                          "Set to a negative value or use --no-memory-guard to disable.")
    ap.add_argument("--no-memory-guard", action="store_true")
    ap.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    ap.add_argument("--dry-run", action="store_true", help="Print planned actions without running anything.")
    ap.add_argument("--resume", action="store_true",
                     help="No-op flag kept for CLI-compatibility with the other campaign runners in this repo "
                          "-- this runner always re-derives what's complete from the filesystem, so resume "
                          "behavior is unconditional, not opt-in.")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"Unknown fold family/families (frozen 7-fold rotation): {sorted(unknown)}")

    memory_guard_gb = None if args.no_memory_guard or args.memory_guard_gb < 0 else args.memory_guard_gb

    state = _load_state(args.state_file)
    state.setdefault("folds", {})

    for fam in families:
        print("=" * 42)
        print(f"=== FOLD: test_family={fam} ===")
        print("=" * 42)
        fold_state = state["folds"].setdefault(fam, {})

        if stage1_valid(fam):
            print(f"[skip] fold={fam} stage1 already valid (resume)")
        else:
            run_stage1(fam, args.python_bin, args.data_read_root, args.dry_run)
            if not args.dry_run and not stage1_valid(fam):
                raise RuntimeError(f"fold={fam}: stage1 ran but did not produce a valid manifest")
        if args.dry_run:
            continue
        fold_state["stage1"] = "complete" if stage1_valid(fam) else "incomplete"
        fold_state["stage1_checked_unix"] = time.time()
        _save_state(args.state_file, state)

        if stage2_valid(fam):
            print(f"[skip] fold={fam} stage2 already valid (resume)")
        else:
            run_stage2(fam, args.python_bin, args.memory_bounded, memory_guard_gb, args.dry_run)
            if not stage2_valid(fam):
                raise RuntimeError(f"fold={fam}: stage2 ran but did not produce valid artifacts")
        fold_state["stage2"] = "complete" if stage2_valid(fam) else "incomplete"
        fold_state["stage2_checked_unix"] = time.time()
        _save_state(args.state_file, state)

        if stage3_valid(fam):
            print(f"[skip] fold={fam} stage3 already valid (resume)")
        else:
            digest = run_stage3(fam, args.dry_run)
            if digest is not None:
                fold_state["sha256"] = digest
        fold_state["stage3"] = "complete" if stage3_valid(fam) else "incomplete"
        fold_state["stage3_checked_unix"] = time.time()
        _save_state(args.state_file, state)

        print(f"=== FOLD {fam} COMPLETE ===" if fold_state["stage3"] == "complete" else f"=== FOLD {fam} INCOMPLETE ===")

    if args.dry_run:
        print("=== DRY RUN: no state was written, nothing was executed ===")
        return

    all_done = all(state["folds"].get(f, {}).get("stage3") == "complete" for f in families)
    print("=== ALL FOLDS COMPLETE ===" if all_done else "=== RUN ENDED WITH INCOMPLETE FOLDS (see state file) ===")
    if not all_done:
        sys.exit(1)


if __name__ == "__main__":
    main()
