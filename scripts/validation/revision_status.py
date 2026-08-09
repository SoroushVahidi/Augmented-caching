"""Read-only status summary for the four KBS second-revision reviewer concerns.

This tool prefers the current worktree as the source of truth and falls back to
known sibling worktrees only when a concern's artifacts are absent locally.
That keeps the canonical branch usable after consolidation while still making
the pre-consolidation multi-worktree layout inspectable.

Never loads a model, never materializes a large CSV into memory, and never
launches or modifies experiments.

Usage:
    python scripts/validation/revision_status.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

FAMILIES = [
    "brightkite",
    "citibike",
    "cloudphysics",
    "metacdn",
    "metakv",
    "twemcache",
    "wiki2018",
]
OBJECTIVES = [
    "objective_eviction_loss",
    "objective_next_arrival",
    "objective_reuse_distance",
    "objective_pairwise",
]
CONDITIONS = ["OFF_POLICY_LRU", "DAGGER_ITER1"]
CAPACITIES = [32, 64, 128]

C4_BLOCKING_PATTERNS = [
    "run_evict_cross_family_pipeline.py",
    "train_evict_value_wulver_v1.py",
    "build_evict_value_dataset_wulver_v1.py",
    "build_supervision_objective_ablation_dataset.py",
    "train_supervision_objective_ablation.py",
    "run_supervision_objective_ablation.py",
    "run_distribution_shift_ablation.py",
    "run_lrb_external_baseline.py",
]


def find_worktrees(repo_root: Path) -> Dict[str, Path]:
    """Return {branch_name: worktree_path} via `git worktree list --porcelain`."""
    out = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    worktrees: Dict[str, Path] = {}
    current_path: Optional[Path] = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            current_path = Path(line[len("worktree ") :])
        elif line.startswith("branch ") and current_path is not None:
            branch = line[len("branch ") :].removeprefix("refs/heads/")
            worktrees[branch] = current_path
    return worktrees


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def _tmux_sessions() -> List[str]:
    result = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.split(":")[0] for line in result.stdout.splitlines() if line.strip()]


def _c4_gate_from_process_list() -> str:
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if "grep" in line or "revision_status.py" in line or "revision_readiness.py" in line:
            continue
        if any(pattern in line for pattern in C4_BLOCKING_PATTERNS):
            return "DEFER"
    return "READY"


def _pick_root(
    repo_root: Path,
    worktrees: Dict[str, Path],
    fallback_branch: str,
    required_paths: Iterable[str],
) -> Optional[Path]:
    """Prefer the candidate with the richest local evidence, tie-breaking to repo_root."""
    candidates: List[Path] = [repo_root]
    fallback = worktrees.get(fallback_branch)
    if fallback is not None and fallback != repo_root:
        candidates.append(fallback)
    required = [Path(p) for p in required_paths]
    best_candidate: Optional[Path] = None
    best_score = -1
    for candidate in candidates:
        score = sum(1 for rel in required if (candidate / rel).exists())
        if score > best_score:
            best_candidate = candidate
            best_score = score
    return best_candidate if best_score > 0 else None


def concern_1_status(root: Path) -> Dict[str, object]:
    data_root = root / "data" / "derived" / "evict_value_v1_cross_family_v1"
    models_root = root / "models"
    datasets_done = sum(1 for family in FAMILIES if (data_root / family / "manifest.json").exists())
    models_done = sum(
        1 for family in FAMILIES if (models_root / f"evict_value_v1_cross_family_v1_{family}.pkl").exists()
    )

    registry_path = root / "analysis" / "reviewer_fairness_cross_family_v1" / "model_registry.json"
    registry_frozen = False
    if registry_path.exists():
        registry_frozen = bool(json.loads(registry_path.read_text(encoding="utf-8")).get("MODEL_SELECTION_FROZEN"))

    eval_csv = root / "analysis" / "reviewer_fairness_cross_family_v1" / "evict_value_v1" / "policy_comparison.csv"
    eval_rows = _count_csv_rows(eval_csv)

    return {
        "root": str(root),
        "datasets_done": datasets_done,
        "datasets_total": len(FAMILIES),
        "models_done": models_done,
        "models_total": len(FAMILIES),
        "registry_frozen": registry_frozen,
        "eval_rows_total": eval_rows,
        "eval_rows_primary_expected": len(FAMILIES) * len(CAPACITIES),
        "status": (
            "TRAINING COMPLETE — READY FOR NEXT STAGE"
            if models_done == len(FAMILIES) and not registry_frozen
            else "TRAINING COMPLETE — EVALUATION PENDING"
            if models_done == len(FAMILIES) and registry_frozen and eval_rows == 0
            else "COMPLETE"
            if eval_rows >= len(FAMILIES) * len(CAPACITIES) * 2
            else "EVALUATION RUNNING"
            if registry_frozen and eval_rows > 0
            else "RUNNING"
            if models_done > 0 or datasets_done > 0
            else "NOT_STARTED"
        ),
    }


def concern_2_status(root: Path) -> Dict[str, object]:
    data_root = root / "data" / "derived" / "supervision_objective_ablation_v1"
    models_root = root / "models" / "supervision_objective_ablation_v1"
    datasets_done = sum(1 for family in FAMILIES if (data_root / family / "manifest.json").exists())
    models_done = sum(1 for obj in OBJECTIVES for family in FAMILIES if (models_root / obj / f"{family}.pkl").exists())

    registry_path = root / "analysis" / "supervision_objective_ablation_v1" / "model_registry.json"
    registry_frozen = False
    if registry_path.exists():
        registry_frozen = bool(json.loads(registry_path.read_text(encoding="utf-8")).get("MODEL_SELECTION_FROZEN"))

    eval_csv = root / "analysis" / "supervision_objective_ablation_v1" / "policy_comparison.csv"
    eval_rows = _count_csv_rows(eval_csv)

    same_example_path = root / "analysis" / "supervision_objective_ablation_v1" / "same_example_audit.json"
    fairness_audit_path = root / "analysis" / "supervision_objective_ablation_v1" / "fairness_audit.json"

    expected_models = len(OBJECTIVES) * len(FAMILIES)
    expected_eval = len(OBJECTIVES) * len(FAMILIES) * len(CAPACITIES)
    return {
        "root": str(root),
        "datasets_done": datasets_done,
        "datasets_total": len(FAMILIES),
        "models_done": models_done,
        "models_total": expected_models,
        "registry_frozen": registry_frozen,
        "eval_rows": eval_rows,
        "eval_rows_expected": expected_eval,
        "same_example_audit_exists": same_example_path.exists(),
        "fairness_audit_exists": fairness_audit_path.exists(),
        "status": (
            "COMPLETE"
            if eval_rows >= expected_eval
            else "TRAINING COMPLETE — EVALUATION PENDING"
            if models_done == expected_models and not registry_frozen
            else "EVALUATION RUNNING"
            if registry_frozen and eval_rows > 0
            else "RUNNING — TRAINING"
            if models_done > 0 or datasets_done > 0
            else "NOT_STARTED"
        ),
    }


def concern_3_status(root: Path) -> Dict[str, object]:
    out_dir = root / "analysis" / "distribution_shift_ablation_v1"
    policy_csv = out_dir / "policy_comparison.csv"
    rows = _count_csv_rows(policy_csv)
    state_path = out_dir / "campaign_state.json"
    completed_folds = json.loads(state_path.read_text(encoding="utf-8")).get("completed_folds", []) if state_path.exists() else []
    expected_rows = len(FAMILIES) * len(CAPACITIES) * len(CONDITIONS)
    return {
        "root": str(root),
        "primary_rows": rows,
        "primary_rows_expected": expected_rows,
        "folds_complete": len(completed_folds),
        "folds_total": len(FAMILIES),
        "status": (
            "COMPLETE"
            if rows >= expected_rows
            else "STOPPED_CLEANLY_PARTIAL"
            if rows > 0
            else "NOT_STARTED"
        ),
    }


def concern_4_status(root: Path) -> Dict[str, object]:
    smoke_dir = root / "analysis" / "practical_significance_ablation_v1"
    smoke_artifacts = sorted(p.name for p in smoke_dir.glob("*.csv")) if smoke_dir.exists() else []
    controlled_marker = smoke_dir / "controlled_final" / "profiler_breakdown.csv"
    return {
        "root": str(root),
        "smoke_artifacts": smoke_artifacts,
        "controlled_campaign_started": controlled_marker.exists(),
        "timing_gate": _c4_gate_from_process_list(),
        "status": (
            "SMOKE_COMPLETE_CONTROLLED_PENDING"
            if smoke_artifacts and not controlled_marker.exists()
            else "CONTROLLED_COMPLETE"
            if controlled_marker.exists()
            else "NOT_STARTED"
        ),
    }


def collect() -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    worktrees = find_worktrees(repo_root)

    concern_1_root = _pick_root(
        repo_root,
        worktrees,
        "feat/reviewer-fairness-protocol",
        [
            "analysis/reviewer_fairness_cross_family_v1/model_registry.json",
            "data/derived/evict_value_v1_cross_family_v1",
            "models/evict_value_v1_cross_family_v1_brightkite.pkl",
            "scripts/experiments/run_cross_family_heldout_eval.py",
        ],
    )
    concern_2_root = _pick_root(
        repo_root,
        worktrees,
        "feat/supervision-objective-ablation",
        [
            "analysis/supervision_objective_ablation_v1/model_registry.json",
            "data/derived/supervision_objective_ablation_v1",
            "models/supervision_objective_ablation_v1/objective_eviction_loss/brightkite.pkl",
            "scripts/experiments/run_supervision_objective_ablation.py",
        ],
    )
    concern_3_root = _pick_root(
        repo_root,
        worktrees,
        "feat/reviewer-fairness-protocol",
        [
            "analysis/distribution_shift_ablation_v1/policy_comparison.csv",
            "analysis/distribution_shift_ablation_v1/campaign_state.json",
            "scripts/experiments/run_distribution_shift_ablation.py",
        ],
    )
    concern_4_root = _pick_root(
        repo_root,
        worktrees,
        "feat/reviewer-fairness-protocol",
        [
            "analysis/practical_significance_ablation_v1/provenance.json",
            "analysis/practical_significance_ablation_v1/exact_optimization_equivalence.json",
            "scripts/experiments/run_practical_significance_ablation.py",
        ],
    )

    report: Dict[str, object] = {
        "repo_root": str(repo_root),
        "worktrees": {k: str(v) for k, v in worktrees.items()},
        "active_tmux_sessions": _tmux_sessions(),
    }
    report["concern_1"] = concern_1_status(concern_1_root) if concern_1_root else {"status": "WORKTREE_NOT_FOUND"}
    report["concern_2"] = concern_2_status(concern_2_root) if concern_2_root else {"status": "WORKTREE_NOT_FOUND"}
    report["concern_3"] = concern_3_status(concern_3_root) if concern_3_root else {"status": "WORKTREE_NOT_FOUND"}
    report["concern_4"] = concern_4_status(concern_4_root) if concern_4_root else {"status": "WORKTREE_NOT_FOUND"}
    return report


def format_report(report: Dict[str, object]) -> str:
    lines = []
    lines.append(f"Repository root: {report['repo_root']}")
    lines.append(f"Active tmux sessions: {report['active_tmux_sessions'] or '(none)'}")
    lines.append("")

    c1 = report["concern_1"]
    lines.append("Concern 1 (cross-family retraining):")
    lines.append(f"    root {c1.get('root', '(unknown)')}")
    lines.append(f"    datasets {c1.get('datasets_done', '?')}/{c1.get('datasets_total', '?')}")
    lines.append(f"    models   {c1.get('models_done', '?')}/{c1.get('models_total', '?')}")
    lines.append(f"    registry frozen: {c1.get('registry_frozen', '?')}")
    lines.append(
        f"    evaluation rows {c1.get('eval_rows_total', '?')} "
        f"(primary expected {c1.get('eval_rows_primary_expected', '?')})"
    )
    lines.append(f"    status {c1.get('status')}")
    lines.append("")

    c2 = report["concern_2"]
    lines.append("Concern 2 (supervision-objective ablation):")
    lines.append(f"    root {c2.get('root', '(unknown)')}")
    lines.append(f"    datasets {c2.get('datasets_done', '?')}/{c2.get('datasets_total', '?')}")
    lines.append(f"    models   {c2.get('models_done', '?')}/{c2.get('models_total', '?')}")
    lines.append(f"    registry frozen: {c2.get('registry_frozen', '?')}")
    lines.append(f"    evaluation {c2.get('eval_rows', '?')}/{c2.get('eval_rows_expected', '?')}")
    lines.append(
        f"    same_example_audit exists: {c2.get('same_example_audit_exists', '?')}, "
        f"fairness_audit exists: {c2.get('fairness_audit_exists', '?')}"
    )
    lines.append(f"    status {c2.get('status')}")
    lines.append("")

    c3 = report["concern_3"]
    lines.append("Concern 3 (distribution shift):")
    lines.append(f"    root {c3.get('root', '(unknown)')}")
    lines.append(f"    primary rows {c3.get('primary_rows', '?')}/{c3.get('primary_rows_expected', '?')}")
    lines.append(f"    folds complete {c3.get('folds_complete', '?')}/{c3.get('folds_total', '?')}")
    lines.append(f"    status {c3.get('status')}")
    lines.append("")

    c4 = report["concern_4"]
    lines.append("Concern 4 (practical significance):")
    lines.append(f"    root {c4.get('root', '(unknown)')}")
    lines.append(f"    smoke artifacts: {len(c4.get('smoke_artifacts', []))} file(s)")
    lines.append(f"    controlled campaign started: {c4.get('controlled_campaign_started', '?')}")
    lines.append(f"    C4_CONTROLLED_TIMING_GATE = {c4.get('timing_gate', '?')}")
    lines.append(f"    status {c4.get('status')}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of the text report.")
    args = ap.parse_args()

    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
