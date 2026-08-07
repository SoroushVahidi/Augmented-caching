"""One-command readiness check for the Reviewer 1 revision campaign.
Builds on scripts/revision_status.py's read-only artifact scan and adds a
per-concern readiness classification (training/audits/registry/eval/resume/
timing) plus a single overall NEXT_ACTION line. No writes, no launches.

Usage:
    python scripts/revision_readiness.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import revision_status as status_mod


def _audit_final_state(path: Path) -> str:
    if not path.exists():
        return "NOT_RUN"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "CORRUPT"
    return "COMPLETE" if payload.get("FINAL") else "PARTIAL"


def concern_1_readiness(c1: Dict[str, object]) -> Dict[str, str]:
    training = "COMPLETE" if c1.get("models_done") == c1.get("models_total") else "RUNNING"
    if c1.get("registry_frozen"):
        registry = "FROZEN"
    elif training == "COMPLETE":
        registry = "READY_TO_FREEZE"
    else:
        registry = "BLOCKED"
    eval_ = "READY" if c1.get("registry_frozen") else "BLOCKED"
    if eval_ == "READY" and c1.get("eval_rows_total", 0) >= c1.get("eval_rows_primary_expected", 21) * 2:
        eval_ = "COMPLETE"
    return {"training": training, "registry": registry, "eval": eval_}


def concern_2_readiness(c2: Dict[str, object], ablation_root: Optional[Path]) -> Dict[str, str]:
    training = "COMPLETE" if c2.get("models_done") == c2.get("models_total") else "RUNNING"
    audits = "NOT_RUN"
    if ablation_root is not None:
        same_ex = _audit_final_state(ablation_root / "analysis" / "supervision_objective_ablation_v1" / "same_example_audit.json")
        fair = _audit_final_state(ablation_root / "analysis" / "supervision_objective_ablation_v1" / "fairness_audit.json")
        if same_ex == "COMPLETE" and fair == "COMPLETE":
            audits = "COMPLETE"
        elif same_ex != "NOT_RUN" or fair != "NOT_RUN":
            audits = f"PARTIAL (same_example={same_ex}, fairness={fair})"
    if c2.get("registry_frozen"):
        registry = "FROZEN"
    elif training == "COMPLETE":
        registry = "READY_TO_FREEZE"
    else:
        registry = "BLOCKED"
    eval_ = "COMPLETE" if c2.get("eval_rows", 0) >= c2.get("eval_rows_expected", 84) else (
        "READY" if c2.get("registry_frozen") else "BLOCKED"
    )
    note = None
    if training == "RUNNING":
        note = ("NOTE: the running objective_ablation_pipeline tmux session already auto-chains "
                "dataset build -> train -> registry freeze -> 84-row eval with NO manual step in "
                "between -- it does not wait for the same-example/fairness audits above. If those "
                "audits are meant to gate the eval, the pipeline must be interrupted before Stage 4 "
                "fires, which is a deliberate, audited decision -- not something this tool does.")
    return {"training": training, "audits": audits, "registry": registry, "eval": eval_, **({"note": note} if note else {})}


def concern_3_readiness(c3: Dict[str, object], active_tmux) -> Dict[str, str]:
    rows = c3.get("primary_rows", 0)
    expected = c3.get("primary_rows_expected", 42)
    campaign = "COMPLETE" if rows >= expected else ("PARTIAL" if rows > 0 else "NOT_STARTED")
    session_active = "distribution_shift_2h_continue" in active_tmux
    if campaign == "COMPLETE":
        resume = "N/A"
    elif session_active:
        resume = "ALREADY_RUNNING"
    else:
        resume = "READY"
    completion_audit = "READY" if campaign == "COMPLETE" else "BLOCKED"
    return {"campaign": campaign, "resume": resume, "completion_audit": completion_audit}


def concern_4_readiness(c4: Dict[str, object]) -> Dict[str, str]:
    smoke = "COMPLETE" if c4.get("smoke_artifacts") else "NOT_STARTED"
    gate = c4.get("timing_gate", "DEFER")
    controlled_timing = "BLOCKED_BY_ACTIVE_JOBS" if gate == "DEFER" else "READY"
    if c4.get("controlled_campaign_started"):
        controlled_timing = "COMPLETE"
    return {"smoke": smoke, "controlled_timing": controlled_timing}


def next_action(c1r, c2r, c3r, c4r) -> str:
    # Freezing a registry is cheap metadata work (no real CPU contention),
    # so it's actionable regardless of what else is running. Resuming C3,
    # by contrast, launches a genuinely heavy competing training+eval job
    # -- only surface it as the top action when C1/C2 aren't already
    # actively consuming CPU for their own training, to avoid recommending
    # a 4th heavy job on top of 2-3 already running.
    other_heavy_job_running = c1r["training"] == "RUNNING" or c2r["training"] == "RUNNING"
    if c1r["registry"] == "READY_TO_FREEZE":
        return "CONCERN_1_READY_FOR_NEXT_STAGE"
    if c2r["registry"] == "READY_TO_FREEZE" and c2r["training"] == "COMPLETE":
        return "CONCERN_2_READY_FOR_NEXT_STAGE"
    if c3r["resume"] == "READY" and not other_heavy_job_running:
        return "CONCERN_3_READY_TO_RESUME"
    if c4r["controlled_timing"] == "READY":
        return "CONCERN_4_TIMING_READY"
    return "WAIT_FOR_RUNNING_JOBS"


def collect_readiness() -> Dict[str, object]:
    report = status_mod.collect()
    repo_root = Path(__file__).resolve().parent.parent
    worktrees = status_mod.find_worktrees(repo_root)
    ablation_root = worktrees.get("feat/supervision-objective-ablation")

    c1r = concern_1_readiness(report["concern_1"])
    c2r = concern_2_readiness(report["concern_2"], ablation_root)
    c3r = concern_3_readiness(report["concern_3"], report["active_tmux_sessions"])
    c4r = concern_4_readiness(report["concern_4"])

    return {
        "concern_1": c1r, "concern_2": c2r, "concern_3": c3r, "concern_4": c4r,
        "NEXT_ACTION": next_action(c1r, c2r, c3r, c4r),
        "_status": report,
    }


def format_readiness(r: Dict[str, object]) -> str:
    lines = []
    lines.append("Concern 1:")
    for k, v in r["concern_1"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Concern 2:")
    for k, v in r["concern_2"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Concern 3:")
    for k, v in r["concern_3"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Concern 4:")
    for k, v in r["concern_4"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"NEXT_ACTION = {r['NEXT_ACTION']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = collect_readiness()
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(format_readiness(r))


if __name__ == "__main__":
    main()
