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

    # Audit states come from the richer status fields (already computed in revision_status).
    same_ex_pass = bool(c2.get("same_example_final_pass"))
    fair_pass = bool(c2.get("fairness_final_pass"))
    same_ex_exists = bool(c2.get("same_example_audit_exists"))
    fair_exists = bool(c2.get("fairness_audit_exists"))

    if same_ex_pass and fair_pass:
        audits = "BOTH_FINAL_PASS"
    elif same_ex_exists or fair_exists:
        audits = f"PARTIAL (same_example_final_pass={same_ex_pass}, fairness_final_pass={fair_pass})"
    else:
        audits = "NOT_RUN"

    if c2.get("registry_frozen"):
        registry = "FROZEN"
    elif training == "COMPLETE":
        registry = "READY_TO_FREEZE"
    else:
        registry = "BLOCKED"

    # Use the eval_state from status module for precise eval classification.
    eval_state = c2.get("eval_state", "NOT_STARTED")
    eval_rows = c2.get("eval_rows", 0)
    eval_expected = c2.get("eval_rows_expected", 84)

    if eval_state == "COMPLETE_AND_AUDITED":
        eval_ = "COMPLETE_AND_AUDITED"
    elif eval_state in ("FINAL_AUDITS_RUNNING",):
        eval_ = "FINAL_AUDITS_RUNNING"
    elif eval_state == "BLOCKED_BY_AUDIT":
        eval_ = "BLOCKED_BY_AUDIT"
    elif eval_state == "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED":
        # 84/84 rows exist but the final audits have not been produced yet.
        eval_ = "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED"
    elif eval_state == "CURRENT_RUN_RUNNING_PRE_GATE":
        # Currently running but audits were not run before eval started.
        # This is the live run that began before the gate was installed.
        eval_ = "CURRENT_RUN_RUNNING_PRE_GATE"
    elif eval_state == "CURRENT_RUN_PARTIAL_STOPPED_PRE_GATE":
        eval_ = "CURRENT_RUN_PARTIAL_STOPPED_PRE_GATE"
    elif eval_state == "NOT_STARTED":
        if registry == "FROZEN" and training == "COMPLETE":
            eval_ = "READY_TO_START"
        else:
            eval_ = "NOT_STARTED"
    elif eval_state in ("INVALID_OVERFLOW", "INVALID_DUPLICATE_KEYS", "INVALID_NAN_INF", "EVALUATION_FAILED"):
        eval_ = "FAILED"
    else:
        eval_ = "BLOCKED"

    result: Dict[str, str] = {
        "training": training,
        "audits": audits,
        "registry": registry,
        "eval": eval_,
        "eval_rows": f"{eval_rows}/{eval_expected}",
    }

    # Acceptance note for the current live run that started without the audit gate.
    if eval_state == "CURRENT_RUN_RUNNING_PRE_GATE":
        result["acceptance_note"] = (
            "CURRENT RUN started before the audit gate was installed. "
            "Run both final audits after 84/84 rows complete. "
            "If both pass with FINAL=true, the run is scientifically acceptable."
        )
    elif eval_state == "CURRENT_RUN_PARTIAL_STOPPED_PRE_GATE":
        result["acceptance_note"] = (
            "CURRENT RUN evaluation stopped before reaching 84/84 (pre-gate run). "
            "Resume the evaluator to 84/84, then run both final audits."
        )
    elif eval_state == "CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED":
        result["acceptance_note"] = (
            "Evaluation is 84/84 but the final audits have not yet run. "
            "Run audit_supervision_objective_examples.py and "
            "audit_supervision_objective_fairness.py (both FINAL=true) "
            "to reach COMPLETE_AND_AUDITED."
        )
    elif eval_state == "BLOCKED_BY_AUDIT":
        result["acceptance_note"] = (
            "One or both final audits FAILED. Acceptance is blocked until the "
            "underlying training/model/data issue is fixed and the audits pass."
        )

    return result


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
    expected = c4.get("smoke_artifacts_expected", 9)
    present = c4.get("smoke_artifacts_count", 0)
    missing = c4.get("smoke_artifacts_missing", [])
    smoke = "COMPLETE" if present == expected and not missing else (
        "INCOMPLETE" if present > 0 else "NOT_STARTED"
    )
    gate = c4.get("timing_gate", "DEFER")
    controlled_timing = "BLOCKED_BY_ACTIVE_JOBS" if gate == "DEFER" else "READY"
    if c4.get("controlled_campaign_started"):
        controlled_timing = "COMPLETE"
    result: Dict[str, str] = {"smoke": smoke, "controlled_timing": controlled_timing}
    if missing:
        result["smoke_missing"] = str(missing)
    return result


def next_action(c1r, c2r, c3r, c4r) -> str:
    # Freezing a registry is cheap metadata work (no real CPU contention),
    # so it's actionable regardless of what else is running. Resuming C3,
    # by contrast, launches a genuinely heavy competing training+eval job
    # -- only surface it as the top action when C1/C2 aren't already
    # actively consuming CPU for their own training, to avoid recommending
    # a 4th heavy job on top of 2-3 already running.
    other_heavy_job_running = (
        c1r["training"] == "RUNNING"
        or c2r["training"] == "RUNNING"
        or c2r["eval"] in ("RUNNING", "CURRENT_RUN_RUNNING_PRE_GATE")
    )
    if c1r["registry"] == "READY_TO_FREEZE":
        return "CONCERN_1_READY_FOR_NEXT_STAGE"
    if c2r["registry"] == "READY_TO_FREEZE" and c2r["training"] == "COMPLETE":
        return "CONCERN_2_READY_FOR_NEXT_STAGE"
    if c2r["eval"] in ("CURRENT_RUN_EVAL_COMPLETE_AUDITS_REQUIRED", "FINAL_AUDITS_RUNNING"):
        return "RUN_C2_FINAL_AUDITS"
    if c2r["eval"] == "BLOCKED_BY_AUDIT":
        return "REVIEW_C2_AUDIT_FAILURE"
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
