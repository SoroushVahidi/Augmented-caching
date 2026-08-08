from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

from experiments.controllers import AdaptiveMinExpandConfig, AdaptiveMinExpandController, BranchState


@dataclass
class MethodResult:
    correct: int
    actions: int
    expansions: int
    verifications: int
    surviving_branches: int
    budget_exhausted: int


def _make_branches(rng: random.Random, branch_count: int, budget: int) -> List[BranchState]:
    out: List[BranchState] = []
    for i in range(branch_count):
        s = rng.uniform(0.0, 1.0)
        out.append(
            BranchState(
                branch_id=i,
                score=s,
                depth=rng.randint(1, 4),
                expansions=0,
                remaining_budget=budget,
                pre_verify_score=s,
                post_verify_score=s,
            )
        )
    return out


def _grow_branch(rng: random.Random, branch: BranchState, remaining_budget: int) -> int:
    branch.expansions += 1
    branch.depth += 1
    branch.remaining_budget = remaining_budget
    branch.pre_verify_score = branch.score
    # lightweight verification/update proxy
    verify_inc = 1 if rng.random() < 0.35 else 0
    branch.verifications += verify_inc
    if branch.verifications > 0:
        branch.post_verify_score = min(1.0, max(0.0, branch.score + rng.uniform(-0.1, 0.2)))
    else:
        branch.post_verify_score = branch.score
    branch.score = branch.post_verify_score
    return verify_inc


def _trace_row(branch: BranchState, method: str, episode_id: int, action_idx: int, gold_branch_id: int) -> Dict[str, object]:
    return {
        "method": method,
        "episode_id": episode_id,
        "action_idx": action_idx,
        "branch_id": branch.branch_id,
        "gold_branch_id": gold_branch_id,
        "eventually_correct_branch": int(branch.branch_id == gold_branch_id),
        "raw_score": float(branch.score),
        "depth": int(branch.depth),
        "expansions": int(branch.expansions),
        "verifications": int(branch.verifications),
        "remaining_budget": int(branch.remaining_budget),
        "pre_verify_score": float(branch.pre_verify_score),
        "post_verify_score": float(branch.post_verify_score),
        "score_delta": float(branch.post_verify_score - branch.pre_verify_score),
        "survived_pruning_steps": int(branch.survived_pruning_steps),
        "alive": int(branch.alive),
    }


def _simulate_one(
    rng: random.Random,
    method: str,
    budget: int,
    branch_count: int,
    min_expand: int,
    episode_id: int,
    learned_model_path: Optional[str],
) -> tuple[MethodResult, List[Dict[str, object]]]:
    branches = _make_branches(rng, branch_count, budget)
    gold_branch_id = rng.randrange(branch_count)

    controller = None
    if method.startswith("adaptive_"):
        ranking_mode = method.replace("adaptive_", "")
        controller = AdaptiveMinExpandController(
            AdaptiveMinExpandConfig(
                min_initial_expansions=min_expand,
                ranking_mode=ranking_mode,  # type: ignore[arg-type]
                progress_weight=0.10,
                learned_model_path=learned_model_path,
            )
        )

    actions = 0
    expansions = 0
    verifications = 0
    budget_exhausted = 0
    trace_rows: List[Dict[str, object]] = []

    for step in range(budget):
        actions += 1
        remaining_budget = budget - step - 1
        if method == "greedy_single_path":
            targets = [max(branches, key=lambda b: (b.score, -b.branch_id))]
        elif method == "best_of_n":
            targets = list(branches)
        elif method == "fixed_width_beam":
            targets = sorted(branches, key=lambda b: (b.score, b.depth), reverse=True)[: min(3, len(branches))]
        else:
            assert controller is not None
            targets = [controller.choose_branch(branches)]

        for target in targets:
            verify_inc = _grow_branch(rng, target, remaining_budget)
            expansions += 1
            verifications += verify_inc
            trace_rows.append(_trace_row(target, method, episode_id, step, gold_branch_id))

        cutoff = 0.12
        for b in branches:
            if b.alive and b.score < cutoff and b.expansions >= min_expand:
                b.alive = False
            if b.alive:
                b.survived_pruning_steps += 1

        if sum(1 for b in branches if b.alive) == 0:
            budget_exhausted = 1
            break

    alive = [b for b in branches if b.alive]
    if not alive:
        alive = branches
    final = max(alive, key=lambda b: (b.score, b.depth, b.expansions, -b.branch_id))
    correct = int(final.branch_id == gold_branch_id)

    return (
        MethodResult(
            correct=correct,
            actions=actions,
            expansions=expansions,
            verifications=verifications,
            surviving_branches=sum(1 for b in branches if b.alive),
            budget_exhausted=budget_exhausted,
        ),
        trace_rows,
    )


def run_pilot(
    samples: int,
    budget: int,
    branch_count: int,
    min_expand: int,
    seed: int,
    methods: Sequence[str],
    learned_model_path: Optional[str],
) -> tuple[Dict[str, Dict[str, float]], List[Dict[str, object]]]:
    table: Dict[str, Dict[str, float]] = {}
    all_rows: List[Dict[str, object]] = []
    for m_idx, method in enumerate(methods):
        rng = random.Random(seed + m_idx)
        outcomes = [
            _simulate_one(rng, method, budget, branch_count, min_expand, ep, learned_model_path) for ep in range(samples)
        ]
        rows = [x[0] for x in outcomes]
        traces = [tr for _, ts in outcomes for tr in ts]
        all_rows.extend(traces)
        table[method] = {
            "accuracy": mean(r.correct for r in rows),
            "avg_actions": mean(r.actions for r in rows),
            "avg_expansions": mean(r.expansions for r in rows),
            "avg_verifications": mean(r.verifications for r in rows),
            "avg_surviving_branches": mean(r.surviving_branches for r in rows),
            "budget_exhaustion_rate": mean(r.budget_exhausted for r in rows),
        }
    return table, all_rows


def _default_methods(compare_learned_only: bool) -> List[str]:
    if compare_learned_only:
        return [
            "fixed_width_beam",
            "adaptive_score_plus_progress",
            "adaptive_relative_rank",
            "adaptive_learned_branch_score",
        ]
    return [
        "greedy_single_path",
        "best_of_n",
        "fixed_width_beam",
        "adaptive_raw_score",
        "adaptive_score_plus_progress",
        "adaptive_relative_rank",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--branch-count", type=int, default=5)
    ap.add_argument("--min-expand", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-id", type=str, default=datetime.utcnow().strftime("%Y%m%d_ranking_ablation"))
    ap.add_argument("--learned-model-path", type=str, default=None)
    ap.add_argument("--compare-learned-only", action="store_true")
    args = ap.parse_args()

    methods = _default_methods(args.compare_learned_only)
    if "adaptive_learned_branch_score" in methods and not args.learned_model_path:
        raise SystemExit("--learned-model-path is required when running adaptive_learned_branch_score")

    out_dir = Path("outputs") / "pilot" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, trace_rows = run_pilot(
        args.samples,
        args.budget,
        args.branch_count,
        args.min_expand,
        args.seed,
        methods,
        args.learned_model_path,
    )

    payload = {
        "provider": "synthetic_offline",
        "model": "none",
        "pilot_size": args.samples,
        "budget": args.budget,
        "branch_count": args.branch_count,
        "min_expand": args.min_expand,
        "methods": methods,
        "learned_model_path": args.learned_model_path,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "branch_traces.jsonl").open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row) + "\n")
    print(str(out_dir))


if __name__ == "__main__":
    main()
