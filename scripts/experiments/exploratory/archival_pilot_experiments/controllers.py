"""Lightweight branch-allocation controllers for ranking-ablation pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

from experiments.learned_scorer import LearnedBranchScorer

RankingMode = Literal[
    "raw_score",
    "score_plus_progress",
    "relative_rank",
    "learned_branch_score",
]


@dataclass
class BranchState:
    branch_id: int
    score: float
    depth: int
    expansions: int = 0
    alive: bool = True
    verifications: int = 0
    remaining_budget: int = 0
    pre_verify_score: float = 0.0
    post_verify_score: float = 0.0
    survived_pruning_steps: int = 0


@dataclass
class AdaptiveMinExpandConfig:
    min_initial_expansions: int = 1
    ranking_mode: RankingMode = "raw_score"
    progress_weight: float = 0.10
    learned_model_path: Optional[str] = None


class AdaptiveMinExpandController:
    """Adaptive controller with minimum-expansion safeguard and pluggable ranking.

    Ranking formulas:
      - raw_score: rank_value = score
      - score_plus_progress: rank_value = score + progress_weight * depth
      - relative_rank: rank_value = rank(score) + rank(depth)
      - learned_branch_score: rank_value = P(promising=1 | branch_features)

    For relative_rank, rank(position) is 1 for worst and N for best in pool.
    """

    def __init__(self, config: AdaptiveMinExpandConfig | None = None) -> None:
        self.config = config or AdaptiveMinExpandConfig()
        self._scorer: Optional[LearnedBranchScorer] = None
        if self.config.ranking_mode == "learned_branch_score":
            if not self.config.learned_model_path:
                raise ValueError("learned_model_path is required for learned_branch_score mode")
            self._scorer = LearnedBranchScorer.load(self.config.learned_model_path)

    def choose_branch(self, branches: Iterable[BranchState]) -> BranchState:
        active = [b for b in branches if b.alive]
        if not active:
            raise ValueError("No active branches available.")

        for b in sorted(active, key=lambda x: (x.expansions, x.branch_id)):
            if b.expansions < self.config.min_initial_expansions:
                return b

        mode = self.config.ranking_mode
        if mode == "raw_score":
            return max(active, key=lambda b: (b.score, b.depth, -b.branch_id))

        if mode == "score_plus_progress":
            return max(
                active,
                key=lambda b: (
                    b.score + self.config.progress_weight * float(b.depth),
                    b.score,
                    b.depth,
                    -b.branch_id,
                ),
            )

        if mode == "relative_rank":
            return self._choose_by_relative_rank(active)

        if mode == "learned_branch_score":
            assert self._scorer is not None
            scored = [(self._scorer.score_branch(b), b) for b in active]
            scored.sort(key=lambda x: (x[0], x[1].score, x[1].depth, -x[1].branch_id), reverse=True)
            return scored[0][1]

        raise ValueError(f"Unsupported ranking mode: {mode}")

    @staticmethod
    def _choose_by_relative_rank(active: List[BranchState]) -> BranchState:
        score_order = sorted(active, key=lambda b: (b.score, b.depth, -b.branch_id))
        depth_order = sorted(active, key=lambda b: (b.depth, b.score, -b.branch_id))

        score_rank = {b.branch_id: idx + 1 for idx, b in enumerate(score_order)}
        depth_rank = {b.branch_id: idx + 1 for idx, b in enumerate(depth_order)}

        return max(
            active,
            key=lambda b: (
                score_rank[b.branch_id] + depth_rank[b.branch_id],
                score_rank[b.branch_id],
                depth_rank[b.branch_id],
                -b.branch_id,
            ),
        )
