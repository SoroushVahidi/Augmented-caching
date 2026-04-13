from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import joblib


@dataclass
class FeatureSpec:
    feature_names: List[str]
    target_name: str = "eventually_correct_branch"
    notes: str = "Proxy target: 1 iff branch_id equals gold branch id for episode."


class LearnedBranchScorer:
    def __init__(self, model, feature_spec: FeatureSpec):
        self.model = model
        self.feature_spec = feature_spec

    @staticmethod
    def feature_dict(branch) -> Dict[str, float]:
        score_delta = branch.post_verify_score - branch.pre_verify_score
        return {
            "raw_score": float(branch.score),
            "depth": float(branch.depth),
            "expansions": float(branch.expansions),
            "verifications": float(branch.verifications),
            "remaining_budget": float(branch.remaining_budget),
            "post_verify_score": float(branch.post_verify_score),
            "score_delta": float(score_delta),
            "survived_pruning_steps": float(branch.survived_pruning_steps),
        }

    def score_branch(self, branch) -> float:
        feats = self.feature_dict(branch)
        x = [[feats[name] for name in self.feature_spec.feature_names]]
        proba = self.model.predict_proba(x)[0][1]
        return float(proba)

    def save(self, out_dir: str) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        (path / "feature_spec.json").write_text(json.dumps(asdict(self.feature_spec), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, artifact_dir: str) -> "LearnedBranchScorer":
        path = Path(artifact_dir)
        model = joblib.load(path / "model.joblib")
        spec = FeatureSpec(**json.loads((path / "feature_spec.json").read_text(encoding="utf-8")))
        return cls(model=model, feature_spec=spec)
