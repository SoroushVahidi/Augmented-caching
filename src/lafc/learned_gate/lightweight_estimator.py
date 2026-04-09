from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LinearProbabilityEstimator:
    """Tiny dependency-light estimator with sklearn-like predict_proba interface."""

    feature_columns: List[str]
    feature_weights: Dict[str, float]
    intercept: float = 0.0

    def _linear_score(self, row: List[float]) -> float:
        val = float(self.intercept)
        for idx, col in enumerate(self.feature_columns):
            val += float(self.feature_weights.get(col, 0.0)) * float(row[idx])
        return val

    def predict_proba(self, x: List[List[float]]) -> List[List[float]]:
        probs: List[List[float]] = []
        for row in x:
            score = self._linear_score([float(v) for v in row])
            p1 = 1.0 / (1.0 + pow(2.718281828459045, -score))
            probs.append([1.0 - p1, p1])
        return probs
