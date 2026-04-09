from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Iterable, List

from lafc.learned_gate.features import ML_GATE_FEATURE_COLUMNS
from lafc.learned_gate.lightweight_estimator import LinearProbabilityEstimator


@dataclass
class LearnedGateModel:
    pipeline: object
    feature_columns: List[str]
    threshold: float = 0.5

    @classmethod
    def new_logistic(cls, random_state: int = 7) -> "LearnedGateModel":
        _ = random_state
        estimator = LinearProbabilityEstimator(
            feature_columns=list(ML_GATE_FEATURE_COLUMNS),
            feature_weights={c: 0.0 for c in ML_GATE_FEATURE_COLUMNS},
            intercept=0.0,
        )
        return cls(pipeline=estimator, feature_columns=list(ML_GATE_FEATURE_COLUMNS), threshold=0.5)

    def fit(self, rows: Iterable[Dict[str, float]], y: Iterable[int]) -> None:
        data = list(rows)
        labels = [int(v) for v in y]
        if not data:
            raise ValueError("No rows provided to LearnedGateModel.fit")

        pos_rows = [r for r, lbl in zip(data, labels) if lbl == 1]
        neg_rows = [r for r, lbl in zip(data, labels) if lbl == 0]
        pos_mean = {
            c: (sum(float(r[c]) for r in pos_rows) / len(pos_rows)) if pos_rows else 0.0
            for c in self.feature_columns
        }
        neg_mean = {
            c: (sum(float(r[c]) for r in neg_rows) / len(neg_rows)) if neg_rows else 0.0
            for c in self.feature_columns
        }
        weights = {c: float(pos_mean[c] - neg_mean[c]) for c in self.feature_columns}
        intercept = float(-0.5 * sum(weights[c] * (pos_mean[c] + neg_mean[c]) for c in self.feature_columns))
        self.pipeline = LinearProbabilityEstimator(
            feature_columns=list(self.feature_columns),
            feature_weights=weights,
            intercept=intercept,
        )

    def predict_proba_one(self, row: Dict[str, float]) -> float:
        x = [[float(row[c]) for c in self.feature_columns]]
        return float(self.pipeline.predict_proba(x)[0][1])

    def predict_one(self, row: Dict[str, float]) -> int:
        return int(self.predict_proba_one(row) >= self.threshold)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "pipeline": self.pipeline,
                    "feature_columns": self.feature_columns,
                    "threshold": self.threshold,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedGateModel":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        return cls(
            pipeline=payload["pipeline"],
            feature_columns=list(payload["feature_columns"]),
            threshold=float(payload.get("threshold", 0.5)),
        )
