from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, List


@dataclass
class LearnedGateV2Model:
    model_name: str
    estimator: object
    feature_columns: List[str]
    threshold: float = 0.5

    def predict_proba_one(self, row: Dict[str, float]) -> float:
        x = [[float(row[c]) for c in self.feature_columns]]
        if hasattr(self.estimator, "predict_proba"):
            return float(self.estimator.predict_proba(x)[0][1])
        score = float(self.estimator.decision_function(x)[0])
        return 1.0 / (1.0 + pow(2.718281828459045, -score))

    def predict_one(self, row: Dict[str, float]) -> int:
        return int(self.predict_proba_one(row) >= self.threshold)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "estimator": self.estimator,
                    "feature_columns": self.feature_columns,
                    "threshold": self.threshold,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedGateV2Model":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        return cls(
            model_name=str(payload["model_name"]),
            estimator=payload["estimator"],
            feature_columns=list(payload["feature_columns"]),
            threshold=float(payload.get("threshold", 0.5)),
        )
