from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.learned_gate.features import ML_GATE_FEATURE_COLUMNS


@dataclass
class LearnedGateModel:
    pipeline: Pipeline
    feature_columns: List[str]
    threshold: float = 0.5

    @classmethod
    def new_logistic(cls, random_state: int = 7) -> "LearnedGateModel":
        pipe = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return cls(pipeline=pipe, feature_columns=list(ML_GATE_FEATURE_COLUMNS), threshold=0.5)

    def fit(self, rows: Iterable[Dict[str, float]], y: Iterable[int]) -> None:
        x = np.asarray([[float(r[c]) for c in self.feature_columns] for r in rows], dtype=float)
        labels = np.asarray(list(y), dtype=int)
        if x.size == 0:
            raise ValueError("No rows provided to LearnedGateModel.fit")
        self.pipeline.fit(x, labels)

    def predict_proba_one(self, row: Dict[str, float]) -> float:
        x = np.asarray([[float(row[c]) for c in self.feature_columns]], dtype=float)
        return float(self.pipeline.predict_proba(x)[0, 1])

    def predict_one(self, row: Dict[str, float]) -> int:
        return int(self.predict_proba_one(row) >= self.threshold)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedGateModel":
        payload = joblib.load(path)
        return cls(
            pipeline=payload["pipeline"],
            feature_columns=list(payload["feature_columns"]),
            threshold=float(payload.get("threshold", 0.5)),
        )
