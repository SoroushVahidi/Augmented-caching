from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np


@dataclass
class EvictValueV1Model:
    model_name: str
    estimator: object
    feature_columns: List[str]

    def predict_loss_one(self, row: Dict[str, float]) -> float:
        x = np.asarray([[float(row[c]) for c in self.feature_columns]], dtype=float)
        return float(self.estimator.predict(x)[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "estimator": self.estimator,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvictValueV1Model":
        payload = joblib.load(path)
        return cls(
            model_name=str(payload["model_name"]),
            estimator=payload["estimator"],
            feature_columns=list(payload["feature_columns"]),
        )
