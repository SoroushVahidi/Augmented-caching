from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, List


@dataclass
class EvictValuePairwiseV1Model:
    model_name: str
    estimator: object
    delta_feature_columns: List[str]

    def _delta_vector(self, a_features: Dict[str, float], b_features: Dict[str, float]) -> List[float]:
        vals: List[float] = []
        for c in self.delta_feature_columns:
            if c in a_features and c in b_features:
                vals.append(float(a_features[c]) - float(b_features[c]))
                continue
            base = c[6:] if c.startswith("delta_") else c
            vals.append(float(a_features.get(base, 0.0)) - float(b_features.get(base, 0.0)))
        return vals

    def predict_a_beats_b_proba(self, a_features: Dict[str, float], b_features: Dict[str, float]) -> float:
        x = [self._delta_vector(a_features, b_features)]
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(x)[0]
            if len(probs) == 1:
                cls = int(getattr(self.estimator, "classes_", [0])[0])
                return float(cls)
            return float(probs[1])
        pred = float(self.estimator.predict(x)[0])
        return max(0.0, min(1.0, pred))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(
                {
                    "model_name": self.model_name,
                    "estimator": self.estimator,
                    "delta_feature_columns": self.delta_feature_columns,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EvictValuePairwiseV1Model":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        return cls(
            model_name=str(payload["model_name"]),
            estimator=payload["estimator"],
            delta_feature_columns=list(payload["delta_feature_columns"]),
        )
