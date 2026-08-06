"""Model artifact wrapper for the LRB (Learning Relaxed Belady) policy.

Wraps a LightGBM GBDT regressor with the exact hyperparameters used by the
official implementation (``include/webcachesim/caches/lrb.h:491-503``,
commit ``9e8b4423383c01c4528deb447f152f0437a37c3a``)::

    boosting=gbdt, objective=regression (L2), num_iterations=32, num_leaves=32,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, learning_rate=0.1

``lightgbm`` is an optional dependency (``pip install lafc[lrb]``); import is
deferred to the top of this module and fails with a clear message rather than
a bare ``ModuleNotFoundError`` if unavailable, since only this policy needs it.

Deviation from the official code (documented, not silent): the official code
relies on LightGBM's stock internal RNG defaults for ``bagging``/
``feature_fraction`` sampling during training, without pinning them
explicitly. To satisfy this repository's "deterministic seeds" requirement,
:data:`DEFAULT_TRAINING_PARAMS` explicitly pins ``seed``/``bagging_seed``/
``feature_fraction_seed`` and sets ``deterministic=True`` and
``force_row_wise=True``. This is a strictly-safer optional deviation, not a
change to the model family, objective, or any tuned hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Sequence

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - exercised only when lightgbm is absent
    raise ImportError(
        "The 'lrb' policy requires the optional 'lightgbm' dependency. "
        "Install it with: pip install 'lafc[lrb]'"
    ) from exc

import numpy as np


DEFAULT_TRAINING_PARAMS: Dict[str, Any] = {
    "boosting": "gbdt",
    "objective": "regression",
    "num_iterations": 32,
    "num_leaves": 32,
    "num_threads": 1,
    "feature_fraction": 0.8,
    "bagging_freq": 5,
    "bagging_fraction": 0.8,
    "learning_rate": 0.1,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 0,
    "bagging_seed": 0,
    "feature_fraction_seed": 0,
}


@dataclass
class LRBModel:
    """Owns a trained (or not-yet-trained) LightGBM booster for LRB."""

    booster: Optional["lgb.Booster"] = None
    n_features: int = 0

    def is_trained(self) -> bool:
        return self.booster is not None

    def train(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[float],
        *,
        params: Dict[str, Any],
        n_features: int,
    ) -> None:
        if not rows:
            raise ValueError("Cannot train LRBModel on an empty training batch")
        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float32)
        dataset = lgb.Dataset(data=X, label=y, params=params, free_raw_data=True)
        num_boost_round = int(params.get("num_iterations", 32))
        self.booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
        self.n_features = n_features

    def predict(self, rows: Sequence[Sequence[float]]) -> List[float]:
        if self.booster is None:
            raise RuntimeError("LRBModel.predict called before any training round")
        X = np.asarray(rows, dtype=np.float64)
        preds = self.booster.predict(data=X)
        return [float(v) for v in preds]

    def save(self, path: "str | Path") -> None:
        if self.booster is None:
            raise RuntimeError("Cannot save an untrained LRBModel")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"booster_str": self.booster.model_to_string(), "n_features": self.n_features},
                fh,
            )

    @classmethod
    def load(cls, path: "str | Path") -> "LRBModel":
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        booster = lgb.Booster(model_str=payload["booster_str"])
        return cls(booster=booster, n_features=int(payload["n_features"]))
