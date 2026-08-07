"""Reward model for the HALP policy: a shared-weight, pointwise two-layer
MLP `R(x)`, trained via a pairwise Bradley-Terry preference loss on
`R(x_A) - R(x_B)`.

Reference
---------
Song et al. "HALP: Heuristic Aided Learned Preference Eviction Policy for
YouTube Content Delivery Network." NSDI 2023.
Official Google Research blog ("Preference learning with automated feedback
for cache eviction") describes the production reward model as a
"light-weight two-layer multilayer perceptron (MLP)" trained continuously
online from random initialization, scoring candidates for preference
comparisons.

Implementation note (why this is hand-rolled rather than
`sklearn.neural_network.MLPClassifier`): a Bradley-Terry / RankNet-style
preference model requires one *shared* network `R(·)` whose pairwise
difference `R(x_A) - R(x_B)` is passed through a sigmoid and compared to
the preference label. For a LINEAR `R`, this is mathematically equivalent
to fitting a binary classifier directly on the difference vectors
`x_A - x_B` (which is what an earlier draft of this file did, and what
`docs/lrb_method_spec.md`-style linear reward models rely on). That
equivalence does **not** hold for a nonlinear `R`: `R(x_A) - R(x_B) !=
R(x_A - x_B)` in general for a 2-layer MLP. Reusing the
diff-vector-classifier trick with `MLPClassifier` would silently produce an
invalid, internally-inconsistent scoring function (confirmed empirically:
it produced `lbfgs failed to converge` warnings and non-ranking-consistent
scores on real trace data). This module instead implements the correct
shared-weight forward/backward pass directly with full-batch gradient
descent, deterministic given a fixed seed and no minibatching.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class HALPModel:
    """Shared-weight two-layer-MLP pairwise (Bradley-Terry) preference model.

    `R(x) = W2 . relu(W1 x + b1)` (no output bias: it cancels exactly in
    `R(x_A) - R(x_B)` and is therefore unidentifiable from pairwise data
    alone -- this is expected, not a bug).
    """

    def __init__(
        self,
        hidden_units: int = 8,
        alpha: float = 1e-4,
        lr: float = 0.05,
        n_epochs: int = 300,
        seed: int = 42,
    ):
        self._imputer = SimpleImputer(strategy="constant", fill_value=1e6)
        self._hidden_units = hidden_units
        self._alpha = alpha
        self._lr = lr
        self._n_epochs = n_epochs
        self._seed = seed
        self._fitted = False
        self._mean: np.ndarray = None
        self._std: np.ndarray = None
        self._W1: np.ndarray = None
        self._b1: np.ndarray = None
        self._W2: np.ndarray = None

    def fit(self, X_pref: np.ndarray, X_non_pref: np.ndarray) -> None:
        """Fit R(.) on (preferred, non-preferred) candidate pairs.

        Each row pair (A, B) means "A is preferred over B" (A's next
        re-access occurs first); the loss is the standard Bradley-Terry /
        RankNet cross-entropy -log(sigmoid(R(A) - R(B))), so no separate
        negated/symmetrized pairs are needed (unlike a generic binary
        classifier, the pairwise loss already encodes direction).
        """
        if len(X_pref) == 0:
            return

        A = self._imputer.fit_transform(X_pref)
        B = self._imputer.transform(X_non_pref)

        all_x = np.vstack([A, B])
        self._mean = all_x.mean(axis=0)
        self._std = all_x.std(axis=0)
        self._std[self._std == 0] = 1.0
        A = (A - self._mean) / self._std
        B = (B - self._mean) / self._std

        n, d = A.shape
        h = self._hidden_units
        rng = np.random.default_rng(self._seed)
        W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, h))
        b1 = np.zeros(h)
        W2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h,))

        for _ in range(self._n_epochs):
            zA = A @ W1 + b1
            hA = np.maximum(zA, 0.0)
            rA = hA @ W2

            zB = B @ W1 + b1
            hB = np.maximum(zB, 0.0)
            rB = hB @ W2

            p = _sigmoid(rA - rB)
            grad_diff = (p - 1.0) / n  # dL/d(rA - rB), label is always 1

            grad_W2 = (hA - hB).T @ grad_diff + self._alpha * W2
            grad_hA = np.outer(grad_diff, W2)
            grad_hB = -np.outer(grad_diff, W2)
            grad_zA = grad_hA * (zA > 0)
            grad_zB = grad_hB * (zB > 0)
            grad_W1 = A.T @ grad_zA + B.T @ grad_zB + self._alpha * W1
            grad_b1 = grad_zA.sum(axis=0) + grad_zB.sum(axis=0)

            W1 -= self._lr * grad_W1
            b1 -= self._lr * grad_b1
            W2 -= self._lr * grad_W2

        self._W1, self._b1, self._W2 = W1, b1, W2
        self._fitted = True

    def predict_rewards(self, X: np.ndarray) -> np.ndarray:
        """Predict scalar reward scores R(x) for candidates."""
        if not self._fitted:
            return np.zeros(len(X))
        X_imp = self._imputer.transform(X)
        Xs = (X_imp - self._mean) / self._std
        h = np.maximum(Xs @ self._W1 + self._b1, 0.0)
        return h @ self._W2

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model (weights + normalization stats) for
        offline freeze/reload across processes -- pure I/O, no change to
        fit()/predict_rewards()'s math."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "hidden_units": self._hidden_units,
                    "alpha": self._alpha,
                    "lr": self._lr,
                    "n_epochs": self._n_epochs,
                    "seed": self._seed,
                    "fitted": self._fitted,
                    "mean": self._mean,
                    "std": self._std,
                    "W1": self._W1,
                    "b1": self._b1,
                    "W2": self._W2,
                    "imputer": self._imputer,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HALPModel":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        model = cls(
            hidden_units=payload["hidden_units"],
            alpha=payload["alpha"],
            lr=payload["lr"],
            n_epochs=payload["n_epochs"],
            seed=payload["seed"],
        )
        model._fitted = payload["fitted"]
        model._mean = payload["mean"]
        model._std = payload["std"]
        model._W1 = payload["W1"]
        model._b1 = payload["b1"]
        model._W2 = payload["W2"]
        model._imputer = payload["imputer"]
        return model
