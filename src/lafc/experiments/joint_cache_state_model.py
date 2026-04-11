from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class JointSoftmaxConfig:
    hidden_dim: int = 16
    lr: float = 0.03
    epochs: int = 160
    l2: float = 1e-4
    seed: int = 7


class JointSoftmaxVictimModel:
    """Minimal joint-state softmax victim classifier with shared candidate encoding."""

    def __init__(self, cfg: JointSoftmaxConfig) -> None:
        self.cfg = cfg
        self.Wc: np.ndarray | None = None
        self.Wg: np.ndarray | None = None
        self.bh: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def fit(self, decisions: Sequence[Dict[str, object]], *, feature_columns: Sequence[str], global_columns: Sequence[str]) -> None:
        if not decisions:
            raise ValueError("No decisions provided to fit joint softmax model")
        dc = len(feature_columns)
        dg = len(global_columns)
        h = int(self.cfg.hidden_dim)
        rng = np.random.default_rng(self.cfg.seed)

        Wc = rng.normal(0.0, 0.08, size=(h, dc))
        Wg = rng.normal(0.0, 0.08, size=(h, dg))
        bh = np.zeros((h,), dtype=float)
        v = rng.normal(0.0, 0.08, size=(h,))

        for _ in range(int(self.cfg.epochs)):
            for d in decisions:
                candidates = d["candidate_features"]
                oracle = str(d["oracle_victim"])
                cmat = np.asarray([[float(c[col]) for col in feature_columns] for c in candidates], dtype=float)
                g = np.asarray([float(d[col]) for col in global_columns], dtype=float)

                hid = np.tanh((cmat @ Wc.T) + (Wg @ g)[None, :] + bh[None, :])
                scores = hid @ v
                z = scores - np.max(scores)
                probs = np.exp(z)
                probs = probs / np.sum(probs)

                y = np.zeros((len(candidates),), dtype=float)
                idx = next((i for i, c in enumerate(candidates) if str(c["candidate_page_id"]) == oracle), None)
                if idx is None:
                    continue
                y[int(idx)] = 1.0

                ds = probs - y
                dv = hid.T @ ds + self.cfg.l2 * v
                dh = ds[:, None] * v[None, :] * (1.0 - hid * hid)
                dWc = dh.T @ cmat + self.cfg.l2 * Wc
                dWg = np.outer(np.sum(dh, axis=0), g) + self.cfg.l2 * Wg
                dbh = np.sum(dh, axis=0)

                Wc -= self.cfg.lr * dWc
                Wg -= self.cfg.lr * dWg
                bh -= self.cfg.lr * dbh
                v -= self.cfg.lr * dv

        self.Wc, self.Wg, self.bh, self.v = Wc, Wg, bh, v

    def predict_proba(self, *, candidate_features: Sequence[Dict[str, object]], global_features: Dict[str, float], feature_columns: Sequence[str], global_columns: Sequence[str]) -> Dict[str, float]:
        if self.Wc is None or self.Wg is None or self.bh is None or self.v is None:
            raise ValueError("Model is not fitted")
        cmat = np.asarray([[float(c[col]) for col in feature_columns] for c in candidate_features], dtype=float)
        g = np.asarray([float(global_features[col]) for col in global_columns], dtype=float)

        hid = np.tanh((cmat @ self.Wc.T) + (self.Wg @ g)[None, :] + self.bh[None, :])
        scores = hid @ self.v
        z = scores - np.max(scores)
        probs = np.exp(z)
        probs = probs / np.sum(probs)
        return {str(c["candidate_page_id"]): float(p) for c, p in zip(candidate_features, probs)}

    def save(self, path: Path, *, feature_columns: Sequence[str], global_columns: Sequence[str]) -> None:
        if self.Wc is None or self.Wg is None or self.bh is None or self.v is None:
            raise ValueError("Model is not fitted")
        payload = {
            "cfg": {
                "hidden_dim": int(self.cfg.hidden_dim),
                "lr": float(self.cfg.lr),
                "epochs": int(self.cfg.epochs),
                "l2": float(self.cfg.l2),
                "seed": int(self.cfg.seed),
            },
            "feature_columns": list(feature_columns),
            "global_columns": list(global_columns),
            "Wc": self.Wc.tolist(),
            "Wg": self.Wg.tolist(),
            "bh": self.bh.tolist(),
            "v": self.v.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
