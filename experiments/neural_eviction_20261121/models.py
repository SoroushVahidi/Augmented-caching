"""Pointwise and masked permutation-equivariant eviction scorers."""
import os
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

from data import FEATURES


@dataclass(frozen=True)
class NetworkConfig:
    architecture: str
    width: int = 128
    layers: int = 3
    dropout: float = 0.1
    ranking_weight: float = 1.0
    regret_weight: float = 0.5


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def mlp(inputs, width, layers, dropout):
    modules = []
    for _ in range(layers):
        modules.extend([nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)])
        inputs = width
    return nn.Sequential(*modules)


class EvictionNetwork(nn.Module):
    def __init__(self, config, horizon, mean, scale):
        super().__init__()
        if config.architecture not in {"mlp", "deepsets", "set_regression"}:
            raise ValueError(config.architecture)
        self.config = config
        self.horizon = horizon
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        context = config.architecture != "mlp"
        encoder_layers = config.layers - 1 if context else config.layers
        self.encoder = mlp(len(FEATURES), config.width, encoder_layers, config.dropout)
        self.context = mlp(2 * config.width, config.width, 1, config.dropout) if context else None
        self.cost_head = nn.Linear(config.width, 1)
        self.rank_head = nn.Linear(config.width, 1) if config.architecture == "deepsets" else None

    def forward(self, features, mask):
        normalized = (features - self.mean) / self.scale
        hidden = self.encoder(normalized.masked_fill(~mask[..., None], 0.0))
        if self.context is not None:
            pooled = hidden.float().masked_fill(~mask[..., None], 0.0).sum(1, keepdim=True)
            pooled = pooled / mask.sum(1, keepdim=True)[..., None].clamp_min(1)
            hidden = self.context(torch.cat((hidden, pooled.expand_as(hidden)), dim=-1))
        cost = self.cost_head(hidden).squeeze(-1).float() * self.horizon
        score = self.rank_head(hidden).squeeze(-1).float() if self.rank_head is not None else cost
        return cost, score.masked_fill(~mask, torch.inf)


def ranking_terms(scores, costs, mask):
    """Mass on all exact minima and expected regret, averaged over informative groups."""
    best = costs.masked_fill(~mask, torch.inf).amin(1, keepdim=True)
    optimal = mask & (costs == best)
    discriminative = (mask & ~optimal).any(1)
    logits = -scores.masked_fill(~mask, torch.inf)
    logmass = torch.logsumexp(logits.masked_fill(~optimal, -torch.inf), dim=1)
    nll = torch.logsumexp(logits, dim=1) - logmass
    regret = ((costs - best).masked_fill(~mask, 0.0) * logits.softmax(1)).sum(1)
    denom = discriminative.sum().clamp_min(1)
    return nll.masked_fill(~discriminative, 0.0).sum() / denom, regret.masked_fill(~discriminative, 0.0).sum() / denom


def objective(model, features, costs, mask):
    predicted, scores = model(features, mask)
    mse = ((predicted - costs).masked_fill(~mask, 0.0) / model.horizon).square().sum() / mask.sum()
    if model.rank_head is None:
        return mse
    nll, regret = ranking_terms(scores, costs, mask)
    return mse + model.config.ranking_weight * nll + model.config.regret_weight * regret


def save(path, model, provenance):
    payload = {"features": FEATURES, "config": asdict(model.config), "horizon": model.horizon,
               "state": model.state_dict(), "provenance": provenance}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load(path, family, horizon):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if (payload["features"], payload["provenance"]["test"], payload["horizon"]) != (FEATURES, family, horizon):
        raise ValueError(f"Incompatible model checkpoint: {path}")
    state = payload["state"]
    model = EvictionNetwork(NetworkConfig(**payload["config"]), horizon, state["mean"], state["scale"])
    model.load_state_dict(state)
    return model.eval(), payload["provenance"]


@torch.inference_mode()
def predict(model, x, device="cpu"):
    features = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=device)
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
    costs, scores = model(features, mask)
    return costs.cpu().numpy(), scores.cpu().numpy()
