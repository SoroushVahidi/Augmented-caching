"""Candidate scorers with causal request encoders and shared object history encoders."""

from dataclasses import dataclass

import torch
from torch import nn

from features import GLOBAL_NAMES, STATIC_NAMES, object_sequence


@dataclass(frozen=True)
class Architecture:
    name: str
    global_hidden: int = 64
    object_hidden: int = 32
    request_history: int = 128
    object_history: int = 8
    width: int = 64
    dropout: float = 0.10
    order: str = "ordered"
    reset_hidden: bool = False


def block(incoming, outgoing, dropout):
    return nn.Sequential(nn.Linear(incoming, outgoing), nn.GELU(), nn.Dropout(dropout))


class VictimNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        name = config.name
        if name not in {f"R{i:02}" for i in range(1, 9)}:
            raise ValueError(name)
        recurrent = nn.LSTM if name in {"R06", "R07"} else nn.GRU
        self.global_rnn = recurrent(len(GLOBAL_NAMES), config.global_hidden, batch_first=True) if name in {
            "R03", "R05", "R06", "R07"
        } else None
        self.object_rnn = recurrent(2, config.object_hidden, batch_first=True) if name in {
            "R04", "R05", "R06", "R07"
        } else None
        self.history_mlp = block(config.request_history * len(GLOBAL_NAMES), config.global_hidden, config.dropout) if name == "R08" else None
        self.object_mlp = block(config.object_history * 2, config.object_hidden, config.dropout) if name == "R08" else None
        self.attention = name == "R07"
        self.use_pool = name != "R01"
        self.candidate = block(len(STATIC_NAMES), config.width, config.dropout)
        if self.attention:
            self.query = nn.Linear(config.width + config.object_hidden, config.global_hidden, bias=False)
            self.key = nn.Linear(config.global_hidden, config.global_hidden, bias=False)
        object_size = config.object_hidden if self.object_rnn or self.object_mlp else 0
        global_size = config.global_hidden if self.global_rnn or self.history_mlp else len(GLOBAL_NAMES)
        incoming = config.width + object_size + global_size
        if self.attention:
            incoming += config.global_hidden
        self.context = block(incoming, config.width, config.dropout) if self.use_pool else None
        self.fusion = nn.Sequential(block(incoming + (config.width if self.use_pool else 0), 128, config.dropout),
                                    block(128, 64, config.dropout))
        self.heads = nn.Linear(64, 3)

    def global_outputs(self, sequence, state=None):
        if self.global_rnn is None:
            return sequence, None
        if self.config.reset_hidden:
            batch, steps, width = sequence.shape
            out, _ = self.global_rnn(sequence.reshape(batch * steps, 1, width))
            return out.reshape(batch, steps, -1), None
        return self.global_rnn(sequence, state)

    def object_encode(self, values, known):
        if self.object_rnn is None and self.object_mlp is None:
            return None
        if self.config.order == "permuted":
            # Reverse only observed events, retaining padding and every marginal value.
            n = values.shape[-1]
            positions = torch.arange(n, device=values.device)
            count = known.sum(-1, keepdim=True).long()
            index = torch.where(known.bool(), (2*n - count - 1 - positions), positions)
            values = values.gather(-1, index)
        observations = torch.stack((values, known), dim=-1)
        if self.object_mlp:
            return self.object_mlp(observations.flatten(-2))
        flat = observations.reshape(-1, self.config.object_history, 2)
        result, _ = self.object_rnn(flat)
        representation = result[:, -1].reshape(values.shape[:-1] + (-1,))
        return representation * known.any(-1, keepdim=True)

    def scores(self, x, mask, current, history=None, history_mask=None, objects=None):
        x = x.masked_fill(~mask[..., None], 0)
        features = self.candidate(x)
        if objects is None:
            objects = self.object_encode(*object_sequence(x, self.config.object_history))
        context = current
        if self.history_mlp:
            context = self.history_mlp(history.flatten(-2))
        parts = [features, context.unsqueeze(-2).expand(*features.shape[:-1], context.shape[-1])]
        if objects is not None:
            parts.append(objects)
        if self.attention:
            q = self.query(torch.cat((features, objects), dim=-1))
            attention = (q @ self.key(history).transpose(-2, -1)) / self.config.global_hidden ** 0.5
            weights = attention.masked_fill(~history_mask.unsqueeze(-2), -torch.inf).softmax(-1)
            parts.append(weights @ history)
        joined = torch.cat(parts, dim=-1)
        if self.use_pool:
            projected = self.context(joined).masked_fill(~mask[..., None], 0)
            pooled = projected.sum(-2, keepdim=True) / mask.sum(-1).clamp_min(1)[..., None, None]
            joined = torch.cat((joined, pooled.expand(*joined.shape[:-1], pooled.shape[-1])), dim=-1)
        return self.heads(self.fusion(joined))

    def forward(self, sequence, times, x, mask, burn_in):
        if burn_in:
            with torch.no_grad():
                prefix, state = self.global_outputs(sequence[:, :burn_in])
            suffix, _ = self.global_outputs(sequence[:, burn_in:], state)
            output = torch.cat((prefix, suffix), dim=1)
        else:
            output, _ = self.global_outputs(sequence)
        rows = torch.arange(len(times), device=times.device)[:, None]
        current = output[rows, times]
        history = history_mask = None
        if self.attention or self.history_mlp:
            offsets = torch.arange(1-self.config.request_history, 1, device=times.device)
            indices = times[..., None] + offsets
            history_mask = indices >= 0
            history = output[rows[..., None], indices.clamp_min(0)]
            history = history.masked_fill(~history_mask[..., None], 0)
        return self.scores(x, mask, current, history, history_mask)
