"""Separate future-only supervision and grouped ranking metrics."""

from collections import defaultdict

import numpy as np
from numba import njit
import torch


def next_positions(stream):
    """Each occurrence's next position. The trace length is the right-censor sentinel."""
    found = {}
    out = np.full(len(stream), len(stream), dtype=np.int64)
    for i in range(len(stream) - 1, -1, -1):
        pid = stream[i]
        out[i] = found.get(pid, len(stream))
        found[pid] = i
    return out


@njit(cache=True)
def continuation_labels(candidates, incoming, future):
    """Force a victim, admit incoming, then count LRU misses from subsequent requests."""
    n = len(candidates)
    order = np.empty_like(candidates)
    results = np.zeros((n, 3), dtype=np.uint8)
    for victim in range(n):
        order[:victim] = candidates[:victim]
        order[victim:n-1] = candidates[victim+1:]
        order[-1] = incoming
        misses = 0
        for step in range(32):
            if step < len(future):
                item = future[step]
                j = n - 1
                while j >= 0 and order[j] != item:
                    j -= 1
                if j < 0:
                    misses += 1
                    j = 0
                for k in range(j, n - 1):
                    order[k] = order[k + 1]
                order[-1] = item
            if step == 3:
                results[victim, 0] = misses
            elif step == 15:
                results[victim, 1] = misses
        results[victim, 2] = misses
    return results


def targets(next_at, distances, cf, mask, objective):
    if objective == "T01":
        return -cf
    return next_at


def objective_terms(predicted, next_at, observed, distance, cf, mask, objective, horizon):
    """Heads are victim logit (higher evicts), log reuse, and normalized miss cost."""
    rank, reuse, value = predicted.unbind(-1)
    rank_target = targets(next_at, distance, cf, mask, objective)
    valid = mask.any(-1)
    if not valid.any():
        raise ValueError("No observed decisions in the minibatch")
    rank, reuse, value = rank[valid], reuse[valid], value[valid]
    mask, observed = mask[valid], observed[valid]
    distance, cf, rank_target = distance[valid], cf[valid], rank_target[valid]
    rank_target = rank_target.masked_fill(~mask, -torch.inf)
    optimal = mask & (rank_target == rank_target.amax(-1, keepdim=True))
    logits = rank.masked_fill(~mask, -torch.inf)
    informative = (mask & ~optimal).any(-1)
    log_mass = torch.logsumexp(logits, -1) - torch.logsumexp(rank.masked_fill(~optimal, -torch.inf), -1)
    ranking = log_mass[informative].sum() / informative.sum().clamp_min(1)
    error = reuse - distance
    reuse_loss = (torch.where(observed, error, error.clamp_max(0)).square() * mask).sum() / mask.sum()
    value_loss = ((value - cf / horizon).square() * mask).sum() / mask.sum()
    losses = {
        "T01": (ranking, value_loss),
        "T02": (ranking,),
        "T03": (ranking, 0.5 * reuse_loss, 0.25 * value_loss),
    }
    return sum(losses[objective]), (ranking, reuse_loss, value_loss)


class OfflineMetrics:
    def __init__(self):
        self.totals = defaultdict(float)

    def add(self, predicted, nxt, observed, distance, cf, mask, objective, horizon):
        total, terms = objective_terms(predicted, nxt, observed, distance, cf, mask, objective, horizon)
        valid = mask.any(-1)
        p = predicted[valid]
        m, n, d, o, c = mask[valid], nxt[valid], distance[valid], observed[valid], cf[valid]
        rank_target = targets(n, d, c, m, objective).masked_fill(~m, -torch.inf)
        optimal = m & (rank_target == rank_target.amax(-1, keepdim=True))
        logits = p[..., 0].masked_fill(~m, -torch.inf)
        top = logits.argmax(-1, keepdim=True)
        hit = optimal.gather(-1, top).squeeze(-1)
        informative = (m & ~optimal).any(-1)
        # Binary NDCG uses the exact optimal set as relevant. This remains defined for censored ties.
        order = logits.argsort(dim=-1, descending=True, stable=True)
        discount = torch.log2(torch.arange(2, 2+m.shape[-1], device=m.device)).reciprocal()
        dcg = (optimal.gather(-1, order) * discount).sum(-1)
        ideal = ((torch.arange(m.shape[-1], device=m.device) < optimal.sum(-1, keepdim=True)) * discount).sum(-1)
        preference = rank_target.unsqueeze(-1) > rank_target.unsqueeze(-2)
        preference &= m.unsqueeze(-1) & m.unsqueeze(-2)
        difference = p[..., 0].unsqueeze(-1) - p[..., 0].unsqueeze(-2)
        reuse_error = (p[..., 1] - d).abs()
        value_error = (p[..., 2] * horizon - c).abs()
        regret = c.gather(-1, top).squeeze(-1) - c.masked_fill(~m, torch.inf).amin(-1)
        count = len(p)
        for name, value in dict(
            decisions=count, candidates=m.sum(), informative=informative.sum(), correct=hit.sum(),
            correct_informative=(hit & informative).sum(), ndcg=(dcg / ideal).sum(),
            pairs=preference.sum(), pair_correct=((difference > 0) & preference).sum()
            + 0.5*((difference == 0) & preference).sum(), reuse_count=(m & o).sum(),
            reuse_absolute=(reuse_error * (m & o)).sum(), value_absolute=(value_error*m).sum(),
            value_square=(value_error.square()*m).sum(), regret=regret.sum(),
            loss=total*count, ranking=terms[0]*informative.sum(),
            reuse_loss=terms[1]*m.sum(), value_loss=terms[2]*m.sum(),
        ).items():
            self.totals[name] += float(value)

    def report(self):
        t = self.totals
        def ratio(x, y):
            return t[x] / t[y] if t[y] else None
        return dict(decisions=int(t["decisions"]), candidates=int(t["candidates"]),
                    loss=ratio("loss", "decisions"), ranking_loss=ratio("ranking", "informative"),
                    optimal_set=ratio("correct", "decisions"), ndcg=ratio("ndcg", "decisions"),
                    discriminative_accuracy=ratio("correct_informative", "informative"),
                    pair_accuracy=ratio("pair_correct", "pairs"), mean_cf_regret=ratio("regret", "decisions"),
                    mean_log_reuse_error=ratio("reuse_absolute", "reuse_count"),
                    cf_mae=ratio("value_absolute", "candidates"),
                    cf_rmse=ratio("value_square", "candidates")**0.5 if t["candidates"] else None,
                    censored=int(t["candidates"]-t["reuse_count"]),
                    tied_fraction=1-t["informative"]/t["decisions"] if t["decisions"] else None,
                    reuse_loss=ratio("reuse_loss", "candidates"), value_loss=ratio("value_loss", "candidates"))
