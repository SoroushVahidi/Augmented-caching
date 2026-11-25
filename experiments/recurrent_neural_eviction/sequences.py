"""Contiguous request records with separately constructed decision supervision."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from features import CausalHistory, GLOBAL_NAMES, STATIC_NAMES
from supervision import continuation_labels, next_positions

HORIZONS = (4, 16, 32)


@dataclass
class Block:
    requests: np.ndarray
    times: list
    features: list
    next_at: list
    observed: list
    distances: list
    values: list
    priority: list


def order_transform(sequence, order):
    # A pair is available only after both observations occurred. Delay by two positions.
    if order == "ordered":
        return sequence
    result = np.zeros_like(sequence)
    if order == "permuted":
        length = (len(sequence) - 2) // 2 * 2
        result[2:2+length] = sequence[:length].reshape(-1, 2, sequence.shape[-1])[:, ::-1].reshape(-1, sequence.shape[-1])
        if 2+length < len(sequence):
            result[-1] = sequence[-2]
    else:
        result[2:] = sequence[:-2]
    return result


def build_blocks(stream, capacity, *, family, split, score_start, end, bins, decisions, burn_in=128,
                 unroll=128, seed=421, legacy_root=None, policy=None, selection="uniform"):
    if not 0 <= score_start < end <= len(stream) or burn_in < 1:
        raise ValueError("Invalid request interval")
    next_at = next_positions(stream[:end])
    rng = np.random.default_rng(seed)
    starts = np.arange(max(score_start, burn_in), end - unroll + 1, unroll)
    if not len(starts):
        raise ValueError(f"Interval too short for a contiguous block: {family}/{capacity}/{split}")
    starts = np.sort(rng.choice(starts, size=min(bins, len(starts)), replace=False))
    scheduled = {int(t): i for i, start in enumerate(starts) for t in range(start, start + unroll)}
    block = [Block(np.empty((burn_in+unroll, len(GLOBAL_NAMES)), np.float32), [], [], [], [], [], [], [])
             for _ in starts]
    x_global = np.empty((end, len(GLOBAL_NAMES)), np.float32)
    legacy = None
    if legacy_root is not None and policy is None and end == len(stream):
        cell = Path(legacy_root) / ("{}-{}".format(family, capacity))
        if (cell / "t.npy").exists():
            legacy = (np.load(cell / "t.npy", mmap_mode="r"), np.load(cell / "y.npy", mmap_mode="r"))
    # Uniformly seed a candidate set of times. Take every available scheduled decision
    # for aggregation, then prefer a bounded ranking of consequential disagreements.
    wanted = {}
    if policy is None:
        for i, start in enumerate(starts):
            wanted[i] = set(rng.choice(np.arange(start, start + unroll),
                                       min(unroll, decisions * 3), replace=False).tolist())
    residents = OrderedDict()
    state = CausalHistory(capacity)
    if policy is not None:
        policy.reset()
    last = {}
    for t, pid in enumerate(stream[:int(starts[-1])+unroll]):
        hit = pid in residents
        observation = state.before(pid, t, hit)
        x_global[t] = observation
        if policy is not None:
            policy.update(observation)
        victim = None
        if hit:
            residents.move_to_end(pid)
        elif len(residents) == capacity:
            index = scheduled.get(t)
            capture = index is not None and (policy is not None or t in wanted[index])
            victims = list(residents) if capture or policy is not None else None
            features = state.candidates(victims, t) if victims is not None else None
            selected = 0 if policy is None else int(np.argmax(policy.score(features, victims, state)))
            victim = next(iter(residents)) if victims is None else victims[selected]
            if capture:
                right = next_at[[last[p] for p in victims]]
                c = continuation_labels(np.asarray(victims, dtype=np.int64), pid,
                                        np.asarray(stream[t+1:min(end,t+33)], dtype=np.int64))
                if legacy is not None:
                    li = np.searchsorted(legacy[0], t)
                    if li == len(legacy[0]) or legacy[0][li] != t:
                        raise ValueError("Matched November supervision lacks an LRU decision")
                    exact = legacy[1][li, :, (0, 2)].T
                    if exact.shape != c[:, :2].shape or not np.array_equal(exact, c[:, :2]):
                        raise ValueError("November target or candidate-order discrepancy")
                    c[:, :2] = exact
                b = block[index]
                b.times.append(t - int(starts[index]) + burn_in)
                b.features.append(features)
                b.next_at.append(right.astype(np.float32))
                b.observed.append(right < end)
                b.distances.append(np.log1p(right - t).astype(np.float32))
                b.values.append(c)
                best = int(right.max())
                delta = np.log1p(best-t) - np.log1p(right[selected]-t)
                b.priority.append((int(victim != victims[0]) * (1 + int(c[selected, 1])-int(c[:, 1].min())) + delta)
                                  if policy is not None else 0.0)
            del residents[victim]
        if not hit:
            residents[pid] = None
        state.after(pid, t, hit, victim)
        last[pid] = t
    result = []
    for b, start in zip(block, starts):
        if not b.times:
            continue
        b.requests[:] = x_global[start-burn_in:start+unroll]
        if selection == "targeted":
            subset = sorted(np.argsort(np.asarray(b.priority), kind="stable")[-decisions:])
        else:
            subset = sorted(rng.choice(len(b.times), min(decisions, len(b.times)), replace=False))
        for attr in ("times", "features", "next_at", "observed", "distances", "values", "priority"):
            values = getattr(b, attr)
            setattr(b, attr, [values[i] for i in subset])
        result.append(b)
    return result


def collate(blocks, device, horizon, order):
    count = len(blocks)
    n = max(len(b.times) for b in blocks)
    cap = max(b.features[0].shape[0] for b in blocks)
    seq = np.asarray([order_transform(b.requests, order) for b in blocks])
    times = np.full((count, n), len(seq[0])-1, dtype=np.int64)
    x = np.zeros((count, n, cap, len(STATIC_NAMES)), dtype=np.float32)
    y = np.zeros((4, count, n, cap), dtype=np.float32)
    mask = np.zeros((count, n, cap), dtype=bool)
    for i, b in enumerate(blocks):
        for j, t in enumerate(b.times):
            length = len(b.features[j])
            times[i, j] = t
            x[i,j,:length] = b.features[j]
            y[0,i,j,:length] = b.next_at[j]
            y[1,i,j,:length] = b.observed[j]
            y[2,i,j,:length] = b.distances[j]
            y[3,i,j,:length] = b.values[j][:, HORIZONS.index(horizon)]
            mask[i,j,:length] = True
    tensor = lambda a: torch.from_numpy(a).to(device)
    return (tensor(seq), tensor(times), tensor(x), tensor(mask), tensor(y[0]), tensor(y[1]).bool(),
            tensor(y[2]), tensor(y[3]))
