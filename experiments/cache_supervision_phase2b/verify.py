"""Bounded correctness and steady-state labeler benchmark."""
import csv
from itertools import permutations, product
import json
from pathlib import Path
import resource
from statistics import median
import sys
from time import perf_counter
import tracemalloc

from numba import njit
import numpy as np

from exact_lru import rollout

HERE = Path(__file__).resolve().parent
sys.path.extend(str(HERE.parent / name) for name in
                ("cache_supervision_forensics_phase2a", "neural_eviction_20261121"))
from sources import CAPACITIES, PROTOCOL, stream_for
from data import labels
from math_core import lru_step, forced_state, paired_lru
from lafc.evict_value_v2_rollout import simulate_rollout_misses
from lafc.types import Request

OUT = HERE.parents[1] / "research-forensics/2026-11-25-Phase-2B-Exact-LRU"
AUDIT = HERE.parents[1].parent / "cache-supervision-forensics-phase2a/experiments/cache_supervision_forensics_phase2a/runs/phase2a"
SEED = 20261125


def write_csv(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@njit(cache=True)
def exhaustive_labels(residents, incoming, future, horizons):
    """Generalize November branch replay to arbitrary horizons and counts."""
    misses = np.zeros((len(residents), len(horizons)), dtype=np.int64)
    for k in range(len(residents)):
        order = forced_state(residents, incoming, k)
        cursor = 0
        total = 0
        for j, horizon in enumerate(horizons):
            stop = min(len(future), horizon)
            while cursor < stop:
                total += lru_step(order, future[cursor])
                cursor += 1
            misses[k, j] = total
    return misses


def simulator(residents, incoming, future):
    requests = [Request(i, p) for i, p in enumerate(future)]
    return [simulate_rollout_misses(
        cache_pages=(*residents[:i], *residents[i+1:], incoming),
        future_reqs=requests, capacity=len(residents), reference_policy="lru",
    ) for i in range(len(residents))]


def check(residents, incoming, future, horizons, check_sim=True):
    h = np.unique(horizons)
    results, _ = rollout(residents, incoming, future, h)
    actual = np.asarray([results[k].misses for k in h]).T
    a, b = np.asarray(residents, dtype=np.int64), np.asarray(future, dtype=np.int64)
    np.testing.assert_array_equal(actual, exhaustive_labels(a, incoming, b, h))
    positive = h > 0
    if positive.any():
        paired, _, _, _, baseline = paired_lru(a, incoming, b, h[positive], int(h[-1]))
        np.testing.assert_array_equal(actual[:, positive], baseline - paired)
    for k, row in results.items():
        assert min(row.advantage) >= -1
        assert row.misses.count(min(row.misses)) >= max(1, len(residents) - min(len(future), k))
        if check_sim:
            np.testing.assert_array_equal(row.misses, simulator(residents, incoming, future[:k]))
        if len(set(future[:k])) >= len(residents):
            assert row.active == 0
    return results


def correctness():
    start = perf_counter()
    for cap in (1, 2, 8):
        resident = tuple(range(cap))
        future = tuple(range(cap + 2)) * 3
        horizons = (0, 1, 4, len(future) + 1)
        expected, _ = rollout(resident, cap, future, horizons)
        checked, early = rollout(resident, cap, future, horizons, stop_when_resolved=True)
        assert checked == expected and early.active == 0
        _, early = rollout(resident, cap, future, stop_when_resolved=True)
        assert early.advantage == expected[horizons[-1]].advantage
    exhaustive = 0
    for cap in (1, 2, 3):
        for options in permutations(range(cap)):
            for length in range(6):
                for future in product(range(cap + 2), repeat=length):
                    check(options, cap, future, tuple(range(length + 2)))
                    exhaustive += 1
    rng = np.random.default_rng(SEED)
    counts = {}
    for i in range(10000):
        cap = (1, 2, 4, 8, 32, 64, 128, 256)[i % 8]
        residents = tuple(map(int, rng.permutation(cap)))
        length = int(rng.integers(0, 35))
        future = tuple(map(int, rng.integers(0, cap + 7, size=length)))
        check(residents, cap, future, (0, 1, 4, 8, 16, length + 2), check_sim=cap <= 8 or i % 97 == 0)
        counts[cap] = counts.get(cap, 0) + 1
    large = 0
    for cap in (1, 2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048):
        residents = tuple(reversed(range(cap)))
        for seq in ([cap] * 4096, list(range(cap, cap + 1024)), list(range(cap + 1)) * 2,
                    list(range(0, cap, 2)) + [cap] * 100):
            check(residents, cap, tuple(seq), (0, 4, 8, 32, min(len(seq), 512), len(seq) + 7), check_sim=cap <= 32)
            large += 1
    write_json("LABELER_CORRECTNESS_TESTS.json", dict(
        unexplained_mismatches=0, exhaustive_case_count=exhaustive, randomized_case_count=sum(counts.values()),
        randomized_by_capacity=counts, structured_case_count=large, elapsed_seconds=perf_counter()-start,
        seed=SEED, authoritative_simulator="lafc.evict_value_v2_rollout.simulate_rollout_misses",
        compiled_comparators=["generalized November exhaustive", "Phase 2A paired ordered coalescence"],
        legacy_horizon_zero_issue="Phase 2A paired replay starts recording at request one. Its archived horizons are positive.",
        archive_validation="recorded separately after verified joins"))


def benchmark():
    rng = np.random.default_rng(SEED)
    rows = []
    for cap in (32, 64, 128, 256, 512, 1024):
        for horizon in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            horizons = np.array([horizon])
            cases = []
            for _ in range(3):
                a = rng.permutation(cap)
                b = rng.integers(0, cap * 2, size=horizon, dtype=np.int64)
                cases.append((a, b, tuple(a.tolist()), tuple(b.tolist())))
            methods = {
                "shared_ordered_dict": lambda a, b, p, q: rollout(p, cap, q, horizons)[1].misses,
                "november_style_exhaustive_compiled": lambda a, b, p, q: exhaustive_labels(a, cap, b, horizons)[:, 0],
                "phase2a_coalescence_compiled": lambda a, b, p, q: paired_lru(a, cap, b, horizons, horizon),
                "repository_simulator": lambda a, b, p, q: simulator(p, cap, q),
            }
            for name, calculate in methods.items():
                if name == "repository_simulator" and cap * horizon > 32768:
                    continue
                calculate(*cases[0])
                seconds = []
                for _ in range(3):
                    started = perf_counter()
                    for case in cases:
                        calculate(*case)
                    seconds.append((perf_counter() - started) / len(cases))
                duration = median(seconds)
                tracemalloc.start()
                calculate(*cases[0])
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append(dict(capacity=cap,horizon=horizon,method=name,cases=len(cases), repeats=3,
                                 seconds_per_decision=duration,decisions_per_second=1/duration,
                                 candidate_labels_per_second=cap/duration, traced_peak_bytes=peak,
                                 reference_matrix_bytes=(horizon+1)*cap*8 if name.startswith("phase2a") else 0,
                                 process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_csv("LABELER_BENCHMARK.csv", rows)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        correctness()
    elif sys.argv[1] == "benchmark":
        benchmark()
