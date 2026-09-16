from __future__ import annotations

import numpy as np

from scripts.pe_publication_learned_retrain import SplitRange, stable_sample, strict_pair_counts


def test_split_range_respects_purge_gaps() -> None:
    sr = SplitRange(
        family="f",
        trace_name="t",
        capacity=32,
        t_min=0,
        t_max=99,
        train_end=59,
        val_start=65,
        val_end=79,
        test_start=85,
        purge_gap_requests=5,
    )
    assert sr.assign(59) == "train"
    assert sr.assign(60) == "purge"
    assert sr.assign(64) == "purge"
    assert sr.assign(65) == "validation"
    assert sr.assign(80) == "purge"
    assert sr.assign(84) == "purge"
    assert sr.assign(85) == "test"


def test_stable_sample_is_repeatable_and_keeps_requested_size() -> None:
    items = [f"d{i}" for i in range(20)]
    first = stable_sample(items, 7, 123, "salt")
    second = stable_sample(list(reversed(items)), 7, 123, "salt")
    assert first == second
    assert len(first) == 7


def test_strict_pair_counts_ignores_equal_targets() -> None:
    y = np.asarray([0.0, 0.0, 1.0, 2.0])
    pred = np.asarray([0.9, 0.1, 0.2, 3.0])
    correct, total = strict_pair_counts(y, pred)
    assert total == 5
    assert correct == 4
