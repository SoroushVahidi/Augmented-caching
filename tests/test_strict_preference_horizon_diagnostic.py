from __future__ import annotations

from scripts.experiments.run_strict_preference_horizon_diagnostic import _jaccard, _fraction


def test_optimal_set_metrics_are_tie_safe():
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert _jaccard(set(), set()) == 1.0
    assert _fraction(1, 2) == 0.5
    assert _fraction(0, 0) is None
