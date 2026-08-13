from __future__ import annotations

from scripts.experiments.run_learned_exact_target_agreement import _fraction, _margin


def test_fraction_and_margin_are_tie_safe():
    assert _fraction(1, 2) == 0.5
    assert _fraction(0, 0) is None
    assert _margin({"a": 1.0, "b": 1.0, "c": 3.0}) == (0.0, 2.0)
    assert _margin({"a": 1.0, "b": 2.0}) == (1.0, 1.0)
