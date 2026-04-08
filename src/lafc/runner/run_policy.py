"""
Simulator runner and CLI entry-point.

Usage
-----
python -m lafc.runner.run_policy \\
    --policy   la_det \\
    --trace    data/example.json \\
    --capacity 3 \\
    --output-dir output/

Supported --policy values:
    lru, weighted_lru, advice_trusting, la_det,
    marker, blind_oracle, follow_the_prediction,
    predictive_marker, trust_and_doubt
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from typing import Dict, List

from lafc.metrics.cost import hit_rate, total_fetch_cost, total_hits, total_misses
from lafc.metrics.prediction_error import (
    compute_discrete_eta,
    compute_eta,
    compute_eta_unweighted,
    compute_weighted_surprises,
)
from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.base import BasePolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.follow_the_prediction import FollowThePredictionPolicy
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.policies.weighted_lru import WeightedLRUPolicy
from lafc.simulator.request_trace import load_trace
from lafc.types import Page, PageId, Request, SimulationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------

POLICY_REGISTRY: Dict[str, BasePolicy] = {
    "lru": LRUPolicy(),
    "weighted_lru": WeightedLRUPolicy(),
    "advice_trusting": AdviceTrustingPolicy(),
    "la_det": LAWeightedPagingDeterministic(),
    # Baseline 2: Lykouris & Vassilvitskii 2018 (unweighted paging)
    "marker": MarkerPolicy(),
    "blind_oracle": BlindOraclePolicy(),
    "follow_the_prediction": FollowThePredictionPolicy(),
    "predictive_marker": PredictiveMarkerPolicy(),
    # Baseline 3: Antoniadis et al. ICML 2020 (TRUST&DOUBT)
    "trust_and_doubt": TrustAndDoubtPolicy(),
}


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------


def run_policy(
    policy: BasePolicy,
    requests: List[Request],
    pages: Dict[PageId, Page],
    capacity: int,
) -> SimulationResult:
    """Run *policy* on *requests* and return the aggregated result.

    Parameters
    ----------
    policy:
        A :class:`~lafc.policies.base.BasePolicy` instance.
    requests:
        Ordered list of requests (with predictions and actual_next filled in).
    pages:
        Page weight dictionary.
    capacity:
        Cache capacity (number of pages).

    Returns
    -------
    :class:`~lafc.types.SimulationResult`
        Aggregated simulation result including per-step events.
    """
    if not requests:
        raise ValueError("Request list must not be empty")
    if capacity <= 0:
        raise ValueError(f"Cache capacity must be >= 1, got {capacity}")

    logger.info(
        "Running policy '%s' on %d requests with capacity=%d",
        policy.name,
        len(requests),
        capacity,
    )

    policy.reset(capacity, pages)
    events = []

    for req in requests:
        event = policy.on_request(req)
        events.append(event)
        logger.debug(
            "t=%d page=%s hit=%s cost=%.2f evicted=%s",
            req.t,
            req.page_id,
            event.hit,
            event.cost,
            event.evicted,
        )

    result = SimulationResult(
        policy_name=policy.name,
        total_cost=total_fetch_cost(events),
        total_hits=total_hits(events),
        total_misses=total_misses(events),
        events=events,
    )

    # Compute prediction error metrics.
    try:
        result.prediction_error_eta = compute_eta(requests, pages)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not compute eta: %s", exc)

    try:
        result.prediction_error_surprises = compute_weighted_surprises(requests, pages)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not compute weighted surprises: %s", exc)

    # For unweighted policies (Baselines 2 and 3), also expose the unweighted η.
    if isinstance(policy, (MarkerPolicy, BlindOraclePolicy, FollowThePredictionPolicy,
                            PredictiveMarkerPolicy, TrustAndDoubtPolicy)):
        try:
            eta_unweighted = compute_eta_unweighted(requests)
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["eta_unweighted"] = (
                None if (eta_unweighted is not None and math.isinf(eta_unweighted))
                else eta_unweighted
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute unweighted eta: %s", exc)

        try:
            eta_discrete = compute_discrete_eta(requests)
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["eta_discrete"] = eta_discrete
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute discrete eta: %s", exc)

    # For Predictive Marker, collect clean-chain diagnostics.
    if isinstance(policy, PredictiveMarkerPolicy):
        try:
            if requests:
                policy.close_final_phase(requests[-1].t)
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["clean_chains"] = policy.compute_clean_chains()
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute clean chains: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_summary(result: SimulationResult, output_dir: str) -> None:
    summary = {
        "policy_name": result.policy_name,
        "total_cost": result.total_cost,
        "total_hits": result.total_hits,
        "total_misses": result.total_misses,
        "hit_rate": hit_rate(result.events),
    }
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary saved to %s", path)


def _save_metrics(result: SimulationResult, output_dir: str) -> None:
    eta = result.prediction_error_eta
    metrics: dict = {
        "prediction_error_eta": None if (eta is not None and math.isinf(eta)) else eta,
    }
    if result.prediction_error_surprises:
        surp = result.prediction_error_surprises
        metrics["total_weighted_surprise"] = surp.get("total_weighted_surprise")
        metrics["total_surprises"] = surp.get("total_surprises")
        metrics["per_class_surprises"] = surp.get("per_class")

    # Include unweighted η, discrete η, and clean-chain diagnostics when present.
    if result.extra_diagnostics:
        if "eta_unweighted" in result.extra_diagnostics:
            metrics["eta_unweighted"] = result.extra_diagnostics["eta_unweighted"]
        if "eta_discrete" in result.extra_diagnostics:
            metrics["eta_discrete"] = result.extra_diagnostics["eta_discrete"]
        if "clean_chains" in result.extra_diagnostics:
            cc = result.extra_diagnostics["clean_chains"]
            metrics["num_clean_phases"] = cc.get("num_clean_phases")
            metrics["num_dirty_phases"] = cc.get("num_dirty_phases")
            metrics["num_clean_chains"] = cc.get("num_clean_chains")
            metrics["total_clean_evictions"] = cc.get("total_clean_evictions")
            metrics["total_dirty_evictions"] = cc.get("total_dirty_evictions")

    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", path)


def _save_per_step(result: SimulationResult, output_dir: str) -> None:
    path = os.path.join(output_dir, "per_step_decisions.csv")
    has_phase = any(e.phase is not None for e in result.events)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if has_phase:
            writer.writerow(["t", "page_id", "hit", "cost", "evicted", "phase"])
            for e in result.events:
                writer.writerow(
                    [e.t, e.page_id, e.hit, e.cost, e.evicted or "", e.phase or ""]
                )
        else:
            writer.writerow(["t", "page_id", "hit", "cost", "evicted"])
            for e in result.events:
                writer.writerow([e.t, e.page_id, e.hit, e.cost, e.evicted or ""])
    logger.info("Per-step decisions saved to %s", path)


def save_results(result: SimulationResult, output_dir: str) -> None:
    """Save all output files to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    _save_summary(result, output_dir)
    _save_metrics(result, output_dir)
    _save_per_step(result, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run a caching policy on a trace.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=list(POLICY_REGISTRY.keys()),
        help="Policy to run.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to a JSON trace file.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=3,
        help="Cache capacity (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output/).",
    )
    parser.add_argument(
        "--perfect-predictions",
        action="store_true",
        help=(
            "If set, replace predictions with perfect oracle predictions "
            "computed from the trace."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    requests, pages = load_trace(args.trace)

    if args.perfect_predictions:
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        requests = compute_perfect_predictions(requests)

    policy = POLICY_REGISTRY[args.policy]
    result = run_policy(policy, requests, pages, args.capacity)
    save_results(result, args.output_dir)

    print(f"Policy:      {result.policy_name}")
    print(f"Total cost:  {result.total_cost}")
    print(f"Hits:        {result.total_hits}")
    print(f"Misses:      {result.total_misses}")
    print(f"Hit rate:    {hit_rate(result.events):.2%}")
    print(f"Output written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
