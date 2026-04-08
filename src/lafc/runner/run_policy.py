"""
Simulator runner and CLI entry-point.

Usage
-----
python -m lafc.runner.run_policy \\
    --policy   blind_oracle_lru_combiner \\
    --trace    data/example_unweighted.json \\
    --capacity 3 \\
    --output-dir output/

Supported --policy values:
    lru, weighted_lru, advice_trusting, la_det,
    marker, blind_oracle, predictive_marker,
    blind_oracle_lru_combiner, offline_belady, trust_and_doubt, atlas_v1, atlas_v2
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
    compute_cache_state_error,
    compute_eta,
    compute_eta_unweighted,
    compute_weighted_surprises,
)
from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.base import BasePolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.policies.la_weighted_paging_det_faithful import LAWeightedPagingDeterministicFaithful
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.offline_belady import OfflineBeladyPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.weighted_lru import WeightedLRUPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.buckets import attach_perfect_buckets, maybe_corrupt_buckets
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
    # Historical interpreted heuristic (kept for backward compatibility).
    "la_det": LAWeightedPagingDeterministic(),
    "la_det_approx": LAWeightedPagingDeterministic(),
    # Faithful-style class-level deterministic algorithm (SODA'22 baseline).
    "la_det_faithful": LAWeightedPagingDeterministicFaithful(),
    # Baseline 2: Lykouris & Vassilvitskii 2018 (unweighted paging)
    "marker": MarkerPolicy(),
    "blind_oracle": BlindOraclePolicy(),
    "predictive_marker": PredictiveMarkerPolicy(),
    # Baseline 4: Wei 2020 (unweighted paging)
    "blind_oracle_lru_combiner": BlindOracleLRUCombiner(),
    "offline_belady": OfflineBeladyPolicy(),
    # Baseline 3: Antoniadis et al. 2020
    "trust_and_doubt": TrustAndDoubtPolicy(),
    # Experimental framework policy (unweighted).
    "atlas_v1": AtlasV1Policy(),
    "atlas_v2": AtlasV2Policy(),
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

    # For unweighted policies (Baseline 2 and 4), also expose the unweighted η.
    if isinstance(
        policy,
        (
            MarkerPolicy,
            BlindOraclePolicy,
            PredictiveMarkerPolicy,
            BlindOracleLRUCombiner,
            OfflineBeladyPolicy,
            TrustAndDoubtPolicy,
            AtlasV1Policy,
            AtlasV2Policy,
        ),
    ):
        try:
            eta_unweighted = compute_eta_unweighted(requests)
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["eta_unweighted"] = (
                None if (eta_unweighted is not None and math.isinf(eta_unweighted))
                else eta_unweighted
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute unweighted eta: %s", exc)

    # For Predictive Marker, collect clean-chain diagnostics.
    if isinstance(policy, PredictiveMarkerPolicy):
        try:
            if requests:
                policy.close_final_phase(requests[-1].t)
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["clean_chains"] = policy.compute_clean_chains()
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not compute clean chains: %s", exc)

    # MTS-style cache-state prediction error (used by TRUST&DOUBT).
    try:
        state_err = compute_cache_state_error(requests, capacity)
        if state_err.get("total_error") is not None:
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["cache_state_error"] = state_err
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not compute cache-state error: %s", exc)

    # For BlindOracleLRUCombiner, collect per-step combiner decision log.
    if isinstance(policy, BlindOracleLRUCombiner):
        try:
            result.extra_diagnostics = result.extra_diagnostics or {}
            result.extra_diagnostics["combiner_step_log"] = [
                {
                    "t": s.t,
                    "page_id": s.page_id,
                    "hit": s.hit,
                    "evicted": s.evicted,
                    "chosen": s.chosen,
                    "bo_misses_before": s.bo_misses_before,
                    "lru_misses_before": s.lru_misses_before,
                }
                for s in policy.step_log()
            ]
            result.extra_diagnostics["shadow_bo_total_misses"] = (
                policy.shadow_bo_misses()
            )
            result.extra_diagnostics["shadow_lru_total_misses"] = (
                policy.shadow_lru_misses()
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not collect combiner step log: %s", exc)

    # Experimental atlas_v1 diagnostics.
    if isinstance(policy, AtlasV1Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["atlas_v1"] = {
            "summary": policy.diagnostics_summary(),
            "decision_log": [
                {
                    "t": d.t,
                    "request_page": d.request_page,
                    "chosen_eviction": d.chosen_eviction,
                    "candidate_buckets": d.candidate_buckets,
                    "candidate_confidences": d.candidate_confidences,
                    "candidate_lambdas": d.candidate_lambdas,
                    "candidate_base_scores": d.candidate_base_scores,
                    "candidate_pred_scores": d.candidate_pred_scores,
                    "candidate_combined_scores": d.candidate_combined_scores,
                    "decision_mode": d.decision_mode,
                }
                for d in policy.decision_log()
            ],
        }
    # Experimental atlas_v2 diagnostics.
    if isinstance(policy, AtlasV2Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["atlas_v2"] = {
            "summary": policy.diagnostics_summary(),
            "time_series": policy.time_series_diagnostics(),
            "decision_log": [
                {
                    "t": d.t,
                    "request_page": d.request_page,
                    "chosen_eviction": d.chosen_eviction,
                    "candidate_buckets": d.candidate_buckets,
                    "candidate_confidences": d.candidate_confidences,
                    "candidate_lambdas": d.candidate_lambdas,
                    "candidate_base_scores": d.candidate_base_scores,
                    "candidate_pred_scores": d.candidate_pred_scores,
                    "candidate_combined_scores": d.candidate_combined_scores,
                    "decision_mode": d.decision_mode,
                    "gamma_before": d.gamma_before,
                    "mismatch_rate": d.mismatch_rate,
                    "tie_break_used": d.tie_break_used,
                    "tie_break_mode": d.tie_break_mode,
                }
                for d in policy.decision_log()
            ],
        }

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

    # Include unweighted η, clean-chain diagnostics, and combiner diagnostics when present.
    if result.extra_diagnostics:
        if "eta_unweighted" in result.extra_diagnostics:
            metrics["eta_unweighted"] = result.extra_diagnostics["eta_unweighted"]
        if "clean_chains" in result.extra_diagnostics:
            cc = result.extra_diagnostics["clean_chains"]
            metrics["num_clean_phases"] = cc.get("num_clean_phases")
            metrics["num_dirty_phases"] = cc.get("num_dirty_phases")
            metrics["num_clean_chains"] = cc.get("num_clean_chains")
            metrics["total_clean_evictions"] = cc.get("total_clean_evictions")
            metrics["total_dirty_evictions"] = cc.get("total_dirty_evictions")
        if "shadow_bo_total_misses" in result.extra_diagnostics:
            metrics["shadow_bo_total_misses"] = result.extra_diagnostics[
                "shadow_bo_total_misses"
            ]
        if "shadow_lru_total_misses" in result.extra_diagnostics:
            metrics["shadow_lru_total_misses"] = result.extra_diagnostics[
                "shadow_lru_total_misses"
            ]
        if "cache_state_error" in result.extra_diagnostics:
            metrics["cache_state_error_total"] = result.extra_diagnostics["cache_state_error"].get("total_error")
        if "atlas_v1" in result.extra_diagnostics:
            atlas = result.extra_diagnostics["atlas_v1"]
            for key, value in atlas.get("summary", {}).items():
                metrics[f"atlas_{key}"] = value
        if "atlas_v2" in result.extra_diagnostics:
            atlas2 = result.extra_diagnostics["atlas_v2"]
            for key, value in atlas2.get("summary", {}).items():
                metrics[f"atlas_v2_{key}"] = value


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


def _save_combiner_decisions(result: SimulationResult, output_dir: str) -> None:
    """Save the combiner's per-step sub-algorithm choices to a separate CSV.

    Written only when ``result.extra_diagnostics`` contains
    ``"combiner_step_log"`` (i.e. the policy was BlindOracleLRUCombiner).
    """
    if not result.extra_diagnostics:
        return
    step_log = result.extra_diagnostics.get("combiner_step_log")
    if not step_log:
        return

    path = os.path.join(output_dir, "combiner_decisions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "t",
                "page_id",
                "hit",
                "evicted",
                "chosen",
                "bo_misses_before",
                "lru_misses_before",
            ]
        )
        for s in step_log:
            writer.writerow(
                [
                    s["t"],
                    s["page_id"],
                    s["hit"],
                    s["evicted"] or "",
                    s["chosen"] or "",
                    s["bo_misses_before"],
                    s["lru_misses_before"],
                ]
            )
    logger.info("Combiner decisions saved to %s", path)


def _save_atlas_diagnostics(result: SimulationResult, output_dir: str) -> None:
    """Save atlas diagnostics when present."""
    if not result.extra_diagnostics:
        return
    atlas = result.extra_diagnostics.get("atlas_v1")
    if atlas:
        path = os.path.join(output_dir, "atlas_v1_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(atlas, fh, indent=2)
        logger.info("ATLAS v1 diagnostics saved to %s", path)
    atlas2 = result.extra_diagnostics.get("atlas_v2")
    if atlas2:
        path = os.path.join(output_dir, "atlas_v2_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(atlas2, fh, indent=2)
        logger.info("ATLAS v2 diagnostics saved to %s", path)


def save_results(result: SimulationResult, output_dir: str) -> None:
    """Save all output files to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    _save_summary(result, output_dir)
    _save_metrics(result, output_dir)
    _save_per_step(result, output_dir)
    _save_combiner_decisions(result, output_dir)
    _save_atlas_diagnostics(result, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run a weighted paging policy on a trace.",
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
        "--derive-predicted-caches",
        action="store_true",
        help="Derive predictor cache states P_t from next-arrival predictions via Blind Oracle conversion.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help=(
            "Additive Gaussian noise standard deviation to apply to predictions "
            "(default: 0.0 = no noise).  Applied after --perfect-predictions if both set."
        ),
    )
    parser.add_argument(
        "--default-confidence",
        type=float,
        default=0.5,
        help="Default confidence λ when per-request confidence is missing (atlas_v1/atlas_v2).",
    )
    parser.add_argument(
        "--bucket-source",
        choices=["trace", "perfect"],
        default="trace",
        help="Source of bucket hints for atlas_v1/atlas_v2 (default: trace).",
    )
    parser.add_argument(
        "--bucket-horizon",
        type=int,
        default=2,
        help="Distance horizon used when generating perfect buckets (atlas_v1/atlas_v2 only).",
    )
    parser.add_argument(
        "--bucket-noise-prob",
        type=float,
        default=0.0,
        help="Probability of corrupting a bucket hint (atlas_v1/atlas_v2 only).",
    )
    parser.add_argument(
        "--bucket-noise-seed",
        type=int,
        default=0,
        help="RNG seed for bucket corruption (atlas_v1/atlas_v2 only).",
    )
    parser.add_argument(
        "--atlas-window",
        type=int,
        default=32,
        help="Rolling mismatch window size for atlas_v2 (default: 32).",
    )
    parser.add_argument(
        "--atlas-rho",
        type=float,
        default=0.3,
        help="Smoothing factor for gamma updates in atlas_v2 (default: 0.3).",
    )
    parser.add_argument(
        "--atlas-initial-gamma",
        type=float,
        default=0.8,
        help="Initial global trust multiplier gamma_0 for atlas_v2 (default: 0.8).",
    )
    parser.add_argument(
        "--atlas-mismatch-threshold",
        type=int,
        default=2,
        help="Max request-distance considered 'too soon' for atlas_v2 mismatch proxy (default: 2).",
    )
    parser.add_argument(
        "--trust-seed",
        type=int,
        default=0,
        help="RNG seed for trust_and_doubt randomized choices (Baseline 3).",
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

    if args.noise_sigma > 0.0:
        from lafc.predictors.noisy import add_additive_noise
        requests = add_additive_noise(requests, sigma=args.noise_sigma)

    if args.derive_predicted_caches:
        from lafc.predictors.offline_from_trace import attach_predicted_caches
        requests = attach_predicted_caches(requests, capacity=args.capacity)

    if args.policy in {"atlas_v1", "atlas_v2"}:
        if args.bucket_source == "perfect":
            requests = attach_perfect_buckets(requests, bucket_horizon=args.bucket_horizon)
        requests = maybe_corrupt_buckets(
            requests,
            noise_prob=args.bucket_noise_prob,
            seed=args.bucket_noise_seed,
        )
        if args.policy == "atlas_v1":
            policy = AtlasV1Policy(default_confidence=args.default_confidence)
        else:
            policy = AtlasV2Policy(
                default_confidence=args.default_confidence,
                atlas_window=args.atlas_window,
                atlas_rho=args.atlas_rho,
                atlas_initial_gamma=args.atlas_initial_gamma,
                atlas_mismatch_threshold=args.atlas_mismatch_threshold,
            )
    elif args.policy == "trust_and_doubt":
        policy = TrustAndDoubtPolicy(seed=args.trust_seed)
    else:
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
