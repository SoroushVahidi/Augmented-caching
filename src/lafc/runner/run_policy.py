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
    marker, blind_oracle, predictive_marker, adaptive_query, parsimonious_caching, robust_ftp_d_marker,
    blind_oracle_lru_combiner, offline_belady, trust_and_doubt, atlas_v1, atlas_v2, atlas_v3, atlas_cga_v1, atlas_cga_v2, rest_v1, ml_gate_v1, ml_gate_v2, evict_value_v1, evict_value_v1_guarded, sentinel_robust_tripwire_v1
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
from lafc.policies.adaptive_query import AdaptiveQueryPolicy
from lafc.policies.atlas_v1 import AtlasV1Policy
from lafc.policies.atlas_v2 import AtlasV2Policy
from lafc.policies.atlas_v3 import AtlasV3Policy
from lafc.policies.atlas_cga_v1 import AtlasCGAV1Policy
from lafc.policies.atlas_cga_v2 import AtlasCGAV2Policy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.ml_gate_v1 import MLGateV1Policy
from lafc.policies.ml_gate_v2 import MLGateV2Policy
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.weighted_lru import WeightedLRUPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.policies.robust_ftp_marker_combiner import RobustFtPDeterministicMarkerCombiner
from lafc.policies.guard_wrapper import EvictValueV1GuardedPolicy
from lafc.policies.sentinel_robust_tripwire_v1 import SentinelRobustTripwireV1Policy
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
    # Baseline 5: Im et al. 2022 parsimonious adaptive querying.
    "adaptive_query": AdaptiveQueryPolicy(),
    "parsimonious_caching": AdaptiveQueryPolicy(),
    # Chłędowski et al. 2021 robust practical switching baseline (deterministic).
    "robust_ftp_d_marker": RobustFtPDeterministicMarkerCombiner(),
    "robust_ftp": RobustFtPDeterministicMarkerCombiner(),
    # Baseline 4: Wei 2020 (unweighted paging)
    "blind_oracle_lru_combiner": BlindOracleLRUCombiner(),
    "offline_belady": OfflineBeladyPolicy(),
    # Baseline 3: Antoniadis et al. 2020
    "trust_and_doubt": TrustAndDoubtPolicy(),
    # Experimental framework policy (unweighted).
    "atlas_v1": AtlasV1Policy(),
    "atlas_v2": AtlasV2Policy(),
    "atlas_v3": AtlasV3Policy(),
    "atlas_cga_v1": AtlasCGAV1Policy(),
    "atlas_cga": AtlasCGAV1Policy(),
    "atlas_cga_v2": AtlasCGAV2Policy(),
    "rest_v1": RestV1Policy(),
    "ml_gate_v1": MLGateV1Policy(),
    "ml_gate_v2": MLGateV2Policy(),
    "evict_value_v1": EvictValueV1Policy(),
    "evict_value_v1_guarded": EvictValueV1GuardedPolicy(),
    "sentinel_robust_tripwire_v1": SentinelRobustTripwireV1Policy(),
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
            AdaptiveQueryPolicy,
            RobustFtPDeterministicMarkerCombiner,
            BlindOracleLRUCombiner,
            OfflineBeladyPolicy,
            TrustAndDoubtPolicy,
            AtlasV1Policy,
            AtlasV2Policy,
            AtlasV3Policy,
            AtlasCGAV1Policy,
            AtlasCGAV2Policy,
            RestV1Policy,
            MLGateV1Policy,
            MLGateV2Policy,
            EvictValueV1Policy,
            EvictValueV1GuardedPolicy,
            SentinelRobustTripwireV1Policy,
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
    # Experimental ml_gate_v1 diagnostics.
    if isinstance(policy, MLGateV1Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["ml_gate_v1"] = {"summary": policy.diagnostics_summary()}

    # Experimental ml_gate_v2 diagnostics.
    if isinstance(policy, MLGateV2Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["ml_gate_v2"] = {"summary": policy.diagnostics_summary()}
    if isinstance(policy, EvictValueV1Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["evict_value_v1"] = {"summary": policy.diagnostics_summary()}
    if isinstance(policy, EvictValueV1GuardedPolicy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["evict_value_v1_guarded"] = {
            "summary": policy.diagnostics_summary(),
            "step_log": [
                {
                    "t": s.t,
                    "page_id": s.page_id,
                    "mode_before": s.mode_before,
                    "mode_after": s.mode_after,
                    "hit": s.hit,
                    "evicted": s.evicted,
                    "base_hit": s.base_hit,
                    "fallback_hit": s.fallback_hit,
                    "early_return_detected": s.early_return_detected,
                    "suspicious_count_window": s.suspicious_count_window,
                    "guard_triggered": s.guard_triggered,
                    "trigger_reason": s.trigger_reason,
                }
                for s in policy.step_log()
            ],
        }
    if isinstance(policy, SentinelRobustTripwireV1Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["sentinel_robust_tripwire_v1"] = {
            "summary": policy.diagnostics_summary(),
            "step_log": [
                {
                    "t": s.t,
                    "page_id": s.page_id,
                    "chosen_line": s.chosen_line,
                    "forced_robust": s.forced_robust,
                    "used_predictor_override": s.used_predictor_override,
                    "risk_score": s.risk_score,
                    "suspicious_count_window": s.suspicious_count_window,
                    "budget_before": s.budget_before,
                    "budget_after": s.budget_after,
                    "guard_remaining_after": s.guard_remaining_after,
                    "robust_hit": s.robust_hit,
                    "predictor_hit": s.predictor_hit,
                    "chosen_hit": s.chosen_hit,
                    "chosen_evicted": s.chosen_evicted,
                }
                for s in policy.step_log()
            ],
        }
    if isinstance(policy, AdaptiveQueryPolicy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["adaptive_query"] = {"summary": policy.diagnostics_summary()}
    if isinstance(policy, RobustFtPDeterministicMarkerCombiner):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["robust_ftp"] = {
            "summary": policy.diagnostics_summary(),
            "switch_points": policy.switch_points(),
            "step_log": [
                {
                    "t": s.t,
                    "page_id": s.page_id,
                    "chosen_expert": s.chosen_expert,
                    "switched": s.switched,
                    "robust_misses_before": s.robust_misses_before,
                    "predictor_misses_before": s.predictor_misses_before,
                    "robust_misses_after": s.robust_misses_after,
                    "predictor_misses_after": s.predictor_misses_after,
                    "hit": s.hit,
                    "evicted": s.evicted,
                }
                for s in policy.step_log()
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
    # Experimental atlas_v3 diagnostics.
    if isinstance(policy, AtlasV3Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["atlas_v3"] = {
            "summary": policy.diagnostics_summary(),
            "time_series": policy.time_series_diagnostics(),
            "decision_log": [
                {
                    "t": d.t,
                    "request_page": d.request_page,
                    "chosen_eviction": d.chosen_eviction,
                    "chosen_lambda": d.chosen_lambda,
                    "chosen_context": d.chosen_context,
                    "candidate_buckets": d.candidate_buckets,
                    "candidate_confidences": d.candidate_confidences,
                    "candidate_contexts": d.candidate_contexts,
                    "candidate_local_trust": d.candidate_local_trust,
                    "candidate_lambdas": d.candidate_lambdas,
                    "candidate_base_scores": d.candidate_base_scores,
                    "candidate_pred_scores": d.candidate_pred_scores,
                    "candidate_combined_scores": d.candidate_combined_scores,
                    "decision_mode": d.decision_mode,
                }
                for d in policy.decision_log()
            ],
        }
    # Experimental atlas_cga_v1 diagnostics.
    if isinstance(policy, AtlasCGAV1Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["atlas_cga_v1"] = {
            "summary": policy.diagnostics_summary(),
            "time_series": policy.time_series_diagnostics(),
            "decision_log": [
                {
                    "t": d.t,
                    "request_page": d.request_page,
                    "chosen_eviction": d.chosen_eviction,
                    "chosen_lambda": d.chosen_lambda,
                    "chosen_context": d.chosen_context,
                    "candidate_buckets": d.candidate_buckets,
                    "candidate_confidences": d.candidate_confidences,
                    "candidate_contexts": d.candidate_contexts,
                    "candidate_local_trust": d.candidate_local_trust,
                    "candidate_pcal_empirical": d.candidate_pcal_empirical,
                    "candidate_pcal_posterior": d.candidate_pcal_posterior,
                    "candidate_pcal_shrunk": d.candidate_pcal_shrunk,
                    "candidate_calibration_weight": d.candidate_calibration_weight,
                    "candidate_lambdas": d.candidate_lambdas,
                    "candidate_base_scores": d.candidate_base_scores,
                    "candidate_pred_scores": d.candidate_pred_scores,
                    "candidate_combined_scores": d.candidate_combined_scores,
                    "decision_mode": d.decision_mode,
                }
                for d in policy.decision_log()
            ],
        }
    if isinstance(policy, AtlasCGAV2Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["atlas_cga_v2"] = {
            "summary": policy.diagnostics_summary(),
            "time_series": policy.time_series_diagnostics(),
            "decision_log": [
                {
                    "t": d.t,
                    "request_page": d.request_page,
                    "chosen_eviction": d.chosen_eviction,
                    "chosen_lambda": d.chosen_lambda,
                    "chosen_context": d.chosen_context,
                    "candidate_buckets": d.candidate_buckets,
                    "candidate_confidences": d.candidate_confidences,
                    "candidate_contexts": d.candidate_contexts,
                    "candidate_local_trust": d.candidate_local_trust,
                    "candidate_pcal_ctx": d.candidate_pcal_ctx,
                    "candidate_pcal_bucket": d.candidate_pcal_bucket,
                    "candidate_pcal_conf": d.candidate_pcal_conf,
                    "candidate_pcal_global": d.candidate_pcal_global,
                    "candidate_pcal_shared": d.candidate_pcal_shared,
                    "candidate_weight_ctx": d.candidate_weight_ctx,
                    "candidate_weight_bucket": d.candidate_weight_bucket,
                    "candidate_weight_conf": d.candidate_weight_conf,
                    "candidate_weight_global": d.candidate_weight_global,
                    "candidate_lambdas": d.candidate_lambdas,
                    "candidate_base_scores": d.candidate_base_scores,
                    "candidate_pred_scores": d.candidate_pred_scores,
                    "candidate_combined_scores": d.candidate_combined_scores,
                    "decision_mode": d.decision_mode,
                }
                for d in policy.decision_log()
            ],
        }
    if isinstance(policy, RestV1Policy):
        result.extra_diagnostics = result.extra_diagnostics or {}
        result.extra_diagnostics["rest_v1"] = {
            "summary": policy.diagnostics_summary(),
            "time_series": policy.time_series_diagnostics(),
            "decision_log": [
                {
                    "t": d.t,
                    "request_page": d.request_page,
                    "chosen_eviction": d.chosen_eviction,
                    "mode": d.mode,
                    "context": d.context,
                    "trust_score_before": d.trust_score_before,
                    "trust_score_after": d.trust_score_after,
                    "candidate_buckets": d.candidate_buckets,
                    "candidate_confidences": d.candidate_confidences,
                    "predictor_scores": d.predictor_scores,
                    "lru_scores": d.lru_scores,
                    "predictor_choice": d.predictor_choice,
                    "lru_choice": d.lru_choice,
                    "blind_oracle_choice": d.blind_oracle_choice,
                    "predictive_marker_choice": d.predictive_marker_choice,
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
        if "atlas_v3" in result.extra_diagnostics:
            atlas3 = result.extra_diagnostics["atlas_v3"]
            for key, value in atlas3.get("summary", {}).items():
                metrics[f"atlas_v3_{key}"] = value
        if "atlas_cga_v1" in result.extra_diagnostics:
            atlas_cga = result.extra_diagnostics["atlas_cga_v1"]
            for key, value in atlas_cga.get("summary", {}).items():
                metrics[f"atlas_cga_v1_{key}"] = value
        if "atlas_cga_v2" in result.extra_diagnostics:
            atlas_cga2 = result.extra_diagnostics["atlas_cga_v2"]
            for key, value in atlas_cga2.get("summary", {}).items():
                metrics[f"atlas_cga_v2_{key}"] = value
        if "rest_v1" in result.extra_diagnostics:
            rest = result.extra_diagnostics["rest_v1"]
            for key, value in rest.get("summary", {}).items():
                metrics[f"rest_v1_{key}"] = value
        if "adaptive_query" in result.extra_diagnostics:
            aq = result.extra_diagnostics["adaptive_query"]
            for key, value in aq.get("summary", {}).items():
                metrics[f"adaptive_query_{key}"] = value
        if "robust_ftp" in result.extra_diagnostics:
            rftp = result.extra_diagnostics["robust_ftp"]
            for key, value in rftp.get("summary", {}).items():
                metrics[f"robust_ftp_{key}"] = value
        if "evict_value_v1_guarded" in result.extra_diagnostics:
            guarded = result.extra_diagnostics["evict_value_v1_guarded"]
            for key, value in guarded.get("summary", {}).items():
                metrics[f"evict_value_v1_guarded_{key}"] = value
        if "sentinel_robust_tripwire_v1" in result.extra_diagnostics:
            sentinel = result.extra_diagnostics["sentinel_robust_tripwire_v1"]
            for key, value in sentinel.get("summary", {}).items():
                metrics[f"sentinel_robust_tripwire_v1_{key}"] = value


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
    if step_log:
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

    robust = result.extra_diagnostics.get("robust_ftp")
    if robust and robust.get("step_log"):
        robust_path = os.path.join(output_dir, "robust_ftp_decisions.csv")
        with open(robust_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "t",
                    "page_id",
                    "chosen_expert",
                    "switched",
                    "robust_misses_before",
                    "predictor_misses_before",
                    "robust_misses_after",
                    "predictor_misses_after",
                    "hit",
                    "evicted",
                ]
            )
            for s in robust["step_log"]:
                writer.writerow(
                    [
                        s["t"],
                        s["page_id"],
                        s["chosen_expert"],
                        s["switched"],
                        s["robust_misses_before"],
                        s["predictor_misses_before"],
                        s["robust_misses_after"],
                        s["predictor_misses_after"],
                        s["hit"],
                        s["evicted"] or "",
                    ]
                )
        logger.info("RobustFtP decisions saved to %s", robust_path)

    guarded = result.extra_diagnostics.get("evict_value_v1_guarded")
    if guarded and guarded.get("step_log"):
        guarded_path = os.path.join(output_dir, "evict_value_v1_guarded_steps.csv")
        with open(guarded_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "t",
                    "page_id",
                    "mode_before",
                    "mode_after",
                    "hit",
                    "evicted",
                    "base_hit",
                    "fallback_hit",
                    "early_return_detected",
                    "suspicious_count_window",
                    "guard_triggered",
                    "trigger_reason",
                ]
            )
            for s in guarded["step_log"]:
                writer.writerow(
                    [
                        s["t"],
                        s["page_id"],
                        s["mode_before"],
                        s["mode_after"],
                        s["hit"],
                        s["evicted"] or "",
                        s["base_hit"],
                        s["fallback_hit"],
                        s["early_return_detected"],
                        s["suspicious_count_window"],
                        s["guard_triggered"],
                        s["trigger_reason"] or "",
                    ]
                )
        logger.info("Guarded EvictValue steps saved to %s", guarded_path)

    sentinel = result.extra_diagnostics.get("sentinel_robust_tripwire_v1")
    if sentinel and sentinel.get("step_log"):
        sentinel_path = os.path.join(output_dir, "sentinel_robust_tripwire_v1_steps.csv")
        with open(sentinel_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "t",
                    "page_id",
                    "chosen_line",
                    "forced_robust",
                    "used_predictor_override",
                    "risk_score",
                    "suspicious_count_window",
                    "budget_before",
                    "budget_after",
                    "guard_remaining_after",
                    "robust_hit",
                    "predictor_hit",
                    "chosen_hit",
                    "chosen_evicted",
                ]
            )
            for s in sentinel["step_log"]:
                writer.writerow(
                    [
                        s["t"],
                        s["page_id"],
                        s["chosen_line"],
                        s["forced_robust"],
                        s["used_predictor_override"],
                        s["risk_score"],
                        s["suspicious_count_window"],
                        s["budget_before"],
                        s["budget_after"],
                        s["guard_remaining_after"],
                        s["robust_hit"],
                        s["predictor_hit"],
                        s["chosen_hit"],
                        s["chosen_evicted"] or "",
                    ]
                )
        logger.info("Sentinel tripwire steps saved to %s", sentinel_path)


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
    atlas3 = result.extra_diagnostics.get("atlas_v3")
    if atlas3:
        path = os.path.join(output_dir, "atlas_v3_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(atlas3, fh, indent=2)
        logger.info("ATLAS v3 diagnostics saved to %s", path)
    atlas_cga = result.extra_diagnostics.get("atlas_cga_v1")
    if atlas_cga:
        path = os.path.join(output_dir, "atlas_cga_v1_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(atlas_cga, fh, indent=2)
        logger.info("ATLAS CGA v1 diagnostics saved to %s", path)
    atlas_cga2 = result.extra_diagnostics.get("atlas_cga_v2")
    if atlas_cga2:
        path = os.path.join(output_dir, "atlas_cga_v2_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(atlas_cga2, fh, indent=2)
        logger.info("ATLAS CGA v2 diagnostics saved to %s", path)
    rest = result.extra_diagnostics.get("rest_v1")
    if rest:
        path = os.path.join(output_dir, "rest_v1_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rest, fh, indent=2)
        logger.info("ReST v1 diagnostics saved to %s", path)


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
        help="Default confidence λ when per-request confidence is missing (atlas_v1/atlas_v2/atlas_v3/atlas_cga_v1/atlas_cga_v2/rest_v1).",
    )
    parser.add_argument(
        "--bucket-source",
        choices=["trace", "perfect"],
        default="trace",
        help="Source of bucket hints for atlas_v1/atlas_v2/atlas_v3/atlas_cga_v1/atlas_cga_v2/rest_v1 (default: trace).",
    )
    parser.add_argument(
        "--bucket-horizon",
        type=int,
        default=2,
        help="Distance horizon used when generating perfect buckets (atlas_v1/atlas_v2/atlas_v3/atlas_cga_v1/atlas_cga_v2/rest_v1 only).",
    )
    parser.add_argument(
        "--bucket-noise-prob",
        type=float,
        default=0.0,
        help="Probability of corrupting a bucket hint (atlas_v1/atlas_v2/atlas_v3/atlas_cga_v1/atlas_cga_v2/rest_v1 only).",
    )
    parser.add_argument(
        "--bucket-noise-seed",
        type=int,
        default=0,
        help="RNG seed for bucket corruption (atlas_v1/atlas_v2/atlas_v3/atlas_cga_v1/atlas_cga_v2/rest_v1 only).",
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
        "--atlas-initial-local-trust",
        type=float,
        default=0.7,
        help="Initial local trust value for unseen contexts in atlas_v3 (default: 0.7).",
    )
    parser.add_argument(
        "--atlas-confidence-bins",
        default="0.33,0.66",
        help="Comma-separated confidence bin thresholds in (0,1) for atlas_v3 contexts.",
    )
    parser.add_argument(
        "--atlas-eta-pos",
        type=float,
        default=0.03,
        help="Positive trust update step for atlas_v3 local trust updates.",
    )
    parser.add_argument(
        "--atlas-eta-neg",
        type=float,
        default=0.12,
        help="Negative trust update step for atlas_v3 local trust updates.",
    )
    parser.add_argument(
        "--atlas-bucket-regret-mode",
        choices=["linear", "exp2", "sqrt"],
        default="linear",
        help="Bucket-to-regret-horizon mapping mode for atlas_v3.",
    )
    parser.add_argument(
        "--atlas-tie-epsilon",
        type=float,
        default=1e-9,
        help="Absolute tie epsilon for atlas_v3 score ties.",
    )
    parser.add_argument(
        "--atlas-adaptive-tie-coef",
        type=float,
        default=0.0,
        help="Adaptive tie epsilon coefficient c for atlas_v3: epsilon_t=max(base, c*std(scores)).",
    )
    parser.add_argument(
        "--atlas-context-mode",
        choices=["bucket_confidence", "bucket_only", "confidence_only", "bucket_group_confidence"],
        default="bucket_confidence",
        help="Context definition mode for atlas_v3 local trust.",
    )
    parser.add_argument(
        "--atlas-bucket-group-size",
        type=int,
        default=2,
        help="Bucket group size when --atlas-context-mode bucket_group_confidence is used.",
    )
    parser.add_argument(
        "--atlas-calibration-prior-a",
        type=float,
        default=1.0,
        help="Beta prior alpha for atlas_cga_v1 per-context calibration.",
    )
    parser.add_argument(
        "--atlas-calibration-prior-b",
        type=float,
        default=1.0,
        help="Beta prior beta for atlas_cga_v1 per-context calibration.",
    )
    parser.add_argument(
        "--atlas-calibration-min-support",
        type=int,
        default=5,
        help="Minimum support before atlas_cga_v1 fully trusts context calibration.",
    )
    parser.add_argument(
        "--atlas-calibration-shrinkage",
        type=float,
        default=10.0,
        help="Shrinkage strength m for atlas_cga_v1 calibration reliability weight n/(n+m).",
    )
    parser.add_argument(
        "--atlas-safe-horizon-mode",
        choices=["bucket_regret", "fixed", "bucket_linear", "bucket_exp2"],
        default="bucket_regret",
        help="Safe-event horizon mode for atlas_cga_v1/atlas_cga_v2 calibration checks.",
    )
    parser.add_argument(
        "--atlas-hier-global-prior-a",
        type=float,
        default=1.0,
        help="Global Beta prior alpha for atlas_cga_v2 hierarchical calibration.",
    )
    parser.add_argument(
        "--atlas-hier-global-prior-b",
        type=float,
        default=1.0,
        help="Global Beta prior beta for atlas_cga_v2 hierarchical calibration.",
    )
    parser.add_argument(
        "--atlas-hier-min-support",
        type=int,
        default=5,
        help="Minimum support threshold used by atlas_cga_v2 support-aware weights.",
    )
    parser.add_argument(
        "--atlas-hier-weight-mode",
        choices=["normalized_support", "uniform_nonzero"],
        default="normalized_support",
        help="Weight rule for atlas_cga_v2 hierarchical sharing.",
    )
    parser.add_argument(
        "--atlas-hier-shrink-strength",
        type=float,
        default=10.0,
        help="Shrink strength for atlas_cga_v2 support-to-weight mapping.",
    )
    parser.add_argument(
        "--trust-seed",
        type=int,
        default=0,
        help="RNG seed for trust_and_doubt randomized choices (Baseline 3).",
    )
    parser.add_argument(
        "--rest-initial-trust",
        type=float,
        default=0.5,
        help="Initial trust score G[ctx] for unseen contexts in rest_v1.",
    )
    parser.add_argument(
        "--rest-eta-pos",
        type=float,
        default=0.05,
        help="Positive trust update step for rest_v1 good TRUST outcomes.",
    )
    parser.add_argument(
        "--rest-eta-neg",
        type=float,
        default=0.10,
        help="Negative trust update step for rest_v1 bad TRUST outcomes.",
    )
    parser.add_argument(
        "--rest-horizon",
        type=int,
        default=2,
        help="Delayed-feedback horizon H for rest_v1 trust outcome checks.",
    )
    parser.add_argument(
        "--rest-confidence-bins",
        default="0.33,0.66",
        help="Comma-separated confidence-bin thresholds in (0,1) for rest_v1 contexts.",
    )
    parser.add_argument(
        "--rest-trust-threshold",
        type=float,
        default=0.5,
        help="Deterministic gate threshold: TRUST iff G[ctx] >= threshold (rest_v1).",
    )
    parser.add_argument(
        "--adaptive-query-b",
        type=int,
        default=2,
        help="Number of sampled query pages b per miss for adaptive_query / parsimonious_caching.",
    )
    parser.add_argument(
        "--adaptive-query-seed",
        type=int,
        default=0,
        help="RNG seed for adaptive_query / parsimonious_caching.",
    )
    parser.add_argument(
        "--evict-value-model-path",
        default="models/evict_value_v1_hist_gb.pkl",
        help="Model artifact path for evict_value_v1 or evict_value_v1_guarded.",
    )
    parser.add_argument(
        "--evict-value-scorer-mode",
        choices=["auto", "artifact", "lightweight"],
        default="auto",
        help="Scorer mode for evict_value_v1 family (auto falls back to lightweight when artifact is missing).",
    )
    parser.add_argument(
        "--evict-value-lightweight-config",
        default="",
        help="Optional JSON config path for evict_value_v1 lightweight scorer (text-only).",
    )
    parser.add_argument(
        "--guard-fallback-policy",
        choices=["lru", "marker"],
        default="lru",
        help="Fallback policy used by evict_value_v1_guarded.",
    )
    parser.add_argument(
        "--guard-early-return-window",
        type=int,
        default=2,
        help="Early-return detector window W for evict_value_v1_guarded.",
    )
    parser.add_argument(
        "--guard-trigger-threshold",
        type=int,
        default=2,
        help="Trigger threshold M: suspicious events needed inside trigger window.",
    )
    parser.add_argument(
        "--guard-trigger-window",
        type=int,
        default=16,
        help="Sliding request window size for counting suspicious events.",
    )
    parser.add_argument(
        "--guard-duration",
        type=int,
        default=8,
        help="Number of requests to stay in fallback mode after a trigger.",
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

    if args.policy in {"atlas_v1", "atlas_v2", "atlas_v3", "atlas_cga_v1", "atlas_cga", "atlas_cga_v2", "rest_v1"}:
        if args.bucket_source == "perfect":
            requests = attach_perfect_buckets(requests, bucket_horizon=args.bucket_horizon)
        requests = maybe_corrupt_buckets(
            requests,
            noise_prob=args.bucket_noise_prob,
            seed=args.bucket_noise_seed,
        )
        if args.policy == "atlas_v1":
            policy = AtlasV1Policy(default_confidence=args.default_confidence)
        elif args.policy == "atlas_v2":
            policy = AtlasV2Policy(
                default_confidence=args.default_confidence,
                atlas_window=args.atlas_window,
                atlas_rho=args.atlas_rho,
                atlas_initial_gamma=args.atlas_initial_gamma,
                atlas_mismatch_threshold=args.atlas_mismatch_threshold,
            )
        elif args.policy == "atlas_v3":
            policy = AtlasV3Policy(
                default_confidence=args.default_confidence,
                atlas_initial_local_trust=args.atlas_initial_local_trust,
                atlas_confidence_bins=args.atlas_confidence_bins,
                atlas_eta_pos=args.atlas_eta_pos,
                atlas_eta_neg=args.atlas_eta_neg,
                atlas_bucket_regret_mode=args.atlas_bucket_regret_mode,
                atlas_tie_epsilon=args.atlas_tie_epsilon,
                atlas_adaptive_tie_coef=args.atlas_adaptive_tie_coef,
                atlas_context_mode=args.atlas_context_mode,
                atlas_bucket_group_size=args.atlas_bucket_group_size,
                bucket_horizon=args.bucket_horizon,
            )
        elif args.policy in {"atlas_cga_v1", "atlas_cga"}:
            policy = AtlasCGAV1Policy(
                default_confidence=args.default_confidence,
                atlas_initial_local_trust=args.atlas_initial_local_trust,
                atlas_confidence_bins=args.atlas_confidence_bins,
                atlas_eta_pos=args.atlas_eta_pos,
                atlas_eta_neg=args.atlas_eta_neg,
                atlas_bucket_regret_mode=args.atlas_bucket_regret_mode,
                atlas_tie_epsilon=args.atlas_tie_epsilon,
                atlas_adaptive_tie_coef=args.atlas_adaptive_tie_coef,
                atlas_context_mode=args.atlas_context_mode,
                atlas_bucket_group_size=args.atlas_bucket_group_size,
                bucket_horizon=args.bucket_horizon,
                atlas_calibration_prior_a=args.atlas_calibration_prior_a,
                atlas_calibration_prior_b=args.atlas_calibration_prior_b,
                atlas_calibration_min_support=args.atlas_calibration_min_support,
                atlas_calibration_shrinkage=args.atlas_calibration_shrinkage,
                atlas_safe_horizon_mode=args.atlas_safe_horizon_mode,
            )
        else:
            if args.policy == "atlas_cga_v2":
                policy = AtlasCGAV2Policy(
                    default_confidence=args.default_confidence,
                    atlas_initial_local_trust=args.atlas_initial_local_trust,
                    atlas_confidence_bins=args.atlas_confidence_bins,
                    atlas_eta_pos=args.atlas_eta_pos,
                    atlas_eta_neg=args.atlas_eta_neg,
                    atlas_bucket_regret_mode=args.atlas_bucket_regret_mode,
                    atlas_tie_epsilon=args.atlas_tie_epsilon,
                    atlas_adaptive_tie_coef=args.atlas_adaptive_tie_coef,
                    atlas_context_mode=args.atlas_context_mode,
                    atlas_bucket_group_size=args.atlas_bucket_group_size,
                    bucket_horizon=args.bucket_horizon,
                    atlas_hier_global_prior_a=args.atlas_hier_global_prior_a,
                    atlas_hier_global_prior_b=args.atlas_hier_global_prior_b,
                    atlas_hier_min_support=args.atlas_hier_min_support,
                    atlas_hier_weight_mode=args.atlas_hier_weight_mode,
                    atlas_hier_shrink_strength=args.atlas_hier_shrink_strength,
                    atlas_safe_horizon_mode=args.atlas_safe_horizon_mode,
                )
            else:
                policy = RestV1Policy(
                    default_confidence=args.default_confidence,
                    rest_initial_trust=args.rest_initial_trust,
                    rest_eta_pos=args.rest_eta_pos,
                    rest_eta_neg=args.rest_eta_neg,
                    rest_horizon=args.rest_horizon,
                    rest_confidence_bins=args.rest_confidence_bins,
                    rest_trust_threshold=args.rest_trust_threshold,
                )
    elif args.policy == "trust_and_doubt":
        policy = TrustAndDoubtPolicy(seed=args.trust_seed)
    elif args.policy in {"adaptive_query", "parsimonious_caching"}:
        policy = AdaptiveQueryPolicy(b=args.adaptive_query_b, seed=args.adaptive_query_seed)
    elif args.policy == "evict_value_v1_guarded":
        policy = EvictValueV1GuardedPolicy(
            model_path=args.evict_value_model_path,
            scorer_mode=args.evict_value_scorer_mode,
            lightweight_config_path=(args.evict_value_lightweight_config or None),
            fallback_policy=args.guard_fallback_policy,
            early_return_window=args.guard_early_return_window,
            trigger_threshold=args.guard_trigger_threshold,
            trigger_window=args.guard_trigger_window,
            guard_duration=args.guard_duration,
        )
    elif args.policy == "evict_value_v1":
        policy = EvictValueV1Policy(
            model_path=args.evict_value_model_path,
            scorer_mode=args.evict_value_scorer_mode,
            lightweight_config_path=(args.evict_value_lightweight_config or None),
        )
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
