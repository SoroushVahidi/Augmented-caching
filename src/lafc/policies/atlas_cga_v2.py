"""ATLAS CGA v2: hierarchical context-sharing calibration (experimental).

Built directly on atlas_cga_v1 / atlas_v3 local-trust score family.

Hierarchy levels for safe-to-evict calibration:
- level 0: global
- level 1: bucket-only and confidence-bin-only
- level 2: full context (bucket, confidence_bin)

For context B=(b,c), use support-aware shared probability:
    pcal_shared(B) = w_ctx * p_ctx(B) + w_bucket * p_bucket(b)
                   + w_conf * p_conf(c) + w_global * p_global
with normalized support-based weights.

This is experimental and does not claim theorem guarantees.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class AtlasCGAV2Decision:
    t: int
    request_page: PageId
    chosen_eviction: Optional[PageId]
    chosen_lambda: float
    chosen_context: Optional[str]
    candidate_buckets: Dict[PageId, int]
    candidate_confidences: Dict[PageId, float]
    candidate_contexts: Dict[PageId, str]
    candidate_local_trust: Dict[PageId, float]
    candidate_pcal_ctx: Dict[PageId, float]
    candidate_pcal_bucket: Dict[PageId, float]
    candidate_pcal_conf: Dict[PageId, float]
    candidate_pcal_global: Dict[PageId, float]
    candidate_pcal_shared: Dict[PageId, float]
    candidate_weight_ctx: Dict[PageId, float]
    candidate_weight_bucket: Dict[PageId, float]
    candidate_weight_conf: Dict[PageId, float]
    candidate_weight_global: Dict[PageId, float]
    candidate_lambdas: Dict[PageId, float]
    candidate_base_scores: Dict[PageId, float]
    candidate_pred_scores: Dict[PageId, float]
    candidate_combined_scores: Dict[PageId, float]
    decision_mode: str


@dataclass
class _PendingCheck:
    page_id: PageId
    evicted_at_t: int
    context: Tuple[str, str]
    bucket_label: str
    conf_label: str
    calibrated_signal: float
    tolerated_horizon: int
    trust_eligible: bool


class AtlasCGAV2Policy(BasePolicy):
    name: str = "atlas_cga_v2"

    def __init__(
        self,
        default_confidence: float = 0.5,
        atlas_initial_local_trust: float = 0.7,
        atlas_confidence_bins: str = "0.33,0.66",
        atlas_eta_pos: float = 0.03,
        atlas_eta_neg: float = 0.12,
        atlas_bucket_regret_mode: str = "linear",
        atlas_tie_epsilon: float = 1e-9,
        atlas_adaptive_tie_coef: float = 0.0,
        atlas_context_mode: str = "bucket_confidence",
        atlas_bucket_group_size: int = 2,
        bucket_horizon: int = 2,
        atlas_hier_global_prior_a: float = 1.0,
        atlas_hier_global_prior_b: float = 1.0,
        atlas_hier_min_support: int = 5,
        atlas_hier_weight_mode: str = "normalized_support",
        atlas_hier_shrink_strength: float = 10.0,
        atlas_safe_horizon_mode: str = "bucket_regret",
        atlas_hier_delta_threshold: float = 0.15,
    ) -> None:
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0,1]")
        if not 0.0 <= atlas_initial_local_trust <= 1.0:
            raise ValueError("atlas_initial_local_trust must be in [0,1]")
        if atlas_eta_pos < 0.0 or atlas_eta_neg < 0.0 or atlas_eta_neg < atlas_eta_pos:
            raise ValueError("atlas eta params invalid")
        if atlas_bucket_regret_mode not in {"linear", "exp2", "sqrt"}:
            raise ValueError("invalid atlas_bucket_regret_mode")
        if atlas_adaptive_tie_coef < 0.0:
            raise ValueError("atlas_adaptive_tie_coef must be >= 0")
        if atlas_context_mode not in {"bucket_confidence", "bucket_only", "confidence_only", "bucket_group_confidence"}:
            raise ValueError("invalid atlas_context_mode")
        if atlas_bucket_group_size < 1 or bucket_horizon < 1:
            raise ValueError("atlas_bucket_group_size and bucket_horizon must be >=1")
        if atlas_hier_global_prior_a <= 0.0 or atlas_hier_global_prior_b <= 0.0:
            raise ValueError("atlas_hier_global_prior_a/b must be > 0")
        if atlas_hier_min_support < 0 or atlas_hier_shrink_strength < 0.0:
            raise ValueError("atlas_hier_min_support/shrink_strength invalid")
        if atlas_hier_weight_mode not in {"normalized_support", "uniform_nonzero"}:
            raise ValueError("atlas_hier_weight_mode must be one of: normalized_support, uniform_nonzero")
        if atlas_safe_horizon_mode not in {"bucket_regret", "fixed", "bucket_linear", "bucket_exp2"}:
            raise ValueError("invalid atlas_safe_horizon_mode")

        self.default_confidence = float(default_confidence)
        self.atlas_initial_local_trust = float(atlas_initial_local_trust)
        self.confidence_thresholds = self._parse_confidence_bins(atlas_confidence_bins)
        self.atlas_eta_pos = float(atlas_eta_pos)
        self.atlas_eta_neg = float(atlas_eta_neg)
        self.atlas_bucket_regret_mode = atlas_bucket_regret_mode
        self.atlas_tie_epsilon = float(atlas_tie_epsilon)
        self.atlas_adaptive_tie_coef = float(atlas_adaptive_tie_coef)
        self.atlas_context_mode = atlas_context_mode
        self.atlas_bucket_group_size = int(atlas_bucket_group_size)
        self.bucket_horizon = int(bucket_horizon)
        self.atlas_hier_global_prior_a = float(atlas_hier_global_prior_a)
        self.atlas_hier_global_prior_b = float(atlas_hier_global_prior_b)
        self.atlas_hier_min_support = int(atlas_hier_min_support)
        self.atlas_hier_weight_mode = atlas_hier_weight_mode
        self.atlas_hier_shrink_strength = float(atlas_hier_shrink_strength)
        self.atlas_safe_horizon_mode = atlas_safe_horizon_mode
        self.atlas_hier_delta_threshold = float(atlas_hier_delta_threshold)

    @staticmethod
    def _parse_confidence_bins(spec: str) -> List[float]:
        vals: List[float] = []
        for chunk in str(spec).split(","):
            t = chunk.strip()
            if not t:
                continue
            v = float(t)
            if not 0.0 < v < 1.0:
                raise ValueError("atlas_confidence_bins thresholds must be in (0,1)")
            vals.append(v)
        return sorted(set(vals))

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        self._bucket_by_page: Dict[PageId, int] = {}
        self._confidence_by_page: Dict[PageId, float] = {}
        self._actual_next_by_page: Dict[PageId, float] = {}

        self._local_trust: Dict[Tuple[str, str], float] = {}
        self._pending_checks: Dict[PageId, _PendingCheck] = {}
        self._ctx_good: Dict[str, int] = collections.defaultdict(int)
        self._ctx_bad: Dict[str, int] = collections.defaultdict(int)
        self._ctx_history: Dict[str, List[float]] = collections.defaultdict(list)

        self._n_global = 0
        self._s_global = 0
        self._n_bucket: Dict[str, int] = collections.defaultdict(int)
        self._s_bucket: Dict[str, int] = collections.defaultdict(int)
        self._n_conf: Dict[str, int] = collections.defaultdict(int)
        self._s_conf: Dict[str, int] = collections.defaultdict(int)
        self._n_ctx: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        self._s_ctx: Dict[Tuple[str, str], int] = collections.defaultdict(int)

        self._decisions: List[AtlasCGAV2Decision] = []
        self._lambda_values: List[float] = []
        self._predictor_dominated = 0
        self._fallback_dominated = 0
        self._tie_region = 0

        self._match_lru_count = 0
        self._match_blind_oracle_count = 0
        self._match_predictive_marker_count = 0
        self._eviction_decision_count = 0

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        self._resolve_pending_checks_for_time(request.t)
        self._resolve_pending_check_for_page(pid, request.t)
        self._update_prediction_state(pid, request)

        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        step_diag: Dict[str, object] = {}
        if self._cache.is_full():
            evicted, step_diag = self._choose_victim(request)
            self._evict(evicted)
            self._order.pop(evicted, None)

        self._add(pid)
        self._order[pid] = None
        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted, diagnostics=step_diag)

    def diagnostics_summary(self) -> Dict[str, object]:
        total = len(self._decisions)
        contexts = sorted(set(self._local_trust) | set(self._n_ctx))
        prior = self._prior_mean()

        per_ctx = {}
        low_support_count = 0
        changed_substantial = 0
        pcal_shared_vals: List[float] = []
        for ctx in contexts:
            b_lbl, c_lbl = ctx
            p_ctx = self._posterior(self._s_ctx[ctx], self._n_ctx[ctx])
            p_bucket = self._posterior(self._s_bucket[b_lbl], self._n_bucket[b_lbl])
            p_conf = self._posterior(self._s_conf[c_lbl], self._n_conf[c_lbl])
            p_global = self._posterior(self._s_global, self._n_global)
            w_ctx, w_bucket, w_conf, w_global = self._hier_weights(
                self._n_ctx[ctx], self._n_bucket[b_lbl], self._n_conf[c_lbl]
            )
            p_shared = w_ctx * p_ctx + w_bucket * p_bucket + w_conf * p_conf + w_global * p_global
            pcal_shared_vals.append(p_shared)
            if self._n_ctx[ctx] < self.atlas_hier_min_support:
                low_support_count += 1
            if abs(p_shared - p_ctx) >= self.atlas_hier_delta_threshold:
                changed_substantial += 1
            per_ctx[self._ctx_to_key(ctx)] = {
                "n": self._n_ctx[ctx],
                "successes": self._s_ctx[ctx],
                "p_ctx": p_ctx,
                "p_shared": p_shared,
                "w_ctx": w_ctx,
                "w_bucket": w_bucket,
                "w_conf": w_conf,
                "w_global": w_global,
            }

        bucket_table = {
            b: {
                "n": self._n_bucket[b],
                "successes": self._s_bucket[b],
                "p_bucket": self._posterior(self._s_bucket[b], self._n_bucket[b]),
            }
            for b in sorted(self._n_bucket)
        }
        conf_table = {
            c: {
                "n": self._n_conf[c],
                "successes": self._s_conf[c],
                "p_conf": self._posterior(self._s_conf[c], self._n_conf[c]),
            }
            for c in sorted(self._n_conf)
        }

        return {
            "average_lambda": mean(self._lambda_values) if self._lambda_values else 0.0,
            "predictor_dominated_decisions": self._predictor_dominated,
            "fallback_dominated_decisions": self._fallback_dominated,
            "tie_region_decisions": self._tie_region,
            "fraction_predictor_dominated": self._predictor_dominated / total if total else 0.0,
            "fraction_fallback_dominated": self._fallback_dominated / total if total else 0.0,
            "fraction_tie_region": self._tie_region / total if total else 0.0,
            "contexts_seen": len(contexts),
            "local_trust_table": {self._ctx_to_key(c): self._local_trust_for_context(c) for c in contexts},
            "context_good_counts": dict(self._ctx_good),
            "context_bad_counts": dict(self._ctx_bad),
            "global_calibration": {
                "n": self._n_global,
                "successes": self._s_global,
                "p_global": self._posterior(self._s_global, self._n_global),
                "prior_mean": prior,
            },
            "bucket_calibration": bucket_table,
            "confidence_calibration": conf_table,
            "context_calibration": per_ctx,
            "num_contexts_low_support": low_support_count,
            "num_contexts_hier_substantial_change": changed_substantial,
            "mean_pcal_shared": mean(pcal_shared_vals) if pcal_shared_vals else prior,
            "match_rate_lru": self._match_lru_count / self._eviction_decision_count if self._eviction_decision_count else 0.0,
            "match_rate_blind_oracle": self._match_blind_oracle_count / self._eviction_decision_count if self._eviction_decision_count else 0.0,
            "match_rate_predictive_marker": self._match_predictive_marker_count / self._eviction_decision_count if self._eviction_decision_count else 0.0,
            "pending_local_checks": len(self._pending_checks),
        }

    def time_series_diagnostics(self) -> Dict[str, object]:
        return {
            "context_trust_evolution": {k: list(v) for k, v in self._ctx_history.items()},
            "contexts_seen_t": [d.chosen_context for d in self._decisions],
            "chosen_lambda_t": [d.chosen_lambda for d in self._decisions],
        }

    def decision_log(self) -> List[AtlasCGAV2Decision]:
        return list(self._decisions)

    def _update_prediction_state(self, pid: PageId, request: Request) -> None:
        bucket = request.metadata.get("bucket")
        if bucket is not None:
            self._bucket_by_page[pid] = int(bucket)
        conf = request.metadata.get("confidence")
        if conf is not None:
            self._confidence_by_page[pid] = min(1.0, max(0.0, float(conf)))
        self._actual_next_by_page[pid] = float(request.actual_next)

    def _confidence_bin(self, confidence: float) -> str:
        for idx, thr in enumerate(self.confidence_thresholds):
            if confidence <= thr:
                return f"bin_{idx}"
        return f"bin_{len(self.confidence_thresholds)}"

    def _context_for_page(self, page: PageId) -> Tuple[str, str]:
        bucket = int(self._bucket_by_page.get(page, 0))
        confidence = float(self._confidence_by_page.get(page, self.default_confidence))
        conf_bin = self._confidence_bin(confidence)
        if self.atlas_context_mode == "bucket_only":
            return (f"bucket={bucket}", "conf=all")
        if self.atlas_context_mode == "confidence_only":
            return ("bucket=all", conf_bin)
        if self.atlas_context_mode == "bucket_group_confidence":
            grp = bucket // self.atlas_bucket_group_size
            return (f"bucket_group={grp}", conf_bin)
        return (f"bucket={bucket}", conf_bin)

    @staticmethod
    def _ctx_to_key(ctx: Tuple[str, str]) -> str:
        return f"{ctx[0]}|{ctx[1]}"

    def _local_trust_for_context(self, ctx: Tuple[str, str]) -> float:
        if ctx not in self._local_trust:
            self._local_trust[ctx] = self.atlas_initial_local_trust
            self._ctx_history[self._ctx_to_key(ctx)].append(self.atlas_initial_local_trust)
        return self._local_trust[ctx]

    def _prior_mean(self) -> float:
        return self.atlas_hier_global_prior_a / (self.atlas_hier_global_prior_a + self.atlas_hier_global_prior_b)

    def _posterior(self, s: int, n: int) -> float:
        return (s + self.atlas_hier_global_prior_a) / (
            n + self.atlas_hier_global_prior_a + self.atlas_hier_global_prior_b
        )

    def _support_alpha(self, n: int) -> float:
        alpha = n / (n + self.atlas_hier_shrink_strength) if self.atlas_hier_shrink_strength > 0 else 1.0
        if n < self.atlas_hier_min_support:
            alpha *= n / max(1, self.atlas_hier_min_support)
        return max(0.0, min(1.0, float(alpha)))

    def _hier_weights(self, n_ctx: int, n_bucket: int, n_conf: int) -> Tuple[float, float, float, float]:
        a_ctx = self._support_alpha(n_ctx)
        a_bucket = self._support_alpha(n_bucket)
        a_conf = self._support_alpha(n_conf)
        if self.atlas_hier_weight_mode == "uniform_nonzero":
            comps = [a_ctx, a_bucket, a_conf]
            nonzero = [x for x in comps if x > 0]
            if not nonzero:
                return 0.0, 0.0, 0.0, 1.0
            per = 1.0 / len(nonzero)
            w_ctx = per if a_ctx > 0 else 0.0
            w_bucket = per if a_bucket > 0 else 0.0
            w_conf = per if a_conf > 0 else 0.0
            return w_ctx, w_bucket, w_conf, 0.0

        total = a_ctx + a_bucket + a_conf
        if total <= 0.0:
            return 0.0, 0.0, 0.0, 1.0
        norm_mass = min(1.0, total)
        w_ctx = norm_mass * (a_ctx / total)
        w_bucket = norm_mass * (a_bucket / total)
        w_conf = norm_mass * (a_conf / total)
        w_global = 1.0 - norm_mass
        return w_ctx, w_bucket, w_conf, w_global

    def _shared_calibration(self, ctx: Tuple[str, str]) -> Tuple[float, float, float, float, float, float, float, float]:
        b_lbl, c_lbl = ctx
        p_ctx = self._posterior(self._s_ctx[ctx], self._n_ctx[ctx])
        p_bucket = self._posterior(self._s_bucket[b_lbl], self._n_bucket[b_lbl])
        p_conf = self._posterior(self._s_conf[c_lbl], self._n_conf[c_lbl])
        p_global = self._posterior(self._s_global, self._n_global)
        w_ctx, w_bucket, w_conf, w_global = self._hier_weights(
            self._n_ctx[ctx], self._n_bucket[b_lbl], self._n_conf[c_lbl]
        )
        p_shared = w_ctx * p_ctx + w_bucket * p_bucket + w_conf * p_conf + w_global * p_global
        return p_ctx, p_bucket, p_conf, p_global, w_ctx, w_bucket, w_conf, p_shared

    def _tolerated_return_horizon(self, bucket: int) -> int:
        b = max(0, int(bucket))
        if self.atlas_safe_horizon_mode == "fixed":
            return self.bucket_horizon
        if self.atlas_safe_horizon_mode == "bucket_linear":
            return max(1, self.bucket_horizon * (b + 1))
        if self.atlas_safe_horizon_mode == "bucket_exp2":
            return max(1, self.bucket_horizon * (2**b))
        if self.atlas_bucket_regret_mode == "exp2":
            return max(1, self.bucket_horizon * (2**b))
        if self.atlas_bucket_regret_mode == "sqrt":
            return max(1, self.bucket_horizon * int((b + 1) ** 0.5))
        return max(1, self.bucket_horizon * (b + 1))

    def _resolve_pending_checks_for_time(self, t: int) -> None:
        expired: List[PageId] = []
        for page, check in self._pending_checks.items():
            if (t - check.evicted_at_t) > check.tolerated_horizon:
                expired.append(page)
        for page in expired:
            check = self._pending_checks.pop(page)
            self._apply_outcome_update(check.context, check.bucket_label, check.conf_label, check.calibrated_signal, False, check.trust_eligible)

    def _resolve_pending_check_for_page(self, pid: PageId, t: int) -> None:
        check = self._pending_checks.get(pid)
        if check is None:
            return
        delta = t - check.evicted_at_t
        is_bad = delta <= check.tolerated_horizon
        self._pending_checks.pop(pid, None)
        self._apply_outcome_update(check.context, check.bucket_label, check.conf_label, check.calibrated_signal, is_bad, check.trust_eligible)

    def _apply_outcome_update(
        self,
        ctx: Tuple[str, str],
        bucket_label: str,
        conf_label: str,
        calibrated_signal: float,
        is_bad: bool,
        trust_eligible: bool,
    ) -> None:
        self._n_global += 1
        self._n_bucket[bucket_label] += 1
        self._n_conf[conf_label] += 1
        self._n_ctx[ctx] += 1
        if not is_bad:
            self._s_global += 1
            self._s_bucket[bucket_label] += 1
            self._s_conf[conf_label] += 1
            self._s_ctx[ctx] += 1

        if not trust_eligible:
            return

        prior = self._local_trust_for_context(ctx)
        if is_bad:
            updated = max(0.0, min(1.0, prior - self.atlas_eta_neg * calibrated_signal))
            self._ctx_bad[self._ctx_to_key(ctx)] += 1
        else:
            updated = max(0.0, min(1.0, prior + self.atlas_eta_pos * calibrated_signal))
            self._ctx_good[self._ctx_to_key(ctx)] += 1
        self._local_trust[ctx] = updated
        self._ctx_history[self._ctx_to_key(ctx)].append(updated)

    def _choose_victim(self, request: Request) -> tuple[PageId, Dict[str, object]]:
        candidates = list(self._order.keys())
        if not candidates:
            raise RuntimeError("No candidate available for eviction")

        base_scores = self._compute_lru_base_scores(candidates)
        pred_scores = self._compute_aggressive_pred_scores(candidates)

        candidate_buckets: Dict[PageId, int] = {}
        candidate_conf: Dict[PageId, float] = {}
        candidate_ctxs: Dict[PageId, str] = {}
        candidate_trust: Dict[PageId, float] = {}
        candidate_pctx: Dict[PageId, float] = {}
        candidate_pb: Dict[PageId, float] = {}
        candidate_pc: Dict[PageId, float] = {}
        candidate_pg: Dict[PageId, float] = {}
        candidate_pshared: Dict[PageId, float] = {}
        candidate_wctx: Dict[PageId, float] = {}
        candidate_wb: Dict[PageId, float] = {}
        candidate_wc: Dict[PageId, float] = {}
        candidate_wg: Dict[PageId, float] = {}
        lambdas: Dict[PageId, float] = {}
        scores: Dict[PageId, float] = {}

        for p in candidates:
            confidence = float(self._confidence_by_page.get(p, self.default_confidence))
            ctx = self._context_for_page(p)
            trust = self._local_trust_for_context(ctx)
            p_ctx, p_bucket, p_conf, p_global, w_ctx, w_bucket, w_conf, p_shared = self._shared_calibration(ctx)
            w_global = 1.0 - (w_ctx + w_bucket + w_conf)
            lam = max(0.0, min(1.0, trust * p_shared))

            candidate_buckets[p] = int(self._bucket_by_page.get(p, 0))
            candidate_conf[p] = confidence
            candidate_ctxs[p] = self._ctx_to_key(ctx)
            candidate_trust[p] = trust
            candidate_pctx[p] = p_ctx
            candidate_pb[p] = p_bucket
            candidate_pc[p] = p_conf
            candidate_pg[p] = p_global
            candidate_pshared[p] = p_shared
            candidate_wctx[p] = w_ctx
            candidate_wb[p] = w_bucket
            candidate_wc[p] = w_conf
            candidate_wg[p] = w_global
            lambdas[p] = lam
            scores[p] = lam * pred_scores[p] + (1.0 - lam) * base_scores[p]
            self._lambda_values.append(lam)

        top_score = max(scores.values())
        avg_score = sum(scores.values()) / len(scores)
        score_std = (sum((v - avg_score) ** 2 for v in scores.values()) / len(scores)) ** 0.5
        effective_tie_eps = max(self.atlas_tie_epsilon, self.atlas_adaptive_tie_coef * score_std)
        top_candidates = [p for p in candidates if abs(scores[p] - top_score) <= effective_tie_eps]
        victim = min(top_candidates, key=lambda p: list(self._order.keys()).index(p))

        predictor_choice = self._argmax_tie_lru(pred_scores, candidates)
        fallback_choice = self._argmax_tie_lru(base_scores, candidates)
        blind_oracle_choice = self._blind_oracle_choice(candidates)
        marker_choice = self._argmax_tie_lru({p: float(self._bucket_by_page.get(p, 0)) for p in candidates}, candidates)

        self._eviction_decision_count += 1
        if victim == fallback_choice:
            self._match_lru_count += 1
        if victim == blind_oracle_choice:
            self._match_blind_oracle_count += 1
        if victim == marker_choice:
            self._match_predictive_marker_count += 1

        decision_mode = "tie-region"
        if victim == predictor_choice and victim != fallback_choice:
            decision_mode = "predictor-dominated"
            self._predictor_dominated += 1
        elif victim == fallback_choice and victim != predictor_choice:
            decision_mode = "fallback-dominated"
            self._fallback_dominated += 1
        else:
            self._tie_region += 1

        chosen_ctx = self._context_for_page(victim)
        chosen_ctx_key = self._ctx_to_key(chosen_ctx)
        b_lbl, c_lbl = chosen_ctx
        self._pending_checks[victim] = _PendingCheck(
            page_id=victim,
            evicted_at_t=request.t,
            context=chosen_ctx,
            bucket_label=b_lbl,
            conf_label=c_lbl,
            calibrated_signal=candidate_pshared[victim],
            tolerated_horizon=self._tolerated_return_horizon(candidate_buckets[victim]),
            trust_eligible=(decision_mode == "predictor-dominated"),
        )

        diag = {
            "decision_mode": decision_mode,
            "chosen_lambda": lambdas[victim],
            "chosen_context": chosen_ctx_key,
            "chosen_pcal_shared": candidate_pshared[victim],
            "chosen_weight_ctx": candidate_wctx[victim],
            "chosen_weight_bucket": candidate_wb[victim],
            "chosen_weight_conf": candidate_wc[victim],
            "chosen_weight_global": candidate_wg[victim],
            "num_contexts_seen": len(self._local_trust),
            "effective_tie_epsilon": effective_tie_eps,
        }

        self._decisions.append(
            AtlasCGAV2Decision(
                t=request.t,
                request_page=request.page_id,
                chosen_eviction=victim,
                chosen_lambda=lambdas[victim],
                chosen_context=chosen_ctx_key,
                candidate_buckets=candidate_buckets,
                candidate_confidences=candidate_conf,
                candidate_contexts=candidate_ctxs,
                candidate_local_trust=candidate_trust,
                candidate_pcal_ctx=candidate_pctx,
                candidate_pcal_bucket=candidate_pb,
                candidate_pcal_conf=candidate_pc,
                candidate_pcal_global=candidate_pg,
                candidate_pcal_shared=candidate_pshared,
                candidate_weight_ctx=candidate_wctx,
                candidate_weight_bucket=candidate_wb,
                candidate_weight_conf=candidate_wc,
                candidate_weight_global=candidate_wg,
                candidate_lambdas=lambdas,
                candidate_base_scores=base_scores,
                candidate_pred_scores=pred_scores,
                candidate_combined_scores=scores,
                decision_mode=decision_mode,
            )
        )

        return victim, diag

    def _compute_lru_base_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        if len(candidates) == 1:
            return {candidates[0]: 1.0}
        denom = len(candidates) - 1
        return {p: 1.0 - (idx / denom) for idx, p in enumerate(candidates)}

    def _compute_aggressive_pred_scores(self, candidates: List[PageId]) -> Dict[PageId, float]:
        if len(candidates) == 1:
            return {candidates[0]: 1.0}
        bucket_values = {p: int(self._bucket_by_page.get(p, 0)) for p in candidates}
        unique = sorted(set(bucket_values.values()))
        if len(unique) == 1:
            return {p: 0.5 for p in candidates}
        rank = {b: i for i, b in enumerate(unique)}
        denom = len(unique) - 1
        return {p: (rank[bucket_values[p]] / denom) ** 2 for p in candidates}

    def _argmax_tie_lru(self, scores: Dict[PageId, float], candidates: List[PageId]) -> PageId:
        best = max(scores[p] for p in candidates)
        ties = [p for p in candidates if scores[p] == best or abs(scores[p] - best) <= self.atlas_tie_epsilon]
        return min(ties, key=lambda p: candidates.index(p))

    def _blind_oracle_choice(self, candidates: List[PageId]) -> PageId:
        score = {p: float(self._actual_next_by_page.get(p, float("inf"))) for p in candidates}
        return self._argmax_tie_lru(score, candidates)
