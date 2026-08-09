from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.supervision_objective_ablation_train import train_scalar_objective
from lafc.types import PageId, Request


CONDITION_C0_BASELINE_LRU = "C0_BASELINE_LRU"
CONDITION_C1_LRU_CONTINUATION_LEARNED_PI1 = "C1_LRU_CONTINUATION_LEARNED_PI1"
CONDITION_C2_PI1_CONTINUATION_LEARNED_PI2 = "C2_PI1_CONTINUATION_LEARNED_PI2"
CONDITION_C1_EXACT_Q_PI0 = "C1_EXACT_Q_PI0"

CONTINUATION_LRU = "lru"
CONTINUATION_FROZEN_PI1 = "frozen_pi1"


class ContinuationModel(Protocol):
    def predict_loss_batch(self, rows: List[Dict[str, float]]) -> List[float]:
        ...


@dataclass(frozen=True)
class ContinuationAblationConfig:
    horizon: int = 4
    history_window: int = 64


@dataclass(frozen=True)
class FrozenPi1Provenance:
    held_out_family: str
    validation_family: str
    training_families: Tuple[str, ...]
    model_path: str
    model_sha256: str
    registry_path: str
    registry_sha256: str
    objective: str = "objective_eviction_loss"
    protocol_id: str = "supervision_objective_ablation_v1"


@dataclass
class ContinuationState:
    order: "collections.OrderedDict[PageId, None]"
    bucket_by_page: Dict[PageId, int]
    confidence_by_page: Dict[PageId, float]
    recent_req_hist: Deque[PageId]
    recent_hit_hist: Deque[PageId]

    def copy(self, history_window: int) -> "ContinuationState":
        return ContinuationState(
            order=collections.OrderedDict((p, None) for p in self.order.keys()),
            bucket_by_page=dict(self.bucket_by_page),
            confidence_by_page=dict(self.confidence_by_page),
            recent_req_hist=collections.deque(self.recent_req_hist, maxlen=history_window),
            recent_hit_hist=collections.deque(self.recent_hit_hist, maxlen=history_window),
        )


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_self_hash(registry: Mapping[str, object]) -> str:
    payload = {k: v for k, v in registry.items() if k != "registry_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_frozen_pi1_from_registry(
    *,
    registry_path: str | Path,
    held_out_family: str,
    folds_dir: str | Path = "configs/fair_cross_family_v1/folds",
    objective: str = "objective_eviction_loss",
) -> Tuple[EvictValueV1Model, FrozenPi1Provenance]:
    """Load the eligible frozen pi1 model for one held-out fold.

    This is deliberately fail-closed: no unfrozen registry, missing record,
    model hash mismatch, held-out leakage, or fold mismatch is allowed.
    """
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("MODEL_SELECTION_FROZEN") is not True:
        raise ValueError(f"MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')}; refusing pi1 fallback")
    actual_registry_hash = _registry_self_hash(registry)
    recorded_registry_hash = str(registry.get("registry_sha256", ""))
    if recorded_registry_hash and actual_registry_hash != recorded_registry_hash:
        raise ValueError(
            f"registry hash mismatch: recorded={recorded_registry_hash} actual={actual_registry_hash}"
        )

    matches = [
        rec for rec in registry.get("records", [])
        if rec.get("objective") == objective and rec.get("held_out_family") == held_out_family
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one registry record for objective={objective} held_out_family={held_out_family}, "
            f"found {len(matches)}"
        )
    rec = matches[0]

    folds_dir = Path(folds_dir)
    fold = json.loads((folds_dir / f"{held_out_family}.json").read_text(encoding="utf-8"))
    if rec.get("fold_id") != fold.get("fold_id"):
        raise ValueError(f"fold_id mismatch for {held_out_family}: registry={rec.get('fold_id')} fold={fold.get('fold_id')}")
    training_families = tuple(str(x) for x in rec.get("training_families", []))
    validation_family = str(rec.get("validation_family", ""))
    if held_out_family in training_families:
        raise ValueError(f"held-out leakage: {held_out_family} appears in pi1 training_families")
    if validation_family == held_out_family:
        raise ValueError(f"held-out leakage: validation_family is held-out family {held_out_family}")
    if tuple(fold.get("training_families", [])) != training_families:
        raise ValueError("training_families mismatch between registry and fold config")
    if str(fold.get("validation_family", "")) != validation_family:
        raise ValueError("validation_family mismatch between registry and fold config")

    model_path = Path(str(rec["model_artifact_path"]))
    expected_name = f"{held_out_family}.pkl"
    if model_path.name != expected_name or model_path.parent.name != objective:
        raise ValueError(f"wrong-fold pi1 model path: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"pi1 model artifact not found: {model_path}")
    actual_model_hash = sha256_of_file(model_path)
    if actual_model_hash != rec.get("model_artifact_sha256"):
        raise ValueError(
            f"pi1 model hash mismatch: registry={rec.get('model_artifact_sha256')} disk={actual_model_hash}"
        )

    model = EvictValueV1Model.load(model_path)
    prov = FrozenPi1Provenance(
        held_out_family=held_out_family,
        validation_family=validation_family,
        training_families=training_families,
        model_path=str(model_path),
        model_sha256=actual_model_hash,
        registry_path=str(registry_path),
        registry_sha256=recorded_registry_hash or actual_registry_hash,
        objective=objective,
        protocol_id=str(rec.get("protocol_id", "supervision_objective_ablation_v1")),
    )
    return model, prov


def _update_request_metadata(state: ContinuationState, req: Request) -> None:
    pid = req.page_id
    if req.metadata.get("bucket") is not None:
        state.bucket_by_page[pid] = int(req.metadata["bucket"])
    if req.metadata.get("confidence") is not None:
        state.confidence_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))


def _feature_rows_for_current_state(
    *,
    state: ContinuationState,
    request: Request,
    history_window: int,
) -> Tuple[List[PageId], List[Dict[str, float]]]:
    candidates = list(state.order.keys())
    if not candidates:
        raise ValueError("cannot score learned continuation with an empty cache")
    req_bucket = int(request.metadata.get("bucket", 0))
    req_conf = float(request.metadata.get("confidence", 0.5))
    feat_rows: List[Dict[str, float]] = []
    for candidate in candidates:
        req_rate = (
            sum(1 for x in state.recent_req_hist if x == candidate) / len(state.recent_req_hist)
        ) if state.recent_req_hist else 0.0
        hit_rate = (
            sum(1 for x in state.recent_hit_hist if x == candidate) / len(state.recent_hit_hist)
        ) if state.recent_hit_hist else 0.0
        feat_rows.append(
            compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=candidate,
                bucket_by_page=state.bucket_by_page,
                confidence_by_page=state.confidence_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()
        )
    return candidates, feat_rows


def _choose_pi1_victim(
    *,
    model: ContinuationModel,
    state: ContinuationState,
    request: Request,
    history_window: int,
) -> PageId:
    candidates, feat_rows = _feature_rows_for_current_state(
        state=state,
        request=request,
        history_window=history_window,
    )
    preds = model.predict_loss_batch(feat_rows)
    if len(preds) != len(candidates):
        raise ValueError(f"pi1 returned {len(preds)} predictions for {len(candidates)} candidates")
    best_idx = min(range(len(candidates)), key=lambda i: (float(preds[i]), i))
    return candidates[best_idx]


def _apply_forced_candidate(
    *,
    base_state: ContinuationState,
    incoming: Request,
    candidate: PageId,
    history_window: int,
) -> ContinuationState:
    state = base_state.copy(history_window)
    if incoming.page_id in state.order:
        raise ValueError("incoming request is already in cache; no forced eviction decision exists")
    if candidate not in state.order:
        raise ValueError(f"forced candidate {candidate!r} is not in the current cache")
    state.order.pop(candidate)
    state.order[incoming.page_id] = None
    state.recent_req_hist.append(incoming.page_id)
    return state


def simulate_pi1_continuation_misses(
    *,
    pre_decision_state: ContinuationState,
    forced_candidate: PageId,
    incoming_request: Request,
    future_reqs: Sequence[Request],
    capacity: int,
    model: ContinuationModel,
    cfg: ContinuationAblationConfig,
) -> int:
    """Counterfactual H-step misses after forcing one candidate eviction.

    The forced candidate action is applied before any continuation step.
    Every later learned decision recomputes online features from the
    resulting state and mutates cache/history exactly like deployment.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if int(cfg.horizon) != len(future_reqs):
        # The caller must pass the exact finite-horizon window being labeled;
        # short trace suffixes should use a config matching that suffix.
        raise ValueError(f"future_reqs length {len(future_reqs)} does not match cfg.horizon {cfg.horizon}")

    state = _apply_forced_candidate(
        base_state=pre_decision_state,
        incoming=incoming_request,
        candidate=forced_candidate,
        history_window=cfg.history_window,
    )
    misses = 0
    for req in future_reqs:
        _update_request_metadata(state, req)
        pid = req.page_id
        if pid in state.order:
            state.order.move_to_end(pid)
            state.recent_req_hist.append(pid)
            state.recent_hit_hist.append(pid)
            continue

        misses += 1
        if len(state.order) >= capacity:
            victim = _choose_pi1_victim(
                model=model,
                state=state,
                request=req,
                history_window=cfg.history_window,
            )
            state.order.pop(victim)
        state.order[pid] = None
        state.recent_req_hist.append(pid)
    return misses


def build_decision_aligned_continuation_rows(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ContinuationAblationConfig,
    pi1_model: ContinuationModel,
    pi1_provenance: FrozenPi1Provenance,
    selected_decision_ids: Optional[set[str]] = None,
    max_decisions: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Build C1/C2 labels on the same decisions and candidates.

    State generation remains LRU so both conditions use identical decision
    IDs. The only label difference is LRU continuation vs frozen pi1
    continuation after the same forced candidate eviction.
    """
    if not pi1_provenance.model_sha256:
        raise ValueError("pi1 provenance must include a model_sha256")
    if pi1_provenance.held_out_family == pi1_provenance.validation_family:
        raise ValueError("pi1 provenance invalid: held-out family equals validation family")
    if pi1_provenance.held_out_family in pi1_provenance.training_families:
        raise ValueError("pi1 provenance invalid: held-out family appears in training_families")

    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    rows: List[Dict[str, object]] = []
    decisions_emitted = 0

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        hit = pid in order
        if hit:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        decision_id = f"{trace_name}|cap={capacity}|t={t}|h={cfg.horizon}"
        candidates = list(order.keys())
        if selected_decision_ids is not None and decision_id not in selected_decision_ids:
            victim = candidates[0]
            order.pop(victim)
            order[pid] = None
            recent_req_hist.append(pid)
            continue
        if max_decisions is not None and decisions_emitted >= max_decisions:
            break

        future = list(requests[t + 1 : t + 1 + cfg.horizon])
        if len(future) < cfg.horizon:
            break
        pre_decision_state = ContinuationState(
            order=collections.OrderedDict((p, None) for p in candidates),
            bucket_by_page=dict(bucket_by_page),
            confidence_by_page=dict(conf_by_page),
            recent_req_hist=collections.deque(recent_req_hist, maxlen=cfg.history_window),
            recent_hit_hist=collections.deque(recent_hit_hist, maxlen=cfg.history_window),
        )
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))

        c1_losses: Dict[PageId, float] = {}
        c2_losses: Dict[PageId, float] = {}
        feature_rows: Dict[PageId, Dict[str, float]] = {}
        for candidate in candidates:
            req_rate = (
                sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)
            ) if recent_req_hist else 0.0
            hit_rate = (
                sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)
            ) if recent_hit_hist else 0.0
            feature_rows[candidate] = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=candidate,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()
            after = [p for p in candidates if p != candidate] + [pid]
            c1_losses[candidate] = float(_simulate_lru_misses(after, future, capacity=capacity))
            c2_losses[candidate] = float(
                simulate_pi1_continuation_misses(
                    pre_decision_state=pre_decision_state,
                    forced_candidate=candidate,
                    incoming_request=req,
                    future_reqs=future,
                    capacity=capacity,
                    model=pi1_model,
                    cfg=cfg,
                )
            )

        c1_best = min(c1_losses.values())
        c2_best = min(c2_losses.values())
        c1_top = min(candidates, key=lambda p: (c1_losses[p], candidates.index(p)))
        c2_top = min(candidates, key=lambda p: (c2_losses[p], candidates.index(p)))
        for candidate in candidates:
            row: Dict[str, object] = {
                "trace_name": trace_name,
                "trace_family": trace_family,
                "capacity": int(capacity),
                "horizon": int(cfg.horizon),
                "decision_id": decision_id,
                "decision_t": int(t),
                "candidate_id": str(candidate),
                "candidate_page_id": candidate,
                "candidate_count": len(candidates),
                "c1_label": float(c1_losses[candidate]),
                "c2_label": float(c2_losses[candidate]),
                "label_delta": float(c2_losses[candidate] - c1_losses[candidate]),
                "c1_regret": float(c1_losses[candidate] - c1_best),
                "c2_regret": float(c2_losses[candidate] - c2_best),
                "c1_top1_candidate_id": str(c1_top),
                "c2_top1_candidate_id": str(c2_top),
                "top1_changed": float(c1_top != c2_top),
                "pi1_hash": pi1_provenance.model_sha256,
                "pi1_model_path": pi1_provenance.model_path,
                "pi1_held_out_family": pi1_provenance.held_out_family,
                "pi1_validation_family": pi1_provenance.validation_family,
                "pi1_training_families": ";".join(pi1_provenance.training_families),
                "continuation_mode_c1": CONTINUATION_LRU,
                "continuation_mode_c2": CONTINUATION_FROZEN_PI1,
                "condition_c1": CONDITION_C1_LRU_CONTINUATION_LEARNED_PI1,
                "condition_c2": CONDITION_C2_PI1_CONTINUATION_LEARNED_PI2,
            }
            row.update(feature_rows[candidate])
            rows.append(row)
        decisions_emitted += 1

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return rows


def label_agreement_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "n_decisions": 0.0,
            "c1_c2_label_agreement": 0.0,
            "mean_abs_label_delta": 0.0,
            "median_abs_label_delta": 0.0,
            "fraction_candidate_rankings_changed": 0.0,
            "fraction_top1_eviction_changed": 0.0,
        }
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)

    labels_equal = [float(row["c1_label"]) == float(row["c2_label"]) for row in rows]
    abs_delta = np.asarray([abs(float(row["label_delta"])) for row in rows], dtype=float)
    ranking_changed = 0
    top1_changed = 0
    for items in grouped.values():
        c1_rank = [str(r["candidate_id"]) for r in sorted(items, key=lambda r: (float(r["c1_label"]), str(r["candidate_id"])))]
        c2_rank = [str(r["candidate_id"]) for r in sorted(items, key=lambda r: (float(r["c2_label"]), str(r["candidate_id"])))]
        ranking_changed += int(c1_rank != c2_rank)
        top1_changed += int(str(items[0]["c1_top1_candidate_id"]) != str(items[0]["c2_top1_candidate_id"]))
    return {
        "n_rows": float(len(rows)),
        "n_decisions": float(len(grouped)),
        "c1_c2_label_agreement": float(sum(labels_equal) / len(labels_equal)),
        "mean_abs_label_delta": float(np.mean(abs_delta)),
        "median_abs_label_delta": float(np.median(abs_delta)),
        "fraction_candidate_rankings_changed": float(ranking_changed / max(len(grouped), 1)),
        "fraction_top1_eviction_changed": float(top1_changed / max(len(grouped), 1)),
    }


def train_pi2_from_c2_labels(
    *,
    train_rows: List[Dict[str, object]],
    val_rows: List[Dict[str, object]],
    seed: int,
    pi1_provenance: FrozenPi1Provenance,
) -> EvictValueV1Model:
    if not pi1_provenance.model_sha256:
        raise ValueError("pi2 training requires frozen pi1 model_sha256 provenance")
    result = train_scalar_objective(
        objective="continuation_policy_causal_ablation_pi2",
        label_column="c2_label",
        direction="min",
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=[],
        seed=seed,
    )
    return result.best_model


__all__ = [
    "CONDITION_C0_BASELINE_LRU",
    "CONDITION_C1_LRU_CONTINUATION_LEARNED_PI1",
    "CONDITION_C2_PI1_CONTINUATION_LEARNED_PI2",
    "CONDITION_C1_EXACT_Q_PI0",
    "ContinuationAblationConfig",
    "ContinuationState",
    "FrozenPi1Provenance",
    "build_decision_aligned_continuation_rows",
    "label_agreement_metrics",
    "load_frozen_pi1_from_registry",
    "sha256_of_file",
    "simulate_pi1_continuation_misses",
    "train_pi2_from_c2_labels",
]
