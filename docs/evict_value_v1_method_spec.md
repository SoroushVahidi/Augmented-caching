# `evict_value_v1` method specification (repository-derived)

This document summarizes **only** what is implemented in code and described in in-repo docs—primarily `src/lafc/policies/evict_value_v1.py`, `src/lafc/evict_value_features_v1.py`, `src/lafc/evict_value_wulver_v1.py`, `scripts/build_evict_value_dataset_wulver_v1.py`, `scripts/train_evict_value_wulver_v1.py`, `scripts/run_policy_comparison_wulver_v1.py`, and `docs/wulver_heavy_evict_value_experiment.md`.

---

## 1. Role in the system

**`evict_value_v1`** is a **direct eviction-value policy**: on a full cache, it assigns a scalar **predicted loss** to each resident page (candidate) and evicts a minimizer (tie-break: LRU order index). Scoring uses a trained regressor when available (`EvictValueV1Model` / sklearn estimator inside a serialized artifact).

---

## 2. Feature list and groups

**Canonical column list:** `EVICT_VALUE_V1_FEATURE_COLUMNS` in `src/lafc/evict_value_features_v1.py` (26 features).

**Informal groups:**

| Group | Features (names) | Meaning |
|-------|------------------|--------|
| Request context | `request_bucket`, `request_confidence` | From current request metadata (prediction-time signals). |
| Candidate identity | `candidate_bucket`, `candidate_confidence`, `candidate_recency_rank`, `candidate_age_norm` | Per-candidate metadata and position in LRU order. |
| vs. predictor / LRU victims | `candidate_predictor_score`, `candidate_lru_score`, `candidate_is_predictor_victim`, `candidate_is_lru_victim`, `score_gap_*`, `bucket_gap_*`, `confidence_gap_*` | From `compute_predictor_scores` / `compute_lru_scores` over current candidates (`lafc.learned_gate.features`). |
| Cache distribution | `cache_bucket_mean/std/min/max`, `cache_unique_bucket_count`, `cache_confidence_mean/std` | Aggregates over current cache set. |
| Disagreement | `predictor_lru_disagree` | Whether predictor-chosen victim ≠ LRU victim. |
| Recent activity | `recent_candidate_request_rate`, `recent_candidate_hit_rate` | Rates derived from deques of recent request/hit page ids (policy state). |

Online policy (`EvictValueV1Policy`) fills these consistently with dataset construction (`compute_candidate_features_v1`).

---

## 3. Target construction (supervised label)

**Dataset builder:** `iter_candidate_rows` in `src/lafc/evict_value_wulver_v1.py`.

- **Decision points:** Full-cache misses (must evict). For each candidate page \(c\) in the LRU-ordered set, the builder forms a **counterfactual cache** after a miss: remove \(c\), insert the requesting page.
- **Horizon:** For each configured `horizon` \(H \in\) `cfg.horizons`, take the next \(H\) requests (`future = requests[t+1:t+1+H]`).
- **Label `y_loss`:** `_simulate_lru_misses(after, fut_h, capacity)` in `lafc/evict_value_dataset_v1.py` — LRU simulation on that counterfactual cache for the length-\(H\) future prefix; **miss count** in that window.
- **`y_value`:** `-y_loss` (stored for convenience).
- Rows are keyed by `decision_id`, `candidate_page_id`, `horizon`, etc., and written to shards with full feature vectors.

**Interpretation:** The label is a **counterfactual short-horizon cost** under LRU dynamics on the chosen eviction, not the global infinite-horizon optimum unless \(H\) is taken to the trace end (it is not).

---

## 4. Replay horizon definition

- **Training horizons:** Configured on the dataset (`WulverDatasetConfig.horizons`) — heavy runbook default **`4, 8, 16`** (`docs/wulver_heavy_evict_value_experiment.md`, `slurm/evict_value_v1_wulver_heavy_train.sbatch`).
- Each training example is duplicated per horizon with the corresponding `y_loss` for that \(H\).

---

## 5. Scorer modes (inference)

From `EvictValueV1Policy` (`src/lafc/policies/evict_value_v1.py`):

| Mode | Behavior |
|------|----------|
| `artifact` | Require model file at `model_path`; load `EvictValueV1Model` and predict. |
| `lightweight` | Use `_LinearTextSurrogateScorer` (debug / no artifact). |
| `auto` | Use artifact if `model_path` exists; otherwise lightweight surrogate. |

**Victim choice:** `argmin` predicted loss over candidates (deterministic tie-break).

---

## 6. Train / validation / test split logic

**Shard-time split** (`assign_split` in `src/lafc/evict_value_wulver_v1.py`):

- **Modes:** `trace_chunk` (default heavy) or `source_family`.
- **`trace_chunk`:** `chunk_id = floor(t / chunk_size)`; key `trace={trace_name}|chunk={chunk_id}`; hashed into \([0,100)\) with `split_seed`; **train** if `< train_pct`, **val** if `< train_pct + val_pct`, else **test**. Defaults: `chunk_size=4096`, `split_train_pct=70`, `split_val_pct=15` → **15% test** implied.
- Rows carry a `split` column consumed by training (`scripts/train_evict_value_wulver_v1.py` loads `train` / `val` / `test` per horizon).

**Training subsampling:** Optional `--max-train-rows`, `--max-val-rows`, `--max-test-rows` shuffle caps per horizon (deterministic RNG seeds per split).

---

## 7. Hyperparameter selection rule

**Per horizon:** Among `ridge`, `random_forest`, `hist_gb`, pick **minimum** tuple  
`(val_mean_regret, val_mae, val_rmse)`  
where `val_mean_regret` is **mean regret vs. oracle** on validation rows (see `_ranking_metrics` in `train_evict_value_wulver_v1.py`).

**Global best:** Same ordering over **all** (horizon, model) rows in `comparison_rows`; best row’s horizon/model written to `best_config` JSON; model copied to `evict_value_wulver_v1_best.pkl` (heavy\_r1 tagged copy in Slurm: `models/evict_value_wulver_v1_best_heavy_r1.pkl`).

**Fixed model hyperparameters** (code): e.g. `Ridge(alpha=1.0)`, RF `n_estimators=80`, `HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=150)`, `random_state=args.seed` (default **7**).

---

## 8. Leakage prevention (what the code enforces)

- **Split isolation:** Training only sees rows with `split=="train"`; validation metrics on `val`; test on `test`. No cross-split mixing in `_load_rows_from_manifest`.
- **Causal ordering:** Labels use only requests **strictly after** decision time \(t\) (`future = requests[t+1:]`).
- **Features at \(t\):** Built from cache state and request metadata available **at** the decision (plus bounded recent deques). No peeking at post-decision labels when forming features.
- **Note:** Subsampling reads shards in randomized order but respects split column; extremely large shards could be partially read—training script caps rows for tractability.

---

## 9. Baseline definitions and canonical KBS comparison subset

### 9.1 Eval driver (Slurm `heavy_eval`)

Default **`BASELINE_POLICIES`** (comma-separated) in `slurm/evict_value_v1_wulver_heavy_eval.sbatch`:

`lru`, `blind_oracle`, `predictive_marker`, `blind_oracle_lru_combiner`, `trust_and_doubt`, `rest_v1`, `evict_value_v1`

Implemented in `scripts/run_policy_comparison_wulver_v1.py` via `POLICIES` registry; `evict_value_v1` receives `--evict-value-model` path.

### 9.2 Manuscript bundle (table/figures)

`TABLE3_POLICIES` / `MAIN_PERF_POLICIES` in `scripts/paper/build_kbs_main_manuscript_artifacts.py` and `scripts/paper/manuscript_figure_common.py` — **six policies** for main tables/figures:

`lru`, `predictive_marker`, `trust_and_doubt`, `blind_oracle_lru_combiner`, `rest_v1`, `evict_value_v1`

**`blind_oracle` is run in the eval CSV but filtered out** of the manuscript main comparison table/figures. Extra policies in an unsuffixed legacy CSV are **not** interchangeable with `*_heavy_r1` (per `docs/wulver_heavy_evict_value_experiment.md`).

---

## 10. Related but out of scope for this spec

- **`evict_value_v1_guarded`:** `lafc/policies/guard_wrapper.py` — not part of canonical `EVIDENCE_FILES` for KBS tables in the builder.
- **Pairwise line:** separate scripts/policies; not `evict_value_v1` pointwise path.
