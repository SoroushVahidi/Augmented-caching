# Method detail support note — `evict_value_v1` (Wulver / KBS line)

**Purpose:** Internal consolidation of **methodological facts already stated** in this repository for the **`evict_value_v1`** pointwise line (dataset → train → replay eval design). This is **not** manuscript prose, **not** a new experimental result, and **not** a substitute for canonical `heavy_r1` eval artifacts.

**Primary sources (read for edits):** `docs/evict_value_v1_method_spec.md`, `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`, `docs/wulver_heavy_evict_value_experiment.md`, `docs/evict_value_v1_kbs_canonical_artifacts.md`, `docs/decision_aligned_eviction_targets.md` (scope boundary vs v1), `src/lafc/evict_value_features_v1.py`, `src/lafc/evict_value_wulver_v1.py`, `slurm/evict_value_v1_wulver_heavy_train.sbatch`.

---

## 1. Learning target

- **Supervision target (per row):** `y_loss` = number of **cache misses** in the **next \(H\) requests** after a hypothetical eviction, simulated on the **counterfactual cache** for that candidate; `y_value = -y_loss` (convenience).
- **Code path:** `iter_candidate_rows` in `src/lafc/evict_value_wulver_v1.py` uses `_simulate_lru_misses(...)` from `src/lafc/evict_value_dataset_v1.py` for the length-\(H\) future prefix (`docs/evict_value_v1_method_spec.md` §3).
- **Interpretation (repo-stated):** finite-horizon **counterfactual short-horizon cost**, not infinite-horizon optimal cost unless \(H\) spans the entire trace (it does not in the heavy defaults).

---

## 2. Candidate-example construction

- **When:** **Full-cache misses** only — a decision point where an eviction is required (`docs/evict_value_v1_method_spec.md`).
- **Who:** For each **resident page** \(c\) (candidate victim) in the **LRU-ordered** candidate set at that time:
  1. Form the **counterfactual cache** after the miss: remove \(c\), insert the **current requesting** page.
  2. For each configured **horizon** \(H \in\) dataset `horizons`, take `future = requests[t+1 : t+1+H]`.
  3. Compute `y_loss` via LRU continuation simulation on that prefix (see §3).
- **Row identity:** Rows carry keys such as `decision_id`, `candidate_page_id`, `horizon`, plus the feature vector; written to **shards** under the derived dataset directory (`docs/wulver_heavy_evict_value_experiment.md` output layout).
- **Horizon replication:** Each `(decision, candidate)` is duplicated **once per \(H\)** with the matching `y_loss` (`docs/evict_value_v1_method_spec.md` §4).

---

## 3. Continuation rule used for labels (**v1**)

- **For `evict_value_v1` Wulver labels:** continuation inside the label simulation is **LRU**: `_simulate_lru_misses` on the counterfactual cache for the next \(H\) requests (`docs/evict_value_v1_method_spec.md` §3; import in `src/lafc/evict_value_wulver_v1.py`).
- **Not the same object as v2 rollouts:** `docs/decision_aligned_eviction_targets.md` describes an **experimental v2** pipeline where rollout labels can use continuation policies **`lru` (default) or `blind_oracle`** over the same finite window. That document explicitly frames v2 as a **separate** dataset family (`build_evict_value_v2_rollout_dataset.py`, etc.). **Do not** describe v1 heavy supervision as offering a configurable oracle continuation unless the manuscript is clearly scoped to v2.

---

## 4. Feature families and concrete feature names

**Canonical ordered list (26 columns):** `EVICT_VALUE_V1_FEATURE_COLUMNS` in `src/lafc/evict_value_features_v1.py` — reproduced here for rewriting convenience (order matches training data):

1. `request_bucket`
2. `request_confidence`
3. `candidate_bucket`
4. `candidate_confidence`
5. `candidate_recency_rank`
6. `candidate_age_norm`
7. `candidate_predictor_score`
8. `candidate_lru_score`
9. `candidate_is_predictor_victim`
10. `candidate_is_lru_victim`
11. `score_gap_to_predictor_best`
12. `score_gap_to_lru_victim`
13. `bucket_gap_to_predictor_best`
14. `bucket_gap_to_lru_victim`
15. `confidence_gap_to_predictor_best`
16. `confidence_gap_to_lru_victim`
17. `cache_bucket_mean`
18. `cache_bucket_std`
19. `cache_bucket_min`
20. `cache_bucket_max`
21. `cache_unique_bucket_count`
22. `cache_confidence_mean`
23. `cache_confidence_std`
24. `predictor_lru_disagree`
25. `recent_candidate_request_rate`
26. `recent_candidate_hit_rate`

**Semantic grouping (from `docs/evict_value_v1_method_spec.md` §2):**

| Group | Role (repo wording) |
|-------|---------------------|
| Request context | `request_*` from current request metadata (prediction-time). |
| Candidate identity | `candidate_bucket`, `candidate_confidence`, LRU position / normalized age. |
| Predictor vs LRU structure | Predictor and LRU scores, victim flags, gaps to “best” predictor / LRU victim. |
| Cache distribution | Mean/std/min/max bucket, unique bucket count, confidence mean/std over current cache. |
| Disagreement | `predictor_lru_disagree` — predictor-chosen victim ≠ LRU victim. |
| Recent activity | Rates from **bounded deques** of recent request/hit page ids (policy-aligned state). |

**Construction parity:** Online policy `EvictValueV1Policy` and offline builder both use `compute_candidate_features_v1` (`docs/evict_value_v1_method_spec.md` §2).

**Trace-side fields feeding metadata:** JSONL ingestion attaches optional per-record `metadata.bucket` and `metadata.confidence` into `Request` / page structures (`src/lafc/evict_value_wulver_v1.py` `_parse_jsonl_trace` — summarized in `docs/evict_value_v1_method_spec.md`).

**Config default touching features:** `WulverDatasetConfig.history_window` defaults to **64** in `src/lafc/evict_value_wulver_v1.py` (used in the Wulver dataset path alongside chunking/splitting).

---

## 5. Data splits and leakage prevention

### 5.1 Split mode (`trace_chunk`, heavy default)

- `chunk_id = floor(t / chunk_size)` with default **`chunk_size = 4096`** (`WulverDatasetConfig`, `slurm/evict_value_v1_wulver_heavy_train.sbatch`).
- Split key: `trace={trace_name}|chunk={chunk_id}`.
- Stable hash into \([0,100)\) with **`split_seed`** (heavy train batch default **`SPLIT_SEED=7`**).
- Assignment (`assign_split` in `src/lafc/evict_value_wulver_v1.py`):
  - **`train`** if bucket `< split_train_pct` (default **70**),
  - **`val`** if `< split_train_pct + split_val_pct` (default **70 + 15**),
  - else **`test`** (remaining **15%** implied).

**Alternative mode (not heavy default):** `source_family` uses key `source={dataset_source}|family={trace_family}`.

### 5.2 Leakage controls (repo-stated)

From `docs/evict_value_v1_method_spec.md` §8:

- Training uses only `split=="train"`; metrics on `val` / `test` respectively; no cross-split mixing in `_load_rows_from_manifest`.
- Labels use requests **strictly after** decision index \(t\).
- Features at \(t\) use state and metadata available **at** the decision plus bounded history; no peeking at post-decision labels when forming features.
- Subsampling respects `split`; very large shards may be **partially read** under row caps (training script).

### 5.3 Heavy `_heavy_r1` dataset scale (artifact-backed counts)

From `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`:

- **Totals:** 289,304,256 rows; 2,524,527 unique `decision_id`.
- **By split (rows):** train 213,173,952; val 47,277,312; test 28,852,992.
- **Manifest meta:** `split_mode: trace_chunk`, `chunk_size: 4096`, `capacities: [32,64,128,256]`, `horizons: [4,8,16]`, `trace_count: 7`, `shard_count: 594`.

**Family × split:** the same markdown lists row counts per **trace_family** × **split** (e.g. `metacdn` has only train/val rows in that snapshot — cite the file directly when writing, do not paraphrase as “missing test” without the table).

---

## 6. Training / validation / test setup

- **Input:** Shard manifest under `data/derived/evict_value_v1_wulver_{EXP_TAG}/` (for `heavy_r1`: paths in `docs/wulver_heavy_evict_value_experiment.md`).
- **Trainer:** `scripts/train_evict_value_wulver_v1.py` loads rows per horizon respecting `split`.
- **Subsampling caps (heavy train batch defaults):** `MAX_TRAIN_ROWS=800000`, `MAX_VAL_ROWS=250000`, `MAX_TEST_ROWS=250000` per horizon (`slurm/evict_value_v1_wulver_heavy_train.sbatch` comments).
- **RNG:** `--seed` default **7** in `scripts/train_evict_value_wulver_v1.py`.
- **Output artifacts (heavy_r1):** `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`, `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, `analysis/evict_value_wulver_v1_best_config_heavy_r1.json`, model copies `models/evict_value_wulver_v1_best.pkl` / `models/evict_value_wulver_v1_best_heavy_r1.pkl` (`docs/wulver_heavy_evict_value_experiment.md`).

---

## 7. Model families and selection criteria

### 7.1 Families

- **`ridge`**, **`random_forest`**, **`hist_gb`** trained per horizon (`docs/evict_value_v1_method_spec.md` §7).

### 7.2 Fixed hyperparameters (code-level, from method spec)

- `Ridge(alpha=1.0)`
- `RandomForestRegressor` with `n_estimators=80`
- `HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=150)`
- `random_state=args.seed` (default 7)

### 7.3 Selection rule

- **Per horizon:** minimize lexicographic tuple `(val_mean_regret, val_mae, val_rmse)` where `val_mean_regret` is **mean regret vs oracle** on validation rows (`docs/evict_value_v1_method_spec.md` §7; `_ranking_metrics` in `scripts/train_evict_value_wulver_v1.py`).
- **Global best across horizons:** same ordering over all `(horizon, model)` rows; best written to `best_config` JSON and best weights copied to `evict_value_wulver_v1_best.pkl`.

**Committed `heavy_r1` best config snapshot:** `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` currently records `horizon: 4`, `model: hist_gb`, and the string `selection_rule` as in that file.

---

## 8. Experimental scope defaults (heavy `heavy_r1`)

From `docs/wulver_heavy_evict_value_experiment.md` + train batch:

| Knob | Default |
|------|---------|
| Trace manifest | `analysis/wulver_trace_manifest_full.csv` |
| Families (via manifest) | brightkite, citibike, wiki2018, twemcache, metakv, metacdn, cloudphysics |
| Capacities | 32, 64, 128, 256 |
| Horizons | 4, 8, 16 |
| Max requests per trace | 50,000 |
| Split | `trace_chunk`, seed 7, chunk 4096 |
| Slurm | `general` / `standard`, CPU-only |

**Caveat (repo-stated):** `wiki2018` is a pageview-derived proxy; interpret accordingly (`docs/wulver_heavy_evict_value_experiment.md`).

---

## 9. Artifact paths (canonical naming)

**KBS builder inputs (`docs/evict_value_v1_kbs_canonical_artifacts.md`):**

| Role | Path |
|------|------|
| Policy comparison (CSV) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` |
| Policy comparison (MD) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` |
| Dataset summary | `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` |
| Train metrics | `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` |
| Model comparison | `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` |
| Best config | `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` |
| Trace manifest | `analysis/wulver_trace_manifest_full.csv` |

**Derived data (disk, per runbook):** `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json`, `split_summary.csv`, `dataset_summary_extended_heavy_r1.json`.

**Do not interchange:** unsuffixed `analysis/evict_value_wulver_v1_policy_comparison.csv` may include **extra policies** vs the heavy eval driver (`docs/wulver_heavy_evict_value_experiment.md`, `analysis/README.md`).

---

## 10. What is artifact-backed vs exploratory

| Topic | Artifact-backed for KBS `heavy_r1` main line | Exploratory / out of scope |
|-------|-----------------------------------------------|----------------------------|
| Dataset construction + splits + counts | `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`, derived manifest family | Other `EXP_TAG` dirs without cross-walk |
| Offline model/horizon ablation | `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, `best_config_heavy_r1.json`, train metrics JSON | Unsuffixed `*_model_comparison.csv` etc. |
| End-to-end replay table for main claims | **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** only when present | Legacy unsuffixed policy CSV; `heavy_smoke` wiring outputs |
| Guarded policy | Spec in `docs/guarded_robust_wrapper.md` | Not in canonical `EVIDENCE_FILES` list in method spec §10 |
| v2 decision-aligned rollouts / pairwise | N/A for v1 heavy path | `docs/decision_aligned_eviction_targets.md`, v2 scripts |

---

## 11. What can safely be stated in the manuscript **now** (conservative)

**Supported without new runs (methods / setup):** Any statement faithfully summarizing §1–§9 above and pointing to the same source files — feature list, label definition, LRU continuation for v1 labels, split logic, model families, selection tuple, heavy defaults, artifact naming, and the wiki2018 caveat.

**Supported with committed numbers (offline training story):** Tables/figures built from `model_comparison_heavy_r1.csv` / `best_config_heavy_r1.json` / dataset summary — consistent with `reports/manuscript_artifacts/kbs_evidence_alignment_report.md` (offline supplements vs end-to-end).

**Not supported as completed canonical KBS quantitative replay:** Broad claims that the **final** `heavy_r1` multi-trace replay comparison **has been measured and tabulated** in-repo unless `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` is actually present (see `docs/kbs_manuscript_workflow.md`, `reports/manuscript_artifacts/manuscript_artifact_report.md`).

**Evaluated policy set vs table subset:** Heavy eval **driver** default policies include `blind_oracle`; manuscript **main** table/figure subset may **filter** to six policies (see `docs/evict_value_v1_method_spec.md` §9). Align text with whichever subset the paper cites.

---

## 12. Method details still not fully recoverable from **current artifacts** alone

The following may require **reading implementation** beyond this note, or are **genuinely under-documented** in prose artifacts:

- **Exact numeric definitions** of `candidate_predictor_score`, `candidate_lru_score`, and gap fields: repo points to `compute_predictor_scores` / `compute_lru_scores` in `lafc.learned_gate.features` (`docs/evict_value_v1_method_spec.md`) but this support note does not duplicate formulas from that module.
- **Per-dataset preprocessing** beyond JSONL fields (`item_id`, `metadata`): trace preparation pipelines live under `scripts/datasets/` and `docs/datasets.md` — not re-summarized here.
- **Oracle / regret definition** at the row level for `val_mean_regret`: specified qualitatively in the method spec and `_ranking_metrics`; full equation lives in `scripts/train_evict_value_wulver_v1.py`.
- **Whether `history_window` is overridden** from 64 in any committed `heavy_r1` run: defaults are in code/batch; confirm against `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` if the manuscript must state the exact value used for the committed dataset.

When in doubt, cite **`docs/evict_value_v1_method_spec.md`** and the **code files it lists** rather than paraphrasing from memory.
