# Internal author note: `evict_value_v1` method & experiments (code/docs only)

Use while rewriting **Method** and **Experiments**. Source: `src/lafc/evict_value_features_v1.py`, `src/lafc/evict_value_wulver_v1.py`, `src/lafc/policies/evict_value_v1.py`, `scripts/build_evict_value_dataset_wulver_v1.py`, `scripts/train_evict_value_wulver_v1.py`, `scripts/run_policy_comparison_wulver_v1.py`, `slurm/evict_value_v1_wulver_heavy_*.sbatch`, `docs/wulver_heavy_evict_value_experiment.md`.

---

## Notation (candidates)

- \(t\): request index along trace; decision at full cache on a **miss**.
- \(H \in \{4,8,16\}\): **replay horizon** for supervised targets (heavy\_r1 default).
- \(c\): a **candidate** resident page to evict; \(\mathcal{C}_t\): current cache contents (ordered; LRU order used in code).
- **Features** \(\phi(c \mid t) \in \mathbb{R}^{26}\): vector aligned with `EVICT_VALUE_V1_FEATURE_COLUMNS` (fixed order).
- **Label** \(y_{\mathrm{loss}}(c,t,H)\): counterfactual **miss count** in next \(H\) requests after hypothetically evicting \(c\) and inserting requestor, simulated with **LRU** on that counterfactual state (`_simulate_lru_misses`).
- **Prediction** \(\hat{y}(c \mid t)\): regressor output; policy evicts \(\arg\min_c \hat{y}(c \mid t)\) (tie-break: LRU index).

---

## Feature groups (one paragraph each in Method)

1. **Request context:** `request_bucket`, `request_confidence`.
2. **Candidate identity / position:** bucket, confidence, recency rank, normalized age in LRU list.
3. **vs. predictor & LRU victims:** scores, indicator victims, gaps (score/bucket/confidence) to predictor-best and LRU victim (`lafc.learned_gate.features`).
4. **Cache distribution:** mean/std/min/max bucket, unique bucket count, confidence mean/std.
5. **Disagreement:** `predictor_lru_disagree`.
6. **Recent activity:** rates from bounded deques of recent request/hit page ids.

---

## Target definition (single precise paragraph)

At each full-cache miss, for **each** candidate \(c\), build counterfactual cache, simulate LRU on the **next \(H\)** requests, count misses → \(y_{\mathrm{loss}}\). Supervised learning fits \(\hat{y} \approx y_{\mathrm{loss}}\) on **train** shards only.

---

## Split protocol

- **Mode:** `trace_chunk` (heavy default): chunk id \(= \lfloor t / 4096 \rfloor\) per trace; hash key with `split_seed` (heavy train: **7**) into train/val/test by **70 / 15 / 15** percent buckets (`assign_split` in `evict_value_wulver_v1.py`).
- Training script loads rows with `split` ∈ {train,val,test} per horizon; optional max-row caps with deterministic subsampling.

---

## Model & selection rule

- **Models:** Ridge (`alpha=1.0`), RandomForest (80 trees, depth 12, …), HistGradientBoosting (depth 6, lr 0.05, 150 iter) — see `train_evict_value_wulver_v1.py`.
- **Per-horizon winner:** minimize \((\texttt{val\_mean\_regret}, \texttt{val\_mae}, \texttt{val\_rmse})\).
- **Global deployed pick:** same ordering over **all** (horizon, model) pairs → `best_config` JSON; copy to `evict_value_wulver_v1_best.pkl` (+ `*_heavy_r1` copy on cluster).

**Metrics:** regression MAE/RMSE; **ranking**: mean regret vs oracle per decision group, Top-1 eviction match (training script definitions).

---

## Baseline subset — **main paper table/figures** (manuscript bundle)

Six policies **only** in Table 3 / Fig. 2–3 filters:

`lru`, `predictive_marker`, `trust_and_doubt`, `blind_oracle_lru_combiner`, `rest_v1`, `evict_value_v1`

**Eval driver** may also run `blind_oracle` — **excluded** from manuscript main comparison (see `evict_value_v1_method_spec.md` §9).

---

## Figure / table availability (builder)

| Item | Source artifact | When available |
|------|-----------------|----------------|
| Fig. 1 | `make_method_overview_two_panel_figure` | Always (schematic) |
| Table 1 | manifest + dataset summary | Always; caveat if no policy CSV |
| Table 2 | static roster | Always |
| Table 4, Fig. 4 | `model_comparison_heavy_r1.csv` | Always if train CSV committed |
| Table 5, Fig. 5 | best_config + model_comparison | **Only when policy CSV absent** (temporary); placeholders when CSV present |
| Table 3, Fig. 2–3 | `policy_comparison_heavy_r1.csv` | **Only after heavy eval** |

---

*For positioning language, see `docs/kbs_knowledge_framing_note.md`.*
