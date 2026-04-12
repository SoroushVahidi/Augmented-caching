# Consolidated manuscript-support note: subsection → artifacts → claims

**Sources:** `reports/manuscript_evidence_audit.md`, `reports/kbs_safe_paper_package.md`, `docs/kbs_knowledge_framing_note.md`, `docs/evict_value_v1_method_spec.md`, `docs/kbs_manuscript_workflow.md`, `scripts/paper/build_kbs_main_manuscript_artifacts.py`.

**Evidence gate (current repo):** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` — treat as **absent** unless you verify after pull (`reports/manuscript_artifacts/manuscript_artifact_report.md`).

Subsections below follow a **typical KBS-style empirical ML paper** (not a copy of an external outline). Map your own section numbers to these blocks.

---

## Introduction / problem setting

| | Content |
|---|--------|
| **Safe claims now** | Caching under uncertainty; learning-augmented caching as a research area; **high-level** motivation using `docs/kbs_knowledge_framing_note.md` (knowledge = metadata + state features + learned mapping). |
| **Blocked on heavy_r1 eval** | Any sentence implying **multi-trace Wulver replay** shows `evict_value_v1` beats or ties named baselines on misses. |
| **Beyond heavy_r1 (new experiments)** | Guard/fallback story with numbers (`evict_value_v1_guarded`); pairwise line; exploratory campaigns in `analysis/pairwise_*`. |

**Artifacts:** None required for prose-only intro; optional pointer to `docs/wulver_heavy_evict_value_experiment.md` for scope.

---

## Related work (LA caching, predictors, robust combiners)

| | Content |
|---|--------|
| **Safe claims now** | Cite baseline **papers** as documented in `docs/baselines.md` (TRUST&DOUBT, combiner, LA weighted paging, etc.) **as literature**, independent of this repo’s eval. |
| **Blocked on heavy_r1 eval** | Claiming this repo **reproduces** a given baseline’s paper numbers on Wulver traces. |
| **Beyond heavy_r1** | Systematic literature survey beyond what `baselines.md` lists. |

**Artifacts:** `docs/baselines.md` (paper-to-code mapping).

---

## Method (`evict_value_v1`)

| | Content |
|---|--------|
| **Safe claims now** | Full pipeline description from code: features (`EVICT_VALUE_V1_FEATURE_COLUMNS`), target `y_loss`, horizons, scorer modes, eviction rule — see `docs/evict_value_v1_method_spec.md`, `src/lafc/evict_value_features_v1.py`, `src/lafc/evict_value_wulver_v1.py`, `src/lafc/policies/evict_value_v1.py`. **Fig. 1** as schematic (`figures/manuscript/figure1_method_overview.*`). |
| **Blocked on heavy_r1 eval** | Claiming the **deployed** model in replay is optimal in any global sense; any **numeric** claim about end-to-end misses. |
| **Beyond heavy_r1** | Formal theorems; pairwise objective; guard dynamics — only if you add experiments/appendix. |

**Artifacts:** Fig. 1 + `figure1_snippet.tex`; method spec doc.

---

## Experimental setup — data & traces

| | Content |
|---|--------|
| **Safe claims now** | Manifest-driven traces: `analysis/wulver_trace_manifest_full.csv`; offline dataset summary `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`; **Table 1** roster (`tables/manuscript/table1_dataset_summary.*`) with caveat on **Main**/capacities until policy CSV exists. |
| **Blocked on heavy_r1 eval** | “These traces are exactly the end-to-end eval roster” without the policy CSV. |
| **Beyond heavy_r1** | Additional traces not in manifest; new dataset builds (`slurm/evict_value_v1_wulver_dataset*.sbatch`) labeled non-canonical per workflow. |

**Artifacts:** Table 1, `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` (if present).

---

## Experimental setup — baselines and policies

| | Content |
|---|--------|
| **Safe claims now** | **Table 2** roster (`table2_policy_roster.*`) lists **planned** comparison policies; implementation names in `scripts/run_policy_comparison_wulver_v1.py`. Literature mapping in `docs/baselines.md`. |
| **Blocked on heavy_r1 eval** | Empirical ranking or miss counts for those policies. |
| **Beyond heavy_r1** | Extra policies only in exploratory CSVs; `blind_oracle` in eval CSV but **excluded** from manuscript Table 3 subset — explain if you mention full eval driver. |

**Artifacts:** Table 2; `slurm/evict_value_v1_wulver_heavy_eval.sbatch` (full policy list for **eval run**).

---

## Experimental setup — training / offline ablation

| | Content |
|---|--------|
| **Safe claims now** | Split protocol, horizons {4,8,16}, models {ridge, RF, HistGB}, selection rule, metrics — backed by `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, `train_metrics_heavy_r1.json`, `best_config_heavy_r1.json`, **Table 4**, **Fig. 4**; optional **Table 5 / Fig. 5** when policy CSV absent (offline-only). |
| **Blocked on heavy_r1 eval** | — (offline path is independent). |
| **Beyond heavy_r1** | Feature ablations, other model families, different split seeds — new runs. |

**Artifacts:** Table 4, Fig. 4; Table 5, Fig. 5 (temporary supplements).

---

## Results — end-to-end replay (main quantitative)

| | Content |
|---|--------|
| **Safe claims now** | — **Nothing numeric** without `policy_comparison_heavy_r1.csv`. May describe **what will be reported** (mean misses by family/capacity) per `run_policy_comparison_wulver_v1.py`. |
| **Blocked on heavy_r1 eval** | **Table 3**, **Fig. 2**, **Fig. 3** — entire subsection. |
| **Beyond heavy_r1** | Statistical tests not in repo; guard A/B; other trace suites. |

**Artifacts:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` → builder outputs `table3_*`, `figure{2,3}_*`.

---

## Discussion / limitations

| | Content |
|---|--------|
| **Safe claims now** | Short-horizon teacher ≠ global optimum; metadata dependence; scope of Wulver manifest; **absence** of canonical eval artifact if still missing. |
| **Blocked on heavy_r1 eval** | Discussion of **which family** `evict_value_v1` wins/loses without CSV. |
| **Beyond heavy_r1** | Broader deployment claims; cost-aware caching; pairwise theory. |

**Artifacts:** `docs/manuscript_open_questions.md` (optional pointer).

---

## Summary table

| Subsection (typical) | Safe now (artifacts) | Blocked until heavy_r1 eval CSV | Needs new experiments (not in heavy_r1) |
|---------------------|----------------------|----------------------------------|----------------------------------------|
| Intro / framing | Narrative + framing note | Empirical Wulver win claims | Guard numbers, pairwise main story |
| Related work | `baselines.md` citations | — | Extra papers |
| Method | Fig. 1, method spec | End-to-end numeric performance | Theory |
| Data / traces | Table 1, dataset summary md | Confirmed eval roster column | New traces |
| Baselines | Table 2 | Replay results | Extra policies analysis |
| Offline study | T4, F4; T5/F5 if no policy CSV | — | Feature ablations |
| Main results | — | **T3, F2, F3** | Significance tests |
| Limitations | Evidence map, workflow | Family-level discussion | — |
