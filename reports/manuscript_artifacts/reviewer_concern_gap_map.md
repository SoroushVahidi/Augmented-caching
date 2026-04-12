# Reviewer concern gap map (internal repo audit)

**Purpose:** Map the *sample review* concerns below to **current repository artifacts only**. No new experiments were run. This is **not** manuscript prose.

**Evidence gate (canonical KBS `heavy_r1` online replay):** In this workspace,  
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` is **absent** (verified with `test -f` on 2026-04-12).  
`reports/manuscript_artifacts/manuscript_artifact_report.md` records `Policy comparison CSV present: **False**`.

**Status legend:** `FULLY_SUPPORTED` | `PARTIALLY_SUPPORTED` | `NOT_SUPPORTED` (for defensible, manuscript-aligned claims).

**Recommendation legend:** `REWRITE_ONLY` | `REWRITE_PLUS_EXISTING_ARTIFACTS` | `NEEDS_NEW_EXPERIMENTAL_EVIDENCE`

---

## 1. “Strongest evidence is only the offline scorer-selection ablation”

| Field | Content |
|--------|---------|
| **Concern** | Main quantitative story reads as offline model/horizon selection, not online policy quality. |
| **Repo evidence** | **PARTIALLY_SUPPORTED** for offline ablation; **NOT_SUPPORTED** for canonical *primary* online story until `policy_comparison_heavy_r1.csv` exists. |
| **Supporting artifacts** | `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` (horizons × models: regret / top-1); `analysis/evict_value_wulver_v1_best_config_heavy_r1.json`; `tables/manuscript/table4_main_ablation.csv`; `figures/manuscript/figure4_ablation.{pdf,png}`; `tables/manuscript/table5_offline_selection.csv`; `figures/manuscript/figure5_offline_top1_ablation.{pdf,png}`; `reports/manuscript_artifacts/kbs_evidence_alignment_report.md` (explicitly separates offline supplements from end-to-end). |
| **Missing** | Canonical **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** (and rebuilt Table~3 / Fig.~2–3 from `build_kbs_main_manuscript_artifacts.py`). |
| **Revise without new heavy runs?** | **Partially:** tighten claims and captions using existing alignment notes; **cannot** honestly claim completed canonical online bundle. |
| **Recommendation** | **REWRITE_PLUS_EXISTING_ARTIFACTS** (honest scope + cite offline tables) **and** **NEEDS_NEW_EXPERIMENTAL_EVIDENCE** for the intended *primary* online table once eval is allowed to finish. |

---

## 2. “Lacks convincing end-to-end online replay vs strong baselines”

| Field | Content |
|--------|---------|
| **Concern** | Need main miss-count / hit-rate tables across traces and capacities under trace replay, vs strong baselines. |
| **Repo evidence** | **NOT_SUPPORTED** for the **canonical KBS** path today (missing `*_heavy_r1` policy comparison). **PARTIALLY_SUPPORTED** if one counts **non-canonical** replay CSVs (see caveat). |
| **Supporting artifacts** | **Design / protocol:** `docs/wulver_heavy_evict_value_experiment.md` (baseline set, capacities, 50k requests/trace, CPU-only); `slurm/evict_value_v1_wulver_heavy_eval.sbatch`; `docs/evict_value_v1_kbs_canonical_artifacts.md`. **Non-canonical replay numbers (do not substitute for KBS per repo):** `analysis/evict_value_wulver_v1_policy_comparison.csv` (multi-policy replay rows exist but docs warn extra policies vs heavy driver). **Wiring-scale replay:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.csv` (subset; not `heavy_r1`). |
| **Missing** | **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** (+ optional `.md`). |
| **Revise without new heavy runs?** | **No** for a completed canonical online table; **yes** only to describe *planned* eval and current stub state (`tables/manuscript/table3_main_quantitative_comparison.csv` rows mark `NOT_VERIFIED`). |
| **Recommendation** | **NEEDS_NEW_EXPERIMENTAL_EVIDENCE** (complete canonical heavy eval). Optional **REWRITE_ONLY** for transparent “pending artifact” language. |

---

## 3. “Guarded variants not empirically substantiated”

| Field | Content |
|--------|---------|
| **Concern** | Guard / robust wrapper claims need empirical support (triggers, fallback behavior, wins/losses). |
| **Repo evidence** | **NOT_SUPPORTED** on the **canonical `heavy_r1` / builder `EVIDENCE_FILES`** path. **PARTIALLY_SUPPORTED** at **exploratory** level. |
| **Supporting artifacts** | **Specification only:** `docs/guarded_robust_wrapper.md` (trigger rule, parameters, diagnostics paths, limitations). **Implementation:** `src/lafc/policies/guard_wrapper.py`. **Non-canonical campaign rows (example):** `analysis/pairwise_publishability_campaign/jobs/*/online_metrics.csv` include `evict_value_v1_guarded` misses for some trace/capacity settings; `kbs_evidence_alignment_report.md` states guarded diagnostics are not canonical without a designated `*_heavy_r1` artifact. |
| **Missing** | Designated **canonical** multi-trace `heavy_r1` (or builder-listed) CSV comparing `evict_value_v1_guarded` vs baselines under the same protocol as main eval; systematic guard ablations. |
| **Revise without new heavy runs?** | **Partially:** can describe mechanism + cite `docs/guarded_robust_wrapper.md`; cannot claim strong empirical substantiation for KBS main line. |
| **Recommendation** | **REWRITE_ONLY** (scope guard as implementation / future work) **or** **NEEDS_NEW_EXPERIMENTAL_EVIDENCE** if the paper must claim guarded gains. |

---

## 4. “Sensitivity to horizon \(H\)”

| Field | Content |
|--------|---------|
| **Concern** | Show how performance depends on training/eval horizon \(H\). |
| **Repo evidence** | **PARTIALLY_SUPPORTED** for **offline** training metrics across \(H \in \{4,8,16\}\)**; **NOT_SUPPORTED** for **online** replay sensitivity of the *deployed* policy vs \(H\) on the canonical artifact (single selected horizon in `best_config`). |
| **Supporting artifacts** | `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`; `tables/manuscript/table4_main_ablation.csv`; `figures/manuscript/figure4_ablation.{pdf,png}`; `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` (rows by horizon); `docs/evict_value_v1_method_spec.md` §4–§7. |
| **Missing** | Online replay table sweeping **deployed** horizon or multi-model ensemble per horizon on full manifest; uncertainty / seed sweep. |
| **Revise without new heavy runs?** | **Yes** for offline horizon narrative; **no** for full online horizon ablation. |
| **Recommendation** | **REWRITE_PLUS_EXISTING_ARTIFACTS** for offline sensitivity; **NEEDS_NEW_EXPERIMENTAL_EVIDENCE** for online horizon sensitivity. |

---

## 5. “Sensitivity to guard parameters”

| Field | Content |
|--------|---------|
| **Concern** | \(W, M, T, D\), fallback policy, etc. |
| **Repo evidence** | **NOT_SUPPORTED** (no consolidated parameter sweep artifacts located under canonical or `analysis/<experiment>/summary` patterns for guard params). |
| **Supporting artifacts** | Defaults and semantics only: `docs/guarded_robust_wrapper.md`. |
| **Missing** | CSV/JSON summaries of miss rate vs guard parameter grid on representative traces. |
| **Revise without new heavy runs?** | Document defaults only; no empirical sweep to cite. |
| **Recommendation** | **REWRITE_ONLY** (report defaults + defer sweep) **or** **NEEDS_NEW_EXPERIMENTAL_EVIDENCE**. |

---

## 6. “Runtime / overhead”

| Field | Content |
|--------|---------|
| **Concern** | Computational cost of learned policy vs baselines at replay time. |
| **Repo evidence** | **NOT_SUPPORTED** in `analysis/*.csv|json|md` as a systematic benchmark (targeted grep for runtime/throughput fields in `analysis/` did not surface a dedicated policy-comparison runtime table). Slurm walltime **anecdotes** exist (e.g. `analysis/pairwise_publishability_campaign/WULVER_JOB_REPORT_2026-04-11.md` re eval timeout) but are not an overhead study. |
| **Supporting artifacts** | Indirect: CPU-only design note in `docs/wulver_heavy_evict_value_experiment.md`; feature count in `docs/evict_value_v1_method_spec.md`. |
| **Missing** | Instrumented runs: seconds per request, relative speedup vs LRU, memory peak during replay, etc. |
| **Revise without new heavy runs?** | High-level complexity discussion only—not numbers-backed. |
| **Recommendation** | **NEEDS_NEW_EXPERIMENTAL_EVIDENCE** (or instrument + micro-benchmark). |

---

## 7. “Learning component under-specified” (features, splits, targets)

| Field | Content |
|--------|---------|
| **Concern** | Reviewer wants more detail on features, train/val/test splits, label construction. |
| **Repo evidence** | **FULLY_SUPPORTED** at **repository documentation** level; **PARTIALLY_SUPPORTED** as **committed paper tables** (dataset summary is markdown; main method tables are split across builder outputs). |
| **Supporting artifacts** | `docs/evict_value_v1_method_spec.md` (feature list pointer, target `y_loss`, LRU continuation for labels, split logic, selection rule, leakage notes); `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` (row counts by split/family/capacity/horizon); `docs/decision_aligned_eviction_targets.md` / `docs/decision_aligned_targets.md` (continuation-policy discussion for v2-style rollouts—related but not identical to v1 label); `docs/reproducibility_and_artifacts.md`; `docs/wulver_heavy_evict_value_experiment.md` (defaults: `trace_chunk`, `SPLIT_SEED=7`, capacities, horizons). |
| **Missing** | A single reviewer-facing “methods appendix” **file** is not required if the paper cites the above; optional: per-feature ablation evidence (not found as standard artifact). |
| **Revise without new heavy runs?** | **Yes**—lift structured content from `evict_value_v1_method_spec.md` and dataset summary into the manuscript. |
| **Recommendation** | **REWRITE_PLUS_EXISTING_ARTIFACTS**. |

---

## 8. “Continuation-rule justification” (for rollout / counterfactual labels)

| Field | Content |
|--------|---------|
| **Concern** | Why LRU (or other) continuation inside label simulation is justified. |
| **Repo evidence** | **PARTIALLY_SUPPORTED.** v1 label definition uses LRU continuation in counterfactual window (`docs/evict_value_v1_method_spec.md` §3). v2 docs discuss explicit continuation-policy choices for richer labels. |
| **Supporting artifacts** | `docs/evict_value_v1_method_spec.md`; `docs/decision_aligned_eviction_targets.md` (continuation policies `lru` vs `blind_oracle` for rollout datasets, caveats). |
| **Missing** | Empirical sensitivity of **online** policy quality to continuation choice for **v1** (would be new runs). |
| **Revise without new heavy runs?** | **Yes** for conceptual justification + pointer to code/docs; weak without sensitivity study. |
| **Recommendation** | **REWRITE_PLUS_EXISTING_ARTIFACTS**; add **NEEDS_NEW_EXPERIMENTAL_EVIDENCE** only if reviewers demand empirical continuation ablations. |

---

## 9. “Novelty vs prior work not sharply isolated”

| Field | Content |
|--------|---------|
| **Concern** | Contribution relative to LA caching / predictors / robust baselines unclear. |
| **Repo evidence** | **PARTIALLY_SUPPORTED** for **implementation-level** differentiation (baselines roster, policy registry, framing notes); **NOT_SUPPORTED** for a dedicated “novelty experiment” artifact. |
| **Supporting artifacts** | `README.md` (baseline families); `docs/reproducibility_and_artifacts.md`; `docs/kbs_knowledge_framing_note.md`; `docs/manuscript_open_questions.md` (claim hierarchy P1); `docs/manuscript_evidence_map.md` (pairwise line—separate from KBS main path); `tables/manuscript/table2_policy_roster.csv`. |
| **Missing** | Curated related-work positioning is primarily **writing + citations**; repo does not contain a finished novelty benchmark artifact. |
| **Revise without new heavy runs?** | **Yes** for clearer positioning and claim tightening; literature synthesis is not an experiment. |
| **Recommendation** | **REWRITE_ONLY** (plus cite repo baseline roster); optional new experiments only if positioning must be *empirically* unique vs a specific prior system. |

---

## 10. “Repetitive / weak evidence-to-claim ratio”

| Field | Content |
|--------|---------|
| **Concern** | Manuscript repeats without adding proportional evidence. |
| **Repo evidence** | **PARTIALLY_SUPPORTED** that the repo **anticipates** this risk (meta-docs); fixing it is editorial. |
| **Supporting artifacts** | `docs/manuscript_open_questions.md` (P1 claim hierarchy, P4 transparency); `reports/manuscript_artifacts/kbs_evidence_alignment_report.md`; `reports/kbs_safe_paper_package.md` (if present in checkout). |
| **Missing** | N/A (structural writing issue). |
| **Revise without new heavy runs?** | **Yes.** |
| **Recommendation** | **REWRITE_ONLY**. |

---

## 11. “Are current figures/tables enough?”

| Field | Content |
|--------|---------|
| **Concern** | Coverage of tables/figures vs claims. |
| **Repo evidence** | **PARTIALLY_SUPPORTED** for **offline + schematic** bundle; **NOT_SUPPORTED** for **full** main quantitative bundle. |
| **Supporting artifacts** | Present under `tables/manuscript/` and `figures/manuscript/`: Fig.~1, Fig.~4, Fig.~5; Tables 1, 2, 4, 5; Table~3 is a **stub** (`tables/manuscript/table3_main_quantitative_comparison.csv` documents absent canonical CSV). Fig.~2–3 snippets are commented pending `policy_comparison_heavy_r1.csv` (`reports/manuscript_artifacts/latex_snippets/figure2_snippet.tex`, `figure3_snippet.tex`). |
| **Missing** | Evidence-backed Table~3 / Fig.~2–3 from canonical eval. |
| **Revise without new heavy runs?** | Adjust narrative to match stub bundle; cannot add missing figures without CSV or fabricated numbers. |
| **Recommendation** | **REWRITE_ONLY** (align claims to bundle) **+ NEEDS_NEW_EXPERIMENTAL_EVIDENCE** for intended main performance figs/tables. |

---

## Concerns we can already address by rewriting

- **Claim hierarchy / evidence-to-claim ratio / repetition:** use `docs/manuscript_open_questions.md`, `reports/manuscript_artifacts/kbs_evidence_alignment_report.md`.
- **Learning setup detail (features, splits, targets, selection rule):** lift from `docs/evict_value_v1_method_spec.md` + `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`.
- **Continuation rule (conceptual):** `docs/evict_value_v1_method_spec.md` + `docs/decision_aligned_eviction_targets.md` (with clear v1 vs v2 scope).
- **Framing without overstating online results:** `docs/kbs_knowledge_framing_note.md`, `docs/kbs_manuscript_workflow.md`.
- **Guarded variant as mechanism (not validated headline):** `docs/guarded_robust_wrapper.md`.
- **Novelty positioning at prose level:** `README.md`, `docs/reproducibility_and_artifacts.md`, `tables/manuscript/table2_policy_roster.csv` (plus external citations in the paper).

---

## Concerns that remain blocked by missing evidence

- **Canonical multi-trace end-to-end replay table / figs for KBS main line:** blocked on **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** (and rebuild of Table~3 / Fig.~2–3).
- **Guarded variant empirical substantiation** at the level expected by the sample review: no canonical `heavy_r1` guard comparison artifact in `EVIDENCE_FILES`.
- **Guard-parameter sensitivity:** no audited sweep artifacts.
- **Runtime / replay overhead:** no audited benchmark artifacts located.
- **Online sensitivity to training horizon beyond offline tables:** deploying multiple horizon-specific models on full manifest is not evidenced in canonical outputs (`best_config_heavy_r1.json` selects a single horizon/model).

---

## Audit footer

- **Report path:** `reports/manuscript_artifacts/reviewer_concern_gap_map.md`
- **Already addressable now (rewrite / cite existing docs + offline tables):** learning-method detail, continuation-rule explanation (conceptual), novelty positioning at text level, repetition/claim-strength alignment, honest “pending canonical eval” framing, offline horizon/model ablation narrative (Table~4 / Fig.~4 / Table~5 / Fig.~5).
- **Single biggest blocker:** **Missing canonical end-to-end replay artifact** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` — without it, the sample review’s core ask for **main online replay miss tables vs strong baselines on the designated manifest** is **not** defensible as the repo’s **KBS canonical** evidence (see `docs/kbs_manuscript_workflow.md`, `reports/manuscript_artifacts/manuscript_artifact_report.md`).
