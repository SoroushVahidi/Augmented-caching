# KBS Revision Repository Audit

**Manuscript:** "Decision-aligned eviction-value prediction for robust learning-augmented caching"
**Journal / manuscript ID:** KNOSYS-D-26-07461
**Editorial decision:** Revise
**Stated reviewer/editor concerns:** (1) lack of end-to-end miss-ratio evaluation, (2) limited differentiation from existing/baseline approaches, (3) need for stronger experimental support, positioning, and reproducibility clarity.
**Audit date:** 2026-06-17
**Auditor:** Claude Code, performed as a read-only repository audit. No research artifacts were modified, deleted, or rewritten. Nothing was pushed to GitHub.

---

## 1. Executive Summary

**Status: PARTIALLY READY.**

The repository is in unusually good methodological shape for a single-author project: it has an explicit canonical/exploratory evidence convention, a single source-of-truth manuscript-artifact builder script, a 283-test passing test suite, and — most importantly — its own internal audit trail (`reports/manuscript_artifacts/*`) that *already* diagnoses the exact same three concerns the reviewers raised, with timestamps going back to 2026-04-12. The manuscript itself (found only inside a zip uploaded 2026-06-17, not as git-tracked LaTeX source) was written conservatively: it does not reference Table 3 or Figures 2/3/6/7/8 anywhere, and the figures/tables it *does* cite (Fig. 1 method overview, Fig. 4 offline ablation) are numerically traceable end-to-end to real 289M-row Wulver-scale data.

The blocking gap is narrow and well-understood by the repo's own internal documentation: **one file**, `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`, is missing. Its absence is the root cause of reviewer concern (1) (no end-to-end miss-ratio table), is a major contributor to concern (3) (reviewers see only offline ablation as "main" evidence), and the repo's own gap-mapping document independently arrives at this same conclusion. Concern (2) (baseline differentiation) is a different, deeper gap: the repo has 9 audited, paper-faithful *internal* baselines (LRU, Marker, Predictive Marker, Trust-and-Doubt, Blind-Oracle/LRU combiner, REST-v1, etc.) but **zero** implementations of or empirical comparisons against the closest systems-level learned-caching prior work cited in related work (PARROT, LRB, Raven, HALP, Mockingjay, MUSTACHE) — this is a positioning/novelty gap, not purely an engineering one, and is not closeable by rerunning a job.

Last known status of the blocking job (Slurm `evictv1-heavy-eval`, job 910352, `EXP_TAG=heavy_r1`): **PENDING**, blocked by a cluster maintenance reservation through 2026-04-16, with a code fix landed the same day (commit `b9df6f1`). There is no evidence in this repository checkout that the job was ever resubmitted or completed between 2026-04-16 and the current HEAD (2026-06-17) — a roughly two-month gap with no recorded activity. **This cannot be verified locally**; it requires checking current Wulver Slurm queue/accounting state directly (see Section 7 and Section 13).

---

## 2. Repository Identity

| Field | Value |
|---|---|
| Clone URL | `https://github.com/SoroushVahidi/Augmented-caching.git` |
| Branch checked out | `main` |
| HEAD commit | `e9ac132c0aba12c640b0663af8c8bb6c946e81e1` |
| HEAD commit subject | "Add files via upload" |
| HEAD commit date | 2026-06-17 22:08:24 -0400 |
| Working tree | Clean (`git status --short` empty) |
| Package name (`pyproject.toml`) | `lafc` (Learning-Augmented Caching), version `0.1.0` |
| Python version used for checks | 3.12.3 |

**Match to manuscript:** Confirmed. The repository's `CANONICAL_KBS_SUBMISSION.md` names the exact manuscript method (`evict_value_v1`) and experiment tag (`heavy_r1`) that anchor the paper. The manuscript itself — extracted from `Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip` at repo root (added in the HEAD commit) — has matching title, single-author attribution (Soroush Vahidi, NJIT), and numeric content (Table 4 / Fig. 4) that is traceable byte-for-byte to files generated from this repo's `analysis/` outputs. There is no ambiguity that this is the correct code repository for KNOSYS-D-26-07461.

**Note:** The full manuscript LaTeX source (`main.tex`, `cover-letter.tex`, `author-agreement.tex`, `refs.bib`, figure PNGs) exists **only** inside the single zip blob committed in the HEAD commit — it was never committed as plain-text source under version control. Only individual generated table/figure artifacts were incrementally committed earlier (commits `7b522a6`, `198f2ae`, `640c65b`, `826fe14`, `4feba34`, all 2026-04-10/11/12).

---

## 3. Canonical Manuscript Path

The repository defines an explicit canonical-vs-exploratory convention, documented authoritatively in `CANONICAL_KBS_SUBMISSION.md`:

- **Method:** `evict_value_v1`
- **Experiment tag:** `EXP_TAG=heavy_r1` — any file/path containing `_heavy_r1` is canonical KBS evidence; anything unsuffixed, `_heavy_smoke`-suffixed, under `analysis/*_light/`, or under `analysis/pairwise_*_campaign/` is **not** canonical and must not be cited as manuscript evidence.
- **Canonical inputs:**
  - `analysis/wulver_trace_manifest_full.csv` (7 real-world trace families: brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018)
  - `models/evict_value_wulver_v1_best_heavy_r1.pkl` (gitignored — not present in this clone; existence outside Wulver scratch is **unverified**)
- **Canonical outputs already present and verified by direct read:**
  - `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` — 289,304,256 total rows, 2,524,527 unique decisions, across capacities {32,64,128,256} and horizons {4,8,16}.
  - `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` — 9 rows (3 horizons × 3 models: hist_gb, random_forest, ridge), feeding Table 4 / Fig. 4.
  - `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` — single selected (horizon, model) configuration.
- **Canonical output that is MISSING:**
  - `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (and its `.md` companion) — the multi-trace, multi-capacity online replay comparison across all baseline policies plus `evict_value_v1`. This is the single file required for Table 3 and Figures 2–3.

This is not an inference — `tables/manuscript/table3_main_quantitative_comparison.csv` literally contains a stub row: `status=NOT_VERIFIED, detail="Canonical file analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv is absent."` and `reports/manuscript_artifacts/manuscript_artifact_manifest.json` records `"policy_comparison_present": false`.

---

## 4. Main Scripts and Artifacts

| Stage | Script | Status |
|---|---|---|
| Build evict_value_v1 training dataset (Wulver-scale) | `slurm/evict_value_v1_wulver_heavy_train.sbatch` → `scripts/build_evict_value_dataset_v1.py` | Completed (outputs exist, see Section 3) |
| Train + select model | same sbatch → `scripts/train_evict_value_v1.py` | Completed (`model_comparison_heavy_r1.csv`, `best_config_heavy_r1.json` present) |
| Multi-trace online policy replay (the missing piece) | `slurm/evict_value_v1_wulver_heavy_eval.sbatch` → `scripts/run_policy_comparison_wulver_v1.py` | **Not completed** — job pending/unresolved as of last recorded check (2026-04-12/16) |
| Build all manuscript tables/figures from `EVIDENCE_FILES` | `scripts/paper/build_kbs_main_manuscript_artifacts.py` | Runs successfully and exits 0 even when the policy-comparison CSV is absent — it emits stub Table 3 and skips Fig. 2/3, substituting offline-only Table 5 / Fig. 5 |
| Smaller manuscript figure regeneration (Fig. 1, Fig. 4 only) | `scripts/paper/regenerate_evidence_aligned_manuscript_figures.py` | Available, used for the two figures actually cited in `main.tex` |
| Guard-wrapper figure builder | `scripts/paper/build_guard_wrapper_manuscript_figure.py` | Exploratory-only; not fed by canonical `EVIDENCE_FILES` |

`build_kbs_main_manuscript_artifacts.py` is the **single source of truth**: its `EVIDENCE_FILES` dict and `TABLE3_POLICIES = ("lru", "predictive_marker", "trust_and_doubt", "blind_oracle_lru_combiner", "rest_v1", "evict_value_v1")` tuple define exactly which six policies and which files feed Table 3 / Figs. 2–3. The eval driver's `BASELINE_POLICIES` env default is broader (includes `blind_oracle`, `blind_oracle_lru_combiner`) — the builder silently filters extra policies, which is documented behavior (`reports/kbs_safe_paper_package.md`) and not a bug.

**Important documentation conflict found and verified:** `docs/evict_value_v1_kbs_canonical_artifacts.md` implies the builder script *fails* without the canonical CSV. This is false — I confirmed (via `reports/manuscript_artifacts/heavy_r1_execution_audit.md`, itself dated 2026-04-12, and consistent with my own review of the script's structure) that the builder degrades gracefully to stub output rather than erroring. This conflict should be corrected as part of the revision so future contributors aren't misled about the failure mode.

---

## 5. End-to-End Miss-Ratio Evaluation

**Answer: NO — not currently present for the canonical `heavy_r1` line.**

- **What exists:** Offline training-time metrics only (`val_mean_regret`, `test_mean_regret`, `val_top1`, `test_top1` per horizon/model in `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`). These measure how well the learned scorer predicts simulated counterfactual loss — they are **not** miss-ratio/hit-ratio numbers from an actual cache replay.
- **What is missing:** A genuine end-to-end replay table: miss count / miss ratio per (trace family × capacity) for each policy in `TABLE3_POLICIES`, replayed via `scripts/run_policy_comparison_wulver_v1.py` against the full 7-trace manifest at capacities {32,64,128,256} with `EVAL_MAX_REQUESTS=50000` per trace.
- **Exact command that would produce it** (already configured, not hypothetical — see `slurm/evict_value_v1_wulver_heavy_eval.sbatch`):
  ```
  sbatch --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_eval.sbatch
  ```
  which internally runs:
  ```
  python scripts/run_policy_comparison_wulver_v1.py \
    --trace-manifest analysis/wulver_trace_manifest_full.csv \
    --capacities 32,64,128,256 \
    --max-requests-per-trace 50000 \
    --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
    --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
    --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv \
    --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md
  ```
- **Why it hasn't happened:** A 24h attempt (job 908326) previously timed out before writing output, prompting a switch to 72h walltime. The resubmitted job (910352) was queued PENDING on 2026-04-12 due to a scheduled maintenance reservation (`Apr_14_Sched_Maint3`, 2026-04-14T09:00–2026-04-16T21:00) that blocked the full 72h window from completing before maintenance. A model-loading bugfix (commit `b9df6f1`, 2026-04-16: joblib fallback + Slurm preflight check) landed the same day maintenance ended. **There is no artifact in this repository indicating the job was ever resubmitted after the fix or that it ever completed.** This must be checked directly against current Wulver Slurm state — see Section 13.
- **Non-canonical replay data that exists but must NOT be cited:** `analysis/evict_value_wulver_v1_policy_comparison.csv` (unsuffixed — extra/different policy set, not `heavy_r1`) and `analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.csv` (small-scale wiring check only). Both are explicitly flagged as non-canonical in `docs/wulver_heavy_evict_value_experiment.md` and `reports/kbs_safe_paper_package.md`.

---

## 6. Baseline Differentiation

Two distinct layers exist, and they should not be conflated in the response to reviewers:

**(a) Internal baseline implementations — audited, mostly solid.**
`analysis/baseline_audit/baseline_audit_summary.csv` rates 9 implemented baselines for faithfulness to their literature definitions:

| Baseline | Faithfulness | Bug risk | Paper-comparison readiness |
|---|---|---|---|
| `lru` | High confidence faithful | Low | Ready |
| `blind_oracle` | High confidence faithful | Low | Ready |
| `advice_trusting` | High confidence faithful | Low | Ready |
| `marker` | Plausible, partly interpreted | Low | Ready |
| `predictive_marker` | Plausible, partly interpreted | Low | Ready |
| `blind_oracle_lru_combiner` | Plausible, partly interpreted | Low | Ready (recommend adding forced-switch tests) |
| `weighted_lru` | Plausible, partly interpreted | — | — |
| `trust_and_doubt` | Plausible, partly interpreted | **Medium** | Needs invariant-focused tests before paper-grade claims |
| `la_det` (LA weighted paging, deterministic) | Likely approximate / not faithful enough | **Medium** | Recommend relabeling as "approximate interpretation" in the paper |

This is real, useful differentiation work — six of these (`lru`, `predictive_marker`, `trust_and_doubt`, `blind_oracle_lru_combiner`, `rest_v1`, `evict_value_v1`) are exactly the `TABLE3_POLICIES` set intended for the main comparison.

**(b) Differentiation from external systems-level prior work — not present.**
The repository's own related-work table notes (`reports/manuscript_artifacts/table6_related_work_uncertainties.md`) name PARROT, LRB, Raven, HALP, Mockingjay, and MUSTACHE as the closest neighbors, and explicitly flag that several of the one-line contrasts in Table 6 (e.g., vs. HALP, vs. MUSTACHE) need primary-source re-verification before being stated more sharply. **None of these six systems are implemented or empirically compared against in this repo.** The novelty argument is currently positioning-only ("candidate-level finite-horizon eviction-value supervision is a different formulation"), not empirically demonstrated against the nearest published systems. This is consistent with — and almost certainly the direct cause of — reviewer concern (2). Closing this gap fully would require new implementation work (not just rerunning an existing job), which is a materially larger undertaking than the missing end-to-end CSV.

---

## 7. Reproducibility Audit

**Tooling and tests: solid.**
- `pip install -e ".[dev]"` succeeds cleanly in a fresh venv (`/tmp/kbs_audit_venv`).
- `pytest tests/ -q` → **283 passed, 14 warnings in 10.60s**, exactly matching the count documented in `AGENTS.md`. Warnings are benign (two docstring `SyntaxWarning`s for invalid escape sequences; PuLP deprecation notices).

**README quick-start: two verified bugs.**
I personally exercised the documented quick-start path (outputs redirected to `/tmp/kbs_quickstart_check`, nothing written into the repo):

1. Step 1 (`scripts/build_evict_value_dataset_v1.py --max-rows 200000 --out-dir ...`) — **works**, but the README's flag name is slightly off in spirit only if a user copies an `--output-dir` guess; the actual flag is `--out-dir` (confirmed via the script's own usage string). Once the correct flag is used, this step succeeds (verified at `--max-rows 5000`, rows_total=438).
2. Step 2 (`scripts/train_evict_value_v1.py --horizon 8 ...`) — **works** (winner_by_val_regret="ridge" at this tiny smoke scale).
3. Step 3 (`scripts/run_evict_value_v1_first_check.py`, no arguments, exactly as documented) — **fails on a clean clone**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'models/ml_gate_v2_random_forest.pkl'
   ```
   Root cause (verified by reading `src/lafc/policies/ml_gate_v2.py` and `ml_gate_v1.py`): these policies hardcode default model paths `models/ml_gate_v2_random_forest.pkl` and `models/ml_gate_v1.pkl`. No `models/` directory exists anywhere in a fresh clone (it's gitignored), and the documented 3-step quick-start never produces these specific files. A new reader following the README verbatim will hit this crash.
4. The script that *would* produce one of the missing files, `scripts/train_ml_gate_v1.py`, itself crashes when run against the repo's own committed `data/derived/ml_gate_train.csv`:
   ```
   AttributeError: 'LinearProbabilityEstimator' object has no attribute 'named_steps'
   ```
   This exactly matches a bug already documented in `AGENTS.md`'s "Gotchas" section — I independently reproduced it rather than just trusting the doc.

**Data/model availability gap (important, partly unverifiable locally):**
`.gitignore` excludes `/models/*.pkl` and `/data/derived/evict_value_v1_wulver*/`. This means the trained `heavy_r1` model and the 289M-row derived dataset are **not** in git — only their summary statistics (`*_heavy_r1.md`, `*_heavy_r1.csv`, `*_heavy_r1.json`) are committed. **I cannot verify locally whether `models/evict_value_wulver_v1_best_heavy_r1.pkl` still exists anywhere outside Wulver scratch space.** If it has been purged (Wulver scratch is typically subject to retention policies), the 36h `heavy_train` job would need to be rerun before the 72h `heavy_eval` job — adding significant time to closing the Section 5 gap. This must be checked directly on the cluster (`ls -la models/*.pkl` on Wulver, or `sacct`/`squeue` for job history) — not something this audit could determine from the GitHub clone.

**Internal self-audits are unusually thorough.** The repo contains its own internal evidence-alignment, gap-mapping, and "what NOT to cite" documentation (`reports/manuscript_artifacts/*`, `CANONICAL_KBS_SUBMISSION.md`, `reports/kbs_safe_paper_package.md`) dated as recently as 2026-04-12–16. This is a genuine reproducibility strength: a future reader (or reviewer) has clear, explicit guidance on what is and isn't safe to cite, rather than having to infer it.

---

## 8. Experimental Strengths

- **Real, large-scale, traceable offline evidence.** 289,304,256 rows across 7 real-world trace families (brightkite check-ins, Citibike trips, Wikipedia 2018 pageviews, Twemcache, MetaKV, MetaCDN, Alibaba CloudPhysics block traces) — not synthetic. Numbers in `main.tex` Table 4 match `tables/manuscript/table4_main_ablation.csv` and the underlying `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` to 4 decimal places, end to end.
- **Conservative manuscript writing.** `main.tex` does not reference Table 3 or any of Figures 2/3/5/6/7/8 — confirming the author did not overstate evidence that doesn't exist. This is good scientific practice, though it is also literally why a reviewer would flag "no end-to-end evaluation": the gap is real, not a writing oversight.
- **Explicit canonical/exploratory separation convention** (`_heavy_r1` suffix discipline), enforced via `.gitignore`, `CANONICAL_KBS_SUBMISSION.md`, and a single-source-of-truth artifact builder.
- **9 internally audited baseline implementations** with explicit faithfulness/bug-risk ratings rather than unaudited "we assume this is correct" baselines.
- **283/283 passing tests**, clean fresh-venv install.
- **Detailed self-diagnostic paper trail** already covering exactly the reviewers' three concerns (`reviewer_concern_gap_map.md` maps 11 anticipated review-style concerns against repo evidence with explicit FULLY/PARTIALLY/NOT_SUPPORTED ratings).

---

## 9. Experimental Gaps

1. **Missing canonical end-to-end policy-comparison CSV** (`*_heavy_r1`) — blocks Table 3 and Figs. 2–3 entirely. Single biggest blocker (Section 5).
2. **No empirical comparison against external systems-level prior work** (PARROT, LRB, Raven, HALP, Mockingjay, MUSTACHE) — novelty is asserted, not demonstrated (Section 6).
3. **Guarded/robust wrapper variant is implemented (`src/lafc/policies/guard_wrapper.py`, spec in `docs/guarded_robust_wrapper.md`) but has no canonical empirical substantiation** — only exploratory campaign data (`analysis/pairwise_publishability_campaign/...`) shows `evict_value_v1_guarded` numbers, and these are explicitly non-canonical.
4. **No guard-parameter sensitivity sweep** (W, M, T, D) — defaults are documented but never swept.
5. **No horizon sensitivity at the online-replay level** — only offline training metrics are swept across H∈{4,8,16}; the canonical `best_config_heavy_r1.json` selects and deploys a single (horizon, model) pair.
6. **No runtime/overhead benchmark** — no instrumented seconds-per-request, throughput, or memory measurements found anywhere under `analysis/`.
7. **Continuation-rule choice (LRU continuation for v1 labels) is justified conceptually in `docs/evict_value_v1_method_spec.md` but has no empirical sensitivity study** comparing it to alternative continuation policies for the v1 label specifically (v2 docs discuss this more, but v2 is not the canonical KBS line).
8. **Two reproducibility bugs on the documented quick-start path** (Section 7) that should be fixed or the README adjusted before resubmission, since reviewers/AEs sometimes attempt to run the quick-start themselves.
9. **Internal documentation conflict**: `docs/evict_value_v1_kbs_canonical_artifacts.md` incorrectly implies the manuscript-artifact builder fails without the canonical CSV; it actually degrades gracefully. Low-priority but worth fixing for contributor clarity.

---

## 10. Reviewer-Concern Mapping

This table is built directly from the repository's own internal gap analysis (`reports/manuscript_artifacts/reviewer_concern_gap_map.md`, 184 lines, read in full), cross-referenced against the three concerns named in the editorial decision and independently verified file-by-file where practical.

| Reviewer/AE concern | Existing repository evidence | Missing evidence | Recommended action | Priority |
|---|---|---|---|---|
| **(1) Lack of end-to-end miss-ratio evaluation** — main story reads as offline ablation only | Offline ablation is real and traceable: `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, Table 4, Fig. 4, Table 5/Fig. 5 (offline-only supplements) | Canonical `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` → Table 3, Figs. 2–3 | Complete the blocked `heavy_eval` Slurm job (Section 13); rebuild manuscript artifacts; until then, keep wording honest about offline-only scope | **High** |
| **(1a) Guarded/robust variant not empirically substantiated** | Mechanism implemented and specified (`guard_wrapper.py`, `docs/guarded_robust_wrapper.md`); exploratory-only numbers in pairwise campaign data | Canonical `heavy_r1`-tagged comparison of `evict_value_v1_guarded` vs. baselines under the same protocol as main eval | Either scope guard as "implementation / future work" in text (no new experiments needed), or run a dedicated guarded-variant eval if the paper must claim guarded gains | **Medium** |
| **(1b) Sensitivity to horizon H** | Full offline metrics sweep across H∈{4,8,16} (Table 4/Fig. 4) | Online replay sensitivity of the *deployed* policy to H (currently single horizon selected) | Cite offline sweep now (no new work); online horizon ablation only if reviewers explicitly demand it | **Low–Medium** |
| **(1c) Sensitivity to guard parameters (W,M,T,D)** | Defaults documented in `docs/guarded_robust_wrapper.md` | No parameter-grid sweep artifacts anywhere | Report defaults + defer sweep to future work, unless reviewer insists | **Low** |
| **(1d) Runtime / overhead** | CPU-only design noted in `docs/wulver_heavy_evict_value_experiment.md`; no measurements | Instrumented seconds/request, throughput, memory vs. baselines | Add a lightweight runtime instrumentation pass to the eval driver if time allows; otherwise discuss complexity qualitatively | **Low–Medium** |
| **(2) Limited differentiation from baselines / prior work** | 9 internally audited, paper-faithful baseline implementations with explicit faithfulness ratings (`analysis/baseline_audit/baseline_audit_summary.csv`); policy roster table (Table 2) | No implementation of or comparison against external systems-level prior work (PARROT, LRB, Raven, HALP, Mockingjay, MUSTACHE); Table 6 contrasts are positioning-level, not empirical, and some need primary-source re-verification | Sharpen related-work text using `table6_related_work_uncertainties.md`; verify HALP/MUSTACHE supervision-target claims against primary sources before publishing stronger contrasts; empirical comparison against an external system is optional/future-work unless explicitly demanded | **High** (positioning) / **Medium** (empirical) |
| **(3) Need stronger experimental support / reproducibility clarity** | Learning setup (features, splits, targets, leakage notes) fully documented in `docs/evict_value_v1_method_spec.md` + dataset summary; 283 passing tests; explicit canonical/exploratory file-naming discipline | Single reviewer-facing methods appendix not yet assembled (optional); two reproducibility bugs on the documented quick-start path (ml_gate model files, `train_ml_gate_v1.py` AttributeError) | Lift structured content from method-spec docs directly into manuscript/appendix; fix or clearly flag the two quick-start bugs before resubmission | **Medium–High** |
| **(3a) Repetitive / weak evidence-to-claim ratio** | Repo's own `docs/manuscript_open_questions.md` (claim hierarchy) and `kbs_evidence_alignment_report.md` already anticipate this risk | None (structural writing issue, not an experimental gap) | Editorial pass on the manuscript text to tighten claim strength to match available evidence | **Medium** |
| **(3b) Are current figures/tables sufficient?** | Fig. 1, Fig. 4, Table 1, Table 2, Table 4, Table 5 present and manuscript-safe | Table 3 is a stub; Figs. 2–3 not built; Figs. 6–8 exist as scripts/data but are unused in `main.tex` | Same as (1): blocked on the missing CSV | **High** |

---

## 11. Concrete Revision Plan

Ordered by dependency and impact:

1. **Verify current Wulver cluster state** (cannot be done locally from this clone): check whether job 910352 (or a resubmission) completed, and whether `models/evict_value_wulver_v1_best_heavy_r1.pkl` and the derived `data/derived/evict_value_v1_wulver*/` dataset still exist on scratch. This determines whether step 2 or step 3 below is the actual starting point.
2. **If the trained model/dataset no longer exist:** resubmit `slurm/evict_value_v1_wulver_heavy_train.sbatch` (36h walltime) with `EXP_TAG=heavy_r1`.
3. **Resubmit the eval job:** `sbatch --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_eval.sbatch` (72h walltime; confirm no overlapping maintenance reservation first — check `sinfo -T` / scheduler reservation list before submitting, given the prior maintenance-induced PENDING failure).
4. **Once `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` exists:** rerun `python scripts/paper/build_kbs_main_manuscript_artifacts.py` to regenerate Table 1 (now with confirmed Main-roster alignment), Table 3, Figs. 2–3, and to retire the offline-only Table 5/Fig. 5 supplements per the builder's documented behavior.
5. **Sanity-check the new Table 3 / Figs. 2–3** against `TABLE3_POLICIES` filtering and confirm extra eval-only policies (e.g. `blind_oracle`) are excluded as designed.
6. **Update the manuscript** to incorporate the new end-to-end results, replacing/softening any "pending artifact" language.
7. **Sharpen related-work positioning** (Table 6) using `table6_related_work_uncertainties.md` — verify HALP and MUSTACHE supervision-object claims against primary sources before stating stronger contrasts; decide whether an empirical comparison against one external system is warranted given reviewer feedback and time budget.
8. **Fix or clearly document the two quick-start reproducibility bugs** (Section 7, items 3–4) — either patch `run_evict_value_v1_first_check.py` / `train_ml_gate_v1.py`, or adjust the README to mark the ml_gate quick-start step as optional/separate from the `evict_value_v1` path it's documented alongside.
9. **Fix the documentation conflict** in `docs/evict_value_v1_kbs_canonical_artifacts.md` regarding builder failure behavior.
10. **Assemble a response-to-reviewers / rebuttal document** — none currently exists anywhere in the repo (confirmed via grep across all files for "response to reviewer", "rebuttal", "point-by-point", "reviewer 1/2/#1"; zero matches). This should be written once steps 1–7 are resolved, using the file list in Section 14.
11. **Update the cover letter** (`cover-letter.tex`) — it currently reads as a fresh-submission letter ("has not been previously rejected by Knowledge-Based Systems", with a commented-out alternate resubmission paragraph) rather than a revision-specific cover letter. This needs to be replaced with revision-appropriate language addressing the "Revise" decision directly.
12. **Optional, lower priority:** add guard-parameter sweep, runtime/overhead benchmark, and online horizon-sensitivity table if time and reviewer demand justify the additional experiment time.

---

## 12. Exact Commands to Reproduce Current Evidence

All of the following were verified to work in this audit (in a throwaway venv / scratch output dir, nothing written into the repo other than the report being delivered):

```bash
# Install (verified: succeeds)
python3 -m venv /path/to/venv && source /path/to/venv/bin/activate
pip install -e ".[dev]"

# Full test suite (verified: 283 passed, 14 warnings, ~10.6s)
pytest tests/ -v

# Quick-start steps 1-2 of README (verified: succeed)
python scripts/build_evict_value_dataset_v1.py --max-rows 200000 --out-dir <scratch_dir>
python scripts/train_evict_value_v1.py --horizon 8 \
  --data-dir <scratch_dir> \
  --metrics-json <scratch_dir>/metrics.json \
  --comparison-csv <scratch_dir>/comparison.csv \
  --models-dir <scratch_dir>/models

# Regenerate the manuscript artifact bundle from already-existing canonical inputs
# (will emit Table 1,2,4,5 + Fig.1,4 again; Table 3/Fig.2-3 remain stubs/missing
#  because the canonical policy-comparison CSV is absent)
python scripts/paper/build_kbs_main_manuscript_artifacts.py

# Regenerate just Fig.1 and Fig.4 (the only two figures cited in main.tex)
python scripts/paper/regenerate_evidence_aligned_manuscript_figures.py
```

**Quick-start step 3 is NOT reproducible as documented** — see Section 7, item 3 (fails with `FileNotFoundError` for `models/ml_gate_v2_random_forest.pkl`). Do not include it in a clean-room reproduction walkthrough for reviewers without first fixing it or removing it from the documented sequence.

---

## 13. Exact Commands or Scripts Needed for Missing Evidence

**Primary blocker — end-to-end policy comparison (Section 5, Section 9 item 1):**

```bash
# Step 0 (manual, on Wulver login node — not run by this audit):
# Check current job/reservation state before resubmitting, given the prior
# maintenance-induced PENDING failure of job 910352.
squeue -u sv96
sacct -j 910352 --format=JobID,JobName,State,Reason,Start,End
sinfo -T   # list active/scheduled reservations

# Step 1 (only if models/evict_value_wulver_v1_best_heavy_r1.pkl and the
# derived heavy_r1 dataset no longer exist on scratch):
sbatch --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_train.sbatch
# Expected inputs: analysis/wulver_trace_manifest_full.csv
# Expected outputs (already present in this clone, verify they still exist on
# cluster scratch): analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md,
# analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv,
# analysis/evict_value_wulver_v1_best_config_heavy_r1.json,
# models/evict_value_wulver_v1_best_heavy_r1.pkl
# Walltime: 36:00:00, partition=general, qos=standard, mem=128G, cpus=32

# Step 2 (the actual missing-evidence job):
sbatch --export=ALL,EXP_TAG=heavy_r1 slurm/evict_value_v1_wulver_heavy_eval.sbatch
# Expected output (this is the file everything is blocked on):
#   analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
#   analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md
# Placement: directly under analysis/ at repo root (already wired into
# EVIDENCE_FILES in scripts/paper/build_kbs_main_manuscript_artifacts.py —
# no path changes needed once the file exists)
# Walltime: 72:00:00, partition=general, qos=standard, mem=64G, cpus=16

# Step 3 (after step 2 completes):
python scripts/paper/build_kbs_main_manuscript_artifacts.py
# This will now populate tables/manuscript/table3_main_quantitative_comparison.{csv,tex}
# and figures/manuscript/figure2_main_performance_comparison.{pdf,png},
# figures/manuscript/figure3_improvement_vs_lru.{pdf,png}, and will retire
# table5_offline_selection / figure5_offline_top1_ablation automatically.
```

**Secondary gaps (Section 9, items 2–6) — no existing script wired up; would require new work:**

- **External baseline comparison:** no script exists. Would require implementing at least one of PARROT/LRB/Raven/HALP/Mockingjay/MUSTACHE (or a faithful simplified surrogate) inside `src/lafc/policies/` and adding it to the eval driver's `--policies` list. This is new engineering work, not a rerun.
- **Guard-parameter sweep:** no script exists. Would need a new driver iterating `guard_wrapper.py`'s (W,M,T,D) parameters across a representative trace subset, writing a CSV under `analysis/guard_param_sweep_heavy_r1.csv` (suggested path/name, not currently defined anywhere).
- **Runtime/overhead benchmark:** no script exists. Would need wall-clock/memory instrumentation added to `scripts/run_policy_comparison_wulver_v1.py` (e.g., `time.perf_counter()` around each policy's per-request decision call), emitting a new column or sidecar file — suggested: `analysis/evict_value_wulver_v1_runtime_heavy_r1.csv`.
- **Online horizon sensitivity:** would require running the eval driver multiple times with different `--evict-value-model` paths (one per horizon-specific trained model) and concatenating results — currently `best_config_heavy_r1.json` only retains one horizon's model.

---

## 14. Files That Should Be Cited in the Response to Reviewers

No response-to-reviewers document currently exists in the repository (verified by exhaustive grep for "response to reviewer", "rebuttal", "point-by-point", "reviewer 1/2/#1" — zero matches anywhere). When writing it, the following repo files are the most defensible, already-vetted sources to cite or quote from:

- `CANONICAL_KBS_SUBMISSION.md` — for explaining the canonical evidence convention and exactly what is/isn't manuscript-grade.
- `reports/manuscript_artifacts/reviewer_concern_gap_map.md` — directly maps anticipated reviewer concerns to evidence status; can be paraphrased almost directly into rebuttal language for concerns about evidence scope.
- `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` — dataset scale and composition (289M rows, 7 real traces) — addresses "is the offline evidence substantial?" sub-concerns.
- `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` / `tables/manuscript/table4_main_ablation.csv` — the real, traceable offline ablation numbers.
- `analysis/baseline_audit/baseline_audit_summary.csv` — for responding to "are baselines correctly implemented?" with explicit per-baseline faithfulness/risk ratings rather than a bare assertion.
- `reports/manuscript_artifacts/table6_related_work_uncertainties.md` — for transparently acknowledging which related-work contrasts are positioning-level vs. verified, which is more credible to reviewers than overclaiming.
- `docs/evict_value_v1_method_spec.md` — for responding to "learning component under-specified" (features, splits, target construction, leakage notes).
- `docs/guarded_robust_wrapper.md` — for responding to guard-related concerns by clearly scoping the guard as a documented mechanism, not (yet) an empirically validated headline claim.
- **Once Section 13 step 2 completes:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`/`.md` and the rebuilt Table 3/Figs. 2–3 become the primary citation for directly answering reviewer concern (1).

---

## 15. Final Recommendation

Given the **July 8, 2026** revision deadline (approximately three weeks from this audit's date):

**The repository is not yet revision-ready, but the path to readiness is well-defined and narrower than the manuscript's reviewer comments might suggest.** The single highest-priority action is determining the live status of the Wulver eval job and, if needed, resubmitting the train→eval pipeline (Section 13). Given the documented history (24h timeout → 72h resubmission → maintenance-blocked PENDING → bugfix landed 2026-04-16 → no further recorded activity through 2026-06-17), this should be treated as urgent and checked **today**, not deferred — a 36h+72h ≈ 4.5-day pipeline (plus possible queue wait) leaves limited margin against a 3-week deadline if the train stage must also be rerun, and very little margin at all if there are further scheduling delays.

In parallel (no dependency on cluster results), the response-to-reviewers document and the related-work sharpening (Section 11, items 7, 10–11) can be drafted now using the already-vetted internal audit files listed in Section 14 — this work is not blocked and should start immediately to avoid being squeezed against the deadline once eval results arrive.

The external-baseline-comparison gap (Section 6b) is realistically **out of scope** for a 3-week revision window given it requires new implementation work, not a rerun. The recommended response-to-reviewers strategy is to address it primarily through sharpened, source-verified positioning language (Table 6) plus an explicit "future work" framing, reserving a true empirical comparison against an external system for a follow-up paper unless the editor's letter explicitly demands it as a condition of acceptance — that determination depends on the actual reviewer text, which was not provided to this audit and should be checked before finalizing this part of the response.

**If the eval job completes successfully and on schedule:** status moves to **revision-ready** for concern (1) and significantly strengthened for concern (3); concern (2) still requires the positioning-level response described above.
**If the eval job cannot complete before the deadline:** the manuscript should be revised to make the offline-only evidence scope fully explicit (the groundwork for this is already drafted in `reports/manuscript_artifacts/kbs_evidence_alignment_report.md`), accompanied by a rebuttal that commits to delivering the end-to-end results in a short post-acceptance/camera-ready window if the venue allows it — a materially weaker but not indefensible position, given the genuine scale and care already visible in the offline evidence.
