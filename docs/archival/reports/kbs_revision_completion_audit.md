# KBS Revision Completion Audit (2026-06-19)

Read-only audit. Nothing was launched, stopped, merged, or pushed while producing
this report. Machine: local/cloud, no Slurm/sbatch involved.

Journal: Knowledge-Based Systems · Manuscript ID: KNOSYS-D-26-07461 ·
Title: "Decision-aligned eviction-value prediction for robust learning-augmented
caching" · Decision: Revise · Due date: 2026-07-08.

## A. Executive summary

- **Update 2026-06-19, later same day**: this file's earlier "no policy-
  comparison process is running anywhere on the machine right now" statement
  is now historical only. A new tmux-backed job,
  `kbs_full_policy_comparison_cap32_with_sieve`, is currently running on this
  same local/cloud machine. Read-only checks in the present pass show the
  tmux session exists, `pgrep` shows the corresponding
  `scripts/run_policy_comparison_wulver_v1.py` process, and there is still no
  `CAP32_WITH_SIEVE_EXIT=` marker. This pass did not touch that job.
- **Correction to the premise of this audit**: the cap32 chunk is **not** still
  running. It completed before this audit started — started
  Thu 2026-06-18 19:18:11 EDT, finished Fri 2026-06-19 00:31:13 EDT (~5h13m),
  `CAP32_EXIT=0`. The tmux session (`kbs_full_policy_comparison_cap32`) is still
  open but idle at a shell prompt; `pgrep` confirms no policy-comparison/heavy_r1
  process is running anywhere on the machine right now. cap64/cap128/cap256 have
  **not** been launched (per instruction).
- **What's already done**: the entire heavy_r1 data/training/validation
  provenance chain (dataset build, dataset summary, model training, validation
  eval, validation sanity audit, platform-consistency audit) is complete and
  documented with sha256-level provenance in
  `reports/kbs_revision_evidence_ledger.md`. The first real chunk of the
  canonical online-replay sweep (cap32, 49 rows = 7 traces × 7 policies) is also
  done. A large amount of planning/report infrastructure exists (ledger,
  manifest, gap tracker, execution plan, next-actions list).
- **New finding this session**: the actual submission package is **not**
  missing-everything, as a prior planning doc (`kbs_submission_package_checklist.md`
  on the unmerged parallel branch) claimed. A manuscript zip
  (`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`,
  added in commit `e9ac132`, 2026-06-17 22:08:24 EDT) contains a real LaTeX
  manuscript (`main.tex`, ~8,919 words), a cover letter, and an author agreement.
  Figures inside it are independently confirmed 300dpi. This corrects several
  "not started" items below to "exists, needs revision" — but also surfaces a
  **new, concrete problem**: the manuscript's embedded offline-ablation table
  uses numbers that do **not** match the current heavy_r1 retrain (see Concern
  Matrix row 7) — the zip predates the training run that produced today's
  canonical `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`.
- **What's remaining**: the multi-day canonical sweep (cap64/128/256 + merge);
  a scope decision on SIEVE/FIFO-Reinsertion (currently a clean, total gap);
  computational-overhead and label-construction-scalability writeups (data
  likely already sitting in existing logs, just not distilled); refreshing the
  stale offline-ablation table; and the full submission-package finalization
  (Highlights, CRediT, revision-specific cover letter, DOCX-format question).
- **Update 2026-06-19 (real reviewer comments now available)**: the bullet
  immediately below is **historical and corrected** — the full verbatim
  AE/Reviewer #2/Reviewer #3 text is no longer blocked. It was supplied
  directly by the user in this session and is recorded in
  `reports/kbs_real_reviewer_comments.md`. See §G below for the current
  blocker ranking, which supersedes this bullet.
- ~~**Blocked**: the full reviewer/AE comment text (only the 12-item summary list
  supplied this session is available — still not verbatim Editorial Manager
  text);~~ the canonical CSV (compute, ~2–3 more days); the DOCX-format question
  for the main manuscript (needs the actual KNOSYS Editorial Manager submission
  instructions, not derivable locally) remain blocked as described.
- **Highest-priority next action**: while no compute is running, do the
  zero-compute items now (log-mining for overhead/scalability numbers, stale
  ablation-table refresh, bibliography fixes, parallel-branch merge review) —
  see §F (Recommended next actions) — rather than immediately launching cap64.

## B. Reviewer/AE concern matrix

**SUPERSEDED 2026-06-19.** This matrix (rows 1-14 below) was built from a
12-item category-level paraphrase, before the real reviewer comments were
available. It is kept for historical/provenance reasons — do not use it as
the current source of truth. The current, real-comment-sourced matrix (22
rows, exact source labels AE/R2-MC1-3/R3-Summary/R3-Issue1-7/R3-Minor8-9/
R3-Rec1-8) is `reports/kbs_complete_reviewer_comment_matrix.md`. Many of the
findings below (e.g. row 4's SIEVE/FIFO-Reinsertion gap, row 3's HALP
treatment, row 7's stale-ablation-table discovery) are independently
confirmed by the real text (R3-Issue2/3, R2-MC1) and remain valid as
evidence — only the "source" framing (paraphrase vs. verbatim) is outdated.

Source for all rows below is the 12-item concern list supplied directly in this
session's prompt (paraphrased as "AE-confirmed item N" against that list's order),
plus two repo-internal items (rows 13–14 partly) the user's Step 4 checklist also
asked to be tracked. Status is graded strictly — "exists in some form" is not the
same as "addresses the concern."

| # | Priority | Source | Concern | Current status | Evidence already available | Remaining action | Blocker | Risk if not addressed |
|---|---|---|---|---|---|---|---|---|
| 1 | Critical | AE item 1 | End-to-end miss-ratio evaluation | **PARTIALLY DONE** | cap32 chunk complete (49 rows, `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv/.md`); validation pass (1 trace × cap256); validation-sanity (1 trace × cap32 × 5k). 3 independent data points, all showing `evict_value_v1` worse than LRU. | Launch+complete cap64/cap128/cap256 (~remaining ~1.5–3 days per execution plan); merge 4 CSVs (script exists, unrun); write the `.md` aggregation follow-up (not written yet); regenerate Table 3/Figs 2–3; decide the result narrative. | Multi-day compute + a research-narrative decision once the full picture exists. | Single most load-bearing concern; manuscript currently has **zero** online-replay results at all. |
| 2 | Critical/Important | AE item 3 | Insufficient baseline comparisons | **PARTIALLY DONE** | 6 internal baselines wired and exercised (`lru, predictive_marker, trust_and_doubt, blind_oracle_lru_combiner, rest_v1` + `evict_value_v1`); `blind_oracle` implemented but excluded from `TABLE3_POLICIES`. `kbs_baseline_positioning_plan.md` (parallel branch) scopes external systems (PARROT/LRB/Raven/HALP/Mockingjay/MUSTACHE) as positioning-only, explicitly recommending no new empirical comparisons this cycle. | Decide, once canonical CSV lands, whether the internal-baseline set reads as "sufficient," or whether reviewers specifically want an external SOTA method actually run (see row 3/4). | Canonical CSV (row 1) for the narrative; a scope decision for external baselines. | If "insufficient" meant external SOTA specifically, positioning text alone may not satisfy reviewers even with the canonical CSV in hand. |
| 3 | Important | AE item 11 (HALP) | HALP comparison / differentiation | **PARTIALLY DONE** | `main.tex` (in the manuscript zip) already has a substantive Related Work paragraph differentiating from HALP ("HALP... uses candidate-level preference learning from future re-access outcomes rather than direct action imitation... rather than explicit supervision on finite-horizon candidate-specific downstream harm"); `refs.bib` has `song2023halp`. This is real manuscript prose, not just an internal note. | Verify this HALP characterization against the primary HALP paper (flagged, not yet re-verified independently this session); decide if textual differentiation is enough or an empirical HALP run is expected. | **NEEDS DECISION** — implement HALP empirically vs. keep positioning-only. | If reviewers want empirical numbers, no amount of prose differentiation will satisfy them. |
| 4 | Important | AE item 11 (SIEVE/FIFO-Reinsertion) | SIEVE / FIFO-Reinsertion / modern lightweight baselines | **NOT STARTED** | Confirmed by direct grep: zero mentions of "sieve" or "fifo-reinsertion" anywhere in the repo (any branch/worktree) **or** in the real manuscript (`refs.bib`, `main.tex`). `kbs_baseline_positioning_plan.md` does not mention either algorithm. | **NEEDS DECISION**: (a) implement SIEVE and/or FIFO-Reinsertion as new eval-pipeline policies and run them (adds real compute on top of the existing sweep timeline), or (b) add positioning-only Related Work paragraphs (no empirical numbers), mirroring the existing HALP/MUSTACHE treatment. | Scope decision; if (a), additional multi-day compute. | Most concrete, easily fact-checked gap in the entire revision — explicitly named, currently zero response of either kind. |
| 5 | Critical | AE item 5 | Computational overhead analysis | **NOT STARTED** | `kbs_full_policy_comparison_execution_plan.md` documents the O(capacity)-per-miss inference cost of `evict_value_v1` (code-verified, `src/lafc/policies/evict_value_v1.py:179-200`) — but this is an internal engineering note, not a manuscript-ready overhead table; nothing in `main.tex`. | Mine wall-clock/throughput numbers from existing chunk run logs (`logs/kbs_full_policy_comparison/cap32/...`) or run a small dedicated micro-benchmark; write a short overhead subsection/table comparing policies. | **NEEDS DECISION**: mine existing logs (cheap) vs. dedicated benchmark (more compute). | O(capacity) is a real, fairly serious scalability property reviewers will likely probe hard; silence reads as avoidance. |
| 6 | Important | AE item 7 | Label construction scalability / preprocessing time | **PARTIALLY DONE** | 96G heavy_r1 dataset build has a real timing log (`logs/kbs_heavy_r1/build_dataset.log`, per ledger §4) — raw timing data exists but isn't distilled into a manuscript-ready statement/table. | Extract throughput numbers (rows/sec, GB/hour, total wall-clock for 662 shards) from the existing log; write a short paragraph. | None identified — appears to be writing-only, no new compute. | Moderate; one of the cheapest items to close. |
| 7 | Critical | AE item 6 | Replay horizon H justification and sensitivity | **PARTIALLY DONE** | Training already sweeps H∈{4,8,16} × 3 models (`analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, 9 rows); selects H=4/random_forest by val_mean_regret. **New finding this session**: (a) selection is based on only 5/7 trace families (~3.2% of available validation rows, a known sampling artifact, disclosed in the gap tracker but not in the manuscript); (b) the manuscript's embedded ablation table (`tab:evict-value-ablation` in `main.tex`) does **not** match the current heavy_r1 CSV at all — e.g. manuscript shows H=4/hist_gb val regret 0.0091 vs. the current CSV's 0.02100, with a different best-model ranking. The zip (committed 2026-06-17 22:08) predates the heavy_r1 retrain. | (1) Refresh `main.tex`'s ablation table with current heavy_r1 numbers; (2) disclose the 5/7-family validation-coverage caveat; (3) add an explicit H-sensitivity discussion (why H=4 over H=8/16; the close hist_gb/random_forest margin). | None for the refresh itself (mechanical, no compute); broader sensitivity narrative is writing-only. | **High** — newly discovered, concrete mismatch between the draft manuscript and the repo's own verified-current numbers; resubmitting as-is would contradict the provenance trail. |
| 8 | Critical | AE item 4 | Fallback mechanism validation or demotion | **PARTIALLY DONE** | `main.tex`'s Limitations section already demotes the fallback/guard mechanism in prose — explicitly calls its window/threshold/interval/duration/policy parameters heuristic and the layer itself "a practical robustness layer... rather than a fully established empirical contribution" (paraphrased). The demotion is already written. | Decide whether existing textual demotion is a sufficient response, or whether reviewers expect at least one minimal empirical validation (guard-on vs. guard-off comparison). `figures/manuscript/figure6_guard_wrapper_evict_value_v1.*` exists — needs checking whether it's wired to real or placeholder data. | **NEEDS DECISION** (textual demotion vs. empirical validation); if empirical, also depends on canonical CSV. | Moderate — manuscript is already conservative here, which helps, but being told "we know it's unvalidated" may not be what reviewers consider resolved. |
| 9 | Important | AE item 8 | Workload-specific breakdowns | **BLOCKED** | cap32's `.md` aggregate already contains a real per-trace-family breakdown (7 families, win/tie/loss vs. LRU) — proves the data shape works, just only for 1 of 4 capacities, and not yet in the manuscript. | Once the canonical CSV is complete, build a per-workload breakdown table/figure. | Canonical CSV (row 1). | Moderate; mechanically straightforward once row 1 is done. |
| 10 | Important | AE item 9 | Paper framing / excessive hedging | **NOT STARTED** (diagnosed, not edited) | Direct read of `main.tex`'s Discussion, Summary of Findings, Implications, and Limitations sections this session found the same scope-limiting disclaimer ("offline-only," "should be interpreted as... rather than a complete demonstration of superiority") repeated near-verbatim across 4 separate sections — concrete, located evidence, not a hypothesis. | Defer the bulk of this rewrite until after row 1 lands — a real online-replay result narrows what needs hedging. Cheap de-duplication could start now if time allows. | Partially blocked on row 1 (the hedging exists largely *because* there's no online-replay evidence yet). | Moderate; explicitly named, cosmetic in nature. |
| 11 | Important | AE item 10 | Manuscript shortening (excessive verbosity/repetition) | **NOT STARTED** | Same textual evidence as row 10 — repeated near-duplicate hedging is the most visible source of avoidable length in an ~8,919-word draft. | Same deferral logic as row 10; budget a dedicated edit pass near the end of the cycle. | Partially blocked on row 1, same reason as row 10. | Low-moderate; purely editorial. |
| 12 | Important | AE item 12 | Single-author / AI-tool credibility and validation depth | **PARTIALLY DONE** | `main.tex` already has an explicit "AI Declaration" section (ChatGPT, Cursor, Codex, Copilot, Gemini, Perplexity, with an explicit author-takes-full-responsibility statement); `author-agreement.tex` confirms sole authorship/originality. This is a real, already-existing disclosure, not a gap. | If the concern is about *credibility/rigor* rather than just disclosure, the response letter should point to objective rigor evidence already in this repo (283-test pytest suite, sanity-audit/platform-consistency-audit process, full sha256 provenance ledger) as counter-evidence. | None for the disclosure itself; response-letter framing is writing-only. | Low; one of the better-prepared concerns already. |
| 13 | Important | repo-internal (Step 4) | Reproducibility package | **DONE on unmerged branch / NOT STARTED on main** | Branch `kbs-revision-parallel-cleanup` (commits `64e39bb`, `9a1642a`) has 2 real reproducibility bugfixes + a CSV structural verifier script. Diffed against `main`: 14 files changed, 925 insertions, 8 deletions. Confirmed not merged, not pushed. | **NEEDS DECISION**: review and merge the parallel branch into `main` so these fixes are part of the citable package. | Human merge decision only — no compute. | Low if merged soon; moderate if left split indefinitely (confusion risk). |
| 14 | Submission/package | AE item 13 | DOCX submission package requirements | **NEEDS DECISION** | The real existing package (zip, commit `e9ac132`) is entirely LaTeX (`elsarticle.cls` + 3 `.bst` + `main.tex` + `refs.bib` + `cover-letter.tex` + `author-agreement.tex`), not DOCX. Figures inside independently confirmed 300dpi. No CRediT statement, no Highlights, no standalone Declaration-of-Interest file exist in **any** format (Declaration of Competing Interest is present but only inline inside `main.tex`). | Determine what "DOCX submission package requirements" actually means for KNOSYS-D-26-07461: (a) main manuscript must be Word (full LaTeX→DOCX conversion, non-trivial with equations/tables/bib), or (b) only certain ancillary files (cover letter, highlights, response letter) need to be Word/DOCX while the manuscript stays LaTeX/PDF (more common Elsevier pattern). | Missing external information — same category of blocker as the missing verbatim reviewer letter; cannot resolve from local files. | Moderate; if (a), LaTeX→DOCX conversion of an elsarticle document is error-prone and should be budgeted for, not left to the end. |

## C. Step 4 — completion audit

### Experiments / data

| Item | Status | Notes |
|---|---|---|
| 96G heavy_r1 dataset build | **DONE** | 662 shards, 7 trace families, capacities 32/64/128/256, horizons 4/8/16. |
| Dataset summary | **DONE** | `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` + extended JSON. |
| heavy_r1 model training | **DONE** | `TRAIN_EXIT=0`; best = horizon 4, random_forest, val_mean_regret≈0.0208 (see row 7 caveat on validation-set family coverage). |
| Validation eval | **DONE** | 1 trace × cap256, pipeline-correctness/timing-calibration only — explicitly not manuscript-citable on its own. |
| Validation sanity audit | **DONE** | Conclusion A: trustworthy; `evict_value_v1`'s worse-than-LRU result is real, not a wiring bug. |
| Platform consistency audit | **DONE** | Conclusion A: safe to run locally with tmux; no manuscript edit required. |
| cap32 canonical chunk | **DONE** | **Corrected from the task premise of "currently running."** Completed Fri 2026-06-19 00:31:13 EDT, exit 0, 49 rows. Aggregate: `evict_value_v1` loses to LRU on 6/7 trace families, ties on `wiki2018` (universal 100% miss rate there), -4.84% vs. LRU overall. |
| cap64 / cap128 / cap256 | **cap64 DONE (2026-06-20, see §L/§M) / cap128 DONE 2026-06-20/21, ANOMALY AUDITED (see §Q) / cap256 NOT STARTED, on hold** | `cap64_with_sieve_fifo` completed `CAP64_WITH_SIEVE_FIFO_EXIT=0`, ~10h04m, 56/56 rows — `evict_value_v1` gap vs LRU/SIEVE/FIFO-Reinsertion roughly halved vs cap32 (see `reports/kbs_cap64_result_analysis.md`). `cap128_with_sieve_fifo` completed `CAP128_WITH_SIEVE_FIFO_EXIT=0`, ~22h08m, 56/56 rows — gap vs LRU **reversed and widened** (+2.26%→+11.76%), concentrated in brightkite/citibike. A dedicated sanity audit (`reports/kbs_cap128_anomaly_sanity_audit.md`) found no bug across 6 independent checks; conclusion is likely-real model/policy behavior, mechanism unconfirmed. cap256 confirmed not launched via `tmux list-sessions`/`pgrep` — on hold pending review of the anomaly finding, per the audit's recommendation. |
| Final canonical merged CSV | **NOT STARTED (verification + draft trend tooling prepared through cap128, see §P/§Q)** | Blocked on the decision of whether/how to include cap256. `scripts/paper/verify_kbs_policy_chunks.py` has now been run against cap32+cap64+cap128 together (PASS, no errors/warnings) and `scripts/paper/build_kbs_policy_trend_artifacts.py` has been run against cap32+cap64 (PASS; draft-only outputs) — but no final merge into a single canonical CSV has been performed. cap128's anomaly does not block this merge technically (data is clean); it is a narrative/cap256 decision, not a data-readiness one. |
| Manuscript artifact regeneration after canonical results | **NOT STARTED** | `tables/manuscript/table3_main_quantitative_comparison.csv` stub still reads `NOT_VERIFIED`; `manuscript_artifact_manifest.json` has `"policy_comparison_present": false`. |

### Reports / planning

| Item | Status | Notes |
|---|---|---|
| Evidence ledger | **DONE** | `reports/kbs_revision_evidence_ledger.md`, 643 lines / 14 sections + appendix; gets a new §15 this session (see §E below). |
| Evidence manifest | **DONE** | `reports/kbs_revision_evidence/evidence_manifest.json`, 48 entries; gets a 49th this session. |
| Existing-results file inventory | **DONE** | Covered across ledger §4–§7 and the gap tracker. |
| Full execution plan | **DONE** | `kbs_full_policy_comparison_execution_plan.md`. |
| Validation sanity audit | **DONE** | (see above). |
| Platform consistency audit | **DONE** | (see above). |
| Revision gap tracker | **DONE** | `kbs_revision_gap_tracker.md`, current as of 2026-06-18. |
| Next-actions-without-rerun | **DONE** | `kbs_next_actions_without_rerun.md`. |
| Reviewer-response planning package | **PARTIALLY DONE** | Concern-class skeleton exists only on the unmerged parallel branch (`kbs_response_to_reviewers_skeleton.md`); not point-by-point since verbatim reviewer text is still unavailable. |
| Manuscript revision checklist | **DONE (unmerged)** | `kbs_manuscript_revision_change_checklist.md`, parallel branch only. |
| Submission package checklist | **PARTIALLY DONE / NEEDS CORRECTION** | `kbs_submission_package_checklist.md` (parallel branch) claims no cover letter/author agreement/manuscript source exists anywhere — **this is false**, corrected by this session's zip discovery. Needs updating before being trusted again. |

### Submission package / Word files

| Item | Status | Notes |
|---|---|---|
| DOCX skeleton directory | **NOT FOUND** | No `submission_kbs_revision_docx`-style directory anywhere in the repo. |
| Response to reviewers DOCX | **NOT STARTED** | Only a markdown skeleton exists, and only on the unmerged branch. |
| Cover letter DOCX | **NOT STARTED as DOCX** | Exists as `cover-letter.tex` (in the zip) but written for an **initial** submission ("Please consider the enclosed manuscript... for publication"), not this revision. Needs rewriting regardless of final format. |
| Highlights DOCX | **NOT STARTED** | No Highlights file or section in any format — confirmed by grep across all `.tex` files in the zip. |
| CRediT statement DOCX | **NOT STARTED** | No CRediT section/file in any format — confirmed by grep. |
| Declaration of interest DOCX | **PARTIALLY DONE** | A "Declaration of Competing Interest" section exists, but only inline inside `main.tex`, not as a standalone file in any format. May or may not satisfy Editorial Manager's separate-file requirement. |
| Author agreement DOCX | **NOT STARTED as DOCX** | Exists as `author-agreement.tex` (in the zip); content appears complete and reusable as-is. |
| Revised manuscript DOCX | **NOT STARTED as DOCX** | Exists as LaTeX source (`main.tex`, in the zip) but (a) has zero online-replay/Table-3 content yet, (b) its offline-ablation table is confirmed stale vs. the current heavy_r1 retrain (row 7), (c) needs the hedging/length edits from rows 10–11. Full consistency audit of `main.tex` itself now done (see §M) — finds the prose is already conservatively worded (no "robust/practical superiority," "validated fallback," etc.), but flags real staleness/omission gaps (SIEVE/FIFO-Reinsertion missing from Related Work/Table 2/6, three disagreeing policy rosters, Contribution #3/Limitations fallback-claim asymmetry). |

## D. Remaining Work (ranked)

### Critical before scientific revision can be credible

1. Complete the canonical 7×4×7 miss-ratio sweep — launch cap64/cap128/cap256, merge the 4 chunk CSVs, write the still-missing `.md`-aggregation follow-up script, regenerate Table 3/Figs 2–3 (matrix row 1).
2. Decide and execute the SIEVE/FIFO-Reinsertion scope — implement-and-run vs. positioning-only citation (row 4). This is the single cleanest, most checkable gap.
3. Computational overhead analysis — mine existing chunk-run logs or run a small dedicated benchmark; write a short table (row 5).
4. Refresh `main.tex`'s offline-ablation table with the current heavy_r1 numbers and disclose the 5/7-family validation-coverage caveat (row 7) — time-sensitive: the draft currently contradicts the repo's own verified-current numbers.
5. Decide the fallback-mechanism response — keep the existing textual demotion as sufficient, or add a minimal empirical validation (row 8).
6. Decide whether HALP needs an empirical reimplementation or can stay positioning-only, given the differentiation prose already exists (row 3).
7. Once the canonical CSV lands, decide whether the internal-baseline set reads as sufficient or an external SOTA method needs to actually be run (row 2).

### Important for manuscript quality

1. Workload-specific breakdown table/figure, once canonical CSV exists (row 9).
2. Hedging/scope-framing rewrite of Discussion/Summary of Findings/Implications/Limitations — defer bulk of this until after item 1 above narrows the claim (row 10).
3. Manuscript shortening / de-duplication pass — same deferral logic (row 11).
4. Related Work update to add SIEVE/FIFO-Reinsertion citations regardless of the empirical-scope decision (row 4).
5. Review and merge the parallel branch's reproducibility fixes into `main` (row 13).
6. Strengthen the response letter's framing of the AI-tool/single-author credibility point using existing rigor evidence (283-test suite, sanity/platform audits, provenance ledger) rather than re-stating only the existing AI declaration (row 12).
7. Fix the 4 incomplete BibTeX entries (`lykouris2018competitive`, `bansal2022weightedpaging`, `wei2020lacaching`, `chledowski2021robustlacaching`) — still open from `kbs_next_actions_without_rerun.md`.
8. Resolve the refs.bib duplication: the zip's `refs.bib` (37 entries, the real/complete manuscript bibliography) vs. the repo's `refs/related_work_table6.bib` (11 entries, a smaller auxiliary file) — confirm which one is canonical going forward so edits land in the right place.

### Submission / package tasks

1. Clarify the actual KNOSYS Editorial Manager DOCX requirements for this manuscript ID — cannot be resolved locally (row 14).
2. Rewrite `cover-letter.tex` for the revision context (currently addressed to an initial submission).
3. Write Highlights (format/length per the journal's actual requirement).
4. Write a CRediT author statement.
5. Decide whether the Declaration of Competing Interest needs to also exist as a standalone file (currently only inline in `main.tex`).
6. Draft the real point-by-point response-to-reviewers once both the verbatim reviewer text and the canonical CSV exist.
7. Final figure/table/citation consistency check once `main.tex` is updated with the new Table 3/Figs 2–3 — figures are already confirmed 300dpi, so this is a consistency check, not a re-export.

## E. Recommended next actions (ordered)

Be specific and conservative — no new compute job is implied by any of these
unless explicitly stated.

1. **Right now (no chunk is currently running — cap32 already finished, cap64/128/256 not yet launched)**: do the zero-compute items first. Mine `logs/kbs_heavy_r1/build_dataset.log` and the cap32 run log for the overhead/label-scalability numbers (matrix rows 5–6); refresh the stale offline-ablation table and disclose the validation-coverage caveat in `main.tex` (row 7); review the parallel branch's diff and decide on the merge (row 13); fix the 4 incomplete BibTeX entries.
2. **Immediately after the next chunk (e.g., cap64) finishes**: spot-check its aggregate `.md` exactly as was done for cap32 here (per-family win/tie/loss vs. LRU) to get an early read on whether the worse-than-LRU pattern holds at larger capacities, before committing further compute to cap128/cap256.
3. **Decide before launching cap64**: (a) get explicit confirmation that the team wants to spend the remaining ~1.5–3 days of compute given that cap32 already shows `evict_value_v1` losing on 6/7 families — vs. considering whether the model/horizon needs rework first; (b) finalize the SIEVE/FIFO-Reinsertion scope decision now, since if "implement and run" is chosen, those policies should be added to the *same* remaining chunked runs rather than triggering a second multi-day sweep later.
4. **Can be done without any more heavy compute, any time**: bibliography fixes; cover-letter and author-agreement revision-context updates; Highlights and CRediT drafting; the parallel-branch merge; the AI-disclosure response-letter framing; an initial hedging/length edit pass on the parts of Discussion/Limitations that are redundant independent of new results.

## F. Zero-compute consolidation pass update (2026-06-19)

Item 1 above ("do the zero-compute items first") is now substantially done.
Seven reports were produced this session, all read-only/zero-compute:

- `reports/kbs_cap32_policy_comparison_report.md` — full cap32 verification
  (closes the "spot-check cap32" half of matrix row 1).
- `reports/kbs_overhead_and_scalability_evidence.md` — closes matrix rows
  5–6 (computational overhead, label-construction scalability) with
  measured numbers, separated explicitly from proposed-but-unrun
  benchmarks.
- `reports/kbs_stale_artifact_refresh_plan.md` — root-causes matrix row 7's
  stale-ablation-table problem to an exact commit boundary; **plan only,
  refresh script intentionally not executed**.
- `reports/kbs_baseline_gap_action_plan.md` — closes matrix rows 3–4 (HALP,
  SIEVE/FIFO-Reinsertion) with a strict classification table and a
  recommendation (SIEVE first). Surfaces a new sub-finding: `rest_v1` has
  no Table 2 row in `main.tex`, and the similarly-named "R-FTP+Marker" row
  is actually a different, unwired policy.
- `reports/kbs_complete_reviewer_comment_matrix.md` and
  `reports/kbs_response_to_reviewers_skeleton.md` — created fresh on
  `main` (differently-structured drafts of the same filenames already
  existed, unreconciled, only on the unmerged `kbs-revision-parallel-cleanup`
  branch). Both carry an explicit caveat that verbatim AE/Reviewer #2/
  Reviewer #3 numbered text has still never been supplied in this repo's
  history.
- `reports/kbs_docx_submission_package_report.md` — closes matrix row 14's
  local-inspection half (confirms zero DOCX files, no skeleton directory,
  package is LaTeX-only); the actual Editorial Manager requirement remains
  an external-information blocker, same as before.

**Still not done from item 1**: the BibTeX fixes, the parallel-branch merge
review/decision, and actually running the artifact-refresh script (plan
exists, execution does not). **Still not done from item 3**: no go/no-go
decision has been made on cap64 — that remains the user's call, not
something resolved by this consolidation pass.
5. **Should wait until the final canonical CSV exists**: Table 3 and Figs. 2–3 generation; the workload-specific breakdown table; the final point-by-point response-to-reviewers numeric claims; locking in the Discussion's narrative wording about how `evict_value_v1` compares to LRU (cap32's pattern is consistent with 2 other independent data points, but it is still only 1 of 4 capacities — don't finalize the prose until all 4 are in).

## G. Real-reviewer-comments correction pass (2026-06-19)

The user supplied the full verbatim AE / Reviewer #2 / Reviewer #3 text
directly in this session. It is now recorded in
`reports/kbs_real_reviewer_comments.md` and used as the source for a rebuilt
`reports/kbs_complete_reviewer_comment_matrix.md` (22 rows, real source
labels) and `reports/kbs_response_to_reviewers_skeleton.md` (rebuilt around
AE/Reviewer #2/Reviewer #3 headings with a fixed 6-tag status vocabulary).
§B above, the old 13-section response skeleton, and several other reports
that previously said reviewer comments were "missing" are now superseded —
each one was updated in place with a pointer here rather than deleted, per
instruction.

**This does not, by itself, resolve any compute or baseline-scope blocker.**
What changed is which document is authoritative for *what the reviewers
asked for*, not how much of it has been answered. The remaining-work ranking
in §D above is superseded by the following, sharper, 5-item ranking, now
informed by exact reviewer language (R3-Issue1-3, R3-Rec1-2/5) instead of
category-level paraphrase:

1. **Final canonical end-to-end miss-ratio sweep** (cap64/cap128/cap256 +
   merge). Directly answers R3-Issue1/R3-Rec1, R2-MC3, and is a prerequisite
   for R3-Issue4/Rec4 (title/scope framing) and R3-Minor9/Rec8 (workload
   breakdowns). Still not launched, per instruction. See
   `reports/kbs_before_cap64_baseline_decision_memo.md` for whether it
   should wait on baseline additions first.
2. **Baseline-scope decision for SIEVE / FIFO-Reinsertion / HALP**
   (R3-Issue2/3, R3-Rec2). SIEVE: implemented, with `cap32_with_sieve`
   currently running. FIFO-Reinsertion: now implemented, source-verified,
   tested, and judged scientifically defensible in
   `reports/kbs_fifo_reinsertion_baseline_audit.md`; the remaining question
   is whether to commit to it in the final canonical policy set before
   cap64. HALP: empirical reimplementation judged infeasible before
   2026-07-08; citation+differentiation prose already exists. This decision
   determines whether item 1 above should be delayed to add baselines first.
3. **Overhead / timing evidence** (R2-MC2, R3-Issue5/Rec3). Complexity-class
   claim (O(capacity) vs O(1)) is already code-verified; measured wall-clock
   numbers exist but are confounded across different chunk runs. A small,
   scoped, controlled timing benchmark (proposed in
   `reports/kbs_overhead_and_scalability_evidence.md` Part 2, not yet run)
   would close this with well under an hour of new compute — independent of
   item 1.
4. **Fallback validate-or-demote decision** (R3-Issue6/Rec5, R2-MC3's
   guarded-fallback half). No fallback-specific ablation artifact exists
   anywhere in the repo. Needs a decision on whether to validate it with new
   evidence or demote/remove the contribution claim.
5. **Manuscript rewrite and DOCX package** (R3-Minor8-9, R3-Rec4/6/7,
   R3-Issue7, AE roll-up, plus the still-unresolved DOCX-format question in
   `reports/kbs_docx_submission_package_report.md`). Lowest urgency of the
   five because most of it is pure writing that should happen last, once the
   content decisions above are settled — rewriting hedging/length/framing
   before knowing the canonical sweep result risks redoing the same prose
   twice.

## H. Zero-compute reviewer-response package pass (2026-06-19, cap32_with_sieve running)

This later pass was explicitly constrained to read-only monitoring of the
running `cap32_with_sieve` job plus report/manuscript-planning work. It did
not launch, stop, restart, or edit any experiment.

New or refreshed outputs in this pass:

- `reports/kbs_parallel_reviewer_work/non_compute_reviewer_work_package.md`
- `reports/kbs_halp_fifo_source_verification.md`
- `reports/kbs_fallback_revision_strategy.md`
- `reports/kbs_horizon_h4_revision_strategy.md`
- `reports/kbs_overhead_manuscript_text_draft.md`
- `reports/kbs_manuscript_shortening_and_reframing_plan.md`
- `reports/kbs_docx_package_action_plan.md`
- tracker/skeleton/ledger updates reflecting those drafts and the current
  `cap32_with_sieve` run state

Net effect:

- HALP/FIFO source verification is now documented as done.
- Fallback handling now has a conservative demotion strategy draft.
- H=4 now has a stronger evidence-backed explanation draft.
- Overhead now has manuscript-ready text separating measured evidence from
  still-missing timing benchmarks.
- Manuscript-shortening and DOCX planning are both concretely drafted.

## I. FIFO-Reinsertion readiness audit pass (2026-06-19, still while cap32_with_sieve runs)

This additional later pass audited the already-present
`fifo_reinsertion` baseline without touching the running canonical job.

New outputs:

- `reports/kbs_fifo_reinsertion_baseline_audit.md`
- `reports/kbs_final_baseline_set_decision_before_cap64.md`

Checks completed:

- confirmed code path: `src/lafc/policies/fifo_reinsertion.py`
- confirmed runner wiring in both canonical and general runners
- confirmed dedicated tests already exist
- ran the requested targeted pytest slice successfully inside the project
  venv: `39 passed, 260 deselected`
- ran a fresh tiny smoke test with separate `_smoke_audit` outputs only

Main conclusion:

- FIFO-Reinsertion is no longer just a definitional placeholder; it is
  implemented, tested, source-verified, and ready for canonical-sweep
  consideration.
- If the final canonical policy set includes it, the currently running
  `cap32_with_sieve` chunk will not be sufficient on its own and cap32 will
  need a later rerun with both SIEVE and FIFO-Reinsertion.

## J. Results-independent manuscript/rebuttal drafting pass (2026-06-19, while `cap32_with_sieve_fifo` runs)

This pass was explicitly constrained to read-only status checks of the
running `cap32_with_sieve_fifo` job plus drafting work. It did not launch,
stop, restart, or edit that job, and did not launch cap64/cap128/cap256.

New outputs, all in `reports/kbs_manuscript_rebuttal_drafts/`:

- `started_at.txt`
- `response_paragraph_bank.md` — 11 results-independent rebuttal paragraphs
  (AE; R2-MC1-3; R3-Issue2,3,5,6,7; R3-Minor8,9), each either usable today
  or carrying an explicit `[INSERT FINAL CANONICAL RESULT: ...]`
  placeholder. None claim `evict_value_v1` outperforms LRU, SIEVE, or
  FIFO-Reinsertion; the cap32 (no-SIEVE/no-FIFO) result — losing to LRU on
  6/7 families, -4.84% mean misses — is stated directly.
- `manuscript_insertions_draft.md` — cautious manuscript-body draft text
  for 8 subsections ("End-to-end policy evaluation", "Baselines",
  "Computational overhead and scalability", "Horizon selection", "Relation
  to HALP", "Fallback mechanism and limitations", "Reproducibility and
  validation", "Limitations").
- `title_and_contribution_reframing_options.md` — 3 title options
  (conservative/moderate/stronger, tied to which sweep outcome would
  justify each) plus revised contribution bullets emphasizing the
  decision-aligned target, end-to-end audit, modern baselines,
  reproducibility artifacts, and honest negative-finding reporting.
- `docx_package_text_bank.md` — reusable text for a revision-specific
  cover letter, Highlights, CRediT statement, Declaration of Competing
  Interest (matches the existing inline `main.tex` text), and an AI-tool
  disclosure (extends the existing inline `main.tex` text). Grounded by
  reading the real `cover-letter.tex` and `main.tex` AI/competing-interest
  text out of the existing submission zip
  (`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`).

Net effect: every reviewer-comment item and every manuscript section that
does *not* require a finished canonical sweep now has draft text ready to
insert; everything that does require it has a clearly marked placeholder
rather than a fabricated number. Tracking files
(`kbs_response_to_reviewers_skeleton.md`,
`kbs_complete_reviewer_comment_matrix.md`, `kbs_revision_gap_tracker.md`,
`kbs_next_actions_without_rerun.md`, `kbs_revision_evidence_ledger.md`,
`kbs_revision_evidence/evidence_manifest.json`) were updated to point at
this new package, marked prepared-but-not-final throughout.

## K. cap32_with_sieve_fifo completion and result analysis (2026-06-19)

The corrected canonical cap32 chunk finished successfully on the current
local/cloud machine:

- `CAP32_WITH_SIEVE_FIFO_EXIT=0`
- Runtime ~5h16m (Fri 14:42:27 → 19:58:48 EDT)
- Outputs: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md` (56/56 rows, 8 policies)
- Policy list: `lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

Key findings (full tables in `reports/kbs_cap32_with_sieve_fifo_result_analysis.md`):

- Best on aggregate: LRU ≡ rest_v1 (34,667 mean misses); fifo_reinsertion essentially tied (−0.06%)
- SIEVE: −1.86% vs LRU aggregate
- `evict_value_v1`: −4.84% vs LRU, −2.92% vs SIEVE, −4.77% vs FIFO-Reinsertion; no strict wins on any family
- Common policies (lru, rest_v1, evict_value_v1, predictive_marker, blind_oracle_lru_combiner) match prior no-SIEVE cap32 exactly; trust_and_doubt has minor drift (≤73 misses)

New reports:

- `reports/kbs_cap32_with_sieve_fifo_result_analysis.md`
- `reports/kbs_after_cap32_decision_memo.md` (recommends **Option C: cap64 only** — not launched)

cap64/cap128/cap256 remain not launched. Merged canonical CSV still missing.

## L. cap64_with_sieve_fifo launch (2026-06-19)

Option C (recommended in §K above) was approved by the user. The cap64
chunk was launched using the same final 8-policy baseline set as the
completed `cap32_with_sieve_fifo` chunk, capacity 64 only, on this same
local/cloud machine (not Wulver, no Slurm), in tmux so SSH/network
disconnects cannot interrupt it.

- tmux session: `kbs_full_policy_comparison_cap64_with_sieve_fifo`
- Runner PID `261643`, tee PID `261644`
- Started: `Fri Jun 19 10:29:23 PM EDT 2026`
- Policy list: `lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`
- Expected outputs (not yet written, job still running):
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv/.md`
- Expected runtime ~5-7h by analogy to cap32's actual 5h16m
- Pre-flight checks passed: branch `main`, `git status` reviewed, trace
  manifest/model/dataset-manifest/policy source files present, targeted
  `pytest tests/test_sieve.py tests/test_fifo_reinsertion.py` → 16/16
  passed, disk/memory/load healthy, output paths confirmed absent before
  launch
- New report: `reports/kbs_cap64_with_sieve_fifo_launch_report.md`

**cap128/cap256 explicitly not launched** — that remains a separate
decision pending review of the cap64 result, consistent with the Option C
rationale in `reports/kbs_after_cap32_decision_memo.md`. Nothing was
pushed, merged, deleted, overwritten, or renamed during this pass.

## M. Manuscript consistency audit (zero-compute, 2026-06-19, while cap64_with_sieve_fifo runs)

Read-only audit of the actual submission package against current repo
state, performed entirely while `cap64_with_sieve_fifo` continued running
untouched. No experiments run; no files inside the zip modified.

- **Source**: `Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
  — the only place `main.tex`/`refs.bib`/`cover-letter.tex`/
  `author-agreement.tex` exist; confirmed via `find` that no loose copies
  exist anywhere else in the repo. Extracted read-only to
  `/tmp/kbs_manuscript_audit_readonly` (outside the repo, not committed).
- **Headline finding**: the specific overclaiming phrases named in the
  audit brief ("robust superiority," "practical superiority," "end-to-end
  improvement," "lightweight production readiness," "validated fallback,"
  "broad generality of H=4") do **not** appear anywhere in `main.tex` —
  every related sentence found was already explicitly hedged. The
  manuscript text is in materially better shape on overclaiming than the
  audit brief's framing assumed.
- **Real gaps found** (staleness/omission, not overclaiming):
  1. SIEVE and FIFO-Reinsertion — implemented, tested, results-backed
     (cap32 done, cap64 running) — are absent from Related Work, Table 2,
     and Table 6 entirely.
  2. Three mutually inconsistent policy rosters exist: `main.tex`'s
     embedded Table 2 (7 entries), `tables/manuscript/table2_policy_roster.csv`
     (6 entries), and the actual canonical 8-policy CLI list. None match.
  3. The offline-ablation table (`tab:evict-value-ablation` /
     `table4_main_ablation.csv`) is confirmed stale vs. the current
     uncommitted heavy_r1 retrain (pinned to commit `53726ce`).
  4. No online end-to-end results table/figure exists anywhere in
     `main.tex` — not stale, never written; gated on a canonical
     multi-capacity CSV that has never existed and, under Option C, may
     never exist as a single file.
  5. Contribution #3 (Introduction) states the guarded-fallback mechanism
     in confident language not matched by its heavily hedged treatment in
     Limitations — the one real claim/hedge asymmetry found; recommended
     fix is the already-drafted demotion in
     `reports/kbs_fallback_revision_strategy.md`.
- **Deliverable**: `reports/kbs_manuscript_consistency_audit.md` — 8
  sections (executive summary; claims to remove/soften; claims that can
  remain; tables/figures requiring regeneration or net-new creation;
  per-section rewrite classification; reviewer-comment coverage mapped to
  real labels; recommended immediate zero-compute edits; edits that must
  wait on cap64/cap128/256).
- **Job status re-verified at audit time**: tmux session
  `kbs_full_policy_comparison_cap64_with_sieve_fifo` alive, PID `261643`
  alive, log 0 bytes (no exit marker), started `Fri Jun 19 10:29:23 PM EDT
  2026`. Not touched, stopped, or restarted during this audit.
- **cap128/cap256 not launched.** Nothing pushed, merged, deleted, or
  overwritten. The manuscript rewrite itself is **not** performed in this
  pass — only the audit and tracker updates.
Tracking files updated in this pass.

## N. Safe manuscript-source edits applied (zero-compute, 2026-06-19, while cap64_with_sieve_fifo runs)

This pass acted on §M's audit findings — the audit itself was read-only;
this pass made the actual text edits, still entirely independent of
cap64/128/256 results. Performed entirely while `cap64_with_sieve_fifo`
continued running, untouched, on the local/cloud machine (not Wulver, no
Slurm).

- **Durable working copy created**: the submission zip
  (`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`)
  was extracted **read-write** into a new, repo-tracked directory,
  `manuscript_source/`, since the original zip must not be overwritten
  (safety rule) but the task explicitly authorized "manuscript-source
  edits." `manuscript_source/main.tex` and `manuscript_source/refs.bib` are
  now the live working copy; the zip itself was never touched.
- **Citations**: added `zhang2024sieve` (NSDI'24, full BibTeX entry,
  author/venue/pages sourced from the already-completed
  `reports/kbs_sieve_source_verification.md`) to `manuscript_source/refs.bib`.
  Used for both the new SIEVE baseline and the FIFO-Reinsertion/CLOCK-family
  discussion in a new Related Work paragraph. `yang2023s3fifo` deliberately
  **not** added — no dedicated source-verification pass exists for it in this
  repo (unlike SIEVE), so fabricating a BibTeX entry from memory was judged
  too risky; flagged as an open TODO instead. `song2023halp` verified already
  adequate (existing differentiation prose makes no empirical-comparison
  claim) — left untouched.
- **Fallback/guard demotion**: applied the already-drafted demotion language
  from `reports/kbs_fallback_revision_strategy.md` to 5 actual locations in
  `main.tex` (Abstract, Contribution #3, Method section ×2, Datasets/Baselines
  cross-reference sentence) — "implementation safeguard," "optional guard,"
  "evaluated only as an ablation," explicit "we do not claim that it improves
  end-to-end miss ratio unless future experiments support it." This closes
  the claim/hedge asymmetry §M (item 5) flagged between Contribution #3 and
  Limitations.
- **Policy roster reconciliation**: fixed all three locations §M (item 2)
  found inconsistent. `main.tex`'s Table 2 (`tab:main_policy_families`) and
  the Datasets/Baselines prose now name exactly the canonical 8 policies
  (`lru, sieve, fifo_reinsertion, predictive_marker,
  blind_oracle_lru_combiner, trust_and_doubt, rest_v1, evict_value_v1`),
  replacing two non-canonical rows (Marker, R-FTP+Marker) that were never
  run in any cap32/64 chunk. This pass also fixed a real, previously
  undiscovered omission: `rest_v1`/REST had **zero** mentions anywhere in
  `main.tex` before this edit, despite being part of the actual canonical
  comparison set. `tables/manuscript/table2_policy_roster.csv/.tex` and
  `reports/manuscript_artifacts/latex_snippets/table2_snippet.tex` were
  regenerated (not hand-edited) by adding `sieve`/`fifo_reinsertion` rows to
  `_build_table2_policy_roster()` in
  `scripts/paper/build_kbs_main_manuscript_artifacts.py` and re-running only
  that function — so future regenerations won't silently revert the fix.
  `blind_oracle` (diagnostic-only) confirmed still excluded everywhere, now
  stated explicitly in the Table 2 caption.
- **Overhead and horizon placeholders**: inserted two new Discussion
  subsections, "Overhead and Scalability" and "Replay Horizon Selection"
  (`main.tex`, before `\section{Conclusions and Future Research}` — nothing
  inside Conclusions touched), using only already-measured, already
  source-backed evidence from `reports/kbs_overhead_manuscript_text_draft.md`
  and `reports/kbs_horizon_h4_revision_strategy.md`. Each carries a
  `% TODO(revision)` LaTeX comment (not `\todo{}`, which is undefined in this
  document's preamble) flagging the still-missing controlled timing
  benchmark and the still-missing citibike/metakv validation coverage /
  per-capacity horizon interaction, respectively. The horizon subsection
  explicitly discloses the 5-of-7-family validation-coverage caveat in the
  manuscript text itself for the first time (previously only in internal
  reports) — this also closes the matching open checklist item in
  `reports/kbs_next_actions_without_rerun.md`.
- **Compile verification**: `tectonic` (already installed) compiled
  `manuscript_source/main.tex` end-to-end twice, scratch output only — exit
  code 0 both times, valid `main.pdf` produced, zero undefined
  citation/cross-reference warnings (confirms `zhang2024sieve` and all new
  `\ref`/`\label` usage resolve correctly).
- **Left untouched** (cap64/128/256-dependent): the entire Conclusions
  section; no Table 3 exists anywhere to mark (confirmed again by grep); no
  sentence anywhere asserts a cap64/128/256 numeric result.
- **Full detail**: `reports/kbs_safe_manuscript_source_edits_report.md` (8
  required sections: files edited, exact nature of edits, citations,
  fallback wording, roster reconciliation, untouched sections, checks run,
  risks/TODOs).
- **Job status re-verified at the end of this pass**: tmux session
  `kbs_full_policy_comparison_cap64_with_sieve_fifo` still running, no exit
  marker. Not touched. cap128/cap256 **not launched**. Nothing pushed,
  merged, deleted, or overwritten.

## O. cap64_with_sieve_fifo completion, result analysis, and cap128 launch (2026-06-20)

- **cap64 status check (read-only)**: `kbs_full_policy_comparison_cap64_with_sieve_fifo`
  found idle at a shell prompt (job already finished by the time this pass
  began). Log confirms `CAP64_WITH_SIEVE_FIFO_EXIT=0`, runtime ~10h04m
  (Fri 22:29:17 PM → Sat 08:33:51 AM EDT). Both output files present:
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv/.md`
  (56/56 rows). All other tmux sessions confirmed idle; no cap128/cap256
  tmux session or process existed at the start of this pass.
- **cap64 result analysis**: `reports/kbs_cap64_result_analysis.md`. Key
  finding: `evict_value_v1`'s aggregate gap narrowed sharply vs cap32 — LRU
  (−4.84%→−2.26%), SIEVE (−2.92%→−0.13%, nearly closed), FIFO-Reinsertion
  (−4.78%→−2.19%) — and this narrowing is *not* a generic
  capacity-convergence artifact, since SIEVE's and FIFO-Reinsertion's own
  gaps vs LRU stayed flat over the same span. The narrowing is concentrated
  in 2/7 families (metacdn: 22.4%→2.7%; metakv: 1.7%→~0%); brightkite and
  cloudphysics are flat-to-slightly-worse; wiki2018 remains degenerate
  (0% hit rate, all policies, both capacities). `evict_value_v1` already
  wins outright vs SIEVE on twemcache, and that win widened. No outright
  win vs LRU or FIFO-Reinsertion on any family at either capacity.
- **Decision memo**: `reports/kbs_after_cap64_decision_memo.md` — three
  options considered (A: stop/reframe now, B: cap128 only then reassess,
  C: cap128+cap256 together). Recommended **Option B**, reasoning that two
  data points (32, 64) show a direction but not a confirmed trend, that a
  prior single-trace cap256 validation point had already reversed once
  before, and that the run script has no mid-run checkpointing — making a
  combined cap128+cap256 commitment riskier than two staged single-chunk
  decisions, consistent with how cap32→cap64 was sequenced.
- **Option B approved and launched**: `cap128_with_sieve_fifo` launched in
  tmux session `kbs_full_policy_comparison_cap128_with_sieve_fifo` (runner
  PID `276106`) on the local/cloud machine — not Wulver/Slurm, no
  `sbatch`/`squeue`/`sacct` used. Started `Sat Jun 20 09:31:03 AM EDT 2026`.
  Pre-flight gates passed before launch: `git branch --show-current` →
  `main`; `tests/test_sieve.py` + `tests/test_fifo_reinsertion.py` → 16/16
  passed; required manifest/model/derived-manifest files confirmed present;
  cap128 output CSV/MD confirmed absent (`test ! -e`) before launch; disk
  499G free, memory 59Gi available, load average ~0.02 (idle) at launch
  time. Same final 8-policy set as cap32/cap64
  (`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`).
  Full detail: `reports/kbs_cap128_with_sieve_fifo_launch_report.md`.
- **cap256 confirmed not launched** — out of scope for this pass per
  instruction; next decision point is after cap128 completes and is
  analyzed, mirroring the cap32→cap64 pattern. Nothing pushed, merged,
  deleted, renamed, or overwritten in this pass.

## P. Post-cap128 tooling prepared while cap128_with_sieve_fifo runs (2026-06-20)

- **cap128 status re-verified (read-only)**: `kbs_full_policy_comparison_cap128_with_sieve_fifo`
  tmux session live, driver PID `276106` in state `R`, ~6h12m-6h41m elapsed
  across the checks in this pass, no `CAP128_WITH_SIEVE_FIFO_EXIT=` marker,
  no output CSV/MD yet. Not touched at any point. cap256: no tmux session,
  no process — confirmed not launched.
- **Verification script created**: `scripts/paper/verify_kbs_policy_chunks.py`.
  Checks column schema, expected capacities/policies/traces, duplicate
  (trace, capacity, policy) rows, per-(trace, capacity) policy completeness,
  numeric sanity (nonnegative misses, `hit_rate` in [0, 1]), and rejects the
  diagnostic-only `blind_oracle` policy (distinct from the fair
  `blind_oracle_lru_combiner`) unless explicitly allowed via
  `--allow-diagnostic-blind-oracle`. Run against
  `evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv`
  + `..._cap64_with_sieve_fifo.csv` with `--expected-capacities 32,64` and the
  canonical 8-policy roster: **PASSED**, zero errors, zero warnings.
- **Draft trend-artifact script created**: `scripts/paper/build_kbs_policy_trend_artifacts.py`.
  Computes per-capacity mean misses by policy, relative gap vs LRU,
  per-trace-family ranking, `evict_value_v1` gap vs LRU/SIEVE/FIFO-Reinsertion,
  and the cap32→cap64(→cap128→cap256) trend direction. Every output is
  stamped with a `DRAFT / AVAILABLE CAPACITIES ONLY` label; the script never
  writes to the canonical `table3_main_quantitative_comparison.csv` filename,
  only to new `*_available_capacities` filenames, and refuses to imply
  finality even when all `--capacities` requested happen to be present.
  Run on the cap32+cap64 chunks only: produced
  `analysis/kbs_policy_trend_available_capacities.csv`,
  `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`,
  `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`.
  Headline (cap32→cap64, draft): `evict_value_v1` gap vs LRU narrows
  +4.84%→+2.26%; vs SIEVE +2.92%→+0.13%; vs FIFO-Reinsertion +4.77%→+2.19%
  (all still losses, consistent with `kbs_cap64_result_analysis.md`).
- **Templates prepared (fill-in only, no results written yet)**:
  `reports/kbs_post_cap128_decision_template.md` (7 sections: completion
  status, headline result, cap32→64→128 trend, gap-vs-baselines table,
  improvement/stall/reversal classification, cap256 go/no-go, manuscript
  framing) and `reports/kbs_cap256_launch_template.md` (exact tmux command
  mirroring the cap128 launch convention, explicitly marked **DO NOT RUN
  UNTIL USER APPROVES AFTER CAP128 ANALYSIS**, not executed).
- **Confirmed throughout this pass**: cap128 untouched (no stop/start/restart);
  cap256 not launched; no canonical merged CSV created (per instruction, since
  cap128 has not finished); no existing manuscript artifact overwritten — all
  three new draft-output paths (`analysis/kbs_policy_trend_available_capacities.csv`,
  `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`,
  `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`)
  were new files, confirmed absent before this pass; nothing pushed, merged,
  deleted, or committed.

## Q. cap128 completion, anomaly discovery, and sanity/root-cause audit (2026-06-20/21)

- **cap128_with_sieve_fifo completed**: `CAP128_WITH_SIEVE_FIFO_EXIT=0`,
  runtime ~22h08m, 56/56 rows in
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv/.md`.
- **Anomaly identified (read-only result analysis)**: `evict_value_v1`'s gap
  vs LRU, which had been *narrowing* from cap32 (+4.84%) to cap64 (+2.26%),
  **reversed sharply** at cap128 (+11.76%) — concentrated in `brightkite`
  (+50.9% gap at cap128) and `citibike` (+45.5%). This breaks the
  cap32→cap64 improving trend that motivated the Option B decision to run
  cap128 in the first place.
- **Sanity/root-cause audit performed** (this pass, ~1h read-mostly budget,
  no cap256, no full sweep, no overwrite of canonical outputs):
  `reports/kbs_cap128_anomaly_sanity_audit.md`. Seven checks: (1) raw
  cap128 row verification for brightkite/citibike/metacdn/twemcache —
  clean, no duplicates, correct request counts; (2)
  `verify_kbs_policy_chunks.py` run across cap32+cap64+cap128 together —
  **PASSED**, zero errors/warnings; (3) targeted reproducibility check —
  the full-scale (50k-request) brightkite+citibike-only recheck was
  estimated at ~6.3h (exceeds the ~1h budget) and was **not run**; instead,
  six small-scale (1,000-request) throwaway probes were run in `/tmp/`
  (outside the repo, not tracked), confirming determinism (byte-identical
  reruns) and that the same reversal direction reproduces even at 1/50th
  scale; (4) full code-path audit of `evict_value_v1.py`,
  `evict_value_features_v1.py`, `evict_value_model_v1.py`,
  `evict_value_dataset_v1.py` — no capacity-specific branching anywhere in
  the codebase (grep-confirmed), train/serve feature parity confirmed, one
  raw (non-rank-normalized) feature (`cache_unique_bucket_count`) flagged
  as a plausible but unconfirmed contributor; (5) model/distribution
  check — cap128 was included in training (87,004,032 rows total,
  confirmed by independently summing shard files), the same single
  multi-capacity model serves cap32/64/128, and brightkite/citibike are
  the two least-represented families in training at every capacity
  (constant relative share, not uniquely thin at cap128) — a credible
  aggravating factor but not a sufficient explanation for the
  cap128-specific widening on its own.
- **Conclusion**: likely real model/policy behavior, not a bug or data
  artifact — every integrity/structural/code-path check came back clean.
  The leading (unconfirmed) hypothesis is compounding distribution shift
  from the documented single-step LRU-continuation training-label
  methodology (`docs/evict_value_v1_method_spec.md`), aggravated by
  brightkite/citibike's thinner training representation.
- **Recommendation**: do **not** launch cap256 yet. Report cap32→cap64→cap128
  as a genuine non-monotonic capacity-sensitivity finding tied to the known
  labeling limitation, rather than suppressing or compute-ing past it. The
  staged ~6.3h full-scale 2-trace recheck remains available as a
  cheaper-than-cap256 next confirmation step if the user wants higher
  confidence before deciding on cap256, but requires separate approval.
- **Confirmed throughout this pass**: cap128 was not re-run, stopped, or
  modified (it had already completed before this pass began); cap256 not
  launched; no canonical cap32/cap64/cap128 output overwritten; no
  push/commit/merge/delete/rename performed. New artifacts: this section,
  `reports/kbs_cap128_anomaly_sanity_audit.md`, and the throwaway
  `/tmp/cap128_probe/` probes (outside the repo, not part of the
  deliverable).

## R. Revision-writing pass — manuscript and response-to-reviewers updated using cap32/64/128 evidence (2026-06-21)

- **Instruction**: proceed with revision-writing using the completed
  cap32/cap64/cap128 evidence and the cap128 anomaly audit; do not launch
  cap256; do not run heavy experiments; do not overwrite raw CSV/MD
  outputs; do not push/commit/merge/delete/rename anything; manuscript/
  report editing is allowed; keep all claims honest (`evict_value_v1` does
  not beat LRU/SIEVE/FIFO-Reinsertion end-to-end).
- **Capacity-explicit tables/figures rebuilt** (zero new compute — reused
  already-complete chunk CSVs): re-ran
  `scripts/paper/build_kbs_policy_trend_artifacts.py` with
  `--inputs <cap32,cap64,cap128 with_sieve_fifo> --capacities 32,64,128`,
  refreshing `analysis/kbs_policy_trend_available_capacities.csv`,
  `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`,
  `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md` —
  all still explicitly labeled "DRAFT / AVAILABLE CAPACITIES ONLY ... cap256
  NOT run" and still distinct from the canonical
  `table3_main_quantitative_comparison.csv` (left untouched in its
  `NOT_VERIFIED` stub state; that pathway is gated on a different,
  capacity-blind merged CSV that does not exist and is out of scope here).
  Created a new script, `scripts/paper/build_kbs_available_capacities_figure.py`,
  producing `figures/manuscript/figure_available_capacities_trend_DRAFT.pdf/.png`
  and a LaTeX snippet — a two-panel figure (mean misses by capacity per
  policy; `evict_value_v1`'s gap vs LRU/SIEVE/FIFO-Reinsertion by capacity).
  The PNG was copied into `manuscript_source/figures/` to match that
  directory's flat-file convention (it is separate from the repo-root
  `figures/manuscript/` path).
- **Manuscript edits applied to `manuscript_source/main.tex`** (7 edits):
  Abstract; two new subsections ("End-to-End Online Replay Evaluation
  Across Available Capacities," "Workload-Specific Breakdown"); Discussion
  and Analysis; Limitations point 3; Summary of Findings (opening and
  closing paragraphs); a figure-path fix. Every edit keeps the claim honest:
  `evict_value_v1` does not outperform LRU, SIEVE, or FIFO-Reinsertion at
  any of cap32/64/128; the cap128 non-monotonic widening is explained via
  the single-step LRU-continuation compounding-shift hypothesis (not hidden
  or asserted as fact); SIEVE/FIFO-Reinsertion are described as strong,
  low-overhead baselines; cap256 is stated as not run, not implied as done
  anywhere. Full change-by-change rationale and exact numbers:
  `reports/kbs_manuscript_change_log_2026-06-21.md`. A discovery made while
  writing the Workload-Specific Breakdown: computing exact per-family gaps
  directly from the three chunk CSVs shows the cap128 anomaly is broader
  than the audit's original two-family (brightkite/citibike) framing —
  MetaCDN (+17.62%) and Twemcache (+14.06%) also show double-digit gaps at
  cap128, with MetaCDN U-shaped across capacities and Twemcache steadily
  increasing. This four-family characterization is what was written into
  the manuscript, not the audit's narrower framing.
- **Response-to-reviewers updated**: `reports/kbs_response_to_reviewers_skeleton.md`
  (authoritative tracker) — 11 edits covering the new banner, AE, R2-MC3,
  R3-Issue1/3/4/6/7, R3-Minor8/9, and the Recommended-Revisions list, plus a
  full recomputation of the bottom status-summary table (counts moved from
  `[PENDING CAP128/CAP256]`=6 to 0, `[IN PROGRESS]` from 8 to 13,
  `[PENDING BASELINE DECISION]` from 5 to 4, `[PENDING MANUSCRIPT REWRITE]`
  from 6 to 3; `[DONE]` remains 0). `submission_kbs_revision_docx/response_to_reviewers_skeleton.md`
  (externally-facing letter rendering) mirrored with the same substantive
  content per its own stated convention ("update the tracker first, then
  mirror relevant changes here"); its `.docx` render was **not**
  regenerated in this pass (pandoc not re-run) and is now stale relative to
  its own `.md` source — tracked as a remaining step.
- **New reports created**: `reports/kbs_manuscript_change_log_2026-06-21.md`
  (exhaustive change-by-change manuscript diff rationale) and
  `reports/kbs_revision_writing_progress_report.md` (overall progress
  summary, status-count recomputation, and remaining-work list).
- **Confirmed throughout this pass**: cap256 was not launched and is not
  claimed anywhere in any edited or new file; no heavy experiment was run
  (only an existing trend-builder script re-run on already-complete inputs,
  plus one new zero-compute plotting script); no raw CSV/MD chunk output
  was modified (chunk CSVs were only read, including via direct Python
  computation for exact per-family percentages); nothing was pushed,
  committed, merged, deleted, or renamed. Lightweight verification
  performed in the same pass: `scripts/paper/verify_kbs_policy_chunks.py`
  on cap32+cap64+cap128 together — PASSED, zero errors/warnings; `tectonic
  main.tex` clean rebuild — exit 0, `main.pdf` written (42 pages), zero
  undefined reference/citation warnings, all 5 new labels confirmed
  defined exactly once each.

## S. Final revision audit and cleanup pass (2026-06-21)

A reviewer-focused audit explicitly constrained to **no cap256, no new
heavy experiments, no modification of raw CSV/MD outputs, no push/commit/
merge/delete/rename**. Produced five new decision/analysis reports; made
**zero edits to `main.tex` or the response skeleton** — every decision
below is recorded but not yet applied to submission-facing text.

1. **Manuscript shortening audit** — `reports/kbs_manuscript_shortening_execution_plan.md`.
   Current word count 11,682 (+31% vs. the 8,919-word pre-cap128 baseline),
   against R3-Rec6's 30-40% reduction demand. 9 itemized cuts identified
   (duplicate equation; §1.1/1.2 overlap; 5x-repeated mixed-conclusion
   caveat; workflow/algorithm redundancy; Datasets/Baselines/Tables
   redundancy; Related Work SIEVE/FIFO redundancy; Discussion/Summary/
   Implications 3-way merge; fallback-caveat repetition; hedging
   tightening). Honest finding: full execution reaches only ~19-25%, short
   of the 30-40% target without cutting evidentiary content other
   reviewers asked to see added. Zero cuts applied.
2. **Fallback contribution decision** — `reports/kbs_fallback_final_decision_report.md`.
   Re-confirmed zero empirical evidence anywhere in the repo (guard not
   wired into the canonical pipeline; the one related ablation,
   `sentinel_budgeted_guard_v2`, favors the *unguarded* baseline) against
   19 mentions across 13 subsections, including a numbered Contributions
   slot (`main.tex` line 75), a full formal subsection with 4 equations,
   and an Algorithm box. **Decision: KEEP AS OPTIONAL IMPLEMENTATION
   DETAIL** — remove from the numbered Contributions list, shrink the
   Robust Decision Mechanism subsection, fix Table 6's guarded-layer
   clause, leave the unused schematic figure out. Resolves R2-MC3/
   R3-Issue6/R3-Rec5's `[PENDING BASELINE DECISION]` status to "decision
   made, manuscript edit pending." Not yet applied.
3. **HALP positioning audit** — `reports/kbs_halp_positioning_final_audit.md`.
   HALP appears in exactly 2 sentences (Related Work only, `main.tex` lines
   89/91); never in Introduction/Discussion/Limitations. Differentiation
   judged thin (omits pairwise-vs-pointwise, online-vs-offline, and the
   sharpest distinction, factual-vs-counterfactual supervision).
   Limitations discloses no scope-limit for the HALP comparison, unlike
   the analogous cap256/fallback disclosures elsewhere in the same
   section. Produced exact ready-to-paste text for a sharper Related Work
   paragraph (~95 words), a new Limitations sentence (~70 words), and the
   response skeleton's still-unfilled R3-Issue2 placeholder bracket (lines
   250-265) — replacement text had existed in
   `kbs_halp_fifo_source_verification.md` since 2026-06-19 but was never
   copied in. None of these three edits applied yet.
4. **Reviewer-response completeness audit** — `reports/kbs_final_reviewer_coverage_audit.md`.
   Strict 22-row table using a 4-level scale (NOT/PARTIALLY/SUBSTANTIALLY/
   FULLY ADDRESSED): 0 FULLY ADDRESSED, 11 SUBSTANTIALLY ADDRESSED, 7
   PARTIALLY ADDRESSED, 4 NOT ADDRESSED (R3-Issue6, R3-Minor8, R3-Rec5,
   R3-Rec6). Independently re-derives the response skeleton's own
   "zero `[DONE]`" finding. New finding from direct re-read of `main.tex`
   lines 640-642: the AI Declaration section is a generic 3-sentence
   tool-disclosure paragraph that does **not** contain the validation-
   methodology argument (schema verification script, cap128 anomaly
   audit) the drafted R3-Issue7 response relies on — confirming the
   skeleton's own self-flagged caveat was real, not a hedge.
5. **Submission-readiness assessment** — `reports/kbs_submission_readiness_assessment.md`.
   Reviewer concerns completed ~50%, manuscript readiness ~65%,
   response-letter readiness ~35%, submission-package readiness ~20%,
   overall readiness ~48%. Directly re-verified via `diff` that
   `submission_kbs_revision_docx/response_to_reviewers_skeleton.md` no
   longer matches the canonical `reports/kbs_response_to_reviewers_skeleton.md`,
   and its `.docx` (built 2026-06-19 23:39) predates the entire cap128
   update and all three decisions above. Top-10 remaining tasks ranked by
   importance, all writing/editing against already-completed analysis.

**The four items with a completed decision but zero manuscript-edit
follow-through** (the highest-leverage remaining work, per
`kbs_final_reviewer_coverage_audit.md`): fallback demotion, HALP
sharpening, shortening cuts, and the AI-declaration cross-check.

**Confirmed for this pass**: cap256 not launched; no new heavy experiment
run (only reading existing files and writing new analysis/report
markdown); no raw CSV/MD experiment output modified; nothing pushed,
committed, merged, deleted, or renamed; `main.tex` was read but not
edited.
