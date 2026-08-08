# KBS revision — next actions that do NOT require rerunning heavy computation

Checklist of work that can proceed right now, on this machine, without launching any
multi-hour/multi-day job and without touching Wulver/Slurm. See
`reports/kbs_revision_evidence_ledger.md` for full provenance behind every item.

**Update 2026-06-19 (zero-compute consolidation pass)**: the items below that
were previously open writing/mining tasks are now done — see
`reports/kbs_cap32_policy_comparison_report.md`,
`reports/kbs_overhead_and_scalability_evidence.md`,
`reports/kbs_stale_artifact_refresh_plan.md`,
`reports/kbs_baseline_gap_action_plan.md`,
`reports/kbs_halp_fifo_source_verification.md`,
`reports/kbs_fallback_revision_strategy.md`,
`reports/kbs_horizon_h4_revision_strategy.md`,
`reports/kbs_overhead_manuscript_text_draft.md`,
`reports/kbs_manuscript_shortening_and_reframing_plan.md`,
`reports/kbs_docx_package_action_plan.md`,
`reports/kbs_complete_reviewer_comment_matrix.md` (now created on `main`,
previously only on the unmerged parallel branch),
`reports/kbs_response_to_reviewers_skeleton.md` (same),
`reports/kbs_docx_submission_package_report.md`. Checkboxes below updated
accordingly; new items added at the end of each section.

## Can do immediately (no compute, no waiting)

- [x] **Retrieve the full reviewer/AE comments** — **done 2026-06-19.** The user
      supplied the verbatim AE / Reviewer #2 (Major Comments 1-3) / Reviewer #3
      (Summary, Issues 1-7, Minor Problems 8-9, Recommended Revisions 1-8) text
      directly in this session; it is now recorded in
      `reports/kbs_real_reviewer_comments.md` and used as the authoritative source
      everywhere in this repo from now on. This was the single highest-leverage
      unblock for the response-to-reviewers letter. The "sample review" items in
      `reports/manuscript_artifacts/reviewer_concern_gap_map.md` are now confirmed
      superseded/hypothetical relative to the real comments (see the banner added to
      that file) — no remaining ambiguity on that point.
- [ ] **Review the diff of branch `kbs-revision-parallel-cleanup`** (commits `64e39bb`,
      `9a1642a`) and decide whether to merge it into `main`. It contains: two
      reproducibility bugfixes (`scripts/run_evict_value_v1_first_check.py` ml_gate
      skip-handling, `scripts/train_ml_gate_v1.py` named_steps fix), a CSV structural
      verifier (`scripts/paper/verify_kbs_heavy_r1_csv.py`), and substantial planning docs
      (`reports/kbs_baseline_positioning_plan.md`,
      `reports/kbs_complete_reviewer_comment_matrix.md`,
      `reports/kbs_manuscript_revision_change_checklist.md`,
      `reports/kbs_submission_package_checklist.md`,
      `reports/need_full_kbs_reviewer_comments.md`,
      `reports/kbs_response_to_reviewers_skeleton.md`). No compute involved — pure review
      + `git merge` decision.
- [ ] **Commit the freshly rebuilt heavy_r1 dataset/training outputs** currently sitting
      modified/untracked in the working tree (`analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`,
      `analysis/evict_value_wulver_v1_best_config_heavy_r1.json`,
      `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`,
      `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`) so the
      twemcache-undercoverage fix (§4 of the ledger) and the `random_forest` selection
      flip are captured in git history. Pure `git add`/`git commit`, zero compute.
- [x] **Disclose the validation-set family-coverage caveat** (citibike/metakv absent from
      the ~4,572-row sample used for model/horizon selection, §5 of the ledger) in
      whichever section of the manuscript or rebuttal discusses model selection. Pure
      writing. **Done 2026-06-19** — the new "Replay Horizon Selection" subsection in
      `manuscript_source/main.tex` (inserted in the safe-manuscript-source-edits pass,
      `reports/kbs_safe_manuscript_source_edits_report.md`) now states directly that the
      H-sensitivity finding covers only 5 of 7 trace families (BrightKite, CloudPhysics,
      MetaCDN, Twemcache, Wikimedia pageviews) and flags Citi Bike/MetaKV coverage and any
      horizon-by-capacity interaction as open, via a `% TODO(revision)` source comment.
- [ ] **Fix the four incomplete BibTeX entries** flagged in
      `reports/kbs_related_work_citation_verification.md` (`lykouris2018competitive`,
      `bansal2022weightedpaging`, `wei2020lacaching`, `chledowski2021robustlacaching` —
      missing volume/page metadata). Pure bibliography editing in
      `refs/related_work_table6.bib`.
- [ ] **Add a BibTeX entry for Im et al. 2022 (ICML), "Parsimonious Learning-Augmented
      Caching"**, if/when the `adaptive_query`/`parsimonious_caching` baseline is
      discussed in the manuscript. Currently cited only in prose in `docs/baselines.md`.
      Pure bibliography editing.
- [ ] **Confirm whether manuscript figure PNG exports meet the 300 dpi requirement**
      (current outputs are `.pdf` with `.png` siblings of unconfirmed DPI). Inspection of
      `scripts/paper/build_kbs_main_manuscript_artifacts.py`'s figure-saving calls, or the
      files in `figures/manuscript/`, no regeneration needed unless DPI is found to be wrong.
- [x] **Draft the concern-class-level sections of the response-to-reviewers letter** that
      don't depend on the canonical CSV — **done 2026-06-19**,
      `reports/kbs_response_to_reviewers_skeleton.md` on `main` (13 sections mapped to
      the AE concern list, with an explicit summary of which sections have real
      non-placeholder text ready today: §5 partial, §6, §12). Sections needing the
      canonical sweep (§1, §2, §4, §9) explicitly marked blocked, not drafted with
      fabricated numbers.
- [x] **Mine existing logs for overhead/label-construction-scalability evidence** —
      **done 2026-06-19**, `reports/kbs_overhead_and_scalability_evidence.md`. Measured:
      dataset build ~10.3h/96G/662 shards; training ~7-8min; code-verified O(capacity)
      vs O(1) per-miss complexity. Explicitly flagged what still needs a real timing
      benchmark (not run).
- [x] **Root-cause and plan the stale offline-ablation table refresh** — **done
      2026-06-19**, `reports/kbs_stale_artifact_refresh_plan.md`. Pinned to exact commit
      boundary (`53726ce`); fix is mechanical (`build_kbs_main_manuscript_artifacts.py` +
      manual paste) but **not executed** this pass per instruction — still a `[ ]` action
      item below if/when someone decides to run it.
- [ ] **Run `python scripts/paper/build_kbs_main_manuscript_artifacts.py`** to refresh
      `table4_main_ablation`, `table5_offline_selection`, `figure4_ablation`,
      `figure5_offline_top1_ablation` against the current (uncommitted) heavy_r1 retrain,
      then manually paste the refreshed numbers into `main.tex`'s `tab:evict-value-ablation`
      table. Plan exists (`kbs_stale_artifact_refresh_plan.md` §5-7); execution intentionally
      deferred, not yet done.
- [x] **Classify HALP/SIEVE/FIFO-Reinsertion/PARROT/Mockingjay/REST baseline gaps** —
      **done 2026-06-19**, `reports/kbs_baseline_gap_action_plan.md`. SIEVE recommended
      first to implement (genuinely absent, low engineering risk). Also surfaced a new
      manuscript-consistency gap: `rest_v1` (canonically evaluated) has no Table 2 row;
      a similarly-named row ("R-FTP+Marker") is actually a different, unwired policy.
- [x] **Implement SIEVE** as a new policy (`src/lafc/policies/sieve.py`) — **done
      2026-06-19**, verified against the official NSDI'24 paper and reference
      `libCacheSim` code first (`reports/kbs_sieve_source_verification.md`), then
      implemented faithfully and wired into `scripts/run_policy_comparison_wulver_v1.py`'s
      `POLICIES` dict and `src/lafc/runner/run_policy.py`'s `POLICY_REGISTRY` (CLI name
      `sieve`), with 8 new unit tests (`tests/test_sieve.py`, all passing; full suite
      291/291) and a tiny smoke test (SIEVE 496 vs. LRU 503 misses on 1 trace/1000
      requests/cap32). See `reports/kbs_sieve_implementation_report.md`. **Update
      2026-06-19**: the `zhang2024sieve` bib entry has now been added to
      `manuscript_source/refs.bib` and cited in `main.tex`'s Related Work and Table 2
      (`reports/kbs_safe_manuscript_source_edits_report.md`). **Still open**: run SIEVE
      through cap64/128/256 (and optionally rerun cap32) once those chunks are
      launched/complete — not done in this pass.
- [x] **Plan (but do not launch) the SIEVE-inclusive cap32 rerun** — **done
      2026-06-19**, `reports/kbs_cap32_rerun_with_sieve_plan.md`. Naming
      decision: Option B (write new `..._cap32_with_sieve.csv/.md` files,
      leave the existing no-SIEVE cap32 files untouched — tagged
      `CANONICAL CHUNK WITHOUT SIEVE / SUPERSEDED FOR FINAL TABLE IF
      SIEVE-INCLUSIVE CHUNKS ARE USED`). `blind_oracle` excluded from the new
      policy list (diagnostic-only, already outside `TABLE3_POLICIES`).
      Pre-run inventory clean, SIEVE unit tests re-verified passing (8/8),
      exact tmux launch command drafted. **Update later the same day**:
      explicit approval was given, but the launched `cap32_with_sieve` job
      was then stopped as obsolete because it omitted `fifo_reinsertion`;
      the corrected live job is now `cap32_with_sieve_fifo`, see
      `reports/kbs_cap32_with_sieve_aborted_superseded_report.md` and
      `reports/kbs_cap32_with_sieve_fifo_launch_report.md`.
- [x] **Clarify the exact "FIFO-Reinsertion" definition intended** — **done
      2026-06-19**, `reports/kbs_halp_fifo_source_verification.md` and
      `reports/kbs_fifo_reinsertion_baseline_audit.md`. Current conclusion:
      the repo's `fifo_reinsertion` is a scientifically defensible CLOCK /
      Second-Chance-family FIFO-Reinsertion baseline, already implemented and
      tested; the remaining decision is canonical inclusion/rerun scope, not
      definition.
- [x] **Fix the REST/Marker/R-FTP+Marker manuscript-consistency gap** in Table 2
      (`tab:main_policy_families` in `main.tex`) — either add a Table 2 row for "REST"
      (rest_v1's manuscript short-label) or otherwise reconcile the naming, per
      `kbs_baseline_gap_action_plan.md`'s recommendation §5. Pure writing, no compute.
      **Done 2026-06-19** — `reports/kbs_safe_manuscript_source_edits_report.md`. Table 2
      now lists exactly the canonical 8 policies (LRU, SIEVE, FIFO-Reinsertion, PredMk,
      BO/LRU, T&D, REST, EV), replacing the non-canonical Marker/R-FTP+Marker rows; REST
      (`rest_v1`) previously had zero mentions anywhere in `main.tex` despite being part
      of the canonical comparison set — now fixed. `tables/manuscript/table2_policy_roster.csv/.tex`
      regenerated to match via a generator-script fix, not a hand-edit.
- [ ] **Inspect the actual KNOSYS Editorial Manager submission requirements** for DOCX
      vs. LaTeX/PDF formats — `reports/kbs_docx_submission_package_report.md` (done
      2026-06-19) confirms zero DOCX files exist anywhere and the existing package is
      entirely LaTeX, but cannot resolve which formats are actually required without
      checking the external Editorial Manager instructions.
- [ ] **Draft a revision-specific cover letter, CRediT statement, and Highlights** —
      none exist in any format; none depend on the canonical CSV; see
      `kbs_docx_submission_package_report.md`'s "what can be created now" column.
- [x] **Draft the conservative fallback response strategy** — **done
      2026-06-19**, `reports/kbs_fallback_revision_strategy.md`.
- [x] **Draft the H=4 justification / reviewer-response text** — **done
      2026-06-19**, `reports/kbs_horizon_h4_revision_strategy.md`.
- [x] **Draft manuscript-ready overhead text** — **done 2026-06-19**,
      `reports/kbs_overhead_manuscript_text_draft.md`.
- [x] **Draft the manuscript shortening / reframing plan** — **done
      2026-06-19**, `reports/kbs_manuscript_shortening_and_reframing_plan.md`.
- [x] **Create the DOCX package action plan / placeholder folder** — **done
      2026-06-19**, `reports/kbs_docx_package_action_plan.md` and
      `submission_kbs_revision_docx/README_next_steps.md`.
- [x] **Draft a full results-independent manuscript/rebuttal package**
      while `cap32_with_sieve_fifo` runs — **done 2026-06-19**,
      `reports/kbs_manuscript_rebuttal_drafts/`: response paragraph bank
      (11 paragraphs), manuscript insertion draft (8 subsections),
      title/contribution reframing options (3 titles + bullets), DOCX
      package text bank (cover letter/highlights/CRediT/declaration-of-
      interest/AI-disclosure). All four are prepared-but-not-final and use
      explicit placeholders for anything depending on cap32/64/128/256;
      the running job was not touched. **Still open**: inserting this text
      into the actual manuscript/response-letter files, which should wait
      until the canonical sweep and baseline/fallback decisions land.
- [x] **Zero-compute manuscript consistency audit** while
      `cap64_with_sieve_fifo` runs — **done 2026-06-19**,
      `reports/kbs_manuscript_consistency_audit.md`. Read the actual
      submission zip's `main.tex`/`cover-letter.tex` (no loose copies exist
      anywhere else in the repo). Finding: the overclaiming phrases the
      audit was watching for ("robust superiority," "validated fallback,"
      "broad generality of H=4," etc.) are **not present** — the prose is
      already conservatively hedged. Real gaps are staleness/omission:
      SIEVE/FIFO-Reinsertion missing from Related Work + Table 2/6 despite
      being implemented and results-backed; three disagreeing policy
      rosters across `main.tex`/`table2_policy_roster.csv`/the actual
      8-policy CLI list; stale offline-ablation Table 4; no online
      end-to-end table exists yet anywhere (never written, not just
      stale); and a Contribution #3 (fallback) vs. Limitations claim/hedge
      asymmetry — recommended fix is the demote route already drafted in
      `kbs_fallback_revision_strategy.md`. **Update 2026-06-19**: the
      recommended fixes have now been applied directly to
      `manuscript_source/main.tex`/`refs.bib` (SIEVE/FIFO-Reinsertion
      citations, fallback demotion, policy-roster reconciliation including
      the `rest_v1`/REST omission fix, and two new placeholder Discussion
      subsections) — see `reports/kbs_safe_manuscript_source_edits_report.md`
      for the full list. **Still open**: inserting final cap64/128/256
      result-dependent numbers and conclusions, which remain untouched and
      pending the running job.
- [x] **Prepare chunk-verification + draft-merge/trend tooling while
      `cap128_with_sieve_fifo` runs** — **done 2026-06-20.** Created
      `scripts/paper/verify_kbs_policy_chunks.py` (schema/policy/capacity/
      trace/duplicate/numeric-sanity checks, rejects diagnostic-only
      `blind_oracle`) and ran it against the completed
      `cap32_with_sieve_fifo` + `cap64_with_sieve_fifo` chunks — **PASSED**.
      Created `scripts/paper/build_kbs_policy_trend_artifacts.py` and ran it
      on the same two chunks, producing explicitly-labeled **DRAFT /
      AVAILABLE CAPACITIES ONLY** outputs (`analysis/kbs_policy_trend_available_capacities.csv`,
      `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`,
      `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`) —
      distinct from, and not overwriting, any canonical manuscript table.
      Also drafted `reports/kbs_post_cap128_decision_template.md` (fill-in
      only) and `reports/kbs_cap256_launch_template.md` (exact command
      staged, marked **DO NOT RUN UNTIL USER APPROVES AFTER CAP128
      ANALYSIS**, not executed). cap128 confirmed untouched throughout;
      cap256 confirmed not launched. Full detail: §P of
      `reports/kbs_revision_completion_audit.md`.
- [ ] **Cross-check or extend the AI Declaration section** (`main.tex` lines
      640-642) against the validation-methodology argument the response
      letter makes for R3-Issue7 (the schema verification script
      `scripts/paper/verify_kbs_policy_chunks.py`, the cap128 anomaly
      sanity audit). **Added 2026-06-21 (final revision audit pass)**:
      directly re-read this section and confirmed it is currently a
      generic 3-sentence tool-disclosure paragraph (names ChatGPT,
      Perplexity, Cursor, Codex, Copilot, Gemini) with no link to that
      validation argument — the response skeleton's own self-flagged
      caveat for R3-Issue7 is confirmed real, not just a hedge. See
      `reports/kbs_final_reviewer_coverage_audit.md` row R3-Issue7. Purely
      mechanical once the fallback/HALP edits above are applied — no new
      evidence needed.

## Needs a human decision before proceeding (not blocked on compute, blocked on judgment)

- [x] **Decide whether to launch the cap32 chunk** — **done**, launched and completed
      (`CAP32_EXIT=0`, 5:13:02, see `reports/kbs_cap32_policy_comparison_report.md`).
      Result: `evict_value_v1` loses to LRU on 6/7 trace families (-4.84% mean misses).
- [x] **Decide whether to launch cap64 next**, given that cap32 already shows
      `evict_value_v1` underperforming LRU on 6/7 families — **done 2026-06-19,
      Option C approved**: launch cap64 only (same final 8-policy baseline set
      as the completed `cap32_with_sieve_fifo` chunk), decide on cap128/256
      only after reviewing cap64 results. See `kbs_cap32_policy_comparison_report.md`
      §10 and `reports/kbs_after_cap32_decision_memo.md` for the full reasoning.
- [x] **Audit FIFO-Reinsertion readiness before cap64** — **done 2026-06-19**,
      `reports/kbs_fifo_reinsertion_baseline_audit.md` plus
      `reports/kbs_final_baseline_set_decision_before_cap64.md`. Conclusion:
      include `fifo_reinsertion` in the final canonical policy set. **Update
      later the same day**: that decision has now been acted on — the
      obsolete `cap32_with_sieve` job was stopped and the corrected
      `cap32_with_sieve_fifo` canonical chunk is now running.
- [x] **Wait for the corrected `cap32_with_sieve_fifo` job to finish, then
      sanity-check its outputs before any cap64 decision.** — **done 2026-06-19.**
      Job finished `CAP32_WITH_SIEVE_FIFO_EXIT=0`, ~5h16m, 56/56 rows. Analysis:
      `reports/kbs_cap32_with_sieve_fifo_result_analysis.md`. Decision memo:
      `reports/kbs_after_cap32_decision_memo.md` (recommended **Option C: cap64
      only**).
- [x] **Human decision: approve cap64 launch (Option C) or pause/reframe (Option B).**
      — **done 2026-06-19, Option C approved and launched.** `cap64_with_sieve_fifo`
      launched in tmux session `kbs_full_policy_comparison_cap64_with_sieve_fifo`
      on the local/cloud machine (not Wulver/Slurm), same final 8-policy set as
      the completed `cap32_with_sieve_fifo` chunk. See
      `reports/kbs_cap64_with_sieve_fifo_launch_report.md`. cap128/cap256
      explicitly **not** launched — that remains a separate decision pending
      review of the cap64 result.
- [x] **Wait for `cap64_with_sieve_fifo` to finish, then sanity-check its
      outputs before any cap128 decision.** — **done 2026-06-20.** Job
      finished `CAP64_WITH_SIEVE_FIFO_EXIT=0`, ~10h04m, 56/56 rows. Analysis:
      `reports/kbs_cap64_result_analysis.md` — `evict_value_v1`'s gap vs
      LRU/SIEVE/FIFO-Reinsertion roughly halved (nearly closed vs SIEVE)
      relative to cap32, concentrated in 2/7 families (metacdn, metakv); no
      outright win vs LRU/FIFO-Reinsertion on any family. Decision memo:
      `reports/kbs_after_cap64_decision_memo.md` (recommended **Option B:
      cap128 only**).
- [x] **Human decision: approve cap128 launch (Option B), stop/reframe
      (Option A), or run cap128+cap256 together (Option C).** — **done
      2026-06-20, Option B approved and launched.** `cap128_with_sieve_fifo`
      launched in tmux session `kbs_full_policy_comparison_cap128_with_sieve_fifo`
      (PID `276106`) on the local/cloud machine (not Wulver/Slurm, no
      `sbatch`/`squeue`/`sacct`), same final 8-policy set as the completed
      cap32/cap64 chunks. See `reports/kbs_cap128_with_sieve_fifo_launch_report.md`.
      cap256 explicitly **not** launched — that remains a separate decision
      pending review of the cap128 result.
- [x] **Wait for `cap128_with_sieve_fifo` to finish, then sanity-check its
      outputs (via `scripts/paper/verify_kbs_policy_chunks.py` with cap32+
      cap64+cap128) before any cap256 decision.** — **done 2026-06-20/21.**
      Job finished `CAP128_WITH_SIEVE_FIFO_EXIT=0`, ~22h08m, 56/56 rows.
      Verification against cap32+cap64+cap128 together **PASSED** (zero
      errors/warnings). Result: `evict_value_v1`'s gap vs LRU, which had
      been narrowing (+4.84%→+2.26%), **reversed and widened sharply** to
      +11.76% at cap128, concentrated in brightkite (+50.9%) and citibike
      (+45.5%) — breaking the cap32→cap64 trend. This unexpected reversal
      triggered a dedicated sanity/root-cause audit (next item) before any
      cap256 decision, per instruction.
- [x] **Sanity/root-cause audit of the cap128 reversal before any cap256
      decision** — **done 2026-06-21.** `reports/kbs_cap128_anomaly_sanity_audit.md`.
      Seven checks (raw-row verification, cross-capacity schema/structural
      verification, small-scale [1,000-req] reproducibility probes in
      `/tmp/cap128_probe/` confirming determinism and that the same
      reversal direction reproduces at 1/50th scale, full code-path audit
      of `evict_value_v1.py`/feature/model/dataset modules finding no
      capacity-specific branching, and a training-distribution check
      confirming cap128 was in training with brightkite/citibike as the
      two least-represented — but consistently so across all
      capacities — families). **Conclusion: likely real model/policy
      behavior, not a bug or artifact**; precise causal mechanism
      (leading hypothesis: compounding error from the documented
      single-step LRU-continuation training-label methodology) remains
      unconfirmed. **Recommendation: do not launch cap256 yet**; report
      cap32→cap64→cap128 as a genuine non-monotonic capacity-sensitivity
      finding. A full-scale brightkite+citibike-only recheck (~6.3h
      estimated) was identified as the next-cheapest confirmation step but
      **not run** (exceeds this pass's ~1h budget) — staged for separate
      approval.
- [ ] **Human decision: approve cap256 launch, approve the staged ~6.3h
      brightkite+citibike-only recheck first, or hold at cap128 and write
      up the anomaly as a capacity-sensitivity finding.** Blocked on
      review of `reports/kbs_cap128_anomaly_sanity_audit.md`. The cap256
      command remains staged (not run) at `reports/kbs_cap256_launch_template.md`,
      explicitly marked **DO NOT RUN UNTIL USER APPROVES AFTER CAP128
      ANALYSIS** — the anomaly audit is now that analysis, and its
      recommendation is to hold, not to proceed automatically.
- [ ] **Decide whether to merge `kbs-revision-parallel-cleanup` into `main`** (see above) —
      flagged here separately because it's a judgment call (review diff for conflicts/
      quality), not just a mechanical step.
- [x] **Write the cap32/64/128 end-to-end evidence and cap128 anomaly audit
      into actual manuscript and response-to-reviewers text** (rather than
      leaving it as analysis-only reports) — **done 2026-06-21.** Applied 7
      edits to `manuscript_source/main.tex` (Abstract; two new subsections;
      Discussion; Limitations; Summary of Findings ×2; figure-path fix) and
      11 edits to `reports/kbs_response_to_reviewers_skeleton.md`, mirrored
      into `submission_kbs_revision_docx/response_to_reviewers_skeleton.md`.
      Full detail: `reports/kbs_manuscript_change_log_2026-06-21.md`,
      `reports/kbs_revision_writing_progress_report.md`. cap256 still not
      launched, still not claimed anywhere in the new text.
- [x] **Re-run `tectonic` on `manuscript_source/main.tex`** to confirm the
      now heavily-edited document still compiles — **done 2026-06-21.**
      Clean rebuild from scratch: exit 0, `main.pdf` written (42 pages,
      890,312 bytes), zero undefined reference/citation warnings in the
      final pass (only pre-existing cosmetic hbox warnings remain). All 5
      new labels confirmed defined exactly once each. Also re-ran
      `scripts/paper/verify_kbs_policy_chunks.py` on cap32+cap64+cap128
      together: PASSED.
- [ ] **Regenerate the `.docx` render** of
      `submission_kbs_revision_docx/response_to_reviewers_skeleton.md` via
      `pandoc` — the `.md` source was updated this pass (2026-06-21) but the
      `.docx` was not, so it is now stale relative to its own source.
      **Update 2026-06-21 (final revision audit pass)**: re-confirmed via
      direct `diff` that the `submission_kbs_revision_docx/` copy of this
      file no longer matches the canonical `reports/kbs_response_to_reviewers_skeleton.md`
      — both the `.md` source and the `.docx` are now stale, and the
      `.docx` predates the entire cap128 update. See
      `reports/kbs_submission_readiness_assessment.md` §4.
- [ ] **Resolve the fallback validate-or-remove decision** (R2-MC3 fallback
      half, R3-Issue6, R3-Rec5) — open author decision, not compute-blocked;
      no ablation has been run either way.
      **Update 2026-06-21 (final revision audit pass)**: decision made —
      **KEEP AS OPTIONAL IMPLEMENTATION DETAIL** (remove from the numbered
      Main Contributions list; shrink the Robust Decision Mechanism
      subsection; fix Table 6's guarded-layer clause; leave the unused
      schematic figure out). Full rationale and the exact concrete actions:
      `reports/kbs_fallback_final_decision_report.md` §5. **Manuscript edit
      not yet applied** — this item should now read as "apply the decision
      to `main.tex`," not "make the decision."
- [ ] **Resolve the HALP-reimplementation scope decision** (R3-Issue2,
      R3-Rec2 HALP half) — open author decision; the analytical
      differentiation already in the manuscript stands in its place unless
      this is revisited.
      **Update 2026-06-21 (final revision audit pass)**: decision made —
      do not attempt empirical HALP reimplementation (infeasible before
      2026-07-08); instead sharpen the existing analytical differentiation
      and disclose the scope limit in Limitations. Exact ready-to-paste
      text for the Related Work paragraph, the new Limitations sentence,
      and the response letter's still-unfilled R3-Issue2 placeholder
      bracket: `reports/kbs_halp_positioning_final_audit.md` §7.1-7.3.
      **None of the three applied yet.**
- [ ] **Apply the manuscript-shortening pass** (R3-Minor8, R3-Rec6,
      R3-Summary) — deferred; the manuscript grew in this pass (new
      end-to-end sections) rather than shrinking toward the reviewer's
      30-40% reduction target.
      **Update 2026-06-21 (final revision audit pass)**: a concrete
      9-cut execution plan now exists —
      `reports/kbs_manuscript_shortening_execution_plan.md`. Honest
      finding: full execution of all 9 cuts reaches an estimated ~19-25%
      reduction, short of the 30-40% target, because closing the rest
      would require removing evidentiary content other reviewers
      explicitly asked to see added (cap32/64/128 results, per-family
      breakdown, SIEVE/FIFO-Reinsertion detail). **Zero cuts applied.**
      This gap should be disclosed honestly in the response letter rather
      than overstated.

## Blocked — cannot proceed without either compute or external information

- [ ] Canonical full policy-comparison CSV (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`)
      — needs the multi-day sweep (ledger §8).
- [ ] Final Table 3 / Figs. 2–3 — need the canonical CSV above.
- [ ] Final point-by-point response to reviewers — the real reviewer comments are now
      in hand; the remaining hard blocker is the canonical CSV plus any final
      baseline/fallback scope decisions.
- [x] Sanity-audit write-up (`reports/kbs_validation_sanity_audit.md`) — **done**.
      Conclusion A: validation trustworthy, `evict_value_v1`'s worse-than-LRU result is
      real (confirmed by a second cap32/5k data point), `blind_oracle`'s bad number traced
      to a found-but-non-blocking config gap (not in `TABLE3_POLICIES`). Full sweep is
      unblocked; only the canonical CSV itself remains (see "Needs a human decision" above).
