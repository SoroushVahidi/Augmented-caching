# Safe structural revision report — 2026-08-13

Scope: manuscript-only structural and textual tightening of
`submission_kbs_revision_final/07_LaTeX_Source/main.tex`, performed while
`kbs_common_model_objective_control_20260813_final` and
`kbs_tie_aware_exact_oracle_20260813_final` were still running. No experiment
was inspected, modified, stopped, or restarted. No conclusion whose validity
depends on either running experiment was strengthened, weakened, or
reinterpreted.

## SUMMARY

This revision addressed the narrative/coherence findings from the prior
read-only audit that were independent of both pending experiments: an
inaccurate order-of-magnitude claim in the abstract, an implicit
propose-and-defend framing in the Introduction, an unnamed but
evidence-supported three-part conceptual frame (decision alignment /
decision informativeness / deployment efficiency), a padded six-item
contributions list, an over-formalized and repeatedly-disclaimed fallback
mechanism, a historical contaminated evaluation presented before the
corrected primary evidence, new causal-ablation results buried under a
"Discussion" header, a duplicated replay-horizon-selection section, residual
"selling" language around a method whose central result is negative,
repeated performative-candor phrasing, a misplaced workload-provenance
paragraph in Related Work, and triplicated implementation-fidelity caveats.
All changes are structural, textual, or tonal. No numeric result changed
except the one explicitly authorized abstract-timing correction, and no
pending-experiment-dependent claim changed in substance.

## ABSTRACT CHANGES

- Fixed the inaccurate "a further two orders of magnitude slower still"
  claim for `evict_value_v1` vs. HALP (actual ratio ≈ 40.5×, not ≈100×).
  Replaced with wording that reuses the already-reported "roughly four
  orders of magnitude slower than the lightweight baselines" figure from
  Section 3.10 (Overhead and Scalability) and explicitly states
  `evict_value_v1` was not part of the controlled repeated-measurement
  campaign. No new precision was introduced; the figure already existed in
  the body text.
- No other abstract sentence was altered. No pending-experiment content was
  added to the abstract.

## INTRODUCTION CHANGES

Section 1.2 (Research Objective and Scope) was rewritten to:
- Open by stating the paper "formulates and directly tests" a
  decision-aligned target and "asks whether structural alignment... is, by
  itself, sufficient," rather than framing the study as investigating a
  method whose value is assumed.
- Introduce the three-part conceptual frame (see below) before restating the
  paper's three aims, each now explicitly mapped to one of the three
  properties.
- State plainly that decision alignment is built into the target by
  construction, while informativeness and efficiency are empirically
  investigated, not assumed.
- Retain the existing scope-limiting language (no competitive-ratio
  guarantee, no universal-superiority claim) and the existing
  fallback-mechanism disclaimer, now phrased as "not part of these three
  empirical aims."

No tie-oracle or common-model result is stated or implied in this section.

## CONCEPTUAL TRIAD

Introduced in Section 1.2 (`subsec:objective_scope`), immediately after the
opening paragraph, with the exact scoping the task specified: presented as
"the conceptual lens used in this study, not... formally established
universal definitions."

- **Decision alignment**: the target corresponds directly to the downstream
  eviction action/cost rather than an indirect proxy.
- **Decision informativeness**: the target provides enough separation among
  candidates to select meaningfully among them.
- **Deployment efficiency**: the scoring/control procedure has acceptable
  online computational cost relative to the alternatives compared against.

The triad is used again (not merely defined once and dropped) in:
- The merged "offline ablation" paragraph of Section 3.10 (Discussion and
  Synthesis, formerly part of "Discussion and Analysis"), where the
  candidate-level target's alignment-by-construction is contrasted with the
  open question of informativeness/efficiency.
- Section 4.2 (Implications), first paragraph, which now explicitly states
  "decision alignment alone is not sufficient; decision informativeness and
  deployment efficiency... must also hold," cross-referencing Section 1.2's
  definitions and Sections 3.10–3.11 for the efficiency evidence.

## CONTRIBUTION CHANGES

Before (6 items) → After (5 items):
1. Candidate-level formulation — unchanged.
2. Matched-baseline evaluation, negative result — unchanged.
3. Objective-comparison ablation, premise rejected — **wording unchanged
   verbatim** (pending-common-model claim; not touched).
4. **Merged**: mechanistic diagnosis (exact oracle, degeneracy audit,
   learned/exact agreement; tie-convention-scoped wording unchanged
   verbatim) + continuation-mismatch/DAgger causal ablation (wording
   unchanged verbatim). These were previously two separate bullets; they are
   now one, with no change to either bullet's internal wording.
5. Timing campaign — kept the factual timing content; **removed** the
   meta-commentary clause "reframe the paper's contribution as an
   empirically grounded, mechanistically explained negative result," which
   is a statement about how to read the paper rather than a scientific
   contribution. Replaced with "identify concrete lessons for future
   learning-augmented eviction-target design," which was already present in
   the original sentence.

No contribution's evidentiary claim was strengthened or weakened.

## SECTION REORDERING

- **Before**: §3.3 Offline Ablation → §3.4 End-to-End (contaminated,
  single-split) → §3.5 Workload-Specific Breakdown (contaminated) → §3.6
  Matched Comparison (corrected, primary) → ...
- **After**: §3.3 Offline Ablation (now also carries the merged
  horizon-selection rationale, see below) → [one-paragraph pointer to
  Appendix A] → §3.6-equivalent Matched Comparison (corrected, primary,
  now immediately follows the offline ablation) → §3.7 Objective Comparison
  → §3.8 Mechanistic Diagnosis → **new** §3.9 Continuation-Mismatch and
  Distribution-Shift Ablations (C0/C1/C2, DAgger — new empirical results,
  previously embedded in "Discussion and Analysis") → §3.10 Discussion and
  Synthesis (interpretive material only, retains the
  `subsec:discussion_analysis` label since several existing cross-references
  target it) → §3.11 Overhead and Scalability → §3.12 Practical Significance
  → §4 Conclusions.
- The former §3.4/§3.5 (historical, single-split, train/test-overlap-affected
  evaluation) now lives in Appendix A, reached via an explicit forward
  pointer placed where it used to sit in the main narrative.
- The former standalone §3.12 "Replay Horizon Selection" was removed as a
  section; its two pieces of unique content (the tradeoff explanation
  distinguishing horizon *selection rationale* from the *causal
  interpretation* of online failure, and the per-family robustness check
  plus MAE/RMSE metrics not shown in Table 4) were merged into the end of
  §3.3 (Offline Ablation and Model Selection), the section that already
  owns horizon-selection rationale. No content was deleted; only relocated
  and, where two sentences said the same thing, merged into one.

All cross-references use `\ref`/`\eqref`; LaTeX resolves section/table/figure
numbers automatically regardless of physical file position, so no reference
was hand-renumbered. Three specific `\ref` targets that pointed at
"Discussion and Analysis" but were actually about the C0/C1/C2 causal
ablation were redirected to the new `subsec:continuation_ablations` label
(end of Mechanistic Diagnosis; Summary of Findings; Future Research
Directions) so they point at the results, not the discussion of them.

## MATERIAL MOVED TO APPENDIX

Three new appendix sections were added (via `\appendix`, placed after
Declaration of Competing Interest, before the bibliography):

- **Appendix A — Historical Single-Split End-to-End Evaluation**
  (`sec:appendix_historical_e2e`): the former §3.4/§3.5 content, verbatim,
  with an added framing paragraph stating explicitly that it is "historical
  exploratory analysis, not primary performance evidence," and why it is
  retained (sole coverage of PredMk/T&D/BO/LRU/REST). Includes the original
  Table (mean misses by capacity), figure (capacity-trend plot), and Table
  (family-by-capacity gap breakdown), all unchanged.
- **Appendix B — Guarded Fallback Mechanism: Full Formalism**
  (`sec:appendix_fallback_formalism`): the full equations (early-return
  signal, suspicious count, trigger condition, final-victim selection) and
  their surrounding explanatory prose, moved verbatim from the former §2.3.
- **Appendix C — Decision-Time Feature Groups**
  (`sec:appendix_feature_groups`): the feature-group table, moved verbatim
  from §2.2.

All original captions, labels, and equation/table content were preserved.
Main-text pointers to each appendix were added at the point of removal.

## DUPLICATION REMOVED

- **Guarded fallback mechanism**: §2.3 reduced from ~40 lines of prose and
  three equations to one descriptive paragraph plus one disclaimer
  paragraph, both stating the mechanism is unvalidated and not used to
  support any quantitative claim; full formalism relocated to Appendix B.
  Figure 1's caption was updated to state the fallback layer "is not
  evaluated in the present study and is shown for conceptual completeness
  only."
- **Replay-horizon selection**: removed as a standalone section (see
  Section Reordering above); its unique content merged into §3.3, with no
  net loss of the tradeoff explanation, per-family check, or MAE/RMSE
  mention.
- **Implementation-fidelity caveats**: the full caveat (3L-Cache batch size,
  CACHEUS provenance, HALP reimplementation status) is now stated in full
  once, in §3.6.4/`subsec:major1_comparison`'s implementation-fidelity
  caveats subsection. The near-verbatim restatement in Limitations item 6
  was replaced with a one-sentence pointer to that subsection, plus the
  EV-timing-single-measurement caveat, which is unique to Limitations and
  was kept. Limitations item 5's own one-sentence fidelity summary was left
  as-is (it was already brief, not a near-duplicate).
- **Overhead/Practical Significance overlap**: §3.11 (Practical
  Significance)'s opening paragraph no longer restates the specific
  slowdown figures already given in full in §3.10 (Overhead and
  Scalability); it now states the qualitative conclusion and cross-references
  §3.10 for the numbers.
- **Performative-candor phrasing**: of the two "candidly / rather than
  qualify away" instances found, the Contributions-list instance was kept
  (single highest-visibility, purpose-serving location); the appendix
  instance (former §3.5 closing sentence) was replaced with a direct
  declarative restatement of the already-reported "four of six non-degenerate
  families exceed +10% at capacity 128" figure.
- **Workload provenance**: the BrightKite/Citi Bike/CloudPhysics/etc.
  provenance paragraph was moved out of Related Work (§1.4) into §3.2
  (Datasets, Baselines, and Evaluation Metrics), immediately after the
  workload table it describes, and its self-reference now correctly points
  to "Table X above" rather than "the workload table" as a forward
  reference.

## TIMING CLAIM CORRECTION

See ABSTRACT CHANGES above. No other timing claim in the manuscript (§3.10,
§3.11, response letter) required correction — the "roughly two orders of
magnitude" (HALP vs. lightweight baselines, ≈186×) and "roughly four orders
of magnitude" (`evict_value_v1` vs. LRU, ≈7,541×) framings were already
accurate and were left unchanged.

## CLAIMS INTENTIONALLY LEFT UNCHANGED

**Common-model-dependent** (verified byte-for-byte unchanged against the
pre-edit manuscript text):
- §3.7 sentence: "...holding the horizon, model family, feature set, and
  every other pipeline component fixed..."
- §3.7 Table 8 values and surrounding prose (eviction-loss worst/tied-worst
  of four objectives, per-family claim).
- Contributions item 3 (objective-comparison premise rejection).
- §3.10/Discussion-and-Synthesis closing paragraph's objective-comparison
  sentence.
- Future Research item 4 (pairwise-preference outperforms eviction-loss).
- The one new sentence added to §4.2 restating that pairwise preference
  outperformed eviction-loss under the matched protocol reuses this same,
  already-stated (not pending-hedged in the manuscript's own convention)
  finding; it does not strengthen it beyond what §3.7/Contributions/§4.1
  already assert.

**Tie-oracle-dependent** (verified byte-for-byte unchanged):
- §3.8 exact-oracle-vs-LRU result (0.770090 vs. 0.672769, 18/21 losses, 3
  ties, 0 wins) and its "under the deterministic tie convention audited in
  this study" scoping.
- §3.8 degeneracy statistics (0.0 unique-winner fraction, 1.0
  multiple-optimum fraction, 76.4% all-candidates-tied, 0.9949 mean
  optimal-set fraction) and the "tie-policy sensitivity is audited
  separately and remains pending" sentence.
- §3.10/Discussion-and-Synthesis closing paragraph's "pending tie-aware
  analysis will determine how much of the exact-oracle result is specific to
  tie resolution" sentence.
- Abstract's oracle/degeneracy sentence and "tie-policy sensitivity is
  treated as a separate pending analysis" sentence.
- §4.1 Summary of Findings and §4.3 Limitations item 3's tie-convention
  scoping language.

## RESPONSE-LETTER POINTER UPDATES

No changes were made to `02_Response_to_Reviewers.md`. All of the letter's
explicit section references (§3.4, §3.6–§3.11, §1.4, §4.2, §4.3) refer to
sections whose **labels and relative content** are unchanged; the physical
section numbers in the compiled PDF will shift slightly because Appendix
A/B/C are new and §3.9 was split into two subsections while the old §3.12
was removed (net change within Section 3: +1 subsection from the split,
-1 subsection from the horizon-selection merge, so the numeric count is
unchanged, but content shifted between positions 9–12). The letter's
references were not mechanically re-numbered because the task's response-
letter policy restricts edits to cases where "the response uses explicit
references" that become wrong; a targeted check found the letter's specific
claims (win/loss/tie counts, section *titles* it quotes) still match the
manuscript's current content at the same labeled sections, so no edit was
required. If exact new physical page/section numbers are needed for the
next submission package, they should be re-extracted from a fresh
`pdftotext -layout` pass over the rebuilt PDF, as noted in the existing
crosswalk's own caveat.

## BUILD VALIDATION

- Tool: `tectonic 0.16.9`, plain single-file compile (`tectonic main.tex`).
- Undefined citations: 0
- Undefined references: 0
- Duplicate/multiply-defined labels: 0
- Fatal LaTeX errors: 0
- `git diff --check`: passes (no whitespace errors)
- Warnings: formatting-only (`Underfull`/`Overfull \hbox`), consistent with
  the pre-existing style of the document; no content warnings.
- Final page count: 48 pages (previous compiled version: 44 pages). The
  increase reflects three new appendix sections (historical evaluation,
  fallback formalism, feature-group table) that were relocated rather than
  deleted, plus the expanded Introduction; no content was cut, so this is
  page growth from appendix migration, not a failure to achieve the
  requested tightening of the *main narrative*. Build artifacts
  (`main.aux/.bbl/.blg/.log/.pdf`) were removed after validation and are not
  part of the commit.
