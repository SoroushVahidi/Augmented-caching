# Fallback contribution — final decision report (2026-06-21, final revision audit pass)

Directly answers **R3-Issue6** ("Fallback mechanism is unvalidated and
oversold as a contribution... If the fallback is not empirically validated,
it should not be listed as a main contribution") and **R3-Rec5** ("Validate
the fallback mechanism or remove it"), plus the AE's "unvalidated fallback
design" and **R2-MC3**'s "the effectiveness of the 'guarded fallback
mechanism' has not been sufficiently demonstrated." This audit re-checks the
**current** state of `manuscript_source/main.tex` (post the 2026-06-21
revision-writing pass) against the evidence already gathered in
`reports/kbs_fallback_revision_strategy.md` (2026-06-19, written against the
older pre-cap128 manuscript). The underlying empirical-evidence facts in
that report are unchanged — re-verified directly below — but the manuscript
text itself has changed since then, so the three questions below are
answered against the actual current text, not the prior report's quotes.

## 1. Is fallback still listed as a contribution?

**Yes — explicitly, in the numbered Main Contributions list.**

Current text, Contributions item 3 of 5 (`main.tex` line 75):

> "We additionally describe an optional guard, an implementation safeguard
> that can temporarily defer the learned candidate scorer to a conservative
> fallback policy when recent online outcomes suggest locally unsafe
> decisions. This preserves the candidate-level structure of the method
> while providing a lightweight control layer for imperfect predictive
> regimes; we present it as a design extension evaluated only as an
> ablation and as a candidate for future robustness validation, and we do
> not claim that it improves end-to-end miss ratio unless future experiments
> support it (see Limitations)."

The 2026-06-19 demotion pass (`kbs_fallback_revision_strategy.md` §5.2)
correctly softened the *wording* — it no longer claims the mechanism
improves miss ratio — but it did **not** remove the item from the *numbered
list*. The mechanism still occupies one of five numbered "Main
Contributions" slots, with parallel structural weight to Contributions 1, 2,
4, and 5, which **are** backed by real evidence (the candidate-level
framing, the eviction-value framework, the empirical comparison framework,
the supervision-target-design finding). R3-Issue6's wording is unconditional
("it should not be listed as a main contribution," not "it should be listed
with a caveat") — the current text does not yet satisfy this.

**Answer: Yes, still listed — only its wording, not its placement, was
softened.**

## 2. Is fallback still implied to improve miss ratio?

**No — this part is now correctly hedged throughout the manuscript.**

Checked every fallback/guard mention in the current text (full enumeration
in §4 below). Every substantive instance now carries an explicit
non-validation disclaimer:
- Contributions item 3 (line 75): "we do not claim that it improves
  end-to-end miss ratio unless future experiments support it"
- Robust Decision Mechanism intro (line 193): "we treat this guard as an
  implementation safeguard and a candidate design for future empirical
  validation rather than as a demonstrated component of the present
  results"
- Line 233 (end of Robust Decision Mechanism): "We note that this second
  layer's empirical effect on end-to-end outcomes has not yet been measured
  in the present study"
- Datasets/Baselines (line 368): "this guard is an implementation safeguard
  evaluated only as an ablation in the present study, and we do not claim
  that it improves end-to-end miss ratio unless future experiments support
  it"
- Limitations (line 613, and the surrounding paragraph): frames the guard's
  design choices as "heuristic parameters rather than theoretically derived
  constants," not as a proven improvement.

No sentence anywhere in the current manuscript claims or implies the guard
reduces misses. This question is **resolved** and does not need further
action.

**Answer: No — correctly hedged.**

## 3. Is fallback still over-emphasized relative to evidence?

**Yes — substantially, in textual/structural weight even though individual
sentences are now hedged.**

Re-verified directly against the repo (same conclusion as
`kbs_fallback_revision_strategy.md` §1–2, re-checked, unchanged):

- **Zero quantitative evidence exists anywhere in the repository.** The
  guard implementation (`EvictValueV1GuardedPolicy` /
  `GuardWrapperPolicy` in `src/lafc/policies/guard_wrapper.py`) is **not**
  present in `scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES`
  dict — confirmed by direct grep in this pass — which is the only registry
  that feeds the canonical cap32/64/128 pipeline. It has never been run on
  any of the seven real trace families this manuscript reports results for.
- The one dedicated guard-specific figure asset that exists,
  `figures/manuscript/figure6_guard_wrapper_evict_value_v1.{pdf,png}`
  (`reports/manuscript_artifacts/figure6_guard_wrapper_report.md`), is a
  **schematic only** ("no timing diagram, no empirical trigger-rate plot...
  canonical KBS `heavy_r1` quantitative bundle still does not include
  guarded A/B metrics... this figure supports explanation only") and,
  confirmed by grep in this pass, **is not even included anywhere in
  `main.tex`** — it was built but never inserted into the manuscript.
- The only empirical ablation of *any* guard-style mechanism that exists in
  this repo, `analysis/sentinel_budgeted_guard_v2/v2_ablation_report.md`,
  tests a different (related, not identical) design on 3 synthetic traces
  at capacities 2–3, and concludes directly: *"Is any single v2 component
  actually useful? No: none of the single-component variants beat v1 on
  mean misses"* and recommends the **unguarded** `v1_baseline` as the "main
  empirical candidate after this ablation." This is not supporting evidence
  for guards — if anything, it is a cautionary data point from an adjacent
  design in the same family, pointing the opposite direction.

Against this zero/negative evidence base, the manuscript still gives the
mechanism:
- A full numbered slot in Main Contributions (line 75).
- A complete, self-contained ~592-word subsection ("Robust Decision
  Mechanism," lines 190–234) with a full formal apparatus: an early-return
  indicator $E_t$, a sliding suspicious-count $S_t$, a trigger condition,
  and a final-victim selector — four numbered equations (`eq:early_return_
  signal`, `eq:suspicious_count`, `eq:guard_trigger`, `eq:final_victim`).
- A full Algorithm box (Algorithm 1, `alg:evict_value_guarded`, 36 lines of
  pseudocode) built entirely around the guarded decision loop, even though
  the guard half of that loop has never been executed end-to-end.
- A dedicated pipeline stage description ("Optional fallback control,"
  stage 5 of 5 in the Algorithmic Workflow subsection).
- A table-roster entry (Table 6, row "EV"): "Candidate-level finite-horizon
  eviction-value predictor with **optional guarded fallback layer**" —
  presented as part of the policy's identity in the baseline-comparison
  table, even though the guarded layer was never run in that comparison.
- Six separate hedge/disclaimer sentences restating its non-validated
  status (Contributions, twice in Robust Decision Mechanism, Datasets/
  Baselines, Limitations, Future Research) — itself a symptom of
  over-emphasis: a mechanism with no evidence should need one clear
  disclosure, not six.

This combination — a fully-derived mathematical formalism, a dedicated
algorithm box, a contributions-list slot, and a table-identity mention, for
a mechanism with literally zero quantitative evidence anywhere in the
project — is precisely what R3-Issue6 means by "oversold as a contribution."
The hedging fixes the *claims*; it has not yet fixed the *proportion of the
paper* devoted to an unvalidated mechanism, which is a separate and
still-open problem.

**Answer: Yes — over-emphasized in structural weight, even though
individual claims are now correctly hedged.**

## 4. Full enumeration of fallback/guard mentions (current `main.tex`)

| Location | Line(s) | Nature |
|---|---|---|
| Abstract | 29 | Hedged mention ("describe, but do not yet empirically validate") |
| Main Contributions, item 3 | 75 | **Numbered contribution slot** |
| Related Work | 87 | Passing mention ("robustness layers... combined with conservative fallback behavior") |
| Eviction-Value Prediction Framework | 134, 167 | Sets up the guard as a future extension point |
| Robust Decision Mechanism (full subsection) | 190–234 | Full formal mechanism, 4 equations, ~592 words |
| Algorithmic Workflow | 238, 255–256, 259, 288 | Pipeline-stage description |
| Algorithm 1 (full box) | 290–327 | Guarded decision loop pseudocode |
| Method-overview figure caption | 336 | "An optional lightweight fallback layer can temporarily delegate control..." |
| Experimental Setup | 353 | Passing mention of "fallback activity in guarded variants" |
| Datasets, Baselines, and Evaluation Metrics | 368 | Hedged restatement |
| Table 6 (`tab:main_policy_families`), row "EV" | 406 | Table-identity mention |
| Discussion and Analysis | 543 | "guarded variants whose end-to-end effect remains unmeasured" |
| Summary of Findings | 582 | "the overall framework includes lightweight fallback variants..." |
| Limitations | 609, 613 | Hedge disclosures |
| Future Research Directions | 624 | "the robustness layer can be made more adaptive" |

19 distinct locations across 13 of the manuscript's ~20 subsections.

## 5. Recommendation

> **KEEP AS OPTIONAL IMPLEMENTATION DETAIL**

### Why not "REMOVE FROM CONTRIBUTIONS" alone, and why not "KEEP AS VALIDATED CONTRIBUTION"

- **Not "KEEP AS VALIDATED CONTRIBUTION":** ruled out outright — there is no
  validating evidence anywhere in the repository (§3), and generating such
  evidence would require new heavy compute (wiring `EvictValueV1
  GuardedPolicy` into the canonical pipeline and running it across all
  seven trace families and three-plus capacities), which this pass is
  explicitly barred from launching. Selecting this option would assert
  something the evidence directly contradicts.
- **"REMOVE FROM CONTRIBUTIONS" is necessary but, taken literally as
  "remove the line from the numbered list only," is not sufficient** given
  the over-emphasis finding in §3 — the numbered-list line is only one of
  19 locations. A decision that fixes only Contributions item 3 while
  leaving a full algorithm box, four formal equations, and a table-identity
  mention untouched would not fully satisfy R3-Issue6's underlying concern
  (oversold relative to evidence), even though it would satisfy the literal
  sentence reviewers quoted.
- **"KEEP AS OPTIONAL IMPLEMENTATION DETAIL"** is the label that correctly
  describes the combined action needed: (a) remove the mechanism from the
  numbered Main Contributions list entirely (fold its one-sentence mention,
  already drafted in `kbs_fallback_revision_strategy.md` §5.2, into the end
  of the Contributions paragraph or into Limitations/Future Work only), and
  (b) reduce its footprint elsewhere in the manuscript so that the amount
  of text devoted to it is proportionate to having zero validating evidence
  — i.e., treat it the way the manuscript already treats other
  not-yet-validated extensions, with one clear description and one clear
  disclosure, not a fully-derived formal apparatus presented with the same
  weight as the validated parts of the method.

### Concrete actions to implement this decision (manuscript-editing follow-up, not performed in this pass)

1. Remove fallback from the numbered Main Contributions list (line
   70–80); fold a single sentence into Contributions' closing paragraph
   (line 79's existing closing sentence is one natural place) or move it to
   Limitations/Future Work exclusively.
2. Retain the Robust Decision Mechanism subsection and Algorithm 1 (the
   design is real, reusable, implemented, and not unreasonable to describe
   — removing the code/idea entirely would be disproportionate, per
   `kbs_fallback_revision_strategy.md` §4's "demote, do not remove"
   reasoning, which this audit reaffirms), but shorten it — this overlaps
   directly with Cut 8 in `reports/kbs_manuscript_shortening_execution_plan.md`,
   which estimates ~250–400 words recoverable once this decision is
   applied.
3. Remove or rephrase Table 6's "with optional guarded fallback layer"
   table-identity clause for the "EV" row, since that table reports actual
   comparison results and the guarded layer was not part of any reported
   comparison.
4. Decide whether to insert the schematic figure
   (`figures/manuscript/figure6_guard_wrapper_evict_value_v1`) into the
   manuscript at all — it currently exists but is unused; given the
   over-emphasis finding, **recommend leaving it out** rather than adding a
   fourth visual asset for an unvalidated mechanism.
5. Consolidate the six scattered hedge sentences (§3) into one clear
   disclosure in Limitations (already partially done at line 613) and trim
   the repeated ones elsewhere — this is the same action as Cut 8.

### What this decision does NOT do

- Does not delete the guard implementation, documentation, or design from
  the codebase — only from the manuscript's contribution framing.
- Does not run any new ablation or generate any new fallback evidence —
  consistent with this turn's "no new heavy experiments" constraint.
- Does not foreclose a future revision cycle re-promoting the mechanism to
  a validated contribution if a dedicated ablation is eventually run and
  shows a real effect.

## 6. Cross-references

- Evidence base: `reports/kbs_fallback_revision_strategy.md` (2026-06-19,
  re-verified unchanged in this pass).
- Wording-level demotion already applied: `reports/kbs_manuscript_change_log_2026-06-21.md`,
  `reports/kbs_safe_manuscript_source_edits_report.md`.
- Word-count interaction: `reports/kbs_manuscript_shortening_execution_plan.md`
  §2, Cut 8.
- Reviewer-response status: `reports/kbs_response_to_reviewers_skeleton.md`,
  R3-Issue6/R3-Rec5, currently tagged `[PENDING BASELINE DECISION]`. **This
  report resolves that pending decision** — the tag should move to
  `[PENDING MANUSCRIPT REWRITE]` (decision made; manuscript edit not yet
  applied), tracked in Task 6 below.
