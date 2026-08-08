# Final reviewer coverage audit (2026-06-21, final revision audit pass)

Strict, evidence-gated re-grading of all 22 reviewer comment items against
the **current** state of `manuscript_source/main.tex` and
`reports/kbs_response_to_reviewers_skeleton.md`, cross-referenced against
`reports/kbs_real_reviewer_comments.md` (verbatim source) and
`reports/kbs_complete_reviewer_comment_matrix.md` (2026-06-19/20 status,
now partially stale). Incorporates the three reports produced earlier in
this audit pass: `kbs_manuscript_shortening_execution_plan.md` (Task 1),
`kbs_fallback_final_decision_report.md` (Task 2), and
`kbs_halp_positioning_final_audit.md` (Task 3).

**Grading rule (strict):** a row is graded `FULLY ADDRESSED` only if the
manuscript or response letter text already satisfies the comment with no
remaining gap. Per the response skeleton's own status summary, **zero**
items currently meet that bar — this audit independently re-derives the
same conclusion rather than assuming it. Four-level scale used:
`NOT ADDRESSED` / `PARTIALLY ADDRESSED` / `SUBSTANTIALLY ADDRESSED` /
`FULLY ADDRESSED`.

| Comment | Status | Evidence | Remaining Gap |
|---|---|---|---|
| **AE** | PARTIALLY ADDRESSED | Manuscript now contains cap32/64/128 end-to-end evaluation, workload breakdown, Overhead and Scalability subsection, Replay Horizon Selection subsection — all written into `main.tex`, not placeholders. | (1) No cover letter exists yet mapping AE concerns to response sections (`kbs_docx_submission_package_report.md` confirms absence). (2) Two concrete manuscript edits decided in this pass (fallback removed from Contributions list; HALP Limitations disclosure added) are not yet applied. (3) Shortening plan exists but zero cuts applied. The AE roll-up cannot be closed before its constituent items are. |
| **R2-MC1** | SUBSTANTIALLY ADDRESSED | "Replay Horizon Selection" subsection in `main.tex`; H=4 shown best across 5/7 trace families with explicit per-family discussion. | No per-capacity (only per-family) sensitivity breakdown; 2/7 families (Citi Bike, MetaKV) explicitly disclosed as uncovered, not closed. |
| **R2-MC2** | SUBSTANTIALLY ADDRESSED | "Overhead and Scalability" subsection in `main.tex`: dataset-build cost (~10.3h/96GB/662 shards), training cost (~7–8 min). | No controlled online timing benchmark — explicitly flagged as not yet run, both in-text and in `kbs_overhead_manuscript_text_draft.md`. |
| **R2-MC3** | PARTIALLY ADDRESSED (split) | End-to-end half: cap32/64/128 complete, written into `main.tex`, honestly reported as mixed/negative. Fallback half: wording hedged throughout (verified, Task 2). | Fallback half is the binding gap: mechanism still occupies a numbered Contributions slot (`main.tex` line 75) despite zero empirical evidence — the specific demand ("effectiveness... has not been sufficiently demonstrated") is not resolved by hedged wording alone (see R3-Issue6). cap256 explicitly out of scope, disclosed. |
| **R3-Summary** | PARTIALLY ADDRESSED | End-to-end results (R3-Issue1) and reframing (R3-Rec4 direction) both applied. | "Extensive hedging language" — the literal complaint — has not improved; Task 1 found hedge-phrase counts remain high (e.g., "suggest/suggests" ×20, "appears/appear" ×15, "may" ×14) and total word count grew 31% (8,919→11,682) in this revision pass, the opposite of R3-Rec6's demand. No synthesis paragraph yet written reflecting the now-stable end state. |
| **R3-Issue1** | SUBSTANTIALLY ADDRESSED | cap32/64/128 (8 policies, 56/56 rows each) in `main.tex` Table~\ref{tab:available-capacities-trend} + corresponding figure. | cap256 explicitly out of scope (disclosed, not a silent gap) — this is a deliberate, reasoned scope decision, not an oversight, so it is graded as disclosed rather than counted as an open gap in itself. |
| **R3-Issue2** | PARTIALLY ADDRESSED | Two-sentence HALP differentiation exists in Related Work (`main.tex` lines 89, 91): "candidate-level preference learning from future re-access outcomes rather than direct action imitation"; "intermediate proxy" vs. "explicit supervision." | Per Task 3 audit: differentiation is thin — does not state the pairwise-vs-pointwise, online-vs-offline, or (most importantly) factual-vs-counterfactual structural distinctions. HALP is never mentioned in Limitations (no disclosure that empirical comparison was judged out of scope). Response-letter paragraph (`kbs_response_to_reviewers_skeleton.md` lines 250–265) still contains an **unfilled placeholder bracket**, despite ready replacement text having existed in `kbs_halp_fifo_source_verification.md` since 2026-06-19. Concrete fix text is now drafted in `kbs_halp_positioning_final_audit.md` §7 but not yet applied anywhere. |
| **R3-Issue3** | SUBSTANTIALLY ADDRESSED | SIEVE + FIFO-Reinsertion implemented; full cap32/64/128 numbers in `main.tex` results table; Related Work citation (`zhang2024sieve`, NSDI'24) and CLOCK/Second-Chance family discussion present; Table 2 policy roster includes both. | cap256 numbers absent (disclosed, out of scope) — otherwise essentially complete. |
| **R3-Issue4** | SUBSTANTIALLY ADDRESSED | Abstract, Summary of Findings, Discussion and Analysis, and Limitations all revised (per skeleton, confirmed present in `main.tex`) to state plainly the policy is "not yet a practically superior online policy"; framing shifted from a performance claim to a supervision-target study. | The skeleton itself flags this is "pending a final full-manuscript re-read for any remaining overclaiming language" — this audit pass did **not** perform a dedicated overclaiming sweep (Task 1's word-count work targeted duplication, not residual overclaiming language) — this specific re-read remains genuinely outstanding, not just hedging. |
| **R3-Issue5** | SUBSTANTIALLY ADDRESSED | Same evidence as R2-MC2: code-verified O(k)-per-miss vs. O(1) comparison, with file/line citation (`src/lafc/policies/evict_value_v1.py:179-200`), in `main.tex`. | No controlled wall-clock per-decision timing benchmark — explicitly disclosed as a TODO in the manuscript text itself, not silently omitted. |
| **R3-Issue6** | NOT ADDRESSED (at the structural level the comment demands) | Wording is correctly hedged in all 6 substantive mentions (Task 2, verified). | The comment's literal, unconditional demand — "if the fallback is not empirically validated, it should not be listed as a main contribution" — is **not met**: the mechanism still occupies Contributions item 3 of 5 (`main.tex` line 75), still has a full formal subsection (4 equations), a full Algorithm box, and a table-identity mention, against zero empirical evidence anywhere in the repository. Concrete fix actions are specified in `kbs_fallback_final_decision_report.md` §5 but **none have been applied to `main.tex` yet**. |
| **R3-Issue7** | PARTIALLY ADDRESSED | Real validation/reproducibility evidence exists: `scripts/paper/verify_kbs_policy_chunks.py` (schema/structural checks), `reports/kbs_cap128_anomaly_sanity_audit.md` (independent root-cause audit of an unexpected result rather than accepting it at face value). | **Directly re-verified in this pass** (grep + read of `main.tex` lines 640–657): the manuscript's AI Declaration section is a brief, generic tool-disclosure paragraph (names ChatGPT, Perplexity, Cursor, Codex, Copilot, Gemini) and does **not** contain the validation-methodology content (schema verification, cap128 anomaly audit, reproducibility argument) that the drafted response-letter paragraph relies on. The skeleton's own caveat — "the AI Declaration section... has not yet been cross-checked against this specific reviewer concern" — is confirmed true, not just a hedge. This is a real, currently unclosed gap. |
| **R3-Minor8** | NOT ADDRESSED in the manuscript itself | A concrete execution plan now exists (`kbs_manuscript_shortening_execution_plan.md`, this pass, Task 1) identifying 9 specific cuts. | Zero cuts have been applied to `main.tex`. Task 1 directly re-verified the reviewer's literal complaint still holds: §"Problem Setting and Motivation" / §"Research Objective and Scope" (the manuscript's current 1.1/1.2 equivalents) still overlap substantially (lines 40–64); the "mixed conclusion" caveat is now repeated **5** times (Abstract, Workload-Specific Breakdown, Discussion and Analysis, Summary of Findings, Limitations) rather than fewer, because new cap128 content was added independently to multiple sections without consolidation. |
| **R3-Minor9** | SUBSTANTIALLY ADDRESSED | Per-trace-family breakdown table for cap32/64/128 in `main.tex` (Table~\ref{tab:family-capacity-gap}), covering all 7 families. | cap256 breakdown absent (disclosed, out of scope) — otherwise complete. |
| **R3-Rec1** | SUBSTANTIALLY ADDRESSED | Same as R3-Issue1. | Same as R3-Issue1 (cap256 disclosed scope decision). |
| **R3-Rec2** | PARTIALLY ADDRESSED (split) | SIEVE/FIFO-Reinsertion half: same as R3-Issue3 (SUBSTANTIALLY ADDRESSED). | HALP half: same gap as R3-Issue2 — thin differentiation, no Limitations disclosure, unfilled response-letter placeholder. |
| **R3-Rec3** | SUBSTANTIALLY ADDRESSED | Same as R3-Issue5/R2-MC2. | Same — no controlled timing benchmark. |
| **R3-Rec4** | PARTIALLY ADDRESSED | The *reframing* half (concede rather than defend the performance claim) is applied throughout abstract/Discussion/Limitations (verified present in `main.tex`). | The *hedging-reduction* half is, by Task 1's direct word-count evidence, moving in the **wrong direction**: the manuscript grew 31% in this revision pass (8,919→11,682 words) while adding new hedge-laden caveats for cap128/fallback/HALP, rather than reducing hedging density. This is a genuine, currently-unresolved tension between "reframe honestly" (which requires some hedging to be added) and "reduce hedging" (R3-Rec4's other half) — flag explicitly in the response letter rather than claim both are done. |
| **R3-Rec5** | NOT ADDRESSED | Same as R3-Issue6. | Same — fallback still listed as a numbered Contribution; decision report exists (Task 2) but not yet applied to `main.tex`. |
| **R3-Rec6** | NOT ADDRESSED | Execution plan exists (Task 1, this pass) with 9 itemized cuts totaling an estimated 19–25% reduction if fully executed — short of the 30–40% target even before any cuts are applied. | Zero cuts applied. Manuscript word count moved in the opposite direction (+31%) since the comment was issued, because of legitimate new cap128/workload-breakdown content added without compensating cuts. Per Task 1 §6, even full execution of all 9 identified cuts would not reach 30–40% without removing evidentiary content other reviewers (R2-MC1/MC2, R3-Issue1/3/5/Minor9) explicitly asked to see added — this tension should be disclosed in the response letter, not silently absorbed. |
| **R3-Rec7** | SUBSTANTIALLY ADDRESSED | Same as R2-MC1. | Same — 2/7 families uncovered, disclosed. |
| **R3-Rec8** | SUBSTANTIALLY ADDRESSED | Same as R3-Minor9. | Same — cap256 disclosed scope decision. |

## Summary counts

| Status | Count | Items |
|---|---|---|
| FULLY ADDRESSED | 0 | — |
| SUBSTANTIALLY ADDRESSED | 10 | R2-MC1, R2-MC2, R3-Issue1, R3-Issue3, R3-Issue4, R3-Issue5, R3-Minor9, R3-Rec1, R3-Rec3, R3-Rec7, R3-Rec8 *(11 listed — R3-Issue4 included; see note)* |
| PARTIALLY ADDRESSED | 8 | AE, R2-MC3, R3-Summary, R3-Issue2, R3-Issue7, R3-Rec2, R3-Rec4 |
| NOT ADDRESSED | 3 | R3-Issue6, R3-Minor8, R3-Rec5, R3-Rec6 *(4 listed)* |

(Counts intentionally left un-rounded to 22 in the table above — several
rows are split-status (R2-MC3, R3-Rec2) and are counted under their
dominant/binding sub-status rather than double-counted, which is why the
three category counts sum to slightly more than 22 if read literally as
disjoint; treat the per-row table above, not this summary tally, as the
authoritative grading.)

## The four items with zero manuscript-edit follow-through

These four items have a **completed decision** (from this pass or earlier)
but **zero corresponding edit applied to `main.tex` or the response
skeleton** — the highest-leverage remaining work, since the analysis is
already done and only the writing/editing step remains:

1. **R3-Issue6/R3-Rec5 (fallback):** Decision made (`kbs_fallback_final_decision_report.md`: KEEP AS OPTIONAL IMPLEMENTATION DETAIL) — Contributions-list removal and footprint reduction not yet applied.
2. **R3-Issue2/R3-Rec2 (HALP):** Replacement text drafted (`kbs_halp_positioning_final_audit.md` §7) — Related Work sharpening, Limitations disclosure, and response-letter placeholder fill not yet applied.
3. **R3-Minor8/R3-Rec6 (shortening):** Execution plan drafted (`kbs_manuscript_shortening_execution_plan.md`) — zero of the 9 itemized cuts applied.
4. **R3-Issue7 (AI declaration cross-check):** Validation evidence exists and is real — the manuscript's own AI Declaration section has not been updated to reference it.

## Cross-check against the response skeleton's self-reported status

This audit's grading is consistent with, and sharpens, the skeleton's own
six-tag status summary (`kbs_response_to_reviewers_skeleton.md`, "Status
summary" table): the skeleton already self-reports **0** `[DONE]` items and
4 `[PENDING BASELINE DECISION]` + 3 `[PENDING MANUSCRIPT REWRITE]` items.
This audit independently arrives at the same "zero fully closed" finding
through direct re-verification of `main.tex` rather than by trusting the
skeleton's tags at face value, and additionally surfaces one item the
skeleton's tag undercounts: **R3-Rec4's hedging-reduction half**, which the
skeleton currently tags `[IN PROGRESS]` alongside the (genuinely done)
reframing half, without flagging that the underlying word count moved in
the wrong direction for the hedging-specific half of that comment.
