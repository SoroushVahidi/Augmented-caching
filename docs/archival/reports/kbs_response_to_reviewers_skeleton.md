# Response to reviewers — skeleton (rebuilt from real comments, 2026-06-19)

Manuscript: KNOSYS-D-26-07461. **This is a skeleton, not a final
point-by-point response.** The real, verbatim AE/Reviewer #2/Reviewer #3
text is now available (`reports/kbs_real_reviewer_comments.md`) and this
skeleton is organized directly against it — every comment below has a known
real source label. It still cannot be submitted as-is: several sections are
explicitly placeholders pending author-scope decisions (fallback, HALP, and
manuscript-packaging choices) and a final explicit statement that cap256 is
out of scope for the current three-capacity replay. **Do not claim final
four-capacity or canonical-sweep results anywhere in this file.**

**Updated 2026-06-20.** Both `cap32_with_sieve_fifo` and
`cap64_with_sieve_fifo` canonical chunks have now completed (SIEVE +
FIFO-Reinsertion included, 56/56 rows each, both `_EXIT=0`). cap32 shows
`evict_value_v1` losing to LRU (−4.84% mean misses), SIEVE (−2.92%), and
FIFO-Reinsertion (−4.77%) on aggregate; cap64 shows the same losses but
roughly halved (LRU −2.26%, SIEVE −0.13% — nearly closed, FIFO-Reinsertion
−2.19%), concentrated in 2/7 trace families (metacdn, metakv) — see
`reports/kbs_cap64_result_analysis.md`. Per the Option B decision
(`reports/kbs_after_cap64_decision_memo.md`), the `cap128_with_sieve_fifo`
chunk (same 8-policy set, capacity 128 only) was launched 2026-06-20 in
tmux on the local/cloud machine and is **running, not yet complete** — see
`reports/kbs_cap128_with_sieve_fifo_launch_report.md`. cap256 remains
**not launched**, pending review of the cap128 result. Do not state a
final multi-capacity result anywhere in this file until cap128 (and any
cap256 decision) is resolved.

**Updated 2026-06-21 (revision-writing pass).** `cap128_with_sieve_fifo`
has now completed (56/56 rows, `_EXIT=0`); see
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv/.md`.
Unlike the cap32→cap64 trend (gap roughly halving), cap128 **reverses and
widens sharply**: `evict_value_v1` loses to LRU by −11.76% (vs −4.84%/
−2.26% at cap32/cap64), to SIEVE by −9.13%, and to FIFO-Reinsertion by
−11.64%, concentrated most heavily in BrightKite (−50.9%) and Citi Bike
(−45.5%), with MetaCDN and Twemcache also double-digit. A dedicated
sanity/root-cause audit (`reports/kbs_cap128_anomaly_sanity_audit.md`)
ruled out file corruption, schema drift, capacity-specific code branching,
train/serve feature mismatch, and result-scale flakiness, concluding the
reversal is **likely real model/policy behavior**, plausibly a compounding
distribution-shift effect tied to the single-step LRU-continuation
supervision target (documented in `docs/evict_value_v1_method_spec.md`).
**Decision: cap256 is on hold, not launched, pending separate explicit
approval** — the three-capacity (32/64/128) trend is now reported and
discussed honestly in the manuscript rather than extended into a fourth,
more expensive capacity point before doing so. The full cap32/64/128
end-to-end comparison, a workload-specific breakdown, and a Discussion/
Limitations explanation of the cap128 finding have now actually been
written into `manuscript_source/main.tex` — see new subsections
"End-to-End Online Replay Evaluation Across Available Capacities" and
"Workload-Specific Breakdown" in Section~4, and the updated Discussion and
Limitations text (`reports/kbs_revision_writing_progress_report.md` has
the full change log). This resolves the compute-pending half of R2-MC3/
R3-Issue1/R3-Issue4/R3-Minor9/R3-Rec1/R3-Rec8 below — each is updated in
place. Cap256 remains the only explicitly out-of-scope gap for this
revision; do not state or imply a cap256 result anywhere in this file.

This file replaces the prior (2026-06-19, earlier the same day) 13-section
version organized around a 12-13 item AE-concern paraphrase, written before
the real comments were available. That version is superseded — kept in git
history for provenance, not deleted. A separate, differently-organized draft
of the same filename also exists, unreconciled, on the unmerged
`kbs-revision-parallel-cleanup` branch — not touched by this rebuild.

**Status tag vocabulary used below** (exactly these six, no others):
`[DONE]`, `[IN PROGRESS]`, `[PENDING CAP128/CAP256]`,
`[PENDING BASELINE DECISION]`, `[PENDING MANUSCRIPT REWRITE]`,
`[PENDING DOCX PACKAGE]`.

**Companion drafting pass (2026-06-19, while `cap32_with_sieve_fifo`
runs).** A results-independent drafting pass produced fuller paragraph-level
text for every section below, in
`reports/kbs_manuscript_rebuttal_drafts/response_paragraph_bank.md`. That
bank is **prepared but not final** — it carries the same `[INSERT FINAL
CANONICAL RESULT: ...]` placeholders wherever a section here is tagged
`[PENDING CAP128/CAP256]` or `[PENDING BASELINE DECISION]`, and it
does not claim `evict_value_v1` outperforms LRU/SIEVE/FIFO-Reinsertion
anywhere. Use it as the drafting source when this skeleton's placeholder
paragraphs are eventually expanded; it does not change any status tag
below by itself.

**Manuscript consistency audit cross-check (2026-06-19, while
`cap64_with_sieve_fifo` runs).** A zero-compute audit of the actual
`main.tex` text (`reports/kbs_manuscript_consistency_audit.md`) confirmed
every status tag below is still accurate — no relabeling needed. It also
surfaced manuscript-text-level detail that sharpens a few items: R3-Issue3/
Rec2's SIEVE/FIFO-Reinsertion gap is total (zero mentions anywhere in
`main.tex`'s Related Work or Table 2/6, not merely partial); R3-Issue1/
Rec1's end-to-end table does not exist in any form yet (not stale —
never written); and R3-Issue6/Rec5's fallback concern lines up with a
specific sentence in the contributions list that overstates the mechanism relative
to Limitations' hedging — recommended fix is the demote route already
drafted in `reports/kbs_fallback_revision_strategy.md`. None of this
changes the `[PENDING ...]` tags themselves; it only confirms they are
well-founded.

---

## Response to Associate Editor

**AE summary.** *Major concerns: no end-to-end miss-ratio eval, limited
differentiation from prior learned-caching methods, insufficient baselines,
unvalidated fallback, no overhead analysis; replay-horizon choice and
label-construction scalability need deeper justification.*

> We thank the Associate Editor and both reviewers for their detailed and
> constructive feedback. We have substantially revised the manuscript in
> response. First, we added an end-to-end online-replay evaluation across
> three cache capacities (32, 64, 128 slots) and seven trace families,
> reported in a new Results subsection together with a workload-specific
> breakdown; we report this evidence honestly, including the finding that
> our proposed policy does not outperform LRU, SIEVE, or FIFO-Reinsertion
> at any evaluated capacity, and that its gap widens sharply and
> non-monotonically at capacity 128 (a finding we audited for
> bugs/artifacts and discuss as likely real model behavior tied to our
> labeling methodology). Second, we strengthened our baseline pool by
> adding SIEVE and FIFO-Reinsertion, two widely used CLOCK-family
> structural baselines, with both implemented, tested, and included in
> every reported comparison. Third, we added an Overhead and Scalability
> subsection reporting measured offline label-construction cost and a
> code-verified asymptotic comparison of per-decision online cost. Fourth,
> we demoted our guarded fallback mechanism from a validated contribution
> to an explicitly labeled implementation safeguard/design extension, since
> we do not yet have a dedicated ablation isolating its end-to-end effect.
> Finally, we softened the abstract, contributions, and conclusions to
> match exactly what the now-expanded empirical evidence supports, rather
> than the offline-ablation-only evidence available at first submission. We
> address each reviewer's individual comments in detail below.

**Status: `[IN PROGRESS]`** — every concern referenced above is now backed
by actual manuscript text (not a placeholder); the only remaining gap is
cap256, which is explicitly out of scope for this revision (see banner
above), and the fallback validate-or-remove decision (R3-Issue6/Rec5),
which remains a deliberate scope choice rather than a compute gap.

---

## Response to Reviewer #2

### Major Comment 1 (R2-MC1) — replay horizon H justification

> We thank the reviewer for this comment. The replay horizon H was selected
> via an offline ablation across H∈{4,8,16} (Table 4/5; see
> `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`), where
> H=4 achieved the lowest validation regret. [Insert sensitivity-analysis
> discussion / theoretical intuition / explicit acknowledgment that H is an
> empirically tuned hyperparameter, plus a workload-conditioned breakdown if
> available — see R3-Minor9/Rec8.]

**Status: `[IN PROGRESS]`** — the underlying offline ablation data exists,
and the zero-compute draft text now exists in
`reports/kbs_horizon_h4_revision_strategy.md`. **Update 2026-06-19**: a
"Replay Horizon Selection" subsection has now actually been inserted into
`manuscript_source/main.tex` (while `cap64_with_sieve_fifo` runs), stating
the H=4 finding holds across 5 of 7 trace families (BrightKite,
CloudPhysics, MetaCDN, Twemcache, Wikimedia pageviews) and explicitly
disclosing that Citi Bike/MetaKV coverage and horizon×capacity interaction
remain open — see `reports/kbs_safe_manuscript_source_edits_report.md`.
Remaining blocker is only any later decision to add a per-capacity
breakdown; tag stays `[IN PROGRESS]` since this is placeholder/disclosure
text, not a finished sensitivity analysis.

### Major Comment 2 (R2-MC2) — computational overhead of label construction

> We thank the reviewer for raising this. Dataset construction for the full
> 7-trace × 4-capacity × 3-horizon sweep took approximately 10.3 hours of
> wall-clock time, producing 96GB across 662 shards
> (`reports/kbs_overhead_and_scalability_evidence.md`, Part 1). Training all
> 9 model configurations took approximately 7-8 minutes. We have added this
> discussion to the "Overhead and Scalability" subsection.
>
> We have also added a controlled wall-clock latency benchmark on a
> BrightKite prefix of 5,000 requests at capacities 32/64/128, measured on
> the author's local development machine under tmux rather than on
> Wulver/Slurm. The benchmark confirms the expected capacity-dependent cost
> of `evict_value_v1`: its mean per-decision cost increases from 75.0 ms at
> capacity 32 to 316.0 ms at capacity 128, while LRU, SIEVE,
> FIFO-Reinsertion, and REST remain several orders of magnitude faster. We
> present this as a bounded overhead probe, not a claim of representativeness
> across all traces.

**Status: `[IN PROGRESS]`** — the measured numbers and manuscript/rebuttal
draft text are now in `reports/kbs_overhead_manuscript_text_draft.md`.
**Update 2026-06-19**: an "Overhead and Scalability" subsection has now
actually been inserted into `manuscript_source/main.tex` (while
`cap64_with_sieve_fifo` runs), stating the dataset-build cost (~10.3h/96GB/
662 shards), training cost (~7-8 min), and the code-verified O(k)-per-miss
vs. O(1) complexity comparison, with an explicit TODO noting no controlled
timing benchmark exists yet — see
`reports/kbs_safe_manuscript_source_edits_report.md`.
**Update 2026-06-21**: the controlled timing benchmark has now been run
(BrightKite, 5,000-request prefix, capacities 32/64/128, local machine
under tmux, not Wulver/Slurm; see
`analysis/kbs_overhead_benchmark_local_tmux_20260621.csv/.md`) and the
"Overhead and Scalability" subsection in `main.tex` has been updated with
the measured numbers. Tag stays `[IN PROGRESS]` pending a final
cross-check pass rather than `[DONE]`.

### Major Comment 3 (R2-MC3) — offline-vs-online metric gap; fallback under-demonstrated

> We thank the reviewer for this important point. We have expanded our
> evaluation to include end-to-end online-replay results across three cache
> capacities (32, 64, 128) and seven trace families, reported in the new
> "End-to-End Online Replay Evaluation Across Available Capacities"
> subsection together with a companion workload-specific breakdown. We
> report this evidence directly: our proposed policy does not outperform
> LRU, SIEVE, or FIFO-Reinsertion at any of the three evaluated capacities,
> and the gap widens sharply at capacity 128, concentrated in four of seven
> trace families. REST is numerically tied with LRU on the three-capacity
> average, so `evict_value_v1` also does not outperform REST on average. We
> discuss the likely cause (a compounding distribution-shift effect tied to
> our single-step continuation-rule supervision target) in the revised
> Discussion and Limitations. Regarding the guarded fallback mechanism, we
> have demoted it throughout the manuscript from a validated contribution to
> an explicitly labeled implementation safeguard, since we do not yet have a
> dedicated ablation isolating its end-to-end effect; we leave validating or
> removing it to future work (see R3-Issue6/Rec5).

**Status: `[IN PROGRESS]`** for the end-to-end claim — cap32, cap64, and
cap128 (3 of the originally envisioned 4 capacity chunks) are now complete
and reported honestly in the manuscript as a mixed/negative result; cap256
remains explicitly out of scope for this revision (on hold pending separate
approval, see banner above), so this section must not imply a four-capacity
sweep is done. **`[PENDING BASELINE DECISION]`** for the fallback-validation
half (see R3-Issue6/Rec5 below — same underlying decision; demotion wording
is done, the validate-or-remove ablation itself is not).

---

## Response to Reviewer #3

### Summary (R3-Summary)

> We thank the reviewer for this thorough assessment. We agree that the
> original submission's empirical evidence was concentrated in offline
> target-quality metrics. In this revision we have added end-to-end
> miss-ratio results across three cache capacities (32, 64, 128) and all
> seven trace families, added SIEVE/FIFO-Reinsertion baseline comparisons
> at the same three capacities, and revised the manuscript framing so that
> `evict_value_v1` is presented as a decision-aligned supervision study
> rather than as a demonstrated online-superiority result. The new evidence
> is honestly negative for the present online policy instantiation, and the
> point-by-point responses below reflect that directly.

**Status: `[PENDING MANUSCRIPT REWRITE]`** — depends on R3-Issue1 and
R3-Rec1/Rec4/Rec6 below being resolved first; this is a synthesis paragraph,
not its own independent claim.

### Issue 1 (R3-Issue1) — no end-to-end cache miss ratio results

> We have added end-to-end online-replay results across 7 trace families ×
> 3 cache capacities (32, 64, 128) × 8 policies, reported in a new Results
> subsection with a companion workload-specific breakdown. The result is
> mixed-to-negative for our proposed method:
> `evict_value_v1` does not improve on LRU, SIEVE, or FIFO-Reinsertion at
> any evaluated capacity (gap vs LRU: +4.84% at cap32, +2.26% at cap64,
> +11.76% at cap128), with the capacity-128 gap concentrated in four of
> seven trace families. We report and discuss this honestly rather than
> qualify it away; see the revised Discussion and Limitations for our
> analysis of the likely cause. REST is numerically tied with LRU on the
> three-capacity average, so `evict_value_v1` also does not outperform REST
> on average. Capacity 256 has not yet been evaluated end-to-end and is
> explicitly out of scope for this revision.

**Status: `[IN PROGRESS]`** — cap32, cap64, and cap128 (`*_with_sieve_fifo`,
8 policies, 56/56 rows each) are all complete and the corresponding text is
now actually written into `manuscript_source/main.tex`
(Section~\ref{subsec:end_to_end_capacities},
\ref{subsec:workload_breakdown}). cap256 remains not launched — on hold
pending separate explicit approval (see banner above and
`reports/kbs_cap128_anomaly_sanity_audit.md` §7) — so this section
correctly describes a three-capacity, not four-capacity, evaluation. Do not
state a cap256 result here.

### Issue 2 (R3-Issue2) — insufficient differentiation from HALP

> We thank the reviewer for this comparison. Unlike HALP (Song et al., NSDI
> '23), which learns a pairwise preference signal from realized future
> re-access outcomes, our finite-horizon replay target explicitly quantifies
> downstream miss-count harm under counterfactual eviction at each decision
> point. A faithful empirical HALP reimplementation was judged out of scope
> for this revision cycle; the analytical differentiation above stands in
> its place.

**Status: `[PENDING BASELINE DECISION]`** — HALP source verification is now
done (`reports/kbs_halp_fifo_source_verification.md`), and the conclusion is
still conservative: HALP is already cited and differentiable in prose, but a
faithful empirical reimplementation before 2026-07-08 is not realistic.

### Issue 3 (R3-Issue3) — missing comparisons to SIEVE / FIFO-Reinsertion

> We have added SIEVE and FIFO-Reinsertion as additional baselines,
> reported in Table 2 (policy roster), the new three-capacity replay table,
> and the Related Work discussion. Both are now included throughout the
> same 32/64/128 end-to-end replay as the rest of the policy roster.

**Status: `[IN PROGRESS]`** for SIEVE/FIFO-Reinsertion — both policies are
implemented, tested, and now included in the completed 32/64/128 replay.
At capacity 32: SIEVE mean 35,314 misses (+1.86% vs LRU); FIFO-Reinsertion
mean 34,688 misses (+0.06% vs LRU); `evict_value_v1` mean 36,344 misses
(+4.84% vs LRU). At capacity 64: SIEVE mean 34,363 (+2.12% vs LRU);
FIFO-Reinsertion mean 33,672 (+0.07% vs LRU); `evict_value_v1` mean 34,409
(+2.26% vs LRU, and +0.13% vs SIEVE). At capacity 128: SIEVE mean 33,279
(+2.41% vs LRU); FIFO-Reinsertion mean 32,531 (+0.11% vs LRU);
`evict_value_v1` mean 36,317 (+11.76% vs LRU, +9.13% vs SIEVE, +11.64% vs
FIFO-Reinsertion). The three-capacity replay table and companion figure are
now written into `manuscript_source/main.tex`; cap256 remains out of scope
for this revision and is not claimed.

### Issue 4 (R3-Issue4) — self-admitted empirical weakness undermines the contribution

> We thank the reviewer for this direct feedback. With the cap32/64/128
> end-to-end results now in hand, response (b) is the honest one: our
> expanded results do **not** demonstrate robust end-to-end superiority, so
> we have revised the title, abstract, contributions, and conclusions to
> reflect exactly the scope of what is empirically demonstrated. The title
> and framing now center on a decision-aligned supervision target for
> eviction scoring, evaluated through (i) an offline ablation showing the
> target is learnable and benefits from nonlinear short-horizon scoring,
> and (ii) an honestly reported end-to-end evaluation showing the resulting
> policy does not yet outperform LRU, SIEVE, or FIFO-Reinsertion. We
> characterize the contribution as a decision-aligned supervision study,
> not as a demonstrated practically superior online policy, and we no
> longer claim or imply end-to-end competitiveness anywhere in the
> manuscript.

**Status: `[IN PROGRESS]`** — cap32, cap64, and cap128 all show
`evict_value_v1` losing to LRU/SIEVE/FIFO-Reinsertion, with the cap128 gap
widening sharply rather than continuing to close; response (b) has been
applied: the abstract, Summary of Findings, Discussion and Analysis, and
Limitations sections of `manuscript_source/main.tex` have all been revised
to state plainly that the policy is not yet a practically superior online
policy. Tag stays `[IN PROGRESS]` rather than `[DONE]` pending a final
full-manuscript re-read for any remaining overclaiming language missed in
this pass, and because cap256 remains on hold.

### Issue 5 (R3-Issue5) — computational cost not addressed

> `evict_value_v1` performs O(capacity) model-inference calls per cache
> miss (one feature vector + prediction per resident candidate; confirmed
> at `src/lafc/policies/evict_value_v1.py:179-200`), compared to O(1) for
> LRU, SIEVE, and FIFO-Reinsertion. We have now also run a controlled
> per-capacity wall-clock timing benchmark on a BrightKite prefix of 5,000
> requests (capacities 32/64/128, local machine under tmux, not
> Wulver/Slurm): mean per-decision cost for `evict_value_v1` rises from
> 75.0 ms to 316.0 ms across capacities 32 to 128, versus roughly 0.001 ms
> for LRU/FIFO-Reinsertion, 0.002-0.005 ms for SIEVE, and 0.04-0.18 ms for
> REST. This is a single-trace, single-run probe, not a claim of
> representativeness across all seven trace families.

**Status: `[IN PROGRESS]`** — the complexity-class claim is code-verified.
**Update 2026-06-21**: the controlled timing benchmark (separate from, and
much cheaper than, the cap64/128/256 sweep) has now been run; see
`analysis/kbs_overhead_benchmark_local_tmux_20260621.csv/.md`. Tag stays
`[IN PROGRESS]` pending a final cross-check pass rather than `[DONE]`.

### Issue 6 (R3-Issue6) — fallback mechanism unvalidated and oversold

> We have revised the manuscript to demote the guarded fallback mechanism
> from a named contribution to an optional implementation safeguard /
> future-work extension. No dedicated fallback-triggered-vs-disabled
> ablation has been added in this revision, so the mechanism is not
> presented as an empirically validated robustness contribution.

**Status: `[PENDING BASELINE DECISION]`** — no fallback-specific ablation
artifact exists anywhere in the repo today, but the conservative demotion
draft is now written in `reports/kbs_fallback_revision_strategy.md`.
**Update 2026-06-19**: the demotion wording has now actually been applied
to `manuscript_source/main.tex` (contributions list and method-section
mentions revised to call the mechanism an "implementation safeguard" /
"optional guard," not a validated contribution — see
`reports/kbs_safe_manuscript_source_edits_report.md`).

**Update 2026-06-21**: option (b) is confirmed as the chosen direction for
this revision cycle — the demotion is consistent throughout the abstract,
contributions, method, and limitations text after the latest revision-
writing pass, and no new fallback-ablation evidence has been generated.
Tag stays `[PENDING BASELINE DECISION]` because the underlying
validate-or-remove choice (option (a) vs. (b)) has not been exercised with
real ablation data; (b) is applied as the current scope decision, not as a
data-backed validation.

### Issue 7 (R3-Issue7) — single authorship and reliance on AI tools

> We thank the reviewer for raising this directly. As a single-author
> submission making use of AI-assisted coding and drafting tools, we have
> placed correspondingly greater emphasis on independent, reproducible
> verification rather than informal multi-author cross-checking. Every
> canonical evaluation artifact in this revision was independently
> re-verified at the raw-data level: a dedicated schema/structural
> verification pass (`scripts/paper/verify_kbs_policy_chunks.py`) checks
> column consistency, duplicate-key absence, and numeric sanity across all
> capacity chunks; a separate sanity/root-cause audit
> (`reports/kbs_cap128_anomaly_sanity_audit.md`) was performed specifically
> when an unexpected result pattern emerged at capacity 128, cross-checking
> raw rows, re-auditing the implementation for capacity-specific code paths,
> confirming small-scale determinism and reproducibility of the result
> independently of the full-scale run, and auditing the training-data
> distribution rather than accepting the surprising result at face value.
> We disclose this validation methodology explicitly in the manuscript and
> in the reproducibility materials accompanying this submission, and we
> view this depth of independent verification as a direct, practical
> mitigation for the single-author/AI-tool-use concern the reviewer raises.

**Status: `[IN PROGRESS]`** — the validation/reproducibility evidence
referenced above all exists and is real (not aspirational); what remains is
incorporating an explicit AI-tool-use and validation-methodology statement
into the manuscript's own text (the AI Declaration section already exists
in `manuscript_source/main.tex` but has not yet been cross-checked against
this specific reviewer concern in this revision pass).

### Minor Problem 8 (R3-Minor8) — verbosity and repetition

> We thank the reviewer for this observation. This revision necessarily
> grew the manuscript (a new end-to-end evaluation subsection, a
> workload-specific breakdown, and an expanded Discussion/Limitations
> treatment of the capacity-128 finding). We have since applied two narrow,
> no-compute repetition passes (2026-06-21) that removed duplicated
> restatements of already-established claims in the Workload-Specific
> Breakdown, Summary of Findings, and guard/fallback discussion, and
> compressed the overlap between Sections 1.1 ("Problem Setting and
> Motivation") and 1.2 ("Research Objective and Scope") that the reviewer
> specifically identified. This is not the full 30-40% reduction requested
> in R3-Rec6; it targets the most repetitive caveats and the named section
> overlap directly while preserving every substantive limitation and
> caveat.

**Status: `[IN PROGRESS]`** — the two repetition passes above are applied
directly in `manuscript_source/main.tex`, not merely planned. A cut/
reframing plan for a larger reduction still exists
(`reports/kbs_manuscript_shortening_and_reframing_plan.md`), but a full
30-40% cut has not been attempted, so this stays `[IN PROGRESS]` rather
than `[DONE]`.

### Minor Problem 9 (R3-Minor9) — missing workload-specific analysis

> We report per-workload-family breakdowns in the workload-specific
> breakdown table of the revised manuscript (Section~\ref{subsec:workload_breakdown}), covering all seven
> trace families at capacities 32, 64, and 128. The breakdown shows that
> the capacity-128 degradation discussed in the main results is broad-based
> across four of the seven families rather than confined to a single
> outlier trace, while two families (CloudPhysics, MetaKV) remain close to
> LRU at every capacity and one (Wikimedia pageviews) is degenerate at every
> capacity.

**Status: `[IN PROGRESS]`** — per-trace breakdowns now exist for all three
completed capacity chunks (cap32, cap64, cap128) and are written directly
into `manuscript_source/main.tex` as Table~\ref{tab:family-capacity-gap}.
Tag stays `[IN PROGRESS]` rather than `[DONE]` only because cap256 is out
of scope for this revision (on hold, see banner above).

### Recommended Revisions

1. **(R3-Rec1) End-to-end miss ratio results.** Status: `[IN PROGRESS]` — same as Issue 1; cap32, cap64, and cap128 all done and written into `main.tex` (2026-06-21); cap256 explicitly out of scope for this revision.
2. **(R3-Rec2) Direct comparisons to HALP and SIEVE.** Status: SIEVE/FIFO-Reinsertion `[IN PROGRESS]` (canonical cap32/64/128 numbers exist and are in the manuscript table — 2026-06-21, see R3-Issue3), HALP `[PENDING BASELINE DECISION]` — same as Issue 2/3.
3. **(R3-Rec3) Report computational overhead.** Status: `[IN PROGRESS]` — same as Issue 5; the "Overhead and Scalability" subsection is now actually in `main.tex` (2026-06-19, see R2-MC2), and now includes the controlled local/tmux wall-clock benchmark added 2026-06-21 (see Issue 5/R2-MC2 updates above).
4. **(R3-Rec4) Reduce hedging or reframe scope.** Status: `[IN PROGRESS]` — same as Issue 4; reframe direction (b) chosen and applied across abstract, Summary of Findings, Discussion, and Limitations (2026-06-21).
5. **(R3-Rec5) Validate the fallback mechanism or remove it.** Status: `[PENDING BASELINE DECISION]` — same as Issue 6; the demotion wording is now actually in `main.tex` (2026-06-19), but the underlying validate-or-remove decision is still open.
6. **(R3-Rec6) Shorten the manuscript by 30-40%.** Status: `[IN PROGRESS]` — two narrow, no-compute repetition passes applied directly to `main.tex` (2026-06-21, see Minor Problem 8 update); the cut/reframing plan in `reports/kbs_manuscript_shortening_and_reframing_plan.md` covers a larger reduction that has not been attempted, so the full 30-40% target remains open.
7. **(R3-Rec7) Investigate why H=4 works best.** Status: `[IN PROGRESS]` — same as R2-MC1; the "Replay Horizon Selection" subsection is now actually in `main.tex` (2026-06-19, see R2-MC1).
8. **(R3-Rec8) Provide workload-specific breakdowns.** Status: `[IN PROGRESS]` — same as Minor Problem 9; cap32/64/128 breakdown now in `main.tex` (2026-06-21), cap256 out of scope.

---

## Status summary (do not overstate)

| Status tag | Count | Sections |
|---|---|---|
| `[DONE]` | 0 | None — no section is fully finalized; even sections with ready evidence (R2-MC2, R3-Issue5/Rec3) still need a final cross-check, and several depend on the cap256 scope decision remaining open. |
| `[IN PROGRESS]` | 18 | AE, R2-MC1, R2-MC2, R2-MC3 (end-to-end half), R3-Issue1, R3-Issue3, R3-Issue4, R3-Issue5, R3-Issue7, R3-Minor8, R3-Minor9, R3-Rec1, R3-Rec2 (SIEVE/FIFO-Reinsertion half), R3-Rec3, R3-Rec4, R3-Rec6, R3-Rec7, R3-Rec8 |
| `[PENDING CAP128/CAP256]` | 0 | None remaining — cap128 completed 2026-06-21; cap256 is now tracked as an explicit scope decision (see banner above), not a pending-compute tag, and no section claims a cap256 result. |
| `[PENDING BASELINE DECISION]` | 4 | R2-MC3 (fallback half), R3-Issue2, R3-Issue6, R3-Rec2 (HALP half), R3-Rec5 |
| `[PENDING MANUSCRIPT REWRITE]` | 1 | R3-Summary (depends on R3-Rec6 being fully resolved, i.e. the full 30-40% target, not just the narrow passes applied so far) |
| `[PENDING DOCX PACKAGE]` | 0 | Not applicable to any reviewer-comment response section itself — relevant only to the submission package, not this letter (see `reports/kbs_docx_submission_package_report.md`). |

**Updated 2026-06-21.** cap128 has completed and the cap32/64/128
end-to-end evaluation, workload-specific breakdown, and the corresponding
Discussion/Limitations/abstract revisions are now actually written into
`manuscript_source/main.tex` — not merely planned. This moved AE, R2-MC3
(end-to-end half), R3-Issue1, R3-Issue3, R3-Issue4, R3-Minor9, R3-Rec1,
R3-Rec2 (SIEVE/FIFO-Reinsertion half), R3-Rec4, and R3-Rec8 out of
`[PENDING CAP128/CAP256]` and into `[IN PROGRESS]`. No section is `[DONE]`:
the reported end-to-end result is honestly mixed/negative for
`evict_value_v1` (it does not beat LRU/SIEVE/FIFO-Reinsertion at any
evaluated capacity), cap256 remains on hold pending separate explicit
approval (not part of this revision's claims), and the four
`[PENDING BASELINE DECISION]` items (HALP reimplementation scope, fallback
validate-or-remove) remain genuinely open author decisions rather than
compute-blocked items. R3-Issue7 was upgraded from
`[PENDING MANUSCRIPT REWRITE]` to `[IN PROGRESS]` because the
reproducibility/validation-depth evidence it cites is now real and
documented.

**Updated 2026-06-21 (second pass).** Two further narrow, no-compute edits
closed two remaining gaps identified in a final readiness audit. First,
`main.tex` now has a Limitations sentence stating that this is a
single-author revision whose validation cannot substitute for independent
multi-author replication, mitigated by repository-level audit trails,
schema/replay-artifact checks, and negative-result reporting — this is the
in-body text the R3-Issue7 response above refers to, closing the gap noted
in the previous update. Second, the Introduction's "Problem Setting and
Motivation" / "Research Objective and Scope" overlap that R3-Minor8
specifically named has been compressed (duplicated candidate-level framing
and the "predictive information is useful only if it improves the decision"
premise are now stated once, not twice), which moved R3-Minor8 and R3-Rec6
from `[PENDING MANUSCRIPT REWRITE]` to `[IN PROGRESS]`. Neither edit changed
any claim, number, or caveat; the full 30-40% reduction in R3-Rec6 remains
unattempted.
