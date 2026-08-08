# Manuscript shortening and reframing plan (2026-06-19)

Directly answers **R3-Rec6** ("Shorten the manuscript by 30-40%") and
**R3-Rec4** ("Reduce hedging or reframe the paper's scope"), plus
R3-Summary's general complaint about "extensive hedging" and length.
Zero-heavy-compute pass: pure manuscript-source inspection (`find` +
`grep` against the extracted LaTeX source), no manuscript rewrite
performed — this is a precise edit plan plus one sample revised paragraph,
per instruction. Done in parallel with the running
`kbs_full_policy_comparison_cap32_with_sieve` tmux job (not touched).

## 1. Manuscript source located

```
find . -maxdepth 4 -type f \( -iname "main.tex" -o -iname "*.tex" -o -iname "*.docx" \) | sort
```

No `.tex`/`.docx` manuscript source exists directly under the repo tree at
depth ≤4 — only generated table/figure snippets under
`reports/manuscript_artifacts/` and `tables/manuscript/`. The actual
manuscript source is the previously-located extracted submission package
at `/tmp/kbs_manuscript_inspect/main.tex` (from
`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`,
repo root, `elsarticle` class, 559 lines). All analysis below is against
that file.

## 2. Current size and per-section breakdown

**Total: 8,919 words** (confirmed via `grep -v "^%" main.tex | wc -w`,
matches the figure already recorded in
`reports/kbs_revision_evidence_ledger.md`). R3-Rec6's 30-40% cut target
means a final length of **5,350-6,250 words**, i.e., removing
**2,670-3,570 words**.

| Section / subsection | Words | Note |
|---|---|---|
| Problem Setting and Motivation | 461 | Introduction |
| Research Objective and Scope | 378 | Introduction — overlaps heavily with above |
| Main Contributions | 248 | Introduction |
| Related Work | 572 | Introduction |
| System Model and Preliminaries | 522 | Method |
| Eviction-Value Prediction Framework | 685 | Method |
| Robust Decision Mechanism | 557 | Method — the fallback/guard description |
| Algorithmic Workflow and Implementation Details | 825 | Method — largest Method subsection |
| Experimental Setup | 499 | Experiments |
| Datasets, Baselines, and Evaluation Metrics | 913 | Experiments — **largest subsection overall** |
| Offline Ablation and Model Selection | 527 | Experiments |
| Discussion and Analysis | 614 | Experiments — overlaps heavily with Conclusions §below |
| Summary of Findings | 394 | Conclusions — overlaps heavily with Discussion and Analysis above |
| Implications of the Proposed Approach | 391 | Conclusions — overlaps with Summary of Findings |
| Limitations | 426 | Conclusions |
| Future Research Directions | 392 | Conclusions |

## 3. Repeated/duplicated content (the highest-leverage cuts)

Found via direct phrase search, not estimation — each item below is a
near-verbatim claim repeated 2-4 times across sections that are read
sequentially by the same reader, with no new information added on repeat:

1. **"Supervision-target design is a central issue in learning-augmented
   caching"** (or a close paraphrase) appears **four times**: Abstract
   (line 29), Main Contributions item 5 (line 79), Discussion and
   Analysis (line 464), and Summary of Findings (line 482, *the most
   verbose restatement*). **Recommendation: state this claim in full once
   (Abstract or Contributions, not both at full length), and reduce the
   other three instances to a one-clause back-reference** (e.g., "as
   discussed above" / "consistent with our central claim"). Estimated
   savings: ~150-200 words.
2. **"Plausible [and practically grounded/methodologically well-motivated]
   direction"** framing appears **three times** describing the same
   finding: Main Contributions item 5 (line 79), Discussion and Analysis
   (line 468), and Discussion and Analysis again two paragraphs later
   (line 495, "a plausible design pattern"). **Recommendation: keep one
   strong instance, cut the other two or fold into a single sentence.**
   Estimated savings: ~60-80 words.
3. **"Preserves the candidate-level structure of the method [while
   providing a conservative/practical control layer]"** appears **twice**,
   almost word-for-word: Main Contributions item 3 (line 75) and Robust
   Decision Mechanism (line 231). **Recommendation: state once.**
   Estimated savings: ~30 words (small on its own, but the surrounding
   sentences in both locations are also largely redundant — see item 5
   below for the larger structural fix this is part of).
4. **"Should be interpreted as... not as a theorem-backed/formally proved
   [guarantee]"** appears **twice** with the same rhetorical structure:
   Robust Decision Mechanism (line 229) and Limitations (line 507).
   **Recommendation**: this is exactly the kind of hedge R3-Rec4 is
   targeting — keep the Limitations instance (it belongs there), cut the
   Method-section instance entirely (the Method section should describe
   the mechanism, not pre-emptively disclaim it; disclaiming belongs in
   Limitations alone). Estimated savings: ~40 words, plus removes one
   instance of the "theorem/formal guarantee" hedge pattern R3-Summary
   flags generally.
5. **Structural duplication between "Discussion and Analysis"
   (Experiments section, 614 words) and "Summary of Findings" +
   "Implications of the Proposed Approach" (Conclusions section, 394+391 =
   785 words).** These three subsections, read back to back (they are
   only ~10 manuscript pages apart), restate the same three findings
   (short-horizon nonlinear scorers win; supervision-target design
   matters; the result is methodological evidence, not a superiority
   claim) three separate times with different wording each time. **This is
   the single largest cut opportunity in the manuscript.**
   **Recommendation: merge "Summary of Findings" and "Implications of the
   Proposed Approach" into one ~400-500 word "Summary and Implications"
   subsection, and trim "Discussion and Analysis" to focus only on
   material not already covered in the (now-merged) Conclusions section —
   e.g., keep the offline-ablation-specific interpretation in Discussion,
   move the higher-level "what this means for the field" framing to
   Conclusions only.** Estimated savings: ~500-700 words — by itself
   roughly 20% of the 2,670-word minimum cut target.

## 4. Hedging-phrase audit

Direct phrase counts against the manuscript text (case-insensitive):

| Phrase | Count |
|---|---|
| "may" | 14 |
| "suggest"/"suggests" | 13 |
| "appears"/"appear" | 7 |
| "could" | 4 |
| "plausible" | 3 |
| "should be interpreted" | 3 |
| "it remains" | 3 |

This is a real, measurable hedging density consistent with R3-Summary's
explicit complaint ("the extensive hedging..."). Not all 14 "may"s or 13
"suggest"s are problems — many are doing legitimate epistemic work
(e.g., "the optimal horizon may be workload-dependent" is an honest
open question, not overhedging). The actionable subset:

- **Stacked hedges** (two or more hedge words in one sentence,
  compounding rather than just being appropriately cautious) — e.g., line
  468's "this is **best interpreted as** evidence in favor of... rather
  than as a complete demonstration..." immediately followed by line 480's
  "this component **should be interpreted as** a practical robustness
  layer... rather than... a fully established empirical contribution" —
  two near-identical hedge constructions three sentences apart in
  different subsections. **Recommendation**: pick one place to make the
  "this is evidence for the framework, not a superiority claim" statement
  clearly and confidently, and trust the reader to carry it forward rather
  than re-hedging every subsequent claim with the same disclaimer.
- **Replace hedge-heavy sentences with direct, scoped claims** where the
  evidence actually supports a direct statement. E.g., rather than "the
  empirical results show that short-horizon nonlinear scorers **appear**
  to be the strongest configurations" (an actual finding, stated with an
  unnecessary hedge), state directly: "the empirical results show that
  short-horizon nonlinear scorers are the strongest configurations" (this
  is in fact closer to the manuscript's own existing Abstract wording,
  which is already appropriately direct — the hedging is denser in the
  body than in the Abstract, an inconsistency worth fixing either
  direction).
- Net effect of a hedging pass: smaller word-count impact than the
  structural merges in §3 (maybe 100-200 words), but directly answers the
  "reduce hedging" half of R3-Rec4 even where it doesn't save much length.

## 5. Section-by-section cut/merge plan with estimated savings

| Section | Action | Est. savings |
|---|---|---|
| Problem Setting and Motivation + Research Objective and Scope | Merge into one ~550-600 word Introduction subsection (currently 839 words combined, with each restating "why candidate-level eviction-value framing" in slightly different words) | ~250-300 |
| Related Work | Move some prose baseline-by-baseline description into a forward-reference to Table 6 (`tab:related_work`, already exists) rather than describing each twice (once in prose, once in the table) | ~100-150 |
| Robust Decision Mechanism | Trim in line with the fallback-demotion decision (`reports/kbs_fallback_revision_strategy.md` §5) — a shorter, more honestly-scoped description needs less defensive hedging prose around it | ~100-150 |
| Algorithmic Workflow and Implementation Details | Trim prose around Algorithm 2 that restates what the pseudocode already shows; keep the pseudocode itself | ~150-200 |
| Datasets, Baselines, and Evaluation Metrics | Largest subsection (913 words) — convert per-baseline prose descriptions to table cross-references (Table 2 already has a `tab:main_policy_families` description column) | ~200-250 |
| Discussion and Analysis / Summary of Findings / Implications of the Proposed Approach | Merge per §3 item 5 | ~500-700 |
| Duplicated phrases (§3 items 1-4) | Dedupe | ~280-350 |
| General hedging pass (§4) | Tighten | ~100-200 |
| **Total estimated** | | **~1,680-2,300** |

This covers roughly 19-26% of the manuscript's current length — **short
of the 30-40% target on its own.** To close the remaining gap
(~400-1,300 more words), the next-highest-leverage option is a deeper cut
to **Algorithmic Workflow and Implementation Details** (825 words, the
largest Method subsection) and **Experimental Setup** (499 words),
neither of which strictly requires new results — both describe the
existing pipeline, which doesn't change regardless of cap64+ outcomes.
Recommend tackling §3's structural merges and §4's hedging pass first
(highest savings-per-effort, zero dependency on final results), then
returning to a second pass on Method-section prose trimming once the
Conclusions-section merge's actual word count is known.

## 6. Sections whose final wording must wait for the canonical sweep

Do **not** rewrite these now even though they're candidates for
shortening — their content depends on results not yet available
(cap64/128/256, and the FIFO-Reinsertion/SIEVE baseline runs):

- **Datasets, Baselines, and Evaluation Metrics** and **Offline Ablation
  and Model Selection**: will need new rows/sentences for SIEVE and
  FIFO-Reinsertion once those are run through a canonical chunk (not yet
  done — see `reports/kbs_baseline_gap_action_plan.md`,
  `reports/kbs_fifo_reinsertion_implementation_report.md`).
- **Discussion and Analysis** / **Summary of Findings**: the
  "short-horizon nonlinear scorers are the strongest configuration" claim
  is offline-only; whatever the merged Conclusions subsection says about
  *online* competitiveness must wait for the full cap32-cap256 sweep
  (cap32 alone already shows `evict_value_v1` losing to LRU on 6/7
  families — `reports/kbs_cap32_policy_comparison_report.md`). Shortening
  the prose is safe to do now; changing the *substance* of what it claims
  is not, until cap64+ finish.
- **Limitations**: will need to incorporate the new disclosures drafted in
  `reports/kbs_fallback_revision_strategy.md` §5.4 and
  `reports/kbs_horizon_h4_revision_strategy.md` §6.3 and
  `reports/kbs_overhead_manuscript_text_draft.md` §3 — these are net
  *additions*, partially offsetting the shortening elsewhere in this
  section. Do the shortening pass on Limitations' existing prose (e.g.,
  dedupe item 3.4 above) before adding the new disclosures, not after, to
  avoid redoing the trim.

## 7. Title/contribution reframing if results stay negative

Cap32's canonical result (`evict_value_v1` loses to LRU on 6/7 trace
families, -4.84% mean misses, `reports/kbs_cap32_policy_comparison_report.md`)
is one of four planned capacity chunks; cap32_with_sieve is running, and
cap64/128/256 are not yet launched. **If the full sweep confirms this
negative pattern**, the manuscript's current framing is already mostly
well-positioned for that outcome and needs only modest adjustment, not a
wholesale rewrite — because the Abstract and Contributions already frame
the work as methodological ("a decision-aligned supervision target..."
"supervision-target design is a central issue") rather than as a
superiority claim ("our method beats X"). Two concrete reframing levers,
in increasing order of how much they concede:

1. **Minimal (recommended first choice)**: keep the title and abstract's
   existing methodological framing, but make the online-replay result
   explicit and unhedged rather than absent. Currently the manuscript
   reports almost no end-to-end miss-ratio numbers (R3-Summary's central
   complaint) — once cap64+ finish, state the actual online result
   plainly ("the candidate-level scorer underperforms LRU in online replay
   on N of M trace families at capacity K") rather than omitting it or
   burying it in hedges. This directly answers R3-Rec1 (end-to-end
   results) and R3-Rec4 (reframe) simultaneously, and is consistent with
   the existing Discussion's own already-honest framing ("the current
   results therefore support the proposed target and scoring formulation
   more strongly than they support broader claims about downstream
   competitiveness").
2. **Fuller reframing (if reviewers/AE want a sharper pivot)**: adjust the
   title from a method-name-forward framing to an explicitly
   diagnostic/negative-result framing, e.g. *"When does candidate-level
   eviction-value prediction help? A decision-aligned supervision target
   and its limits in online cache replacement"* — this signals up front
   that the paper's contribution is the supervision-target formulation and
   the empirical finding of *when/whether* it helps, not a claim that the
   resulting policy is competitive. This is a more defensive, lower-risk
   framing if the full sweep's result is uniformly negative, at the cost of
   sounding less like a "we built something that works" paper. **Not
   recommended as the first move** — only escalate to this if cap64+
   confirms the negative pattern holds broadly and the AE/reviewers'
   eventual second-round feedback still reads the paper as overclaiming
   under option 1's lighter fix.

## 8. Sample revised contribution paragraph

Demonstrates both the shortening (§3 items 1-3) and the fallback-demotion
edit (`kbs_fallback_revision_strategy.md` §5.2) in one place, replacing
the current 5-item, 248-word Main Contributions list:

**Current** (5 items, 248 words, includes the fallback mechanism as a
co-equal contribution and repeats "candidate-level structure" /
"supervision-target design" framing that's restated elsewhere):

> [existing 5-item list, see `main.tex` lines 68-80, already quoted in
> full in `reports/kbs_fallback_revision_strategy.md` §5.2]

**Revised** (4 items, ~165 words — cuts the fallback mechanism out of the
numbered list per the demotion decision, removes the repeated
"supervision-target design is central" framing since it's already stated
once in the Abstract, and tightens contribution #4/#5's overlapping
"reproducible framework" language into one item):

> The main contributions of this paper are as follows.
>
> 1. We formulate learning-augmented paging from a candidate-level
>    decision perspective, treating each full-cache miss as a structured
>    comparison among the items currently stored in the cache, rather than
>    as a question of global trust in advice alone.
> 2. We introduce an eviction-value prediction framework that uses
>    short-horizon downstream replay to construct a decision-aligned
>    supervision target for each eviction candidate, directly modeling the
>    local downstream harm of an eviction choice.
> 3. We provide a reproducible empirical framework that evaluates this
>    formulation against classical, predictive, and robust reference
>    policies under common replay semantics, with explicit separation
>    between offline scorer analysis and online replay evaluation, and we
>    additionally describe (but do not yet empirically validate) a
>    lightweight guard-style extension compatible with this formulation
>    (see Limitations).
>
> We provide evidence that candidate-level finite-horizon eviction scoring
> is a methodologically grounded direction for incorporating predictive
> context into eviction decisions, and that supervision-target design — not
> only predictive accuracy — is a central factor in learning-augmented
> caching performance.

This drops the original 5-item structure to 3 numbered items plus one
closing synthesis sentence, removes the fallback mechanism's standalone
contribution status (folded into item 3 with an explicit non-validation
flag), and states the "supervision-target design is central" claim once
instead of the original's implicit setup for four total restatements
across the paper.

## 9. What this report does NOT do

- Does not edit `main.tex` itself (no manuscript source lives in this
  repo's tracked tree — only the extracted copy at
  `/tmp/kbs_manuscript_inspect/main.tex`, which is scratch space, not a
  durable repo artifact). All edits above are drafts for a human (or a
  future pass with explicit authorization) to apply.
- Does not commit to a final word count — gives a concrete, itemized path
  to ~19-26% reduction with zero dependency on unfinished results, plus
  named next-highest-leverage targets to close the remaining gap to the
  30-40% target.
- Does not choose between §7's two reframing options — that is an
  editorial judgment call that should be made once cap64+ results are
  in, not preemptively here.
