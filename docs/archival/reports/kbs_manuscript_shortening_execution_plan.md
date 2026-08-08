# Manuscript shortening execution plan (2026-06-21, final revision audit pass)

Directly answers **R3-Minor8** ("Excessive verbosity and repetition... Sections
1.1 and 1.2 overlap substantially. Sections 3.4 and 4.1–4.2 repeat
methodological caution many times") and **R3-Rec6** ("Shorten the manuscript
by 30–40%"). This supersedes the analysis in
`reports/kbs_manuscript_shortening_and_reframing_plan.md` (2026-06-19), which
was written against a now-stale 559-line, ~8,919-word extraction
(`/tmp/kbs_manuscript_inspect/main.tex`) **before** the cap128 end-to-end
evidence, the workload-specific breakdown, and the capacity-128 causal-
hypothesis discussion were added to the manuscript. That prior report's
duplication findings (items 1–4 of its §3) are re-verified below against the
current `manuscript_source/main.tex` (661 lines) and are still present
essentially unchanged; its §3 item 5 (Discussion/Conclusions overlap) has
gotten **worse**, not better, because the revision-writing pass added new
capacity-128 material to all three locations independently rather than to
one. This audit is read-only against `manuscript_source/main.tex` — no edits
were applied in this pass, per the "manuscript editing... allowed" but
audit-first instruction for this turn.

## 0. Headline numbers

- Current manuscript length: **11,682 words** (`grep -v "^%" main.tex | wc -w`
  — the exact method used to produce the prior report's 8,919-word figure,
  so the two numbers are directly comparable).
- Growth since the last shortening audit: **+2,763 words (+31%)** — driven
  entirely by the (necessary, reviewer-mandated) cap32/64/128 end-to-end
  evaluation, workload-specific breakdown, and capacity-128 anomaly
  discussion added in the 2026-06-21 revision-writing pass.
- R3-Rec6 target (30–40% of the **current** length): cut **3,505–4,673
  words**, landing at a final length of **7,009–8,177 words**.
- This plan's itemized, low/medium-risk cuts below total **≈2,250–2,900
  words (19–25% of current length)** — consistent with the prior report's
  own finding that pure deduplication does not reach 30–40% on its own.
  Closing the remaining gap to the full 30–40% target requires the
  higher-risk, deeper compression items in §6, which this plan flags but
  does not pre-commit to, since they trade off explanatory detail rather
  than removing pure duplication.

## 1. Per-subsection word counts (current manuscript)

| Subsection | Words |
|---|---|
| Problem Setting and Motivation | 457 |
| Research Objective and Scope | 374 |
| Main Contributions | 287 |
| Related Work | 716 |
| System Model and Preliminaries | 518 |
| Eviction-Value Prediction Framework | 682 |
| Robust Decision Mechanism | 592 |
| Algorithmic Workflow and Implementation Details | 820 |
| Experimental Setup | 497 |
| Datasets, Baselines, and Evaluation Metrics | **1,037 (largest)** |
| Offline Ablation and Model Selection | 522 |
| End-to-End Online Replay Evaluation Across Available Capacities | 586 |
| Workload-Specific Breakdown | 465 |
| Discussion and Analysis | 937 |
| Overhead and Scalability | 311 |
| Replay Horizon Selection | 179 |
| Summary of Findings | 504 |
| Implications of the Proposed Approach | 386 |
| Limitations | 722 |
| Future Research Directions | 389 |

## 2. Itemized cut list (location | est. words removed | risk | recommendation)

### Cut 1 — Duplicate formal definition of the eviction-loss target
**Location:** System Model and Preliminaries, lines 127–134 (the
`eq:horizon-loss` definition of $L_H(q,t)$) duplicates, almost equation-for-
equation and clause-for-clause, the formal definition of the *same*
quantity given immediately afterward in Eviction-Value Prediction Framework,
lines 140–147 (`eq:eviction_loss`). Both use identical phrasing — "lightweight
LRU-style replay," "deterministic and computationally tractable," "local
proxy... not an offline-optimal objective."
**Est. words removed:** ~170 (replace the System Model passage with one
forward-referencing sentence, e.g. "We formalize this loss target in
Section~\ref{subsec:eviction_value_framework}, Eq.~\eqref{eq:eviction_loss}.").
**Risk:** **Low** — true duplicate; the full, final definition survives
intact in the very next subsection.
**Recommendation:** Cut. Highest-confidence, zero-content-loss item in this
plan.

### Cut 2 — Sections 1.1/1.2 overlap (R3-Minor8's literal first example)
**Location:** "Problem Setting and Motivation" (lines 40–51, 457 words) and
"Research Objective and Scope" (lines 53–64, 374 words). Both independently
state the same core premise — "predictive information is useful only when it
improves the actual decision/action taken" (line 45 vs. line 56) and "the
replacement problem is a candidate-selection/candidate-level problem" (line
47 vs. line 58) — in different words, back-to-back.
**Est. words removed:** ~280–330 (merge into one ~520–570-word subsection,
e.g. "Problem Setting, Motivation, and Research Objective").
**Risk:** **Low** — pure restructuring; no result or claim depends on these
two subsections remaining separate.
**Recommendation:** Merge. This is the exact pair of sections R3-Minor8
names explicitly — addressing it directly answers a named, specific reviewer
complaint and should be prioritized first.

### Cut 3 — The "mixed conclusion" paragraph, repeated four times
**Location:** A near-identical "mixed conclusion" paragraph — *"the proposed
target/scorer is well-motivated and learnable in isolation, but the present
instantiation is not yet a practically superior online [policy/rule],
because it does not improve on LRU/SIEVE/FIFO-Reinsertion end-to-end, and the
gap widens non-monotonically at capacity 128"* — appears, with only
surface-level wording changes, at:
  - Workload-Specific Breakdown, line 530 (95 words)
  - Discussion and Analysis, line 547 (141 words)
  - Summary of Findings, line 586 (152 words)
  - Limitations, line 607 (closing ~120 words of a 353-word paragraph that
    is otherwise unique causal-hypothesis content)
  - Abstract, line 29 (shorter; recommend keeping as-is, since the Abstract
    must stand alone)

  This is precisely the pattern R3-Minor8 flags ("repeat methodological
  caution many times") and is the manuscript's single largest unforced
  repetition. It has **grown** since the prior shortening audit because the
  capacity-128 finding was layered into all four locations independently
  during the 2026-06-21 revision-writing pass.
**Est. words removed:** ~300–400. Keep one full statement (recommend the
Discussion and Analysis version, line 547, since it sits immediately after
the unique causal-mechanism analysis and is the most natural place for a
reader to encounter the full claim first). Reduce Workload-Specific
Breakdown (530) and Summary of Findings (586) to one-sentence
back-references ("see Discussion and Analysis, §X, for our overall
assessment of this finding"). In Limitations (607), keep the unique
diagnostic-checks/causal-hypothesis material (the bulk of the 353-word
paragraph) and cut only its closing restated sentence.
**Risk:** **Low–Medium** — purely a "say it once, reference it elsewhere"
edit; no factual claim changes, but touches four different subsections so
requires care to keep cross-references (`\ref{}`) consistent.
**Recommendation:** Cut, in the order: Workload-Specific Breakdown first
(lowest risk, most clearly redundant), then Summary of Findings, then trim
Limitations' closing sentence only.

### Cut 4 — Algorithmic Workflow prose duplicates Algorithm 1's pseudocode
**Location:** Algorithmic Workflow and Implementation Details, lines
261–288 (the prose walk-through of building the candidate feature matrix,
scoring, and taking the argmin) restates, in sentence form, exactly what
Algorithm 1 (`alg:evict_value_guarded`, lines 290–327) already shows in
pseudocode immediately below it — and what Eqs.~\eqref{eq:feature_vector}–
\eqref{eq:eviction_rule} in the Eviction-Value Prediction Framework
subsection already established formally.
**Est. words removed:** ~150–200 (trim the prose to 2–3 sentences that
introduce Algorithm 1 and highlight only what isn't obvious from the
pseudocode, e.g. the fallback hand-off point).
**Risk:** **Medium** — this is the largest Method subsection (820 words);
some readers benefit from both a prose and pseudocode version, so this is
compression of explanation rather than removal of a true duplicate. Keep the
equations; cut only the prose paragraph that re-narrates them.
**Recommendation:** Cut, but apply after Cuts 1–3 (lower-risk items) and
re-read once for clarity before finalizing.

### Cut 5 — Datasets/Baselines prose duplicates Table 5/Table 6 content
**Location:** Datasets, Baselines, and Evaluation Metrics, lines 364 and 366
(178 words combined) narrate, baseline-by-baseline and family-by-family, the
same descriptions already given in the "Role in this work" column of Table 5
(`tab:trace_families`) and the "Role in evaluation" column of Table 6
(`tab:main_policy_families`). This is the **largest subsection in the
manuscript** (1,037 words) and was already flagged in the prior (superseded)
shortening report at the smaller manuscript size; it is unchanged in the
current version.
**Est. words removed:** ~200–300 (replace per-baseline/per-family prose
narration with one or two sentences that forward-reference the tables by
number, consistent with how the manuscript already treats Table 2's feature
groups).
**Risk:** **Low** — the tables already contain the full content; this is
non-substantive trimming of restated material.
**Recommendation:** Cut.

### Cut 6 — Related Work's SIEVE/FIFO-Reinsertion mechanism detail is a third restatement
**Location:** Related Work, line 95 (146 words) gives a detailed
hand-advancement-vs-physical-reinsertion mechanism description for
SIEVE/FIFO-Reinsertion that is *also* given (more briefly) in Table 6's
"Role in evaluation" column and *also* partially repeated in the
Datasets/Baselines prose at line 366 (counted in Cut 5). Three
descriptions of the same two baselines' mechanics.
**Est. words removed:** ~100–150 (keep the citation and the one-sentence
"contrasts hand-advancement against physical reinsertion" framing; cut the
fuller mechanism restatement, since Table 6 and the policy implementation
itself carry that detail).
**Risk:** **Low.**
**Recommendation:** Cut.

### Cut 7 — Discussion / Summary of Findings / Implications three-way structural overlap
**Location:** This is the structural version of Cut 3, and is the modern
equivalent of R3-Minor8's literal "Sections 3.4 and 4.1–4.2" complaint
(old numbering predates the current section structure, but maps onto these
three subsections): **Discussion and Analysis** (937 words, Experiments
section), **Summary of Findings** (504 words) and **Implications of the
Proposed Approach** (386 words, both Conclusions section) restate, a third
and fourth time beyond Cut 3's narrower paragraph-level overlap, the same
three higher-level claims: (a) short-horizon nonlinear scorers win the
offline ablation, (b) supervision-target design is a central issue, (c) the
work is methodological evidence, not a superiority demonstration. Compare,
e.g., line 537 ("the main contribution of the current results is
methodological... candidate-level downstream supervision is a viable
target") against line 584 ("the main contribution of the paper is both
methodological and empirical: it introduces a candidate-level finite-horizon
supervision target") — same claim, same rhetorical structure, restated with
different connective tissue.
**Est. words removed:** ~300–450, beyond Cut 3. Recommend merging "Summary
of Findings" and "Implications of the Proposed Approach" into one ~450–550
word "Summary and Implications" subsection (this was the prior report's
top-ranked recommendation at the smaller manuscript size and remains valid
and arguably more urgent now); leave "Discussion and Analysis" focused on
content genuinely unique to it (the offline-ablation interpretation and the
capacity-128 causal hypothesis), which Cut 3 already partially achieves.
**Risk:** **Medium** — a structural section merge in the Conclusions
chapter; requires a careful pass to ensure no fact is dropped, but no
result/data is at stake, purely an editorial restructuring.
**Recommendation:** Cut/merge, after Cuts 1–3 are done and stable (so the
merge target reflects the already-deduplicated Discussion text).

### Cut 8 — Fallback-related caveat repetition (cross-references Task 2)
**Location:** The "not yet measured/validated" disclaimer for the
guard/fallback mechanism is restated with only minor wording changes at:
Contributions item 3 (line 75), Robust Decision Mechanism intro (line 193),
line 231, line 233, Datasets/Baselines (line 368), and Limitations (line
613) — six near-identical hedges for one mechanism.
**Est. words removed:** ~250–400, **contingent on the Task 2 decision**
(`reports/kbs_fallback_final_decision_report.md`). If fallback is removed
from the numbered Contributions list (this audit's Task 2 recommendation —
see that report), the Contributions-list sentence and at least two of the
Method-section repeats can be cut outright rather than merely shortened.
**Risk:** **Low–Medium**, gated on the Task 2 decision being applied first.
**Recommendation:** Cut, sequenced after Task 2's decision is adopted in the
manuscript (do not cut this independently of that decision, since the
right *number* of remaining caveat instances depends on whether fallback
stays in Contributions at all).

### Cut 9 — General hedging-density tightening (diffuse, lower priority)
**Location:** Manuscript-wide. Phrase counts in the current text: "rather
than" ×51, "may" ×14, "suggest"/"suggests" ×20, "appears"/"appear" ×15,
"plausible" ×5, "should be interpreted" ×3. Not all instances are
overhedging (many do legitimate epistemic work, e.g. genuinely open
questions about workload-dependence), but the density is consistent with
R3-Summary's explicit complaint ("the extensive hedging language further
weakens the perceived contribution") and is measurably higher than in the
Abstract, which is already appropriately direct by comparison.
**Est. words removed:** ~150–250, achievable only via a full manual
read-through (not a mechanical find/replace, since each instance needs a
judgment call about whether the hedge is load-bearing).
**Risk:** **Low**, but effort-to-savings ratio is worse than Cuts 1–7;
do last.
**Recommendation:** Tighten, as a final pass after the structural cuts
above, not as a standalone first step.

## 3. Summary table

| # | Location | Est. words removed | Risk | Recommendation |
|---|---|---|---|---|
| 1 | System Model duplicate $L_H(q,t)$ definition (lines 127–134) | ~170 | Low | Cut — replace with forward reference |
| 2 | §1.1/§1.2 overlap (lines 40–64) | ~280–330 | Low | Merge into one subsection |
| 3 | "Mixed conclusion" paragraph ×4 (lines 530, 547, 586, 607) | ~300–400 | Low–Medium | Keep one (Discussion, line 547); reference elsewhere |
| 4 | Algorithmic Workflow prose vs. Algorithm 1 (lines 261–288) | ~150–200 | Medium | Trim prose, keep pseudocode |
| 5 | Datasets/Baselines prose vs. Tables 5/6 (lines 364, 366) | ~200–300 | Low | Forward-reference tables |
| 6 | Related Work SIEVE/FIFO-Reinsertion mechanism detail (line 95) | ~100–150 | Low | Cut to one sentence + citation |
| 7 | Discussion/Summary/Implications 3-way overlap | ~300–450 | Medium | Merge Summary + Implications |
| 8 | Fallback caveat repetition ×6 | ~250–400 | Low–Medium (gated on Task 2) | Cut after Task 2 decision applied |
| 9 | General hedging tightening | ~150–250 | Low | Final pass, do last |
| | **Total** | **~1,900–2,650** | | |

(Note: the §0 headline range of 2,250–2,900 includes the upper half of each
row's range; the table above shows the full per-row spread.)

## 4. Recommended execution order

1. Cut 1 (duplicate equation) — zero risk, do immediately.
2. Cut 2 (§1.1/§1.2 merge) — directly answers R3-Minor8's named example.
3. Cut 3 (mixed-conclusion dedup) — directly answers R3-Minor8's "repeat...
   many times" complaint and is the largest single-cluster saving.
4. Resolve Task 2 (fallback decision), then execute Cut 8.
5. Cut 5, Cut 6 (table cross-referencing) — low risk, moderate savings.
6. Cut 7 (Conclusions-chapter merge) — apply once Cuts 1–3 have already
   de-duplicated the source material being merged.
7. Cut 4 (Algorithmic Workflow trim) — medium risk, do with a careful
   re-read.
8. Cut 9 (hedging pass) — last, lowest savings-per-effort.

## 5. Verification requirement after any cuts are applied

Per the standing constraint from the prior revision-writing pass, any
shortening edits applied to `manuscript_source/main.tex` must be followed by
a clean `tectonic main.tex` rebuild from scratch (`rm -f main.pdf main.aux
main.log main.bbl main.blg main.out && tectonic main.tex`) to confirm zero
undefined references — several of the cuts above (3, 4, 7) touch
cross-referenced labels (`\ref{subsec:discussion_analysis}`,
`\ref{subsec:limitations}`, `\ref{alg:evict_value_guarded}`) and a broken
`\ref` would not be caught without a full recompile. **This audit pass did
not apply any of the cuts above** — they are a plan for a future editing
pass, per this turn's instruction to perform audit/reporting work only.

## 6. What it would take to reach the full 30–40% target (not recommended yet)

The itemized cuts in §2 total ~19–25% (1,900–2,900 of the needed 3,505–4,673
words). Reaching the full 30–40% target would additionally require
compressing technical explanation, not just removing duplication, in one or
both of:
- **Algorithmic Workflow and Implementation Details** (820 words, the
  largest Method subsection even after Cut 4) — a deeper cut here would mean
  shortening the five-stage workflow description itself, not just the
  Algorithm-1-duplicate prose.
- **Limitations** (722 words, six numbered points) — beyond Cut 3's
  reduction of point 3's closing sentence, a deeper cut would mean
  compressing the diagnostic-checks enumeration itself (point 3's unique
  content), which risks weakening exactly the evidence that supports the
  capacity-128 "likely real, not a bug" conclusion — **not recommended**
  without explicit sign-off, since this is evidentiary content R3-Issue7
  (validation-depth concern) and R2-MC3 (end-to-end evidence concern)
  specifically reward having, not content to thin out for length alone.

**Recommendation:** Execute §2's ~19–25% cut first, then re-assess whether
the remaining gap to 30% is worth closing via the higher-risk items above,
or whether a response-letter statement ("we have shortened the manuscript by
X%, focusing on duplication and overlapping caveats per Reviewer #3's
specific examples, while preserving the validation evidence requested
elsewhere by Reviewer #2 and Reviewer #3") is the more defensible position.
A mechanical 30–40% cut that removes evidentiary detail to hit a percentage
target would risk reopening exactly the credibility concerns (R2-MC3,
R3-Issue5, R3-Issue7) this revision has otherwise worked to close.
