# HALP positioning — final audit (2026-06-21, final revision audit pass)

Directly answers **R3-Issue2** ("Insufficient differentiation from existing
learned caching methods... HALP (NSDI '23) already learns candidate-level
preferences from future re-access outcomes and evicts the least-preferred
candidate—operationally very close to what this paper proposes. The paper
needs to clearly articulate what the finite-horizon replay target provides
that HALP's preference signal does not, and ideally present direct
empirical comparisons") and **R3-Rec2** ("Add direct comparisons to HALP and
SIEVE"). Audited locations: Related Work, Introduction, Discussion, and the
response-to-reviewers draft, per this task's scope. Builds on, and
re-verifies, `reports/kbs_halp_fifo_source_verification.md` (2026-06-19)
against the **current** `main.tex` and the **current**
`kbs_response_to_reviewers_skeleton.md`.

## 1. Where HALP currently appears

| File | Location | Content |
|---|---|---|
| `main.tex` | Related Work, line 89 | One-sentence characterization: "HALP is especially relevant because it uses candidate-level preference learning from future re-access outcomes rather than direct action imitation." |
| `main.tex` | Related Work, line 91 | Listed among proxies that remain "intermediate" rather than directly supervising downstream harm. |
| `main.tex` | Introduction (Problem Setting, Objective, Contributions) | **Not mentioned anywhere.** |
| `main.tex` | Discussion and Analysis | **Not mentioned anywhere.** |
| `main.tex` | Limitations | **Not mentioned anywhere** — no disclosure that an empirical HALP comparison was not performed. |
| `reports/kbs_response_to_reviewers_skeleton.md` | R3-Issue2 (lines 250–265) | Still contains an unresolved placeholder bracket: "[Insert direct empirical HALP comparison if implemented, or an explicit, honest statement that a faithful empirical HALP reimplementation was judged out of scope...]" — the honest-statement text was never actually written into the response letter, despite the recommended wording already existing in `reports/kbs_halp_fifo_source_verification.md` §1.4 since 2026-06-19. |

**Total: 2 sentences in the entire manuscript, confined to one paragraph of
Related Work.** This is the central finding of this audit: the
differentiation that exists is directionally correct but minimal, appears
nowhere outside Related Work, and is not load-bearing enough for a reviewer
who explicitly named HALP as "operationally very close."

## 2. What does HALP learn? (manuscript's current characterization, checked against the source)

Current manuscript text (line 89): "it uses candidate-level preference
learning from future re-access outcomes rather than direct action
imitation."

This is accurate but thin. Per `kbs_halp_fifo_source_verification.md` §1.2
(NSDI '23 paper, re-verified in this pass — no new web lookups performed,
only the existing verification re-checked against current manuscript text),
HALP's actual mechanism has three properties the current one-sentence
characterization does not surface:
1. **Pairwise, not pointwise, supervision** — HALP's training label is a
   binary comparison ("which of two candidates is accessed further in the
   future"), not a scalar score per candidate.
2. **Pre-filtered candidate set** — HALP first selects a small heuristic
   candidate subset, then re-ranks only within that subset via pairwise
   comparisons, rather than scoring every resident item.
3. **Online, continuous training** — HALP's labels arise from automated
   feedback on realized future accesses in a live production system
   (YouTube CDN), and the model is updated continuously, not trained once
   offline on a fixed dataset.

**Assessment: incomplete but not wrong.** The manuscript's one sentence
captures property (this is "candidate-level... from future re-access
outcomes") but omits all three structural details above, which is exactly
what makes the current differentiation feel thin to a reviewer who knows
the HALP paper well.

## 3. What does this work learn? (cross-check against Method section)

Per the Method section (Eviction-Value Prediction Framework,
`subsec:eviction_value_framework`), `evict_value_v1` learns a scalar,
candidate-specific, finite-horizon **counterfactual** loss: the number of
misses incurred over the next $H$ steps *if* candidate $q$ were forcibly
evicted now and the cache continued under a fixed LRU-style continuation
rule. This is:
1. **Pointwise, not pairwise** — every candidate gets its own scalar
   target, scored independently.
2. **Exhaustive over the full candidate set** — every resident item ($k$ of
   them) is scored at every miss, with no heuristic pre-filtering.
3. **Offline, batch-trained** — the supervised model is fit once on a fixed
   dataset constructed by replaying counterfactual evictions, not updated
   continuously from live feedback.
4. **Counterfactual, not factual** — the label is derived from a simulated
   alternate trajectory (what *would* happen under a fixed continuation
   rule after a forced eviction), not from the single realized future that
   actually occurred.

This is well-documented elsewhere in the manuscript (Eqs.
\eqref{eq:eviction_loss}–\eqref{eq:eviction_rule}); the gap is only that
this characterization is never explicitly placed side-by-side with HALP's.

## 4. Similarities (currently stated; verified accurate)

The manuscript correctly identifies the shared high-level structure: both
methods (a) operate at the level of individual eviction candidates rather
than a single global trust/advice signal, (b) use future-derived
information to construct training supervision, and (c) move beyond
hand-designed heuristics or pure oracle-action imitation (PARROT-style).
This framing (line 89's "These papers are close to the present work in
that they move eviction learning beyond purely hand-designed heuristics")
is accurate and does not need revision.

## 5. Differences (currently understated — this is the actionable gap)

The current text states one difference ("intermediate proxy" vs. "explicit
supervision on finite-horizon candidate-specific downstream harm," line
91) but does not state the four concrete structural distinctions
enumerated in §§2–3 above side-by-side. Given that R3-Issue2 explicitly
asks the paper to "clearly articulate what the finite-horizon replay target
provides that HALP's preference signal does not," a reviewer re-reading the
revision would reasonably expect to see these differences spelled out, not
just asserted in the abstract category of "proxy vs. direct target."

The single sharpest, most defensible distinction — and the one currently
**missing entirely** from the manuscript — is the **counterfactual vs.
factual** distinction (§2 item 4 / §3 item 4): HALP's pairwise label is
built from the single future that actually happened, while
`evict_value_v1`'s label is built from a simulated alternate future that
would have happened under forced eviction. This is a genuine, citable
methodological difference (it is why `evict_value_v1` can construct a label
for *every* candidate, including ones that were not actually evicted, while
HALP's pairwise comparisons are necessarily drawn from realized outcomes)
and is currently left implicit.

## 6. Limitations of the current comparison (not disclosed anywhere — this is the second actionable gap)

The manuscript nowhere states that no empirical HALP comparison was run.
Compare this to how the manuscript handles other comparison gaps: cap256 is
explicitly flagged as "not yet evaluated... pending separate explicit
approval" in multiple locations (Abstract, End-to-End subsection,
Limitations), and the fallback mechanism's lack of validation is disclosed
six times (per `reports/kbs_fallback_final_decision_report.md` §4). HALP
receives no equivalent disclosure anywhere — a reviewer cannot tell, from
the manuscript alone, whether the absence of an empirical HALP row in
Table 7 (the end-to-end results table) is an oversight or a deliberate,
reasoned scope decision. Given that `kbs_halp_fifo_source_verification.md`
§1.3 already concludes empirical HALP reproduction is infeasible before
2026-07-08 (it requires "a distinct online preference-learning loop,"
production-style deferred feedback, and a YouTube-CDN-specific evaluation
context not reproducible from this repo's trace assets) — this is a
defensible, already-reasoned position. It simply has not been written into
the manuscript or the response letter yet.

## 7. Exact text suggestions

### 7.1 Related Work (`main.tex`, replace line 89's HALP sentence)

**Current:**
> "HALP is especially relevant because it uses candidate-level preference
> learning from future re-access outcomes rather than direct action
> imitation \cite{song2023halp}."

**Suggested replacement** (adds the pairwise/pointwise and
factual/counterfactual distinctions explicitly):
> "HALP is especially relevant because it also makes candidate-level
> eviction decisions using future-derived supervision \cite{song2023halp}.
> Two distinctions are central. First, HALP's training signal is a
> *pairwise* preference between two heuristically pre-selected candidates
> (which of the two is accessed further in the future), continuously
> generated from realized online outcomes in a production deployment,
> whereas our target is a *pointwise*, finite-horizon loss computed
> independently for every resident candidate from a batch-constructed
> dataset. Second, and more fundamentally, HALP's label reflects the single
> future that actually occurred, while our target is explicitly
> \emph{counterfactual}: it is constructed by simulating the cache's
> continuation under a fixed rule after a forced eviction, which lets us
> assign a downstream-harm estimate to every candidate, including ones that
> were never actually evicted in the observed trace."

(Estimated addition: ~95 words. This is a net manuscript addition, not a
cut — flag for the shortening pass in
`reports/kbs_manuscript_shortening_execution_plan.md` as content that
should be added even while other sections are trimmed, since it directly
answers a named Major/Recommended-Revision item.)

### 7.2 Limitations (`main.tex`, new sentence — recommend placing alongside the existing fourth limitation point, which already discusses the paper's non-theorem-centered, comparison-scope framing)

**Suggested addition:**
> "We also did not empirically reimplement HALP \citep{song2023halp} in
> this revision. A faithful comparison would require reproducing its
> online, continuously-updated pairwise preference-learning loop and its
> production-feedback data-generation process, which is not reproducible
> from the offline replay traces used in this study; we therefore restrict
> our comparison to the analytical differentiation in
> Section~\ref{subsec:related_work} rather than report an unfaithful
> surrogate implementation."

(Estimated addition: ~70 words.)

### 7.3 Response-to-reviewers skeleton (`reports/kbs_response_to_reviewers_skeleton.md`, R3-Issue2 block, lines 250–265)

The bracketed placeholder should be replaced with the text already drafted
in `reports/kbs_halp_fifo_source_verification.md` §1.4 (verified still
current and accurate in this pass):

> "A faithful empirical HALP reimplementation would require its own online
> preference-learning pipeline and production-motivated feedback loop,
> which we judged out of scope for the present revision timeline; we
> therefore treat HALP as a closely related cited system, sharpened in the
> revised Related Work (Section~\ref{subsec:related_work}) to state
> explicitly that our target is pointwise and counterfactual where HALP's
> is pairwise and drawn from realized outcomes, rather than report an
> unfaithful surrogate comparison."

This text is **drafted and ready to copy in** — no new analysis is needed;
the only remaining step is the copy-paste-and-reconcile edit itself.

## 8. Bottom line

- **What HALP learns / what this work learns:** both correctly identified
  at a high level; **incomplete** at the structural level a reviewer who
  named HALP explicitly will expect (pairwise vs. pointwise, online vs.
  offline, factual vs. counterfactual, pre-filtered vs. exhaustive
  candidate set).
- **Similarities:** accurately stated, no change needed.
- **Differences:** understated — §7.1's suggested replacement closes this
  gap with a concrete, ready-to-insert paragraph.
- **Limitations of the current comparison:** **not disclosed anywhere** in
  the manuscript today — §7.2's suggested addition closes this gap.
- **Response letter:** contains an unresolved placeholder bracket despite
  the necessary text having existed in a separate report since 2026-06-19
  — §7.3 closes this gap by pointing to the exact ready-to-copy text.

None of the three suggested edits require new compute, a baseline
reimplementation, or new evidence — they are writing-only fixes that make
the manuscript's and response letter's HALP positioning match the depth of
analysis that already exists in `kbs_halp_fifo_source_verification.md` but
was never fully transcribed into the submission-facing documents.
