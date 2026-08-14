# Response to Reviewers

Manuscript: KNOSYS-D-26-07461R1
Decision: Second revision
Date: 2026-08-14

This letter responds point-by-point to Reviewer #2's four major comments and
Reviewer #3's primary unresolved issue on the previous revision. All section,
table, and page references below are to the enclosed revised manuscript PDF
(44 pages).

We thank both reviewers for pushing this manuscript toward a substantially
more rigorous and honest empirical study. In preparing this revision we also
identified, and disclose directly here and in the manuscript, a train/test
overlap in the model underlying the manuscript's original end-to-end
evaluation (Table 5, Section 3.4); the corrected, leakage-free evaluation
built to address Reviewer #2's Major Comment 1 also resolves this issue and
is now the paper's primary evidence for `evict_value_v1`'s comparative
performance.

---

## Response to the Associate Editor

> We thank the Associate Editor and both reviewers for their detailed and
> constructive feedback on the previous revision. In this new revision we
> have: (1) added a corrected, leakage-free comparison of `evict_value_v1`
> against four learned cache-replacement systems (LRB, 3L-Cache, CACHEUS,
> HALP) and three classical baselines, requested specifically for LRB and
> 3L-Cache, under a matched evaluation protocol — the result is negative for
> our method; (2) directly tested, rather than merely argued for, the premise
> that our eviction-loss supervision target is better suited to the eviction
> decision than reuse-distance, next-arrival, or pairwise-preference
> alternatives in the deployed pipeline — that pipeline comparison is
> negative, but a matched common-model control that changes only the training
> objective does **not** support blaming the eviction-loss label itself;
> (3) replaced a previously speculative causal account of the offline/online
> performance gap with a mechanistic diagnosis (an exact-target oracle, a
> target-degeneracy audit, a learned/exact agreement analysis, and a
> tie-aware follow-up) and a controlled causal ablation directly testing the
> continuation-label mismatch hypothesis Reviewer #3 identified as untested;
> (4) replaced our earlier single-run local timing benchmark with a
> controlled, repeated-measurement wall-clock campaign and added an explicit
> statement of intended use and current practical limitations; and
> (5) disclosed and addressed a train/test-overlap issue we found in the
> model underlying our original end-to-end table while preparing this
> revision. We report all of this evidence candidly: the revised
> manuscript's central finding is that the deployed method does not
> outperform any tested baseline, that the eviction-loss training objective
> is not established as the cause, and that the previously reported
> deterministic exact-oracle deficit versus LRU is a tie-breaking confound
> rather than proof that the exact target intrinsically loses to LRU.

---

## Response to Reviewer #2

### Major Comment 1 — Insufficient comparison with existing learned cache-replacement methods

**Reviewer comment**

> Insufficient comparison with existing learned cache-replacement methods.
> Direct comparisons are needed, particularly with LRB and 3L-Cache. Such
> comparisons are necessary to determine whether the proposed candidate-level
> eviction-loss objective offers advantages over existing learning-based
> approaches. Fair comparison requires the same traces, capacities,
> preprocessing, request budgets, and metrics.

**Response**

We agree, and have added the requested direct comparison. We retrained
`evict_value_v1` under a leave-one-family-out protocol (no request from an
evaluated family contributes to that family's training, validation,
hyperparameter selection, or early stopping at any point) and evaluated it
against LRB, 3L-Cache, CACHEUS, and HALP — plus LRU, SIEVE, and
FIFO-Reinsertion — under a matched protocol: identical traces (verified by
SHA-256 hash, not filename), identical capacities (32, 64, 128), an identical
history prefix (`[0, 10,000)`), an identical scored suffix (`[10,000,
50,000)`, 40,000 scored requests), identical object-slot/unit-object
semantics, and identical hit/miss accounting.

The answer to the reviewer's specific question is direct: **the comparison
does not establish an advantage for `evict_value_v1`.** It loses on a clear
majority of the 21 matched (family, capacity) cells against every one of the
seven baselines, including LRB (13 losses, 5 wins, 3 ties; $+1.85\%$ relative
miss ratio) and 3L-Cache (13 losses, 5 wins, 3 ties; $+1.13\%$). We report
this candidly rather than qualify it away, and we use the same corrected,
leakage-free evaluation to replace the train/test-overlap-affected model that
had underpinned the manuscript's original end-to-end table.

**Changes in the revised manuscript**

- New Section 3.6, "Matched Comparison Against Learned Cache-Replacement
  Baselines" (pp. 23–25), with subsections on the corrected training
  protocol (§3.6.1), the matched evaluation protocol (§3.6.2), results
  (§3.6.3, Table 7, p. 24), and implementation-fidelity caveats (§3.6.4).
- Related Work (§1.4, p. 5) now introduces 3L-Cache and CACHEUS with proper
  citations alongside the existing LRB and HALP citations, and states that
  all four are directly compared in §3.6.
- Table 3 (Main policy families, p. 17) now lists LRB, 3L-Cache, CACHEUS,
  and HALP as learned baselines, with their evaluation location and
  fidelity noted.
- Section 3.4 (End-to-End Online Replay Evaluation, p. 18) now discloses the
  train/test-overlap limitation of its underlying model directly and points
  to §3.6 as the corrected, primary evidence.
- Contributions list, item 2 (p. 4), and Abstract (p. 1) state the matched
  comparison and its negative result explicitly.

**Additional clarification**

Implementation-fidelity caveats are recorded transparently and are separate
from the evaluation-protocol dimensions above, which are exact for all seven
baselines: LRB and 3L-Cache are independent repository reimplementations
(3L-Cache additionally uses a fixed default batch size not re-tuned for this
study); CACHEUS runs the official authors' own decision engine unmodified,
with a provenance caveat that the external upstream clone is not currently
live-verifiable in this development environment; HALP is an independent
reimplementation because no official public HALP implementation exists (see
also the response to Reviewer #3, Issue 2, below). None of these caveats
affect the matched traces, capacities, windows, budgets, or metrics — only
how faithfully each reimplementation reproduces its source algorithm's exact
internal behavior. We also note, for full transparency, that "matched
evaluation protocol" refers only to evaluation-side inputs: LRB, 3L-Cache,
and CACHEUS are online/adaptive baselines with no cross-family training
corpus at all, HALP trains offline only on each trace's own history prefix,
and `evict_value_v1` trains offline with leave-one-family-out exclusion —
these are intentional, disclosed differences in training mechanics, not an
evaluation-protocol gap, and no baseline in the comparison has access to
future information or another family's data at any point.

---

### Major Comment 2 — Insufficient justification for finite-horizon eviction-loss supervision

**Reviewer comment**

> Insufficient justification for the finite-horizon eviction-loss
> supervision target. Evidence is needed against alternative candidate-level
> objectives such as reuse distance, next-arrival time, and pairwise
> preference, to justify why eviction-loss should be preferred.

**Response**

We agree this premise was previously asserted rather than tested, and we now
test it at two levels.

First, using the same leave-one-family-out training and matched evaluation
protocol as the Major Comment 1 response, we trained and evaluated three
alternative supervision objectives in place of eviction-loss — reuse
distance, next-arrival time, and pairwise preference — in the full
`evict_value_v1` pipeline. In that pipeline, eviction-loss produces the most
total misses of the four objectives (601,569), compared to next-arrival
(573,059), reuse-distance (571,456), and pairwise preference (565,127).
Eviction-loss is the single worst objective within every one of the seven
trace families individually. We report this pipeline result directly.

Second, that comparison does not isolate the training objective as the sole
causal factor. We therefore added a matched common-model control on the same
21 cells: identical folds, windows, features, architecture, and seed; only
the objective changes; and the pairwise objective uses a corrected
orientation. Under this control, eviction-loss is nominally best by total
misses (571,976), ahead of pairwise (577,339), reuse-distance (615,850), and
next-arrival (627,392). Eviction-loss is **not** materially worse than the
alternatives. The matched experiment therefore does **not** support the
hypothesis that the eviction-loss training objective itself explains the
poor performance of the full learned policy.

**Changes in the revised manuscript**

- Section 3.7, "Direct Comparison Against Alternative Supervision
  Objectives", retains Table 8 (full-pipeline comparison) and adds a
  matched common-model V2 table.
- Section 4.2 ("Implications of the Proposed Approach") is revised so that
  the design lesson is decision informativeness and tie resolution, not
  "the eviction-loss label is uniquely harmful."
- Contributions list, item 3, and the Abstract state both the pipeline
  result and the matched-control interpretation explicitly.

**Additional clarification**

We connect this distinction to the mechanistic diagnosis in Section 3.8:
the stronger supported explanation is that the horizon-4 target is highly
action-underdetermined, not that the eviction-loss training objective is
the culprit.

---

### Major Comment 3 — Gap between offline learning and online cache performance; speculative distribution-shift explanation

**Reviewer comment**

> There remains a gap between offline learning performance and online cache
> performance. The manuscript's explanation invoking distribution shift was
> speculative. Controlled analyses/experiments are needed to test this
> explanation rather than assert it.

**Response**

We agree the previous revision's account — a single, unconfirmed hypothesis
attributing the online gap to "compounding distribution shift" from the
LRU-continuation labeling assumption — was speculative, and we replace it
with controlled evidence. We ran complementary diagnostics, all under the
matched leave-one-family-out protocol:

1. **Is inaccurate prediction the cause?** We compare the learned scorer's
   decisions against an exact solver for the same eviction-loss target. The
   learned scorer agrees with the exact target's optimal candidate set in
   97.53% of decisions, disfavoring gross model-fitting failure.
2. **Is the target itself useful, even optimized perfectly?** A
   *deterministic* exact horizon-4 oracle (minimum candidate identifier
   among exact minimizers) loses to LRU on 18 of 21 cells (0 wins, 3 ties;
   $+81{,}750$ total misses). This shows that this specific
   target-plus-tie-resolution instantiation is a poor online policy. It
   does **not**, by itself, prove that the exact target intrinsically loses
   to LRU.
3. **Is that deficit robust to valid within-minimum tie-breaking?** A
   tie-aware follow-up on the same 21 cells finds
   `fraction_tied_decisions = 1.0` on every non-LRU oracle row (mean
   optimal-set fraction $\approx 0.991$). Choosing the LRU-most exact
   minimizer never loses to LRU (16 wins / 5 ties / 0 losses; $413$ fewer
   total misses than LRU). The deterministic deficit is therefore a
   tie-breaking confound. The strongest supported diagnosis is that the
   target is highly action-underdetermined, so downstream selection among
   many exact minimizers matters substantially. We do not claim that tie
   degeneracy explains every aspect of learned-policy underperformance.
4. **Why is the target so underdetermined?** Independent degeneracy and
   reuse-tail diagnostics remain consistent with this picture: no decision
   has a unique optimal candidate, and the horizon-4 window observes only a
   small fraction of eventual reuse
   ($P(T>4\mid\text{resident})=0.9939$).

We then directly test the continuation-mismatch hypothesis with a controlled
causal ablation, changing exactly one thing between two trained policies
(the continuation rule used to build training labels: LRU vs. a frozen
first-stage policy). Correcting the continuation label improves 13 of 21
cells (macro mean $-0.0102$) but worsens 5, including one severe
counter-example (BrightKite, capacity 32, $+0.2433$) — we classify this as
**partially supported and regime-dependent**, a real but secondary
contributor, not a full explanation. A companion experiment tests whether
reducing a *generic* measure of training-state distribution shift (one-step
DAgger-style correction) improves performance: it reduces the measured shift
in 16 of 21 cells while online misses simultaneously *worsen* in 16 of 21 —
a clear negative result for this specific corrective intervention.

**Changes in the revised manuscript**

- Section 3.8, "Mechanistic Diagnosis of the Eviction-Loss Target", now
  includes the deterministic oracle, the degeneracy/agreement diagnostics,
  and the completed tie-aware follow-up (no longer pending).
- Section 3.9 reports the C0/C1/C2 causal ablation and the DAgger
  distribution-shift-correction result.
- Section 4.3, "Limitations", states that these experiments constrain
  interpretation rather than proving one complete causal mechanism, and
  that deterministic exact-oracle replay is tie-sensitive.
- Contributions list, item 4, and the Abstract state the tie-aware
  reinterpretation explicitly.

**Additional clarification**

We deliberately distinguish "distribution shift as a phenomenon" from "this
specific corrective intervention as a remedy": the DAgger result shows the
tested generic shift-reduction method does not help, not that distribution
shift does not exist. We likewise distinguish "the deterministic exact
oracle loses to LRU" from "the exact target intrinsically loses to LRU";
only the former is supported, and it is explained by tie-breaking.

---

### Major Comment 4 — Practical significance: high overhead and poor performance relative to lightweight baselines

**Reviewer comment**

> The practical significance of the proposed method is unclear given its
> high computational overhead and poor performance relative to lightweight
> baselines. The manuscript should clarify intended-use scenarios and when
> the cost could be justified.

**Response**

We agree and address this directly rather than minimizing the cost. We
replaced our earlier single-machine, single-run local timing probe with a
controlled, repeated-measurement campaign: 7 families $\times$ 3 capacities
$\times$ 5 repetitions per policy (420 total timed runs) for LRU,
FIFO-Reinsertion, SIEVE, and a causal HALP reimplementation, executed on a
dedicated compute node. `evict_value_v1` was not included in this controlled
campaign — a full 5-repetition run was judged computationally infeasible
given its substantially higher per-decision cost — so we report its runtime
separately, as the single, non-repeated measurement already available from
the Major Comment 1 evaluation, and we do not place it in the same table as
the four controlled policies.

LRU (4.68 $\mu$s/request mean), FIFO-Reinsertion (5.17 $\mu$s), and SIEVE
(9.52 $\mu$s) remain single-digit-microsecond policies under repeated
measurement. HALP is $186\times$ slower than LRU. `evict_value_v1`'s
single-run measurement is roughly four orders of magnitude slower than LRU.
Combined with the negative miss-ratio results (Major Comments 1–2 above), we
state plainly that the current instantiation is not a candidate for
replacing any evaluated baseline in latency- or throughput-sensitive
deployments, and we do not identify or claim a deployment scenario in which
its current cost/performance combination is justified. We frame its value
instead as diagnostic and methodological: a controlled empirical test that
isolates *why* a representative finite-horizon candidate-level target
underperforms, which directly informs what a future redesign would need to
change.

**Changes in the revised manuscript**

- Section 3.10, "Overhead and Scalability" (pp. 31–33), replaces the earlier
  single-run local benchmark with the controlled 420-run campaign (Table 9,
  p. 32) and the separately reported `evict_value_v1` single-run figure.
- New Section 3.11, "Practical Significance" (p. 33), directly states the
  absence of a currently justified deployment scenario and the narrower
  research-tool framing.
- Contributions list, item 6 (p. 4), and Abstract (p. 1) state the timing
  result and the reframed contribution explicitly.
- Section 4.3, "Limitations" (p. 36), item 6, records that
  `evict_value_v1`'s timing is a single measurement, not part of the
  controlled 5-repetition campaign, and should not be treated as directly
  statistically comparable to it.

**Additional clarification**

We verified the $O(k)$-per-miss vs. $O(1)$ complexity-class comparison
directly against our own implementations' source code (unchanged from the
previous revision), and the new controlled campaign adds repeated-measurement
statistical support (mean, median, 95% CI) rather than relying on a single
run for the four lightweight/HALP policies.

---

## Response to Reviewer #3

### Primary issue — untested causal claim about LRU-continuation label construction

**Reviewer comment**

> The manuscript attributed the offline/online performance gap partly to a
> mismatch between LRU-continuation label construction and the learned
> policy's actual deployed trajectory, but this causal claim had not been
> tested.

**Response**

We thank the reviewer for identifying this gap precisely; it is a direct,
specific, and testable claim, and we agree it should not have been asserted
without a controlled test. We designed and ran a three-arm causal ablation
that isolates exactly this mechanism: **C0** replays LRU end-to-end (the
reference). **C1** trains a policy $\pi_1$ on labels built under LRU
continuation (the original specification) and deploys $\pi_1$ recursively.
**C2** trains a second policy $\pi_2$ on labels built under a corrected
continuation rule — the frozen $\pi_1$'s own decisions rather than LRU — and
deploys $\pi_2$ recursively. This is the *only* change between C1 and C2,
under the leave-one-family-out protocol used throughout this revision.

Across the 21 (family, capacity) cells: C2 improves over C1 in 13 cells, ties
in 3 (the degenerate Wikimedia cells), and worsens in 5, with a macro mean
miss-ratio improvement of $-0.0102$ (aggregate misses: C1 $601{,}569 \to$ C2
$592{,}970$, against a C0/LRU reference of $565{,}126$). One cell — BrightKite
at capacity 32 — regresses sharply ($+0.2433$), which by itself precludes a
claim of uniform benefit.

**Our answer to the reviewer's question is therefore: partially supported and
regime-dependent, not fully supported and not disfavored outright.** The
continuation-label mismatch is a real, measurable, secondary contributor to
the offline/online gap in most family/capacity combinations, but it does not
fully explain the gap, and one specific case shows a large effect in the
opposite direction. We have replaced the manuscript's previous speculative
language with this controlled result throughout.

We also tested, as a related but distinct question, whether correcting a
*generic* measure of training-state distribution shift (rather than the
continuation label specifically) helps: a one-step DAgger-style intervention
reduces the measured shift metric in 16 of 21 cells, yet online misses
*worsen* in 16 of 21 (macro mean $+0.0094$). We report this as a companion
negative result and are careful not to conflate it with the continuation-label
finding above: they test different mechanisms (label construction vs.
broader state-visitation correction), and only the former shows partial
positive support.

**Changes in the revised manuscript**

- Section 3.9, "Discussion and Analysis" (pp. 28–30), presents the full C0/
  C1/C2 causal ablation and the DAgger distribution-shift result, replacing
  the previous single unconfirmed hypothesis.
- Section 4.3, "Limitations" (p. 36), item 3, states the causal account's
  bounded scope (single horizon, one-step continuation correction, one-step
  DAgger correction) and flags multi-round correction as future work
  (Section 4.4, p. 38).
- Contributions list, item 5 (p. 4), and Abstract (p. 1) state the partial,
  regime-dependent finding explicitly, avoiding the word "explains" in favor
  of "partial, family-dependent secondary contributor."

**Additional clarification**

We chose a leave-one-family-out protocol for this ablation (rather than a
single fixed split) specifically to avoid the train/test-overlap issue
disclosed in our response to Major Comment 1 above, so that the causal
ablation's own conclusions are not confounded by the same contamination
concern.

---

### Issue 2 — insufficient differentiation from HALP

**Reviewer comment**

> Insufficient empirical differentiation from HALP, which the previous
> revision addressed only analytically rather than empirically.

**Response**

We agree and have added an empirical HALP comparison. Section 3.6 (Table 7,
p. 24) reports `evict_value_v1` against HALP under the matched
leave-one-family-out protocol: `evict_value_v1` loses on 17 of 21 cells (1
win, 3 ties), a $+3.59\%$ relative miss-ratio disadvantage. Because no
official public HALP implementation exists, our comparison uses an
independent reimplementation, which we disclose as a lower-fidelity
supporting comparison (Section 3.6.4, p. 25) rather than an official
empirical baseline; this is consistent with the reviewer's request for
differentiation from HALP specifically, rather than a claim of an official
reproduction.

**Changes in the revised manuscript**

- Table 7 (p. 24) includes HALP as an empirical row.
- Section 3.6.4 (p. 25) and Section 4.3 (p. 36), item 6, disclose the
  reimplementation-fidelity caveat.
- Related Work (§1.4, p. 5) retains the analytical differentiation from HALP
  and now also points to the empirical result.

---

### Issue 3 — cost not addressed with controlled measurement

**Reviewer comment**

> Computational cost was not addressed with a controlled measurement.

**Response**

Addressed in full above under Major Comment 4; see Section 3.10 (Table 9,
p. 32) for the controlled 420-run, 5-repetition timing campaign.

---

## Summary of scientific-status changes

| Concern | Previous status | Current status |
|---|---|---|
| R2 Major 1 (LRB/3L-Cache comparison) | Not present | Complete; negative result (§3.6) |
| R2 Major 2 (objective justification) | Asserted, not tested | Pipeline ablation: eviction-loss worst of four (§3.7, Table 8). Matched common-model V2: objective-causality **not supported** (§3.7, common-model table) |
| R2 Major 3 (offline/online gap) | Speculative hypothesis | Mechanistically diagnosed (§3.8), including completed tie-aware follow-up; continuation/DAgger tested (§3.9) |
| R2 Major 4 (practical significance) | Earlier single local run | Controlled 420-run campaign (§3.10) + explicit scope statement (§3.11) |
| R3 primary issue (continuation causal claim) | Untested | Tested; partially supported, regime-dependent (§3.9) |
| R3 Issue 2 (HALP differentiation) | Analytical only | Empirical comparison added (§3.6, Table 7) |

No claim in this revision states that `evict_value_v1` outperforms any
evaluated baseline, that the eviction-loss *training objective itself* is
established as the cause of poor performance, that the exact H4 target
intrinsically loses to LRU, that continuation-label mismatch fully explains
the offline/online gap, that the DAgger intervention improves online
performance, or that `evict_value_v1` was part of the controlled
5-repetition timing campaign. The manuscript reports a negative deployed
result whose leading supported diagnosis is action-underdetermined
supervision, and we believe this directly addresses each concern raised by
the Associate Editor and both reviewers.
