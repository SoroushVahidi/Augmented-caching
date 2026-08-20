# Response to Reviewers

Manuscript: KNOSYS-D-26-07461R1  
Decision: Second revision  
Date: 2026-08-14

This letter responds to the Associate Editor and to both review rounds for
Reviewer #2 and Reviewer #3. Section, table, and page references are to the
enclosed 21-page revised PDF.

Code, audited artifacts, and a reviewer verification guide are public at
https://github.com/SoroushVahidi/Augmented-caching.
The reviewer landing page is
https://github.com/SoroushVahidi/Augmented-caching/blob/main/docs/reviewer/START_HERE.md.
Those materials support, rather than replace, independent inspection of the
manuscript evidence.

An earlier single-split end-to-end evaluation contained train/test overlap.
It is no longer primary evidence. The leave-one-family-out matched evaluation
in §3.4 (Table 5, p. 10) replaces it. Family-level LRU gaps under the same
protocol are in Table 4 (p. 8).

---

## Response to the Associate Editor

We thank the Associate Editor and both reviewers. The first-round letter
asked for end-to-end miss-ratio evaluation, stronger and more modern
baselines, clearer differentiation from prior learned methods, computational
overhead, a justified replay horizon, and a fallback design that is not
oversold. The second-round requests asked for a matched LRB / 3L-Cache
comparison, a test of the eviction-loss objective against alternatives, a
non-speculative account of the offline-to-online gap, practical
significance, and a controlled test of the continuation-label claim.

The revised manuscript reports those comparisons and controls, and it
narrows claims where the new experiments do not support the original
interpretation.

- `evict_value_v1` does not outperform any matched baseline overall
  (§3.4, Table 5, p. 10).
- Full-pipeline objective ranking does not favor eviction-loss (Table 6,
  p. 10). A matched common-model control does not support blaming that
  training objective (Table 7, p. 11).
- The 97.53% set-aware agreement is weak evidence of candidate
  discrimination because the exact optimal set contains about 99.1% of
  candidates (§3.6, pp. 11–12).
- Continuation mismatch is partially supported and regime-dependent
  (§3.7, pp. 12–13). The tested DAgger-style correction is negative for
  miss ratio.
- Controlled timing shows the implementation is not
  deployment-competitive (§3.9–§3.10, Table 8, pp. 13–14). No validated
  deployment scenario is claimed.
- Fallback is not part of the reported quantitative evidence (Figure 1,
  p. 5; §4.2–§4.3, pp. 14–15).
- Primary matched capacities are 32/64/128. Capacity 256 remains
  unevaluated and is disclosed (§4.2–§4.3).
- The manuscript is 21 pages.

Point-by-point responses follow. First-round items that are fully answered
by a second-round experiment are cross-referenced rather than repeated at
length.

---

## Response to Reviewer #2

### Second-round Major Comment 1

**Reviewer concern** (second revision)

> Insufficient comparison with existing learned cache-replacement methods,
> and existing comparisons are not clearly fair.

**Response**

We agree and have added a matched comparison. `evict_value_v1` is retrained
under leave-one-family-out (the held-out family contributes no training,
validation, hyperparameter, or early-stopping rows) and evaluated against
LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, and FIFO-Reinsertion. All eight
policies use identical traces (SHA-256), capacities 32/64/128, history
prefix [0, 10,000), scored suffix [10,000, 50,000) (40,000 requests),
unit-object slots, and the same hit/miss accounting.

The comparison does not establish an advantage for `evict_value_v1`. It
loses on a majority of the 21 cells against every baseline:

| Baseline | EV wins / losses / ties | Relative miss ratio |
|---|---|---|
| LRB | 5 / 13 / 3 | +1.85% |
| 3L-Cache | 5 / 13 / 3 | +1.13% |
| CACHEUS | 3 / 15 / 3 | +3.68% |
| HALP | 1 / 17 / 3 | +3.59% |
| LRU | 1 / 16 / 4 | +4.12% |
| SIEVE | 4 / 14 / 3 | +1.89% |
| FIFO-Reinsertion | 2 / 16 / 3 | +4.04% |

The closest mean comparison is 3L-Cache, where EV still loses on 13 of 21
cells. The negative result spans both learned systems and lightweight
queues.

Matching is evaluation-side. LRB, 3L-Cache, and CACHEUS adapt online; HALP
trains on each trace’s own prefix; `evict_value_v1` is leave-one-family-out.
No policy sees future requests or another family’s data.

Implementation fidelity is disclosed separately from that protocol
(§3.4.4, pp. 9–10): LRB and 3L-Cache are independent reimplementations
(3L-Cache uses an untuned default batch size); CACHEUS wraps the authors’
engine, with the upstream clone not live-verifiable here; HALP has no
official public implementation and is a lower-fidelity supporting
comparison.

**Manuscript locations:** §3.4 and Table 5 (discussion pp. 9–10; table
p. 10); Table 4 (p. 8); §1.3 (p. 3); Table 2 (p. 7).

**Supporting materials:**
https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/kbs_final_evidence_20260813/major1_reviewer_summary.md

---

### Second-round Major Comment 2

**Reviewer concern** (second revision)

> Insufficient justification for the finite-horizon eviction-loss
> supervision objective versus plausible alternatives.

**Response**

We agree the previous revision asserted preference for eviction-loss rather
than testing it. The revised manuscript now reports two distinct
experiments. These tables answer different causal questions and are not
contradictory.

**A. Full-pipeline comparison (Table 6, p. 10).** Under the same
leave-one-family-out protocol, we replace only the candidate-level label.
Eviction-loss has the most total misses of the four objectives (601,569),
versus next-arrival (573,059), reuse-distance (571,456), and pairwise
preference (565,127). It is worst or tied-worst in every family. The paper
no longer claims empirical superiority of eviction-loss.

**B. Matched Common-Model V2 (Table 7, p. 11).** Architecture, features,
folds, windows, and seed are held fixed; only the objective changes, with
corrected pairwise orientation. Totals: eviction-loss 571,976; pairwise
577,339; reuse-distance 615,850; next-arrival 627,392. Eviction-loss is
not materially worse. This control does not support blaming the
eviction-loss training objective for poor deployed performance.

The more strongly supported issue is target informativeness /
action-underdetermination (§3.6, pp. 11–12), not a uniquely harmful
eviction-loss label.

**Manuscript locations:** §3.5 / Tables 6–7 (pp. 10–11); abstract and
contribution 3 (pp. 1–2).

**Supporting materials:**
https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/common_model_v2_formal_audit_20260814/AUDIT.md

---

### Second-round Major Comment 3

**Reviewer concern** (second revision)

> Unexplained gap between offline learning quality and online cache
> performance; the distribution-shift explanation is speculative and
> untested.

**Response**

We agree the previous account was speculative. The revised manuscript now
reports controlled diagnostics on the same 21 cells.

The learned scorer selects an element of the exact target-defined optimal
set in 97.53% of decisions. Because that optimal set contains approximately
99.1% of candidates on average, this metric is weak evidence of
discrimination among candidates. Its main implication is that the learned
scorer rarely selects outside the target-defined optimal set. It does not
prove that model fitting is satisfactory.

A deterministic exact horizon-4 oracle (minimum candidate identifier among
exact minimizers) loses to LRU on 18 of 21 cells (0 wins / 3 ties / 18
losses; +81,750 misses). That instantiation is a poor online policy. It
does not prove that the exact target intrinsically loses to LRU.

A tie-aware follow-up finds fraction_tied_decisions = 1.0 on every
non-LRU oracle row (mean all-tied fraction ≈ 0.649; mean optimal-set
fraction ≈ 0.991). LRU-within-minima never loses to LRU (16 wins / 5 ties /
0 losses; −413 misses). MRU-within-minima matches the deterministic deficit
(0 / 3 / 18; +89,135). The old deterministic deficit was
tie-breaking-confounded. This does not mean that H = 4 supervision improves
LRU, that the target adds proven value to LRU, or that the exact H = 4
target is intrinsically better than LRU.

Independent degeneracy diagnostics agree qualitatively: no decision has a
unique optimum, and P(T>4 | resident) = 0.9939. The replay-time all-tied
fraction (≈ 0.649) and the strict-preference all-tied rate (76.4%) are two
instruments of the same qualitative fact, not interchangeable point
estimates.

Continuation mismatch and generic shift correction are tested separately in
§3.7 (see Reviewer #3, second-round primary issue). Continuation correction
is partially supported and regime-dependent. The tested DAgger-style
intervention is a negative result for miss ratio. That result does not
imply that distribution shift is absent.

**Manuscript locations:** §3.6 (pp. 11–12); §3.7 (pp. 12–13); §4.2
(pp. 14–15).

**Supporting materials:**
https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md

---

### Second-round Major Comment 4

**Reviewer concern** (second revision)

> The implementation is much slower per eviction decision than lightweight
> baselines while not outperforming them on miss ratio; practical
> significance is unclear.

**Response**

We replaced the earlier single-run local probe with a controlled campaign:
7 families × 3 capacities × 5 repetitions × 4 policies (420 timed runs)
for LRU, FIFO-Reinsertion, SIEVE, and causal HALP.

Means: LRU 4.68 μs/request; FIFO-Reinsertion 5.17 μs; SIEVE 9.52 μs; HALP
870.66 μs (186.02× LRU). `evict_value_v1` is not in that repeated campaign.
Its available single-run measurement from the matched evaluation is
reported separately and is not statistically comparable to Table 8.

Together with the negative miss-ratio results, the current method is not
deployment-competitive. No deployment scenario is claimed. The present
scientific value is diagnostic: a controlled test of why a representative
finite-horizon candidate-level target underperforms.

Offline label construction required about 10.3 hours and 96 GB (662
shards); fitting all horizon/model configurations took about 7–8 minutes.
Online, each miss scores all k residents (O(k) inferences) versus O(1) for
LRU, SIEVE, and FIFO-Reinsertion.

**Manuscript locations:** §3.9 / Table 8 (p. 13); §3.10 (pp. 13–14); §4.2
(pp. 14–15).

**Supporting materials:**
https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/kbs_final_evidence_20260813/controlled_timing_summary.csv

---

### Closure of first-round Reviewer #2 comments

**Major Comment 1 (first round) — replay horizon H**

> The proposed framework relies heavily on a manually specified replay
> horizon (H) when constructing the eviction-value target. … I encourage
> the author to provide a more thorough sensitivity analysis, theoretical
> intuition, or an adaptive mechanism for horizon selection.

The revised manuscript reports an empirical H = 4 justification from the
offline ablation (Table 3, p. 8; §3.3, pp. 6–8): tree models beat ridge at
every evaluated horizon, and H = 4 is strongest among {4, 8, 16}. A short
horizon ties the label to the scored eviction; longer horizons mix in later
continuation decisions. This is not a claim of online superiority,
theoretical optimality, an adaptive rule, or a workload-universal horizon
choice. Mechanistic results in §3.6 further show that H = 4 is highly
action-underdetermined.

**Major Comment 2 (first round) — label-generation cost**

> The proposed supervision target requires counter-factual replay for every
> eviction candidate at each full-cache miss. … a quantitative analysis of
> the offline cost is necessary.

Reported in §3.9 (p. 13): about 10.3 hours and 96 GB across 662 shards for
label construction, and about 7–8 minutes to fit the horizon/model
configurations. Online scoring cost is in Table 8 and Major Comment 4
above. No claim is made that the current implementation is cheap at
deployment scale.

**Major Comment 3 (first round) — end-to-end evaluation and fallback**

> Conduct more end-to-end evaluations, compare against strong baselines,
> and test across different workloads, cache capacities, and
> prediction-quality regimes. Additionally, the effectiveness of the
> “guarded fallback mechanism” has not been sufficiently demonstrated.

End-to-end matched miss-ratio evaluation is now the primary evidence
(Table 5, p. 10; Table 4, p. 8; capacities 32/64/128). Learned and
lightweight baselines are included. Capacity 256 remains unevaluated and
is disclosed. Fallback is not part of any reported quantitative claim; it
was removed from the evaluated workflow in Figure 1 (p. 5). See also
Reviewer #3, first-round Issues 1, 3, 5, 6, and 9 below.

---

## Response to Reviewer #3

### Second-round primary issue

**Recorded second-revision concern.** The previous revision attributed part
of the offline/online gap to a mismatch between LRU-continuation labels and
the learned policy’s deployed trajectory without a controlled test.

**Response**

We agree the claim required a test. The three-arm ablation isolates that
mechanism: C0 is LRU; C1 trains and deploys π₁ on LRU-continuation labels;
C2 retrains π₂ on frozen-π₁ continuation and deploys π₂. That is the only
C1→C2 change, under leave-one-family-out.

C2 improves on 13 cells, ties 3 (Wikimedia), and worsens 5 (macro mean
−0.0102; misses C1 601,569 → C2 592,970, versus C0 565,126). BrightKite at
capacity 32 regresses +0.2433.

Answer: partially supported and regime-dependent. Continuation mismatch is
a real secondary contributor in most cells, not a full explanation, and not
uniformly beneficial.

A separate one-step DAgger-style correction reduces a generic shift metric
in 16 of 21 cells, yet misses improve in 2, tie in 3, and worsen in 16
(macro +0.0094; 591,604 → 599,537). That is a negative result for this
intervention, not a test of the continuation label itself.

**Manuscript locations:** §3.7 (pp. 12–13); §4.2–§4.3 (pp. 14–15).

**Supporting materials:**
https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/kbs_final_evidence_20260813/c0_continuation_summary.csv
and
https://github.com/SoroushVahidi/Augmented-caching/blob/main/reports/kbs_final_evidence_20260813/distribution_shift_summary.csv

---

### Closure of first-round Reviewer #3 comments

**Issue 1 — no end-to-end miss-ratio results**

> The paper limits its empirical evidence to offline target-quality
> metrics. For a caching paper, the primary evaluation metric must be
> downstream cache performance—miss ratios or hit rates compared against
> strong baselines.

The primary evidence is now matched end-to-end miss counts / miss ratios
on 21 cells (seven families × capacities 32/64/128). `evict_value_v1` does
not outperform any matched baseline overall (Table 5, p. 10). Capacity 256
was not evaluated.

**Issue 2 — insufficient differentiation from PARROT / Mockingjay / HALP**

> HALP (NSDI ’23) already learns candidate-level preferences from future
> re-access outcomes and evicts the least-preferred candidate—operationally
> very close to what this paper proposes. The paper needs to clearly
> articulate what the finite-horizon replay target provides that HALP’s
> preference signal does not, and ideally present direct empirical
> comparisons.

§1.3 (p. 3) locates PARROT, Mockingjay, and HALP. HALP is the closest
operational neighbour: it also scores residents and evicts a
least-preferred candidate, but from a pairwise preference over realized
re-access. We instead label each candidate with a counterfactual
finite-horizon miss count. Table 5 reports the empirical HALP comparison:
`evict_value_v1` loses on 17 of 21 cells (1 win / 3 ties; +3.59%). No
official public HALP code was available; fidelity is disclosed in §3.4.4
(pp. 9–10). LRB and 3L-Cache are compared under the same matched protocol.

**Issue 3 — missing SIEVE / FIFO-Reinsertion**

> Notably absent are SIEVE (NSDI ’24), FIFO-Reinsertion, and other recently
> proposed lightweight algorithms that achieve strong performance with
> minimal overhead.

Both are in the matched comparison (Table 5, p. 10) and in the timing
campaign (Table 8, p. 13). EV loses to SIEVE on 14 of 21 cells (+1.89%)
and to FIFO-Reinsertion on 16 of 21 cells (+4.04%).

**Issue 4 — excessive hedging / unclear scope**

> The title promises robust learning-augmented caching, but the paper
> itself admits it has not demonstrated robust end-to-end performance.

The current title no longer uses “robust.” Abstract, contributions, and
§3.10 state a diagnostic, negative result: the method is not a
deployment-ready replacement policy. Hedging remains where the evidence is
incomplete (for example, capacity 256, fallback, and independent
replication).

**Issue 5 — computational cost absent**

> The method requires constructing a feature vector for every cached item
> at every full-cache miss … The paper provides no wall-clock time
> analysis, no complexity comparison against O(1) baselines like LRU or
> SIEVE, and no discussion of production overhead.

Addressed under Reviewer #2, second-round Major Comment 4 (§3.9–§3.10,
Table 8, pp. 13–14).

**Issue 6 — fallback unvalidated / oversold**

> If the fallback is not empirically validated, it should not be listed as
> a main contribution.

Fallback is not a contribution and is not used in any quantitative claim.
Figure 1 (p. 5) shows only the evaluated workflow (offline labeling and
online arg min eviction). An optional early-return guard exists in the
accompanying code; it is unvalidated. Future validation is listed as
future work (§4.2–§4.3, pp. 14–15).

**Issue 7 — single-authorship and AI / validation concern**

> Single authorship combined with reliance on AI tools raises questions
> about validation depth, code correctness, and result verification.

Code and audited artifacts are public. Reviewer verification and
reproduction documentation are provided at the repository landing page
above. The manuscript’s AI-use statement is explicit. All reported results
were checked against repository artifacts. Independent replication remains
desirable (§4.2, p. 15). Repository checks do not replace independent
replication.

**Issue 8 — excessive verbosity / repetition**

> Sections 1.1 and 1.2 overlap substantially. Sections 3.4 and 4.1–4.2
> repeat methodological caution many times.

The enclosed PDF is 21 pages. Overlapping contribution/method restatements
were compressed; first-round length recommendations are addressed by that
shortening rather than by removing reviewer-critical tables.

**Issue 9 — workload-specific analysis missing**

> Given workload dependence in caching, aggregate-only results would be
> insufficient.

Table 4 (p. 8) reports matched-protocol family-level miss gaps versus LRU
on the primary scored window. Table 5 reports cell-wise wins/losses across
the same 21 family–capacity cells. Wikimedia is degenerate under this
window.

First-round recommended revisions Rec1–Rec8 map onto Issues 1–9 and
Reviewer #2’s horizon/overhead comments above.

---

## Complete Concern-Closure Summary

| Concern | Closed by | Location |
|---|---|---|
| AE: end-to-end miss ratio | Matched 21-cell evaluation; negative | §3.4, Table 5, p. 10 |
| AE: baselines / learned methods | LRB, 3L-Cache, CACHEUS, HALP, SIEVE, FIFO-Reinsertion, LRU | Tables 2 and 5, pp. 7 and 10 |
| AE / R2-R1: horizon H | Empirical H = 4 among {4,8,16}; no adaptive rule claimed | §3.3, Table 3, p. 8 |
| AE / R2-R1: label-construction cost | 10.3 h, 96 GB, 662 shards; 7–8 min fitting | §3.9, p. 13 |
| AE / R3-I6: fallback | Removed from evaluated workflow; unvalidated | Figure 1, p. 5; §4.2–§4.3 |
| R2-R2 MC1: fair learned comparison | Matched protocol; EV loses overall | Table 5, p. 10 |
| R2-R2 MC2: objective justification | Pipeline vs Common-Model V2 | Tables 6–7, pp. 10–11 |
| R2-R2 MC3: offline/online gap | Tie-aware diagnosis; continuation/DAgger tests | §3.6–§3.7, pp. 11–13 |
| R2-R2 MC4 / R3-I5: practical significance | 420-run timing; no deployment claim | Table 8, pp. 13–14 |
| R3-R2: continuation causal claim | C0/C1/C2 partial; DAgger negative | §3.7, pp. 12–13 |
| R3-I1: end-to-end metric | Same as Table 5 | p. 10 |
| R3-I2: HALP / PARROT / Mockingjay | Related Work + empirical HALP row | §1.3, p. 3; Table 5 |
| R3-I3: SIEVE / FIFO-Reinsertion | Matched miss ratio and timing | Tables 5 and 8 |
| R3-I4: scope / hedging | Diagnostic framing; title without “robust” | Abstract; §3.10 |
| R3-I7: single-author / AI | Public artifacts; independent replication still desirable | §4.2, p. 15 |
| R3-I8: verbosity | 21-page compressed manuscript | enclosed PDF |
| R3-I9: workload-specific results | Family gaps under matched protocol | Table 4, p. 8 |
| Capacity 256 / 1024 | Not evaluated; disclosed | §4.2–§4.3 |

---

## Closing

The revised manuscript now reports the requested baseline comparisons,
objective controls, mechanistic and causal analyses, and controlled timing
evidence. Where those experiments do not support the original
interpretation, the claims have been narrowed. No validated deployment
scenario is claimed.
