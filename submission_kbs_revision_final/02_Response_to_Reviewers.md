# Response to Reviewers

Manuscript: KNOSYS-D-26-07461R1
Decision: Second revision
Date: 2026-08-14

This letter responds point-by-point to Reviewer #2's four major comments
from the previous revision and to Reviewer #3's remaining issues. Section,
table, and page references are to the enclosed 21-page revised PDF.

Code, experimental artifacts, and reviewer-verification materials are
available at
https://github.com/SoroushVahidi/Augmented-caching,
with a reviewer-oriented index in
[docs/reviewer/START_HERE.md](../docs/reviewer/START_HERE.md).
Those materials support, rather than replace, the evidence reported in the
manuscript.

An earlier single-split end-to-end model was found to contain train/test
overlap. That result is no longer primary evidence. The leave-one-family-out
matched evaluation in §3.4 (Table 5) replaces it, including the family-level
LRU gaps in Table 4.

---

## Response to the Associate Editor

We thank the Associate Editor and both reviewers. The revision reports the
requested comparisons and controls, and it narrows claims where the new
experiments do not support the original interpretation.

- **Matched learned-baseline comparison.** `evict_value_v1` is compared
  with LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, and FIFO-Reinsertion under
  a leakage-free matched protocol. The result is negative (§3.4, Table 5).
- **Objective control.** Eviction-loss is tested against reuse-distance,
  next-arrival, and pairwise-preference labels. The full pipeline does not
  favor eviction-loss (Table 6). A matched common-model control does not
  support blaming the eviction-loss objective itself (Table 7).
- **Mechanistic and causal diagnosis.** The offline-to-online gap is
  examined with an exact-target oracle, degeneracy/agreement checks, a
  tie-aware follow-up, a continuation ablation, and a DAgger-style control
  (§3.6–§3.7).
- **Timing and practical significance.** A controlled 420-run timing
  campaign replaces the earlier single-run probe (§3.9–§3.10, Table 8). No
  deployment scenario is claimed.
- **Train/test-overlap correction.** The contaminated single-split
  evaluation is disclosed and superseded by the matched protocol.

---

## Response to Reviewer #2

### Major Comment 1 — Insufficient comparison with existing learned cache-replacement methods

**Reviewer concern** (recorded second-revision request)

> Insufficient comparison with existing learned cache-replacement methods,
> and existing comparisons are not clearly fair.

This revision treats LRB and 3L-Cache as the named learned baselines, with a
matched protocol on traces, capacities, windows, and metrics.

**Response**

We agree and added the comparison. `evict_value_v1` is retrained under
leave-one-family-out (the held-out family contributes no training,
validation, hyperparameter, or early-stopping rows) and evaluated against
LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, and FIFO-Reinsertion. All eight
policies use identical traces (SHA-256), capacities 32/64/128, history
prefix `[0, 10,000)`, scored suffix `[10,000, 50,000)` (40,000 requests),
unit-object slots, and the same hit/miss accounting.

**The comparison does not establish an advantage for `evict_value_v1`.**
It loses on a majority of the 21 cells against every baseline:

| Baseline | EV wins / losses / ties | Relative miss ratio |
|---|---|---|
| LRB | 5 / 13 / 3 | $+1.85\%$ |
| 3L-Cache | 5 / 13 / 3 | $+1.13\%$ |
| CACHEUS | 3 / 15 / 3 | $+3.68\%$ |
| HALP | 1 / 17 / 3 | $+3.59\%$ |
| LRU | 1 / 16 / 4 | $+4.12\%$ |
| SIEVE | 4 / 14 / 3 | $+1.89\%$ |
| FIFO-Reinsertion | 2 / 16 / 3 | $+4.04\%$ |

Matching is evaluation-side. LRB, 3L-Cache, and CACHEUS adapt online; HALP
trains on each trace's own prefix; `evict_value_v1` is leave-one-family-out.
No policy sees future requests or another family's data.

Implementation fidelity is disclosed separately from that protocol (§3.4.4):
LRB and 3L-Cache are independent reimplementations (3L-Cache uses an untuned
default batch size); CACHEUS wraps the authors' engine, with the upstream
clone not live-verifiable here; HALP has no official public implementation
and is a lower-fidelity supporting comparison. The empirical ranking is
interpreted with those caveats.

**Changes in the revised manuscript**

- §3.4 / Table 5 (pp. 6–10): matched comparison
- §3.3 / Table 4 (p. 8): matched family-level LRU gaps
- §1.3 (p. 3) and Table 2 (p. 7): learned baselines located
- Abstract (p. 1) and contribution 2 (p. 2): negative result stated

**Supporting materials:**
[major1_reviewer_summary.md](../reports/kbs_final_evidence_20260813/major1_reviewer_summary.md)

---

### Major Comment 2 — Insufficient justification for finite-horizon eviction-loss supervision

**Reviewer concern** (recorded second-revision request)

> Insufficient justification for the finite-horizon eviction-loss
> supervision objective versus plausible alternatives.

**Response**

We agree the previous revision asserted preference for eviction-loss rather
than testing it. We now report two distinct experiments.

**A. Full-pipeline comparison (Table 6).** Under the same leave-one-family-out
protocol, we replace only the candidate-level label. Eviction-loss has the
most total misses of the four objectives (601,569), versus next-arrival
(573,059), reuse-distance (571,456), and pairwise preference (565,127). It
is worst or tied-worst in every family. This ranking does **not** isolate
the training objective as the sole causal factor, and the paper no longer
claims empirical superiority of eviction-loss.

The pairwise row (565,127) is from that full-pipeline ablation, which orients
training pairs by the stored preference label. It is not the superseded
common-model V1 control, which discarded those labels and is invalid.

**B. Matched Common-Model V2 (Table 7).** Architecture, features, folds,
windows, and seed are held fixed; only the objective changes, with corrected
pairwise orientation. Totals: eviction-loss 571,976; pairwise 577,339;
reuse-distance 615,850; next-arrival 627,392. Eviction-loss is not
materially worse. This control does **not** support blaming the
eviction-loss training objective for poor deployed performance.

The more strongly supported issue is target informativeness /
action-underdetermination (§3.6), not a uniquely harmful eviction-loss
label.

**Changes in the revised manuscript**

- §3.5 / Tables 6–7 (p. 10): pipeline ranking vs.\ matched control
- §3.8 (p. 13): design lesson is informativeness, not label uniqueness
- Abstract and contribution 3 (pp. 1–2)

**Supporting materials:**
[Common-Model V2 audit](../reports/common_model_v2_formal_audit_20260814/AUDIT.md)

---

### Major Comment 3 — Offline/online gap and speculative distribution-shift explanation

**Reviewer concern** (recorded second-revision request)

> Unexplained gap between offline learning quality and online cache
> performance; the distribution-shift explanation is speculative and
> untested.

**Response**

We agree the previous account was speculative. We replace it with controlled
diagnostics on the same 21 cells.

The learned scorer selects an element of the exact target-defined optimal
set in 97.53% of decisions. Because that optimal set contains approximately
99.1% of candidates on average, this metric is weak evidence of
discrimination among candidates; its main value is showing that the learned
policy rarely leaves the target-defined optimal set. It does not prove that
model fitting is satisfactory.

A *deterministic* exact horizon-4 oracle (minimum candidate identifier among
exact minimizers) loses to LRU on 18 of 21 cells (0 wins / 3 ties / 18
losses; $+81{,}750$ misses). That instantiation is a poor online policy. It
does not prove that the exact target intrinsically loses to LRU.

A tie-aware follow-up finds `fraction_tied_decisions = 1.0` on every
non-LRU oracle row (mean all-tied fraction $\approx 0.649$; mean optimal-set
fraction $\approx 0.991$). LRU-within-minima never loses to LRU (16 wins / 5
ties / 0 losses; $-413$ misses). MRU-within-minima matches the deterministic
deficit (0 / 3 / 18; $+89{,}135$). The old deterministic deficit was
tie-breaking-confounded: downstream choice within the large optimal set
matters substantially. This does **not** mean that $H=4$ improves LRU, that
the target adds proven value to LRU, or that the exact H4 target is
intrinsically better than LRU.

Independent degeneracy diagnostics agree qualitatively: no decision has a
unique optimum, and $P(T>4\mid\text{resident})=0.9939$. The replay-time
all-tied fraction ($\approx 0.649$) and the strict-preference all-tied rate
($76.4\%$) are two instruments of the same qualitative fact, not
interchangeable point estimates.

Continuation mismatch and generic shift correction are tested separately in
§3.7 (detailed under Reviewer #3). Continuation correction is partially
supported and regime-dependent. The tested DAgger-style intervention is a
negative result for miss ratio. That result does not imply that distribution
shift is absent.

**Changes in the revised manuscript**

- §3.6 (pp. 11–12): oracle, agreement, degeneracy, tie-aware control
- §3.7 (p. 12): continuation and DAgger
- §4.2 (p. 14): interpretation is constrained, not a single complete cause

**Supporting materials:**
[tie-aware oracle audit](../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md)

---

### Major Comment 4 — Practical significance

**Reviewer concern** (recorded second-revision request)

> The implementation is much slower per eviction decision than lightweight
> baselines while not outperforming them on miss ratio; practical
> significance is unclear.

**Response**

We replaced the earlier single-run local probe with a controlled campaign:
7 families $\times$ 3 capacities $\times$ 5 repetitions $\times$ 4 policies
(420 timed runs) for LRU, FIFO-Reinsertion, SIEVE, and causal HALP.

Means: LRU $4.68\,\mu$s/request; FIFO-Reinsertion $5.17\,\mu$s; SIEVE
$9.52\,\mu$s; HALP $870.66\,\mu$s ($\approx 186.02\times$ LRU).
`evict_value_v1` is not in that repeated campaign. Its available single-run
measurement from the matched evaluation is reported separately and is not
statistically comparable to Table 8.

Together with the negative miss-ratio results, the current method is not
deployment-competitive. No deployment scenario is claimed. The present
scientific value is diagnostic: a controlled test of why a representative
finite-horizon candidate-level target underperforms.

**Changes in the revised manuscript**

- §3.9 / Table 8 (p. 13): 420-run campaign
- §3.10 (pp. 13–14): no justified deployment scenario
- §4.2 (p. 14): single-run `evict_value_v1` timing caveat

**Supporting materials:**
[controlled_timing_summary.csv](../reports/kbs_final_evidence_20260813/controlled_timing_summary.csv)

---

## Response to Reviewer #3

### Primary issue — untested continuation-label causal claim

**Recorded second-revision concern.** The previous revision attributed part
of the offline/online gap to a mismatch between LRU-continuation labels and
the learned policy's deployed trajectory without a controlled test.

**Response**

We agree the claim required a test. The three-arm ablation isolates that
mechanism: **C0** is LRU; **C1** trains and deploys $\pi_1$ on
LRU-continuation labels; **C2** retrains $\pi_2$ on frozen-$\pi_1$
continuation and deploys $\pi_2$. That is the only C1$\to$C2 change, under
leave-one-family-out.

C2 improves on 13 cells, ties 3 (Wikimedia), and worsens 5 (macro mean
$-0.0102$; misses C1 $601{,}569 \to$ C2 $592{,}970$, vs.\ C0 $565{,}126$).
BrightKite at capacity 32 regresses $+0.2433$.

**Answer: partially supported and regime-dependent.** Continuation mismatch
is a real secondary contributor in most cells, not a full explanation, and
not uniformly beneficial.

A separate one-step DAgger-style correction reduces a generic shift metric
in 16 of 21 cells, yet misses improve in 2, tie in 3, and worsen in 16
(macro $+0.0094$; $591{,}604 \to 599{,}537$). That is a negative result for
this intervention, not a test of the continuation label itself.

**Changes in the revised manuscript**

- §3.7 (p. 12): C0/C1/C2 and DAgger
- §4.2–§4.3 (pp. 14–15): one-step scope; multi-round correction as future work
- Abstract and contribution 4 (pp. 1–2)

**Supporting materials:**
[c0_continuation_summary.csv](../reports/kbs_final_evidence_20260813/c0_continuation_summary.csv);
[distribution_shift_summary.csv](../reports/kbs_final_evidence_20260813/distribution_shift_summary.csv)

---

### Previously raised Issue 2 — insufficient differentiation from HALP

**Reviewer comment** (first-round letter)

> HALP (NSDI '23) already learns candidate-level preferences from future
> re-access outcomes and evicts the least-preferred candidate—operationally
> very close to what this paper proposes. The paper needs to clearly
> articulate what the finite-horizon replay target provides that HALP's
> preference signal does not, and ideally present direct empirical
> comparisons.

**Response**

Table 5 reports the empirical comparison: `evict_value_v1` loses to HALP on
17 of 21 cells (1 win / 3 ties; $+3.59\%$). No official public HALP
implementation was available, so an independent implementation was used.
That fidelity limitation is disclosed (§3.4.4); the comparison is
interpreted accordingly.

**Changes:** Table 5 (p. 9); §3.4.4 (p. 10); §1.3 (p. 3).

---

### Previously raised Issue 5 — computational cost

**Reviewer comment** (first-round letter)

> The method requires constructing a feature vector for every cached item at
> every full-cache miss, then running a trained model to predict losses for
> all candidates. The paper provides no wall-clock time analysis, no
> complexity comparison against O(1) baselines like LRU or SIEVE, and no
> discussion of production overhead.

**Response**

Addressed under Major Comment 4: §3.9 / Table 8 (p. 13).

First-round Issue 3 (SIEVE / FIFO-Reinsertion) is covered by the matched
comparison in Table 5.

---

## Summary

| Concern | Revision | Main location |
|---|---|---|
| R2 Major 1 | Matched comparison; negative | §3.4, Table 5 |
| R2 Major 2 | Pipeline does not favor eviction-loss; V2 does not blame the objective | §3.5, Tables 6–7 |
| R2 Major 3 | Mechanistic + causal tests; tie confound | §3.6–§3.7 |
| R2 Major 4 | 420-run timing; no deployment claim | §3.9–§3.10, Table 8 |
| R3 continuation | Partial, regime-dependent | §3.7 |
| R3 HALP (Issue 2) | Empirical comparison, disclosed fidelity | §3.4, Table 5 |
| R3 cost (Issue 5) | Controlled timing | §3.9, Table 8 |

The revised manuscript now reports the requested baseline comparisons,
objective controls, mechanistic and causal analyses, and controlled timing
evidence, while narrowing its claims where the new experiments do not
support the original interpretation.
