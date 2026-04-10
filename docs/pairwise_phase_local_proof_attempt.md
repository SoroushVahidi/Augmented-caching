# Pairwise phase-local proof attempt (internal)

This is a conservative proof-attempt artifact, not a completed theorem proof.

## 1) Candidate theorem statement (phase-local)

We work in unweighted paging with cache size `k`, deterministic tie-breaking, and a policy that induces a transitive total ranking over the current full candidate set at each eviction decision.

Let `OPT` denote Belady's policy on the same request trace. At each eviction decision `t` where candidate set `C_t` is identical in the coupled states, define `inv_t` as the number of pairwise inversions between the policy ranking and Belady's ranking on `C_t`.

Fix a phase decomposition into `k`-phases. Define `E_loc` as the number of **phase-local excess misses**, i.e., misses by the policy at time `u` where `OPT` hits at `u` and where the divergence can be attributed to an inversion event from the same phase (no cross-phase attribution).

### Phase-local candidate theorem (PL-T)
Under the above assumptions and restricted to traces/decision events satisfying phase-local attribution,

`E_loc <= Σ_t inv_t`.

The theorem does **not** claim control of cross-phase spillover; those are residual terms outside PL-T.

---

## 2) Lemma plan and status

### Lemma L1 (well-defined local inversion metric on coupled states)
**Claim:** If coupled states have identical candidate sets and ties are deterministic, `inv_t` is well-defined and stable for each decision.

**Status:** **proved from current materials**.

**Reasoning/source alignment:** this is a definitional/assumption-level lemma already aligned with existing theorem assumptions and proof-audit scaffolding.

---

### Lemma L2 (single-event local charge inside phase)
**Claim:** For one inversion event at decision `t` with no additional inversion events in the same phase and no cross-phase spillover, the resulting phase-local excess is at most that event's inversion count.

**Status:** **plausible but unproved**.

**Why plausible:** tiny brute-force evidence shows many one-inversion top-swap events incur +0 or +1 excess and remain same-phase local.

**What is still missing:** a formal coupling argument that maps any same-phase extra miss to a unique inversion-causing displacement without double-counting.

---

### Lemma L3 (additivity across multiple phase-local inversion events)
**Claim:** If all attributed excess misses in a phase are local to inversion events in that phase, total phase-local excess is bounded by the sum of per-event inversion charges.

**Status:** **currently blocked**.

**Missing statement:** a non-interference lemma ensuring per-event charged regions inside a phase do not overlap in a way that allows one extra miss to be "caused" by multiple prior inversions.

**Potential falsifier:** a same-phase trace where two small inversions interact so that one displaced page causes repeated misses that cannot be injectively assigned to one inversion charge.

**Weaker replacement:** partition each phase into verified non-overlap windows (or decision blocks) and prove additivity block-wise plus a residual overlap term.

---

### Lemma L4 (phase boundary containment)
**Claim:** Under phase-local attribution assumptions, all charged divergences terminate by phase end and do not require cross-phase accounting.

**Status:** **currently blocked**.

**Missing statement:** a boundary-reset/coupling-recovery lemma: by phase end, displaced set difference between policy and OPT is neutralized for charged events.

**Potential falsifier:** a one-inversion event near phase end that preserves a far-future page and pushes the harm into next phase despite no further inversions.

**Weaker replacement:** retain PL-T only for decisions with sufficient phase slack (distance to phase end >= `h`), and move boundary spillover to explicit residual.

---

### Lemma L5 (phase-local theorem composition)
**Claim:** If L2-L4 hold, then `E_loc <= Σ_t inv_t` for phase-local-attributed excess.

**Status:** **plausible but unproved** (depends on blocked L3/L4).

---

## 3) Blocked-step register (explicit)

### Block B1 (non-interference/additivity)
- **Missing statement:** injective mapping from same-phase excess misses to inversion charges across multiple inversion events.
- **How to falsify:** construct two-inversion same-phase trace where total local excess exceeds total inversion count despite no cross-phase spillover.
- **Fallback that keeps theorem alive:** block-wise additivity with an overlap residual term.

### Block B2 (phase-boundary containment)
- **Missing statement:** charged divergence cannot survive phase boundary under candidate assumptions.
- **How to falsify:** boundary-adjacent inversion where first extra miss appears in next phase.
- **Fallback that keeps theorem alive:** theorem with phase-end residual or with a minimum slack condition.

---

## 4) Current verdict

**Outcome bucket: (2) phase-local theorem still has one critical unresolved step.**

Operationally, two blocks remain (B1/B2), but the single most critical unresolved lemma is **L3 non-interference/additivity**. If L3 fails, local charging loses compositionality even before boundary issues.

## 5) Next action recommendation inside theory workflow

1. Attempt a direct proof (or falsification search) for **L3 non-interference** on tiny exhaustive traces with at least two inversion events in one phase.
2. If L3 fails, immediately pivot to the weaker block-wise theorem with explicit overlap residual.
3. Keep phase-boundary spillover as a second residual channel unless a clean boundary-reset lemma is obtained.
