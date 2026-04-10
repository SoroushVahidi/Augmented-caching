# Pairwise theorem outline (design package, not a finished proof)

## Scope and status
This note proposes a **conservative target theorem** and a proof roadmap for pairwise learning-augmented caching. It is a theorem-design artifact only. It does **not** claim that a complete checked proof currently exists in the repository.

Relevant prior artifacts:
- `docs/pairwise_theory_roadmap.md` (targets/conjectures and proof-template discussion)
- `analysis/pairwise_inversion_examples.md` (examples showing both harmless and amplified damage from local inversions)

---

## Main target theorem (conservative first version)

### Candidate theorem T0 (restricted, unweighted paging)
Consider unweighted paging with cache size `k` and deterministic tie-breaking. On each full-cache miss, a predictor provides a **total eviction ranking** over current cached pages. Suppose this ranking is consistent (transitive) and can be evaluated for all current candidates.

Define an inversion-style error budget `E` over miss decisions (precise definition below). Then for a BlindOracle-style policy that evicts the top-ranked page (or a policy equivalent to it under ranking input), there exists a bound of the form

`Misses(pairwise_rank_policy) <= Misses(reference_policy) + Phi(E, k, T)`

for a monotone function `Phi` determined by the proof construction and assumptions. In the perfect-ordering special case `E = 0`, the policy agrees with the reference decisions under the same tie rules on the realized trajectory.

**Interpretation:** first theorem goal is not a sharp universal competitive ratio; it is a controlled excess-cost relation under explicit restrictions.

---

## Labeled proof roadmap and status checkpoints

### Step 1) Define ranking-based predictor formally
- **Goal:** Formalize predictor output as a strict/total order on current cache candidates at each miss; include deterministic tie rule.
- **Likely status:** **Likely already available** at conceptual level.
- **Why:** Existing roadmap already defines pairwise preference outputs and induced rankings (e.g., Copeland/Borda-style aggregation options).
- **Checkpoint artifact to produce:** a compact definitions block with symbols for decision times, candidate sets, ranking relation, and selected victim.

### Step 2) Define inversion-style error notion
- **Goal:** Specify one theorem-grade error metric (e.g., local inversion count relative to Belady/reference ranking at each miss; optional phase-normalized version).
- **Likely status:** **Likely needs adaptation**.
- **Why:** Inversion notions exist in roadmap and exploratory examples, but a theorem proof usually needs one precise canonical definition and accounting convention.
- **Checkpoint artifact to produce:** an error-budget definition that is unambiguous under trajectory coupling and tie handling.

### Step 3) Map ranking predictor to proof objects used in existing analysis
- **Goal:** Show how ranking advice can be translated into the advice objects required by existing BlindOracle/LA-style analysis machinery.
- **Likely status:** **Likely needs adaptation**.
- **Why:** Current proofs may expect scalar next-arrival-like predictions or specific cost surrogates; mapping from order-only objects must be explicit.
- **Checkpoint artifact to produce:** a lemma or interface translation: “ranking advice -> admissible proof object” with assumptions listed.

### Step 4) Identify where proof uses only order vs where it uses magnitudes
- **Goal:** Audit each key lemma to classify dependence on (a) relative ordering only, versus (b) scalar magnitude information.
- **Likely status:** **Likely open / uncertain**.
- **Why:** This is the highest-risk bridge. Emerging intuition suggests some Wei-style arguments may be order-driven, but this has not been checked line-by-line in a pairwise adaptation.
- **Checkpoint artifact to produce:** a table of lemmas: `order-only`, `magnitude-used`, `unclear`.

### Step 5) Derive base cost bound
- **Goal:** Obtain an excess-miss bound for the ranking-driven policy under the chosen inversion error budget and assumptions.
- **Likely status:** **Likely open / uncertain**.
- **Why:** Existing roadmap states target forms, but no finalized bound/proof is present.
- **Checkpoint artifact to produce:** theorem statement + full derivation under restricted assumptions (even if loose constants).

### Step 6) Derive combiner consequence (optional second stage)
- **Goal:** If Step 5 succeeds, derive a corollary for a fallback/combiner policy (e.g., pairwise + robust baseline switch/black-box combiner).
- **Likely status:** **Likely open / uncertain**.
- **Why:** Combiner guarantees often require additional switching/accounting properties beyond a learned-policy cost bound.
- **Checkpoint artifact to produce:** conditional corollary with explicit preconditions; otherwise defer to future work.

---

## Practical theorem-work checkpoints (recommended order)
1. Finalize Step 1 + Step 2 definitions in one short note.
2. Complete Step 4 audit before proving anything broad.
3. Prove a minimal Step 5 theorem in a restricted regime.
4. Attempt Step 6 only after Step 5 is stable.

## Non-goals for this first theorem
- Not aiming yet for the strongest possible competitive ratio.
- Not claiming coverage for weighted caching, shortlist restrictions, or non-transitive pairwise tournaments.
- Not claiming that local inversion count alone always tightly predicts damage (counterexamples already exist in exploratory analysis).
