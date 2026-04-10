# L3 attack plan: non-interference / additivity across phase-local events

This is an internal theorem-attack artifact. It is conservative and does not claim a completed proof.

## Context
In the phase-local proof attempt, the critical unresolved step is L3: additivity across multiple inversion events in one phase.

## Strongest plausible L3 statement

### L3-S (strong form)
Fix one phase. Suppose eviction decisions in that phase are coupled to Belady on identical candidate sets, with deterministic tie-breaking and transitive total rankings. Let `e` index inversion events in that phase and let `inv(e)` be local inversion count at event `e`.

Let `E_phase` be total phase-local excess misses attributed to events in that phase.

**Claim L3-S:**

`E_phase <= Σ_e inv(e)`.

This is the clean non-interference/additivity form needed for a clean phase-local theorem.

- **Why it might be true:** if each event induces a disjoint or injectively chargeable divergence footprint, per-event local charges sum without overlap.
- **Current support:** many two-event tiny cases are exactly additive (`extra_ij = extra_i + extra_j`) in current brute force.
- **What kills it:** any pair where each event alone is bounded but together exceed additive charge (super-additive interference).

## Weaker L3 candidates

### L3-W1: bounded-overlap additivity
**Statement:** if event footprints overlap only up to `B` shared affected pages/requests, then

`E_phase <= Σ_e inv(e) + B`.

- **Why plausible:** overlap creates double-count risk; explicit overlap budget can absorb it.
- **Current support:** overlap-subset statistics can be tracked and often stay near additive on small cases.
- **Counterexample type:** cases with tiny measured overlap but large super-additive excess beyond any fixed small `B`.

### L3-W2: separated-return-times additivity
**Statement:** if consecutive inversion events satisfy return-time/event-time separation `gap >= Δ`, then strict additivity holds:

`E_phase <= Σ_e inv(e)`

on that restricted class.

- **Why plausible:** spacing allows re-coupling before the next inversion event interacts.
- **Current support:** separated-time subset has lower super-additivity rate than unrestricted in current search.
- **Counterexample type:** well-separated events that still produce super-additive joint excess.

### L3-W3: one-shot local-events form
**Statement:** if each inversion event alone causes at most one same-phase excess miss and recouples before the next inversion event, then

`E_phase <= number_of_events`.

(Equivalent to additivity with `inv(e)=1` events under this regime.)

- **Why plausible:** this is a direct non-overlapping one-shot charging model.
- **Current support:** many one-event and some two-event traces match this pattern.
- **Counterexample type:** two one-shot singles that merge under composition and exceed the sum.

## Dependency graph for theorem decisions

1. **If L3-S holds:**
   - phase-local theorem can survive in a clean form (modulo boundary lemma).
2. **If L3-S fails but some L3-W* holds:**
   - restricted phase-local theorem survives with explicit restriction/overlap residual.
3. **If all L3 versions fail:**
   - move to reuse-gap-separated theorem as next backup path.

## Current attack verdict from tiny brute force

- Current L3 search found super-additive cases for the strong test condition.
- Therefore **L3-S is currently empirically falsified on the searched tiny grid** (not a formal impossibility proof, but enough to stop clean-L3 assumptions).
- Best currently plausible fallback is **L3-W1 (bounded-overlap with explicit residual)**; strict separation-only additivity (L3-W2) remains uncertain on current tiny-grid evidence.

## Immediate next steps

1. Extract minimal explicit counterexample traces for L3-S and record their interaction mechanism.
2. Re-run search with stricter separation filters to test whether L3-W2 remains stable.
3. If L3-W2 also breaks broadly, pivot phase-local path toward explicit residual and prioritize reuse-gap-separated theorem.
