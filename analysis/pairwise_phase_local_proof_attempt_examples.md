# Phase-local proof-attempt examples (targeted)

This note isolates tiny examples relevant to the phase-local candidate theorem in `docs/pairwise_phase_local_proof_attempt.md`.

## Scope
- Focus: whether phase-local excess appears to obey local inversion charge.
- Data source: existing Step-4 tiny-grid search and prior inversion examples.
- Conservative interpretation only; no theorem claim.

## Examples consistent with phase-local theorem candidate

1. **One inversion, same-phase +1 local damage**
   - Example: `ABCA`, `k=2` (top-swap perturbation at the single eviction decision).
   - Observed: inversion count `1`, extra misses `+1`, extra damage remains in inversion phase.
   - Why relevant: matches candidate bound `E_loc <= Σ inv_t` in the cleanest nontrivial case.

2. **One inversion, zero local damage**
   - Example: `AABC`, `k=2` (top-swap perturbation).
   - Observed: inversion count `1`, extra misses `0`.
   - Why relevant: shows local inversion can be harmless, consistent with an upper-bound style theorem.

## Near-counterexamples / cautionary patterns

1. **Boundary-adjacent risk pattern (not yet a violation in current grid)**
   - Large-gap example such as `DDDDDDCBD`, `k=2`, shows +1 damage with large reuse-gap and late decision index.
   - Interpretation: still bounded locally in current search, but it stresses the phase-boundary containment lemma.

2. **Known nonlocal cascade risk from broader inversion setting**
   - Prior brute-force file shows single-step local inversion can yield larger total damage when inversion-like errors repeat across decisions (e.g., repeated wrong evictions).
   - Interpretation: this does not directly refute the phase-local candidate, but it motivates explicit non-interference and boundary controls.

## Structural separator hypothesis (harmless vs dangerous)

Current tiny evidence suggests the key separator is **state re-coupling speed inside the phase**:
- **Harmless/local:** displaced item is either not reused soon or re-coupling occurs before additional same-phase divergence accumulates.
- **Dangerous:** displaced item is reused under continued divergence (especially with multiple nearby inversion events), making additivity/non-overlap hard.

This points directly to the unresolved proof need: a non-interference lemma for multiple inversion events in one phase.
