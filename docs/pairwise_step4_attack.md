# Pairwise Step-4 attack plan

This note is conservative and exploratory. It does **not** claim a theorem proof.

## Target claim for Step 4 (strongest order-only form)

### S0 (strongest)
For unweighted paging with deterministic tie-breaking, couple a comparator policy to Belady at each eviction decision `t` on the current candidate set. Let `inv_t` be the number of pairwise ranking inversions between comparator and Belady over that candidate set. Let `extra` be total excess misses of the comparator versus Belady on the full trace.

**Claim S0:**

`extra <= Σ_t inv_t`.

Interpretation: each local inversion pays at most one unit of eventual excess cost, with no scalar-distance residual.

- **Why it might be true:** if every wrong local swap creates at most one additional miss event before coupling “self-heals,” local charge can telescope.
- **Current supporting evidence:** many tiny examples with one inversion have +0 or +1 extra misses (`analysis/pairwise_inversion_examples.md`).
- **What falsifies it:** any trace with `extra > Σ_t inv_t`; especially a single-inversion case with +2 or more misses.

## Weaker candidates

### W1: shortlist-only local charge
Restrict the claim to deployments where only a shortlist (e.g., top-`m`) determines eviction, and inversion accounting is computed only inside that shortlist.

**Statement:**

`extra <= Σ_t inv_t(shortlist)`

for decisions where eviction is chosen from the shortlist and non-shortlist order cannot affect the chosen victim.

- **Why it might be true:** lower-rank permutations may inflate global inversion counts without changing action, so shortlist-local accounting may align better with realized damage.
- **Current artifact support:** Step-4 brute force includes lower-rank swaps with zero damage, highlighting shortlist/full-order mismatch (`analysis/pairwise_step4_attack_examples.md`).
- **What falsifies it:** shortlist-internal inversion cases where damage still exceeds shortlist inversion charge.

### W2: reuse-gap-separated local charge
Restrict to states where the two swapped contenders at inversion time have sufficiently separated next-use distances.

**Statement:** there exists a threshold `Δ` such that when each charged inversion satisfies `gap_t >= Δ`,

`extra <= C(Δ) * Σ_t inv_t`

with smaller residual/cascade terms than the unrestricted case.

- **Why it might be true:** larger reuse-gap may reduce ambiguity and stabilize local coupling after a wrong eviction.
- **Current artifact support:** gap-separated subsets often show cleaner behavior than unrestricted subsets in tiny search; not universal yet (`analysis/pairwise_step4_attack_examples.md`).
- **What falsifies it:** persistent violations with large gaps across deterministic exhaustive tiny grids.

### W3: phase-local charge (bounded cascade window)
Charge only inversions whose induced extra misses stay inside the same request phase (or a bounded next `h` decisions).

**Statement:** for phase-local inversion events,

`extra_phase_local <= Σ_t inv_t(phase-local)`

and non-local spillover is handled as explicit residual.

- **Why it might be true:** if divergence does not cross phase boundaries, local amortization has less state-drift risk.
- **Current artifact support:** tiny examples include one-inversion/+1 cases where all extra damage stays in the inversion phase (`analysis/pairwise_step4_attack_examples.md`).
- **What falsifies it:** same-phase restricted cases with repeated `extra > inversion charge`.

## Decision tree

1. **Attempt S0 first** (`extra <= Σ inv_t`).
2. If S0 fails, test **W1 (shortlist-only)** on action-relevant inversions.
3. If W1 fails, test **W3 (phase-local)** and/or **W2 (reuse-gap-separated)** to isolate when local charging still works.
4. If all restricted order-only forms fail on exhaustive tiny search, adopt the **hybrid theorem**:
   - order-only charged term + explicit residual term for cascades/state divergence.

## Practical success criteria for this attack package

- If no S0 counterexample appears on current exhaustive tiny grid: keep S0 as plausible but unproven.
- If S0 counterexamples appear but restricted forms survive on their scoped subsets: promote the strongest surviving restricted theorem.
- If S0 and restricted forms all show robust violations: commit to hybrid theorem as primary target.
