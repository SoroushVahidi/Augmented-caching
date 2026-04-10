# Pairwise bounded-overlap residual theorem candidate

This is an internal theorem-development artifact. It is conservative and does not claim a proof.

## 1) Main candidate theorem (interaction-aware residual)

Consider unweighted paging with cache size `k`, deterministic tie-breaking, and a policy that outputs a transitive total ranking over the full current candidate set at each eviction decision. Couple the policy with Belady on the same trace.

At each charged decision event `e`, let `inv(e)` be the local inversion count. For a phase `P`, define:
- `E(P)`: excess misses of the policy vs Belady attributed to events in `P`.
- `L(P) = Σ_{e in P} inv(e)`: order-only local charge term.
- `R(P)`: explicit overlap/interference residual.

### Candidate theorem BO-T
There exists a simple structural residual `R(P)` and constant `c` such that, on the target class,

`E(P) <= L(P) + c * R(P)`.

This replaces the failed strong additivity claim `E(P) <= L(P)`.

---

## 2) Overlap / interaction quantity definitions

We currently track pair-event interaction using simple event features:
- shared affected page (`same_victim`),
- same phase (`same_phase`),
- near-time proximity (`Δt <= 2`).

A practical composite quantity is:

`interaction_depth = 1 + 1[same_victim] + 1[same_phase] + 1[Δt <= 2]`.

For a phase with many events, residual can aggregate this over interacting pairs.

---

## 3) Residual formalization options

### R1: shared-victim count
`R1(P) = # interacting pairs with same_victim`

- **Why plausible:** collisions on the same displaced page are a direct source of double counting.
- **Current evidence:** captures many interference cases but misses residual cases where victims differ.
- **Falsifier:** any residual>0 case with different victims (observed on tiny grid).

### R2: reinsertion-collision count
`R2(P) = # pairs with same_victim and near-time proximity`

- **Why plausible:** rapid reinsertion/re-eviction loops are a natural cascade trigger.
- **Current evidence:** identifies a strict subset of problematic interactions.
- **Falsifier:** residual>0 cases without this collision signature (observed on tiny grid).

### R3: interaction-depth score (recommended current candidate)
For each interacting event pair `(e_i,e_j)` in phase `P`, define:

`d(e_i,e_j) = 1 + 1[same_victim] + 1[same_phase] + 1[Δt <= 2]`.

Then set `R3(P) = Σ d(e_i,e_j)` over charged interacting pairs.

- **Why plausible:** combines page overlap + temporal overlap + phase overlap, so it tracks multiple interference channels.
- **Current evidence:** on current tiny grid, this score gives a finite small constant fit while simpler binary indicators can fail with no finite constant.
- **Falsifier:** counterexamples where residual grows faster than any fixed `c * R3` on expanded grids.

---

## 4) Current empirical reading

- Strong L3 additivity is already broken on tiny-grid search.
- Residualized bounds remain plausible if the residual captures interaction structure.
- Among tested simple quantities, **R3 (interaction-depth)** is currently the most practical candidate.

This is still empirical guidance, not proof.

---

## 5) Fallback hierarchy

1. **Bounded-overlap residual theorem (BO-T)** with explicit interaction residual.
2. If BO-T fails broadly, use **reuse-gap-separated theorem**.
3. If both fail, move to **hybrid theorem with scalar residual**.

---

## 6) What would most improve confidence next

1. Minimize super-additive counterexamples for documentation and mechanistic diagnosis.
2. Stress-test `R3` on larger tiny grids and confirm constant stability.
3. If `R3` destabilizes, add one more structural component (e.g., recoupling delay) before conceding to full hybrid scalar residual.
