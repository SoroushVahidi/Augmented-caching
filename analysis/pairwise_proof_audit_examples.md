# Pairwise proof-audit examples for F1 (order vs magnitude)

This note gives tiny synthetic thought-experiments for testing hidden scalar-magnitude dependence.  
It is not a proof and does not certify theorem validity.

Related exploratory artifact: `analysis/pairwise_inversion_examples.md`.

---

## Example 1: same ranking, wildly different scalar magnitudes
- Setup at a miss with candidates `{a,b,c}`.
- Ranking in both models: `a` is best eviction, then `b`, then `c`.
- Model A scalar scores: `(a,b,c) = (0.51, 0.50, 0.49)`.
- Model B scalar scores: `(a,b,c) = (1000, 0, -1000)`.

**Observation:** eviction action is identical if rule uses only ranking.  
**Audit implication:** any proof step that changes under this transformation is magnitude-dependent, not order-only.

---

## Example 2: zero inversions but large scalar miscalibration
- Predictor preserves correct relative order at each miss (zero inversion relative to reference order).
- Scalar values are strongly miscalibrated in absolute terms (e.g., predicted distances all compressed or all inflated).

**Observation:** an order-only theorem should be insensitive to this miscalibration.  
**Audit implication:** if a candidate proof bound worsens because magnitudes are miscalibrated despite zero inversions, that step is not order-only.

---

## Example 3: one inversion with tiny magnitude gap
- Two top candidates are nearly tied in scalar value; one comparison flips.
- Inversion count increments by 1.

**Observation:** tiny scalar gap does not prevent order error; action can still change.
**Audit implication:** order-based error accounting can detect this even when magnitude-based residual seems tiny.

---

## Example 4: one inversion with huge magnitude gap
- Same inversion count as Example 3, but flipped pair has very large scalar separation.

**Observation:** inversion metric is identical to Example 3, but any magnitude-sensitive penalty may differ sharply.
**Audit implication:** if proof uses scalar gap to bound harm, this case may produce different constants than inversion-only accounting.

---

## Example 5: local inversion count fixed, downstream damage varies
- Reuse `analysis/pairwise_inversion_examples.md` pattern: sequences with similar local inversion statistics but very different extra misses.

**Observation:** inversion counts alone may not tightly control damage without extra structure.
**Audit implication:** motivates either (a) stronger/localized error metrics, or (b) structural assumptions in theorem statement.

---

## What these examples suggest for F1
1. **S1/S2-type lemmas** are plausibly order-only (action identity under same ranking).
2. **Cost-charge lemmas** are most likely magnitude-sensitive unless carefully reformulated.
3. A **hybrid theorem** (order term + explicit residual) is a realistic fallback if pure order-only derivation fails.
