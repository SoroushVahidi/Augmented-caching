# Internal research summary: eviction-value direction (working note)

> **Status:** Internal research notes only.  
> **Not manuscript text.**  
> **Not canonical `heavy_r1` evidence.**

## 1) Purpose of this note

This note preserves our current project understanding so future manuscript writing can start from a written internal record.

This is a coordination artifact, not a claim of final results.

## 2) Current conceptual position

Our current working view is:

- The strongest novelty is probably **not** candidate-level eviction by itself.
- The strongest novelty is probably **not** fallback/guard mechanisms by themselves.
- The strongest novelty hypothesis right now is the supervision design:
  **finite-horizon counterfactual supervision for candidate-level eviction-value prediction**.

At eviction time, the real action is: **which resident page is safest to remove now**.

## 3) Why eviction-value supervision is more decision-aligned

We can frame several existing label types as useful but indirect tools:

- next-arrival prediction,
- Belady imitation,
- binary good/bad labels,
- preference/pairwise labels.

What we directly care about is downstream harm from the eviction decision.

So the decision-aligned local target is:

- “If I evict candidate page `v` now, how much downstream harm occurs over the next `H` requests?”

Important scope control:

- This is **not** the exact full long-run global objective.
- It is a **local proxy** for downstream harm that is better aligned to the immediate eviction choice.

## 4) Horizon choice and data availability

Horizon selection appears to be a learnability tradeoff (bias-variance style):

- **Shorter horizon:** simpler target, easier to learn with limited data, but may miss longer-range effects.
- **Longer horizon:** richer target, but noisier/harder to learn and more data hungry.

Working implication:

- Horizon should be selected empirically based on available data quality/quantity, not by principle alone.

## 5) Continuation-policy issue in rollout labeling

In rollout labeling, continuation policy matters.

- Most faithful in principle: continue with the same learned algorithm.
- Practical issue: this makes labels recursive, moving, and expensive.

Why fixed continuation (e.g., LRU) is attractive:

- cheap,
- deterministic,
- stable across retrains,
- easy to explain.

Working conclusion:

- The “right” continuation policy should be treated as an empirical question and tested in lightweight comparisons.

## 6) Recent lightweight exploratory findings

> These findings are exploratory only and are **not** canonical manuscript evidence.

### A) History-context ablation (exploratory)

Observed outcome in discussion summary:

- small/inconsistent benefit,
- test top-1 match did not improve,
- test mean regret did not improve,
- downstream replay was mostly ties with a couple of wins.

Interpretation:

- neutral to slightly supportive,
- not strong evidence for broad benefit at this stage.

### B) Hybrid fallback experiment (exploratory)

Observed outcome in discussion summary:

- no material gain over pointwise baseline in tested setup,
- held-out totals were identical,
- selected thresholds were `0.0`,
- held-out trigger frequency was `0.0` (fallback never fired).

Interpretation:

- neutral to mildly weakening for the claim that this simple confidence fallback currently helps.
- still informative: deployment-rule calibration can dominate realized benefit.

## 7) Prior-work positioning and safest current novelty framing

Working prior-work conclusion:

- candidate-level eviction choices already appear in prior work,
- fallback/guard ideas also have prior art.

Safest under-explored angle to emphasize internally:

1. explicit candidate enumeration,
2. finite-horizon counterfactual rollouts,
3. candidate-specific downstream cost/value labels,
4. supervised learning of those labels.

Safe framing direction:

- We are **not** claiming “first ML for eviction” or “first guard”.
- We are exploring a formulation of eviction as candidate-level finite-horizon value prediction with supervision from counterfactual replay, instead of relying only on next-arrival prediction, Belady imitation, binary labels, or offline-RL-style rewards.

## 8) What we should NOT claim

Do **not** claim:

- first use of ML for eviction,
- first candidate-level eviction framing,
- first guard/fallback in caching,
- proven global optimality from finite-horizon local labels,
- exploratory lightweight outcomes as canonical manuscript evidence.

## 9) Immediate implications for future work

Near-term priorities:

- preserve and refine conceptual framing,
- keep exploratory evidence clearly separated from canonical `heavy_r1` artifacts,
- run targeted lightweight empirical checks where framing uncertainty exists (horizon and continuation policy),
- defer manuscript-strength claims until stronger evidence is assembled.

## References / papers to verify and track (families)

Track and verify positioning against these clusters mentioned in discussion:

- PARROT
- HALP
- Cold-RL
- Raven
- LRB
- learning-augmented caching theory papers
- robustification / GUARD-style work
- decision-focused optimization / SPO-style literature

(Internal action item: convert these family labels into precise citations when manuscript drafting begins.)

---

### Boundary reminder

This note is a project-memory artifact.
It should not be cited as canonical experiment evidence or manuscript text.
