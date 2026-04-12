# Internal prior-work comparison matrix: eviction-value project

> Internal positioning note only (not manuscript prose, not canonical `heavy_r1` artifact).

## 1) Purpose of this note

- Preserve a stable internal record of current related-work/novelty understanding for eviction-value.
- Keep manuscript drafting aligned on **safe claim scope** and avoid narrative drift.

## 2) Comparison matrix

| Line / paper family | Core supervision target | Oracle imitation? | Next-arrival / reuse-distance prediction? | Pairwise preference / reward learning? | Candidate-level scoring? | Explicit finite-horizon downstream harm? | Explicit per-candidate counterfactual rollout labels? | Main novelty risk to our current claim | Why we still differ (current framing) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **Our current eviction-value framing** | Candidate-specific downstream eviction harm over a finite horizon | Partial contrast (not primary) | Not primary | Can be auxiliary, not core claim | Yes | **Yes (core)** | **Yes (core)** | Risk if phrased as “new online rule” instead of target-design novelty | Main contribution framed as **training target design**; continuation (LRU) is practical approximation, not theory claim |
| **PARROT** | Learn policy/action from oracle demonstrations | **Yes (main contrast)** | Not central | Not primary | May score candidates implicitly/explicitly | Not explicit as primary target (per current internal read) | Not explicit as primary target | Strong prior for “learning from future-optimal behavior” and imitation framing | Our target is not “imitate oracle action”; it is direct supervision on finite-horizon candidate harm |
| **HALP** | Heuristic-aided learned preference/eviction scoring | Not primary | Not primary | **Yes (major)** | **Yes** | Future-aware, but explicit finite-horizon harm target must be distinguished carefully | Counterfactual rollout labeling not clearly central in current internal understanding | Major risk against broad claims like “candidate future-aware scoring is new” | We must claim narrow target-level distinction, not broad candidate-awareness novelty |
| **LRB** | Belady-relaxed/Belady-derived learning signal for eviction | Related | Belady/future-distance style signals | Possibly ranking-style | Yes | Not our explicit harm target | Not clearly our rollout-label formulation | Narrows claims around Belady-guided learned eviction novelty | Different supervised object: direct finite-horizon per-candidate downstream harm |
| **Raven** | Belady-guided predictive learned caching | Related | Strong future-prediction orientation | May include learned scoring | Yes | Not explicit as our target | Not clearly our rollout-label formulation | Narrows broad novelty around predictive learned eviction | Our claim must stay on explicit harm labels, not “predictive caching is new” |
| **Mockingjay** | Rich candidate-aware future estimate/scoring | Not primary | Future estimate heavy | Could be ranking/score-driven | **Yes** | Future-aware but broad; explicit finite-horizon harm target distinction still needed | Not clearly our rollout-label formulation | Narrows “candidate-level + future-aware” novelty claims | We should differentiate by exact supervised label semantics (finite-horizon eviction harm) |
| **Learning-augmented caching theory line** | Advice quality / robustness / competitive framing | Not typically | Often arrival/reuse advice formulations | Usually not pairwise reward learning | Sometimes indirect | Generally not explicit finite-horizon harm labels | No | Narrows broad “ML + caching novelty” claims | Provides theory context; our framing is empirical target design for per-candidate harm supervision |
| **RL-based cache eviction line (if present in repo discussions)** | Reward-maximization (MDP/offline RL) | No | May use state features with long-horizon return | **Yes (reward/return)** | Yes | Often implicit via returns, not explicit supervised harm label | Typically no direct supervised counterfactual candidate rollout label table | Could collapse our story into generic reward learning if phrased loosely | Our training signal is explicit supervised finite-horizon candidate harm labels, not full RL objective optimization |

## 3) Safest current novelty sentence

- **Safest internal sentence:** prior work already covers oracle imitation, next-arrival/reuse-distance prediction, Belady-derived labels, pairwise preference learning, and RL-style reward learning; what still appears under-explored is training directly on **finite-horizon, candidate-specific downstream eviction harm**.

## 4) What we must not claim

- “Candidate-level eviction is our novelty.”
- “Guard/fallback mechanisms are our novelty.”
- “We introduce a fundamentally new online decision rule” (as primary contribution framing).
- Broad “first learned future-aware candidate scoring” claims.
- “Finite-horizon local labels are the exact global objective.”

## 5) Papers to cite prominently

- PARROT (main oracle-imitation contrast).
- HALP (major candidate-specific future-aware comparator).
- LRB.
- Raven.
- Mockingjay.
- Learning-augmented caching theory line (Lykouris–Vassilvitskii, Wei, Antoniadis et al., Chłędowski et al.).
- RL-based cache eviction papers already used in repo docs/manuscript discussion.

## 6) References to verify / add

- If bibliography is still not committed, keep these as verification items rather than inventing uncertain BibTeX entries:
  - PARROT.
  - HALP.
  - LRB.
  - Raven.
  - Mockingjay.
  - Learning-augmented caching theory anchors (Lykouris–Vassilvitskii; Wei; Antoniadis et al.; Chłędowski et al.).
  - RL-based cache eviction references already mentioned in repo notes.

Maintenance reminder:
- Update this matrix after any major related-work pass so future manuscript drafts inherit stable claim boundaries.
