# Internal prior-work audit: eviction-value / decision-aligned supervision

> **Internal research-management note** (not manuscript text).  
> **Not canonical `heavy_r1` evidence.**  
> Purpose: track prior-work coverage and bibliography gaps for the current eviction-value framing.

## 1) Purpose and scope

This note audits how well the repo currently covers key prior work for our current safest framing:

- not “candidate-level eviction is new,”
- not “guards/fallbacks are new,”
- but **finite-horizon counterfactual supervision for candidate-level eviction-value prediction**.

The goal is project hygiene: identify what is already captured vs what still needs explicit bibliography and manuscript-ready citation support.

## 2) Bibliography-gap check (repo-level)

Current repo state: there is **no committed `.bib` bibliography database** at repo root or under manuscript-support paths; this is also explicitly documented in `reports/kbs_reference_usage_note.md`.

Working implication:

- “status in repo bibliography” is generally **missing** unless/until a real BibTeX file is added.
- Some papers are still documented in prose inside docs/code comments and reference-usage notes.

## 3) Audit of required papers/families

Legend used below:

- **Bibliography status:** present / missing / unclear
- **Docs/manuscript status:** cited / mentioned only / absent

### A. Learning-augmented caching theory + robustification baselines

| Paper / family | Bibliography status | Docs/manuscript status | Relevance to current story | Why it matters now |
|---|---|---|---|---|
| **Lykouris & Vassilvitskii (Competitive Caching with Machine Learned Advice)** | **missing** (no `.bib` file) | **mentioned only** (mainly source comments/policy context; no explicit doc citation line found) | theory baseline; novelty-risk control | Establishes classical LA-caching foundation; prevents overclaiming “ML+eviction novelty.” |
| **Wei (Better and Simpler Learning-Augmented Online Caching)** | **missing** | **cited** in baseline and reference-usage docs | theory/algorithm baseline; empirical baseline context | Already used to justify BO/LRU combiner framing and related baseline identity. |
| **Antoniadis et al. (Online Metric Algorithms with Untrusted Predictions)** | **missing** | **cited** in baseline and reference-usage docs | theory baseline; robust-prediction context | Supports trust/doubt baseline and untrusted-prediction framing. |
| **Chłędowski et al. (Robust Learning-Augmented Caching: An Experimental Study)** | **missing** | **cited** in baseline docs | empirical robustification baseline; novelty-risk control | Shows robust switching/fallback ideas have strong prior art. |

### B. Systems / learned-eviction cluster requested for tracking

| Paper / family | Bibliography status | Docs/manuscript status | Relevance to current story | Why it matters now |
|---|---|---|---|---|
| **PARROT (imitation-learning cache replacement)** | **missing** | **mentioned only** (internal summary list; no dedicated citation note) | systems prior; novelty-risk control | Candidate-level learning-for-eviction precedent risk. |
| **LRB (Learning Relaxed Belady for CDN caching)** | **missing** | **mentioned only** (internal summary list) | systems prior; empirical benchmark context | Close practical prior for Belady-guided learned eviction/value ideas. |
| **Raven (Belady-guided predictive deep learning for caching)** | **missing** | **mentioned only** (internal summary list) | systems prior; novelty-risk control | Very close in “Belady-guided learned caching” neighborhood. |
| **HALP (Heuristic Aided Learned Preference Eviction Policy)** | **missing** | **mentioned only** (internal summary list) | systems prior; preference-learning proximity | Direct relevance to preference-based eviction learning story. |
| **GUARD-style robustification work** | **missing** | **mentioned only** (internal summary list / broad wording) | robustification prior; novelty-risk control | Directly overlaps with fallback/guard claims. |
| **Cold-RL / offline RL for cache eviction** | **missing** | **mentioned only** (internal summary list) | conceptual contrast (RL reward vs supervised counterfactual labels) | Helps define what our method is *not* (not framed as full offline RL). |

### C. Decision-focused optimization framing

| Paper / family | Bibliography status | Docs/manuscript status | Relevance to current story | Why it matters now |
|---|---|---|---|---|
| **Decision-focused / SPO references** | **missing** | **mentioned only** in decision-aligned notes (conceptual mention, not fully pinned citations) | conceptual framing support | Backs the claim that prediction error and decision quality can diverge; supports decision-aligned supervision language. |

## 4) References to verify and add

Do not invent BibTeX blindly. Add these only after metadata verification:

1. Lykouris & Vassilvitskii — *Competitive Caching with Machine Learned Advice*.
2. Wei — *Better and Simpler Learning-Augmented Online Caching*.
3. Antoniadis et al. — *Online Metric Algorithms with Untrusted Predictions*.
4. Chłędowski et al. — *Robust Learning-Augmented Caching: An Experimental Study*.
5. PARROT — *An Imitation Learning Approach for Cache Replacement*.
6. LRB — *Learning Relaxed Belady for CDN Caching*.
7. Raven — *Belady-Guided Predictive Deep Learning for Caching*.
8. HALP — *Heuristic Aided Learned Preference Eviction Policy*.
9. GUARD-style robust caching with predictions (exact citation to pin).
10. Cold-RL / offline RL for cache eviction (exact citation to pin).
11. Decision-focused / SPO family items already used conceptually in decision-aligned notes (exact papers to pin).

## 5) What our current claim must avoid

We should avoid saying or implying:

- “first ML method for eviction,”
- “first candidate-level eviction formulation,”
- “first robust guard/fallback approach in caching,”
- “finite-horizon local labels are the exact global objective.”

## 6) What our current claim can still safely emphasize

Given current repo framing and evidence separation, we can safely emphasize:

- a **decision-aligned local supervision formulation** for eviction,
- explicit **candidate enumeration + finite-horizon counterfactual replay**,
- **candidate-specific downstream harm/value labels**,
- supervised training against those labels as a practical alternative to relying only on next-arrival prediction, Belady imitation, binary labels, or offline-RL reward shaping.

## 7) Biggest novelty-risk papers/families to track first

From the requested set, the highest novelty-risk pressure on our current framing appears to be:

1. **PARROT / HALP / Raven / LRB cluster** (closest systems-level learned-eviction neighbors).
2. **GUARD-style robustification line** (overlap with fallback/guard positioning).
3. **Lykouris–Vassilvitskii + Wei + Antoniadis + Chłędowski** (must-anchor LA-caching baseline/theory context so novelty claims remain precise).

---

### Maintenance note

This file is a living internal audit. Update statuses after:

- adding a real bibliography file,
- pinning exact citations for currently “mentioned only” entries,
- and clarifying where each item appears in manuscript-facing text.
