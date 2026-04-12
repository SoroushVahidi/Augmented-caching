# Internal note: novelty positioning for eviction-value project

> **Internal research note only.**
> This is a project-memory document for team alignment.
> It is **not manuscript text** and **not canonical `heavy_r1` evidence**.

## 1) Purpose of this note

This note captures the current agreed novelty framing so future manuscript drafting starts from the same internal baseline.

Scope of this note:

- novelty positioning only,
- conservative claim boundaries,
- related-work risk control.

This is not a final paper claim set and not a replacement for canonical manuscript evidence files.

## 2) What we are NOT claiming

We should **not** claim the following as core novelty by themselves:

- candidate-level eviction,
- fallback / guard mechanisms,
- “new online decision rule” as the primary contribution,
- broad novelty over all candidate-scoring approaches.

Working primary framing remains:

- the main contribution direction is a **better training target** for eviction decisions.

## 3) Main conceptual contrast against Parrot

Internal contrast to maintain:

- On one side, **Parrot-style oracle imitation** framing: learning to imitate oracle actions/policies.
- On our side, a narrower target-design claim: explicit supervision on finite-horizon, candidate-specific downstream harm from eviction.

Conservative phrasing:

- We are not claiming oracle imitation is “wrong.”
- We are claiming our supervision target emphasizes a different object: candidate-level downstream harm over a finite horizon.

## 4) Why HALP must be treated as major related work

HALP should be treated as high-priority related work because it already occupies nearby conceptual territory:

- candidate-specific,
- future-aware scoring/ranking ideas.

Implication:

- our claim must be narrower and more precise than “future-aware candidate scoring is new.”
- manuscript language should distinguish **what target is supervised** (finite-horizon downstream eviction harm) rather than implying we introduced candidate-aware future sensitivity broadly.

## 5) Why Mockingjay also narrows broad novelty claims

Mockingjay also limits broad novelty claims because it uses richer per-candidate future estimates.

Implication:

- we cannot claim broad novelty on candidate scoring,
- we cannot claim broad novelty simply for moving beyond one oracle action,
- we should focus on the specific supervision-target formulation and its decision alignment.

## 6) Safest current novelty statement

Safest current internal statement:

> Prior work uses oracle imitation or candidate rankings/rewards derived from future outcomes; to the best of our knowledge, it does not explicitly train on finite-horizon, candidate-specific downstream eviction harm as the supervision target.

Supporting boundaries to keep attached:

- this is a **current best-effort** novelty hypothesis pending full related-work verification during manuscript drafting,
- claim scope is about supervision target definition, not universal performance dominance,
- LRU continuation is our default practical continuation rule for label generation, presented as an approximation choice (simple, deterministic, cheap, stable), not as a theoretically preferred continuation policy.

## 7) Risks if we overclaim

Main overclaim risks:

1. **Prior-art contradiction risk:** broad “first candidate scoring” or “first beyond oracle action” claims are likely unsafe.
2. **Reviewer trust risk:** framing fallback/guard or candidate-level eviction as novel can be read as ignoring known literature.
3. **Claim-evidence mismatch risk:** novelty wording may outrun current evidence if we imply theorem-level or universally superior outcomes.
4. **Narrative drift risk:** if we frame as a new online rule, we dilute the core target-design contribution.

## 8) Immediate implications for future manuscript positioning

Near-term positioning rules:

- center contribution text on **training-target design** and decision alignment,
- keep online-rule and fallback mechanisms as implementation/deployment context, not primary novelty,
- explicitly discuss HALP/Mockingjay/Parrot in related-work contrast paragraphs,
- attach caution labels when evidence is exploratory,
- keep canonical `heavy_r1` evidence boundaries explicit and separate.

## 9) Papers that must be handled carefully in related work

At minimum, manuscript drafting should treat these as high-attention citations:

- **Parrot** (oracle-imitation style framing pressure),
- **HALP** (candidate-specific future-aware scoring pressure),
- **Mockingjay** (richer per-candidate future-estimate pressure),
- **Raven** (Belady-guided learned caching pressure),
- **learning-augmented caching theory baselines** (for claim calibration),
- **robustification / GUARD-style lines** (to avoid overclaiming fallback novelty).

Internal reminder:

- During manuscript drafting, convert these labels into precise bibliographic entries and verify wording against the exact published claims.
