# ACM TIST positioning note (working draft)

## Purpose of this note
This document is a **manuscript-support scaffold**, not a final paper narrative. It is meant to keep positioning aligned with the current repository evidence while experiments and theorem work continue.

## Problem framing: an intelligent systems + ML + algorithm design contribution
The current project is best framed as a **learning-augmented caching systems paper** with three coupled layers:

1. **Algorithmic decision structure**: online eviction decisions with explicit fallback/combiner rules in the policy layer (e.g., pairwise + LRU black-box combiner).  
2. **ML supervision design**: candidate comparison supervision (pairwise) versus absolute candidate scoring (pointwise).  
3. **Deployment logic in an online simulator**: downstream miss-rate behavior depends not only on scorer quality, but also on *how* learned signals are integrated (unguarded, guarded, combiner, shortlist).

This is exactly the kind of mixed intelligent-systems contribution that fits TIST: practical algorithm design informed by ML, evaluated through decision-level and online-system outcomes.

## Why this is not best framed as a pure theory paper (yet)
- The theorem path is currently documented as exploratory roadmap/conjectures rather than finished guarantees. See `docs/pairwise_theory_roadmap.md` and the inversion examples note.  
- Existing inversion analyses include counterexamples and open obstructions, which is useful and honest, but not sufficient for a theory-primary manuscript claim.  
- There is no finalized competitive-ratio style theorem with full assumptions, proof, and robustness scope in-repo yet.

## Why this is not best framed as a pure ML paper
- The work is not mainly about introducing a new generic model architecture; instead it is about **decision-coupled learning signals** in an online caching policy context.  
- Current results indicate deployment rule design can dominate marginal scorer tweaks in some local regimes (combiner effect), which is an algorithm-systems message rather than a standalone predictive-modeling message.  
- Evaluation artifacts are policy/outcome centric (misses, downstream replay, fallback behavior), not just offline prediction metrics.

## Candidate title directions (working)
1. **Learning-Augmented Caching with Pairwise Eviction Preferences and Robust Deployment Rules**  
2. **From Pairwise Eviction Signals to Online Cache Policies: Deployment-Centric Learning-Augmented Caching**  
3. **Decision-Aligned Eviction Learning: Pairwise Supervision, Fallback Combiners, and Evidence-Limited Scaling**

## Candidate abstract framing A: deployment-rule centered
**Framing intent:** highlight that reliable online gains currently come most from *how* pairwise advice is deployed.

> We study learning-augmented online caching where eviction advice is produced by learned pairwise candidate preferences. Across controlled local traces, we find that deployment design (unguarded use, guard wrappers, and lightweight LRU black-box combination) can materially change downstream misses even when the underlying scorer is fixed. Our strongest current local policy is a pairwise + LRU combiner, suggesting that robust integration logic is at least as important as score quality at this stage. We provide a structured evidence map separating strong local findings from unresolved scaling and theorem questions, and we outline an inversion-based theory program that is promising but unfinished.

## Candidate abstract framing B: pairwise-vs-pointwise centered
**Framing intent:** foreground the supervision-choice hypothesis while preserving caution on current empirical strength.

> We investigate whether eviction learning should be formulated as pointwise candidate scoring or pairwise candidate preference prediction. Motivated by the ranking nature of eviction, we build a pairwise supervision pipeline and compare it to pointwise baselines under shared feature and replay settings. Current repository evidence supports pairwise learning as a promising direction, especially when coupled with robust deployment rules, but pairwise-only gains are not yet uniformly dominant across all local experiments. We therefore present a conservative synthesis: pairwise supervision appears decision-aligned and theoretically fertile (via inversion-style analyses), while broad-scale empirical superiority remains an open target.

## What would need to improve before upgrading to a more theory-heavy journal?
1. **At least one finalized theorem with proof** connecting a pairwise/inversion error notion to excess misses (with clear assumptions and failure modes).  
2. **A complete assumption stress-test section** showing where the theorem regime does and does not apply, backed by synthetic or trace-derived counterexamples.  
3. **A cleaner bridge between theorem metric and empirical metric** (e.g., inversion proxy trends that predict online regret in controlled studies).  
4. **Robustness statement for fallback/combiners** with a formal guarantee (or impossibility boundary) rather than only empirical ablations.  
5. **Manuscript-ready theorem narrative**: definitions, lemmas, proofs, and relation to prior LA-caching guarantees as a complete theoretical package.
