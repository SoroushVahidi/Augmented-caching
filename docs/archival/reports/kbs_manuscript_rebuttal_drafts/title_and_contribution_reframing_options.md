# Title and contribution reframing options (results-independent, 2026-06-19)

Status: **prepared, not final.** Drafted while `cap32_with_sieve_fifo`
runs. These are options to choose between once the canonical sweep
(cap64/128/256, plus the corrected cap32-with-SIEVE-and-FIFO-Reinsertion
chunk) completes — not a recommendation to adopt any one of them today.
Current title in `main.tex` claims "robust learning-augmented caching"
(per `reports/kbs_complete_reviewer_comment_matrix.md`, R3-Issue4); the
only completed canonical evidence (cap32, no SIEVE/FIFO) shows
`evict_value_v1` losing to LRU on 6/7 trace families, which does not
currently support that title.

---

## Title options

### Option 1 — conservative/accurate, if end-to-end results remain weak

> **"Decision-Aligned Eviction-Value Prediction: An End-to-End Empirical
> Audit of a Learned Caching Policy"**

Use when: the full sweep confirms the cap32 pattern — `evict_value_v1`
underperforms LRU (and/or SIEVE, FIFO-Reinsertion) on most trace
families/capacities, with no clear regime where it wins consistently.

Rationale: drops "robust" and "caching" as the headline contribution noun
(replaced by "audit"), framing the paper's actual demonstrated contribution
— a careful, end-to-end empirical evaluation methodology and a negative/
mixed finding — rather than a performance claim the data does not support.
This is the only option compatible with a fully negative outcome without
requiring a defensive rewrite of the abstract.

### Option 2 — moderate, if the learned policy is competitive on some workloads

> **"Learned, Decision-Aligned Eviction Value Estimates: When They Help and
> When They Don't"**

Use when: the full sweep shows a mixed picture — `evict_value_v1` matches
or beats LRU/SIEVE/FIFO-Reinsertion on a meaningful subset of trace
families or capacities, but not uniformly.

Rationale: makes workload-dependence the headline finding instead of
hiding it in a "robust" claim that the per-family breakdown would
contradict. Directly answers R3-Minor9 (workload-specific breakdowns
needed) at the title level, and sets up the per-family table as the central
piece of evidence rather than a buried appendix.

### Option 3 — stronger, only if later canonical results clearly support it

> **"Decision-Aligned Eviction Value Prediction for Robust
> Learning-Augmented Caching"**

Use when: the full sweep shows `evict_value_v1` consistently matching or
outperforming LRU, SIEVE, and FIFO-Reinsertion across most trace families
and most/all four capacities — i.e., the original title's claim becomes
empirically supported rather than aspirational.

Rationale: closest to the current title; should only be adopted if cap64/
128/256 reverse or substantially narrow the cap32 gap. **Do not adopt this
option by default** — it is the current title, and it is the one R3-Issue4
specifically flagged as unsupported by the paper's own admitted empirical
weakness. Re-adopting it requires the data to have changed, not just the
prose.

**Recommendation for now:** assume Option 1 or 2 unless the full sweep
gives a clear, broad-based reason to use Option 3. Do not finalize any of
these until cap64/128/256 complete.

---

## Revised contribution bullets

Drafted to emphasize the five themes requested, written so that each bullet
is true regardless of how the canonical sweep resolves (no placeholder
numbers embedded directly in the bullets; specifics deferred to the body).

1. **Decision-aligned supervision target.** We propose a supervision target
   for learned cache-eviction policies constructed from finite-horizon
   counterfactual replay: for each eviction candidate at each full-cache
   miss, we estimate the downstream miss-count harm that evicting that
   specific candidate would cause under a fixed continuation policy, over a
   bounded horizon H. This target is directly aligned with the decision the
   policy must make, in contrast to proxy targets such as next-access-time
   prediction.

2. **End-to-end empirical audit, not just offline validation.** Rather than
   reporting only offline target-quality metrics (regret, Top-1 agreement),
   we evaluate the resulting policy end-to-end, via online cache replay,
   across multiple trace families and cache capacities, against a set of
   classical and modern baselines. We report this evaluation in full,
   including results that do not favor our proposed method, rather than
   selectively reporting favorable conditions.

3. **Comparison against modern lightweight baselines.** We extend our
   baseline set beyond LRU and prior learned-caching heuristics to include
   SIEVE (NSDI '24) and FIFO-Reinsertion, both implemented and verified
   against their published/reference specifications, addressing a gap in
   our original baseline coverage identified by reviewers.

4. **Reproducibility artifacts.** We release the full evaluation pipeline,
   trained models, dataset-construction scripts, and per-run evidence
   logs underlying every reported number, including the offline
   label-construction cost and online per-decision complexity analysis, to
   support independent verification given this is single-author,
   AI-assisted work.

5. **Honest reporting of limitations and negative findings.** We report,
   rather than omit or soften, the conditions under which our proposed
   method does not improve on simpler baselines, and we revise our framing
   of the guarded fallback mechanism to reflect that it has not been
   empirically validated as a contribution of this work, only proposed as a
   candidate extension for future study.

**Note:** bullet 5 is a structural commitment, not a placeholder — it
should remain in the contributions list even if later results turn out
favorable on some workloads, because the per-family breakdown (R3-Minor9)
will still show some families where the method does not help. Do not
delete this bullet if the aggregate number ends up positive.
