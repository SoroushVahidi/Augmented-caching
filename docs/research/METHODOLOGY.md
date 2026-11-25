# Cache eviction research methodology

## Label and state contract

A decision group denotes a full-cache miss at request index t. Candidate actions choose the resident to evict. In mandatory admission every branch inserts the same incoming object. The default finite-horizon miss cost counts misses in requests t+1 through t+H, truncated by the observation end. A positive LRU-relative advantage is Q(LRU) minus Q(candidate). Include state generator, shortlist construction, continuation policy, horizon and suffix length when describing any target. Optimal continuation and the feasible online deployment policy answer different questions.

Alternative continuation comparisons start from the same resident set. LFU aging, SIEVE reference bits and hand state, and LRU-K access history require explicit prefix reconstruction or documented transfer. Optional admission has a distinct bypass action and needs its own state and objective. Oracle results bound opportunity under their specified assumptions.

## Tie ambiguity and forecast supervision

Optimal sets contain every minimum-cost action. Pairwise rankings omit equal-cost pairs. Longer horizons, first-miss positions and discounted costs can add supervision; report how often they actually change ordering. First-miss or next-reuse observation windows must travel with their labels. Censored examples record a lower bound on reuse time. Reuse probabilities conditional on surviving to age a use 1 minus S(a+H)/S(a) when S(a) is positive.

History and online features stop at the decision. Compare MLP and DeepSets controls with recurrent and temporal encoders under the same available histories and shortlist. Compare forecast methods on calibration and then on actual cache behavior. Offline label separation and candidate ordering do not establish lower replay miss counts.

## Evaluation and feasibility

Freeze hyperparameters, checkpoints, gates, and tie breaking on development data. Label every result with its stage and population. A complete seven-fold retrospective portfolio has 21 family/capacity cells, normally 840,000 scored requests after warm-up. Exact windows and reference models depend on the campaign. Keep independent confirmation access separate.

Miss ratio is misses divided by scored requests. Percentage improvement uses the matched baseline's misses as denominator. Sum counts before calculating a pooled rate and retain the family-level distribution. Compare with KBS EV only when model identity and cells match. Archive CPU results with hardware, affinity, thread count, warm-up, timing boundary, repetitions and memory method. Compute relative CPU against LRU measured on the same path. Native comparisons also require comparably optimized native LRU and must account for object lookup. The project advancement criteria include a minimum 0.5% miss reduction and a maximum 5× historical LRU CPU ratio. Additional stage gates are defined in each frozen protocol.

Preserve negative results, incomplete outer stages, and validation-only results with their receipts. A successful scheduler exit and a scientific advancement decision are different records.

## Reproduction and data provenance

See [datasets](../datasets.md) and the [independently versioned LAFC-Evict release repository](https://github.com/SoroushVahidi/lafc-evict-dataset). Trace rights and attribution are governed per upstream family. Publisher snapshots and archived source/checkpoint combinations are pinned separately from evolving documentation. Check manifest integrity before comparing results, and use a fresh output directory when reproducing historical commands.
