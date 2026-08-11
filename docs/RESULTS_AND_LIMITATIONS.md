# Results and Limitations

This is the most important document for understanding what this repository
currently supports as evidence, and what it does not. It is written to be
transparent, including about negative and inconclusive results.

**Status note:** this page describes an active research investigation. Some
of the evidence below (the mechanistic diagnostics in particular) comes from
work developed on the `kbs/second-revision-science` branch, not yet merged
to `main`. Where that applies it is stated explicitly per finding, along
with a pointer to [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) for the
current branch location. Nothing below should be read as a final published
conclusion; treat it as the current best understanding, subject to revision
as the underlying experiments complete.

## A. What worked

- The core simulator and trace-replay framework (`src/lafc/`,
  `python -m lafc.runner.run_policy`) reproduces a broad set of
  literature-faithful baselines (`lru`, `marker`, `predictive_marker`,
  `trust_and_doubt`, `robust_ftp_d_marker`, `blind_oracle_lru_combiner`,
  `offline_belady`) plus general-caching LP+rounding — see
  [`baselines.md`](baselines.md).
- `evict_value_v1`, the candidate-level learned eviction scorer, trains and
  evaluates end to end, produces reproducible artifacts (CSV/JSON/Markdown),
  and its offline target can be learned from data: in a deeply audited
  diagnostic cell, the trained model achieves 96.5% agreement with the exact
  value of its own training target and a low mean regret against it (0.035)
  -- the model is not failing to fit what it is asked to fit.
- An implementation-equivalence check found that a much faster, vectorized
  version of the candidate-scoring computation makes exactly the same
  eviction decisions as the reference implementation across all checked
  trace/capacity pairs -- the method can be made significantly cheaper
  without changing any decision, in the cells checked.

## B. What did not work (or is not yet established)

- In the same deeply audited diagnostic cell, **exact optimization of the
  finite-horizon `eviction_loss` training target performed worse than plain
  LRU** (higher miss ratio). This is a single-cell result (one trace family,
  one capacity, one horizon), not a general claim, but it is a genuine and
  somewhat counter-intuitive finding worth stating plainly: a supervision
  target can look reasonable and even be learnable, while still not being a
  good target for the actual online objective (minimizing misses).
- Interestingly, the *learned* approximation of that same target performed
  better than exact optimization of the target itself (fewer misses) --
  suggesting the model's departure from the target, not its fidelity to it,
  is doing useful work in that cell. See
  [`HYPOTHESIS_MAP.md`](HYPOTHESIS_MAP.md) H2/H3 for the interpretation.
- A frozen four-way objective comparison (comparing `eviction_loss` against
  `next_arrival`, `reuse_distance`, and a pairwise/ranking objective, all
  under the same protocol) found `eviction_loss` to be the worst-performing
  objective in nearly every trace family tested. Because the other two
  scalar objectives (`next_arrival`, `reuse_distance`) also outperformed
  `eviction_loss`, this argues against a simple "scalar regression is the
  problem" explanation and points toward something specific to how
  `eviction_loss` is constructed. *(Currently on `kbs/second-revision-science`.)*
- A direct measurement of that target's information content in the same
  audited cell found it severely degenerate: essentially all eviction
  candidates tied for "optimal" under the finite-horizon label almost all of
  the time. Extending the horizon 8x only broke a minority of those ties.
  *(Currently on `kbs/second-revision-science`.)*
- Whether simply training on more data closes the gap between offline
  quality and online performance is not yet established either way. A
  same-target learning-curve comparison, audited up through a moderate
  training-data fraction, shows essentially flat offline and downstream
  metrics across a large increase in training data for the cells completed
  so far -- weak evidence against "not enough data" as the primary
  explanation, but the campaign is still running and this is not a final
  answer. *(Currently on `kbs/second-revision-science`.)*
- A preliminary check for sequential distribution shift (the mismatch
  between how training labels are constructed and how the policy actually
  behaves once deployed) found substantial trajectory divergence, but one
  round of a natural corrective procedure (DAgger-style relabeling) made
  measured downstream misses *worse*, not better, in the one trace family
  checked -- so this is a real phenomenon whose fix is not yet obvious, not
  a solved problem. *(Currently on `kbs/second-revision-science`.)*
- A causally cleaner test of continuation-policy sensitivity (whether the
  fixed-LRU assumption used to construct training labels matters, versus a
  more consistent alternative) is implemented and tested but has not yet
  been run beyond a tiny smoke scale -- no result exists yet either way.
  *(Currently on `kbs/second-revision-science`.)*

## C. What we learned from the negative results

Negative and mixed results are reported here as scientific findings in their
own right, not as failures to hide:

1. **A target can be learnable without being a good target.** The clearest
   single finding above (exact optimization of the finite-horizon target
   loses to LRU) is a caution that applies more broadly than this specific
   method: designing a finite-horizon supervision signal for a sequential
   decision problem is not automatically safe just because a model can fit
   it well.
2. **A learned model's imperfections can be load-bearing.** The model
   outperforming exact optimization of its own target (rather than simply
   approximating it) suggests the model's inductive bias (its features, its
   regularization) was doing something the raw target label was not. This
   is worth further mechanistic study rather than treating "high agreement
   with the target" as automatically good news.
3. **Representation is not the same as target construction.** An earlier
   objective comparison found a pairwise/ranking-style objective
   outperforming scalar regression, which could be (mis)read as "pairwise
   representation is inherently better." Comparing scalar-vs-pairwise
   representations of the *same, fixed* underlying target directly tested
   and did not support that simplification -- the advantage tracks the
   underlying target construction, not the representation choice alone.

## D. Mechanistic evidence (diagnostics, not primary claims)

The findings above rest on a mix of evidence strength -- see
[`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) for the full evidence-level
classification per experiment. In particular:

- The exact-target-oracle and target-degeneracy findings above come from
  **one deeply audited cell** (one trace family, one cache capacity, one
  horizon). They are mechanistic diagnostics, explicitly not yet shown to
  generalize across families or capacities. Do not read them as a
  workload-general conclusion.
- The distribution-shift finding covers one trace family out of the
  intended full set.
- The learning-curve finding covers a partial, ongoing campaign.

## E. Limitations

- Several of the diagnostics above are single-cell or single-family;
  broader replication is an explicit open item, not an oversight.
- The corrected, non-contaminated cross-family held-out comparison against
  baselines is not yet complete; an earlier fair-window comparison was
  found to have train/test overlap and must not be cited.
- A learned-baseline comparison used in this line of work includes an
  independently reimplemented HALP-style policy adapted to these benchmark
  traces -- there is no public official HALP implementation to compare
  against directly, and this should be described as *"a causal HALP
  reimplementation adapted to unit-size/object-miss benchmark traces,"*
  never as an official or exact production reproduction.
- Computational-overhead numbers currently rest on smoke-scale timing only
  and explicitly should not be cited as a final controlled result.
- Some of the deeper diagnostic work described here currently lives on the
  `kbs/second-revision-science` development branch and has not yet been
  merged to `main`; see [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)
  for exactly which experiments that applies to.

## F. Invalid or superseded artifacts

These specific artifacts must not be used to support a claim, even though
they remain on disk for provenance. Historical evidence is never deleted;
it is labeled instead.

| Artifact | Why invalid / superseded | Replacement | Safe to use? | Status |
|---|---|---|---|---|
| An earlier `evict_value_v1` fair-window comparison CSV (`kbs/second-revision-science`) | Confirmed train/test overlap: the same trace streams were used for both training and evaluation | The corrected held-out cross-family replay (experiment 9 in [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)), currently `COMPLETE_PARTIAL_SCOPE` | No -- never cite | `CONTAMINATED_DO_NOT_USE` |
| `deployment_full_stream`-tagged rows in any fairness-protocol comparison | Not the primary controlled-evaluation-window comparison; included for supporting context only | The `primary_controlled_window`-tagged rows in the same files | Supporting context only, not for a primary comparison | Exclude from primary claims |
| An earlier, smaller continuation-policy exploration (predates the causal C1/C2 formulation in experiment 7) | Superseded by a purpose-built causal test of the same hypothesis | Experiment 7's causal continuation-policy comparison (`kbs/second-revision-science`, currently `IMPLEMENTATION_READY`/`SMOKE_ONLY`) | No -- do not conflate with experiment 7's result once it exists | `HISTORICAL_SUPERSEDED`, kept for provenance only |
| Smoke-scale computational-overhead timing numbers | Explicitly non-canonical; the artifact itself is not a controlled timing measurement | The pending controlled timing campaign (experiment 8) | No -- do not cite as a final performance number | `SMOKE_ONLY`, not citable as final |
| Treating a same-target scalar-vs-pairwise finding (experiment 6) as equivalent to the objective-comparison finding (experiment 3) | These compare different things: experiment 6 fixes the target and varies representation; experiment 3 varies the target itself. They use similarly-named conditions that are easy to conflate | Check which experiment (3 vs. 6) and which exact condition a result comes from before citing it | N/A -- a citation-care note, not an artifact | Always verify source experiment before citing |

## G. What remains under investigation

See [`HYPOTHESIS_MAP.md`](HYPOTHESIS_MAP.md) for the full mechanistic
hypothesis matrix (why the offline-to-online gap exists) and
[`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) for the current status of
every experiment, including which are still running or not yet started.
