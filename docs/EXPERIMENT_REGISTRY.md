# Experiment Registry

Canonical index of the experiments behind this project's evidence. Organized
by scientific question, not by any external review process.

**Branch note:** this repository's `main` branch contains the core
simulator, the `evict_value_v1` learned-eviction method, and the
literature-faithful baseline suite. A deeper diagnostic investigation
(objective comparisons, mechanistic diagnostics, an expanded fairness
comparison against several modern learned baselines, and a training-data
learning curve) is being developed on the `kbs/second-revision-science`
branch. Two of these diagnostics -- the exact-target-oracle diagnostic
(experiment 4) and the target-degeneracy diagnostic (experiment 5) -- have
had their general-purpose source code, CLI entry points, and tests promoted
to `main` as of this pass; the rest remain on the development branch. Each
row below states clearly where its code currently lives, and whether that
means the *code* runs on `main`, the *result* is reproducible on `main`, or
both. Read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before running
anything.

## Status vocabulary

| Term | Meaning |
|---|---|
| `FINAL_VALIDATED` | Execution complete, audited, no known integrity gaps |
| `COMPLETE_DIAGNOSTIC` | Complete for its intended (often small) scope; a mechanistic diagnostic, not a primary result |
| `COMPLETE_PARTIAL_SCOPE` | Complete only for part of the intended campaign |
| `RUNNING` | Actively executing |
| `PENDING` | Not started |
| `IMPLEMENTATION_READY` | Code/tests/config exist and are frozen; no scientific result yet beyond a smoke test |
| `SMOKE_ONLY` | A smoke-scale run exists and is explicitly non-canonical |
| `NOT_MERGED` | Implemented and has results, but currently lives on a development branch, not `main` |

## Evidence-strength hierarchy

| Class | Meaning |
|---|---|
| `PRIMARY_EVIDENCE` | Directly supports a main comparison/claim |
| `SUPPORTING_EVIDENCE` | Strengthens or contextualizes primary evidence |
| `MECHANISTIC_DIAGNOSTIC` | Explains *why*, usually single-cell/small-scope; never a standalone general claim |
| `IMPLEMENTATION_ONLY` | Code/tests/config exist; no scientific weight yet |
| `HISTORICAL_SUPERSEDED` | Kept for provenance; not current evidence |

---

## 1. Core method and simulator

- Question: does the simulator/policy framework faithfully reproduce
  literature baselines and support a learned candidate-level eviction
  method end to end?
- Scope: full baseline roster (`lru`, `marker`, `predictive_marker`,
  `trust_and_doubt`, `robust_ftp_d_marker`, `blind_oracle_lru_combiner`,
  `offline_belady`, general caching LP+rounding) plus `evict_value_v1`
  training/eval pipeline
- Source: `src/lafc/`, `python -m lafc.runner.run_policy` (this branch)
- Status: `FINAL_VALIDATED`
- Evidence level: `PRIMARY_EVIDENCE`
- Reproducibility: see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); fully
  reproducible on `main` today
- Limitations: none specific to this layer itself

## 2. Learned-baseline fairness comparison (LRB and others)

- Question: how does the method compare to modern learned caching
  baselines under a matched, controlled protocol?
- Scope: LRB is implemented and validated on `main`; a broader comparison
  (3L-Cache, CACHEUS, a causal HALP-style reimplementation) is implemented
  on the development branch
- Source: LRB -- this branch (`src/`, `docs/lrb_method_spec.md`); the
  broader comparison -- `kbs/second-revision-science` (`NOT_MERGED`)
- Status: LRB `FINAL_VALIDATED` on `main`; broader comparison
  `COMPLETE_DIAGNOSTIC`, `NOT_MERGED`
- Evidence level: `PRIMARY_EVIDENCE`, with an explicit fidelity caveat for
  the HALP-style comparator: there is no public official HALP
  implementation, so this is *a causal reimplementation adapted to
  unit-size/object-miss benchmark traces*, not an official or exact
  production reproduction. CACHEUS uses official, unmodified upstream
  source; LRB and a compared 3L-Cache-style baseline are independent
  reimplementations.
- Limitations: implementation/tuning parity across baselines is not
  claimed to be perfect; see [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md).

## 3. Supervision-objective comparison

- Question: which supervision target (the current `eviction_loss` target,
  a next-arrival-time target, a reuse-distance target, or a pairwise/
  ranking target) produces the best downstream caching performance?
- Scope: a frozen four-objective comparison across multiple trace families
  and cache capacities
- Source: `kbs/second-revision-science` (`NOT_MERGED`)
- Status: `COMPLETE_DIAGNOSTIC` for the completed scope; `NOT_MERGED`
- Evidence level: `SUPPORTING_EVIDENCE`
- Limitations: see [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md)
  for the finding and its interpretation caveats

## 4. Exact-target oracle diagnostic

- Question: does exact optimization of the frozen finite-horizon
  `eviction_loss` target actually beat a simple baseline (LRU), and does
  the learned model reproduce or outperform that exact target?
- Scope: one deeply audited cell (one trace family, one cache capacity);
  the tooling is parameterized to run on other cells, not hardcoded
- Source: `main` (`src/lafc/oracle_diagnostics.py`,
  `scripts/experiments/run_exact_target_oracle_diagnostic.py`,
  `tests/test_oracle_diagnostics.py`) -- diagnostic code and tests were
  promoted from `kbs/second-revision-science` and pass on `main` today.
  The previously-recorded one-cell numeric result (brightkite, capacity 64,
  H=4) was produced on the development branch and is not independently
  regenerated by this promotion: the "learned model" comparison leg needs a
  trained model registry (`analysis/supervision_objective_ablation_v1/model_registry.json`)
  produced by the objective-ablation campaign, which itself remains
  `NOT_MERGED`. Run with `--no-learned` on `main` to exercise the exact-
  oracle-vs-LRU-vs-Belady legs without that dependency.
- Status: `COMPLETE_DIAGNOSTIC` (single cell; code `FINAL_VALIDATED` on
  `main`, underlying result `NOT_MERGED` pending the campaign artifact
  above)
- Evidence level: `MECHANISTIC_DIAGNOSTIC`
- Limitations: single-cell result; do not generalize across trace families
  or capacities without further replication

## 5. Target-degeneracy and horizon-resolution diagnostic

- Question: how much distinguishing information does the finite-horizon
  target actually carry among eviction candidates, and how much does
  extending the horizon recover?
- Scope: same single cell as experiment 4, plus a longer-horizon
  tie-resolution extension
- Source: `main` (`src/lafc/target_degeneracy.py`,
  `scripts/experiments/analyze_eviction_loss_target_degeneracy.py`,
  `tests/test_target_degeneracy.py`) -- diagnostic code and tests were
  promoted from `kbs/second-revision-science` and pass on `main` today. The
  core tie/entropy metrics run directly from trace data
  (`--no-learned` skips the trained-model comparison leg, same caveat as
  experiment 4); the previously-recorded one-cell result was produced on
  the development branch and is not independently regenerated by this
  promotion.
- Status: `COMPLETE_DIAGNOSTIC` (single cell; code `FINAL_VALIDATED` on
  `main`, underlying result `NOT_MERGED`) -- currently the
  strongest-supported mechanistic finding in this line of work (see
  [`HYPOTHESIS_MAP.md`](HYPOTHESIS_MAP.md) H3)
- Evidence level: `MECHANISTIC_DIAGNOSTIC`
- Limitations: single-cell; a genuine base-horizon sensitivity sweep
  (varying the horizon itself, distinct from the longer-horizon
  tie-resolution extension already run) remains open

## 6. Training-data learning curve (same-target, scalar vs. pairwise)

- Question: does more training data close the offline-to-online gap, and
  does a pairwise representation of the *same* target outperform scalar
  regression on it?
- Scope: an ongoing campaign across training-data fractions and all trace
  families; partially complete as of this writing
- Source: `kbs/second-revision-science` (`NOT_MERGED`)
- Status: `RUNNING` overall; see the development branch for exact live
  progress -- not tracked here to avoid stale numbers in a durable document
- Evidence level: `MECHANISTIC_DIAGNOSTIC`
- Limitations: partial-campaign results must not be treated as final; the
  campaign explicitly separates a representation question (scalar vs.
  pairwise on the same target) from a target-construction question (this
  target vs. other targets, see experiment 3) -- the two are not
  interchangeable

## 7. Distribution-shift / continuation-policy diagnostics

- Question: does the mismatch between how training labels are constructed
  (assuming a fixed continuation policy) and how the policy actually
  behaves once deployed explain part of the gap?
- Scope: a preliminary trajectory-divergence check on one trace family; a
  causally cleaner, purpose-built comparison is implemented but has only
  been run at a tiny smoke scale
- Source: `kbs/second-revision-science` (`NOT_MERGED`)
- Status: preliminary check `COMPLETE_PARTIAL_SCOPE`; causal comparison
  `IMPLEMENTATION_READY`, `SMOKE_ONLY`
- Evidence level: `MECHANISTIC_DIAGNOSTIC`
- Limitations: single trace family for the preliminary check; no full
  result yet for the causal comparison

## 8. Computational-overhead / practical-significance check

- Question: is the fine-grained, candidate-level scoring computation
  practical to run, or does it carry prohibitive overhead?
- Scope: a smoke-scale implementation-equivalence check (does a faster
  implementation make the same decisions as the reference implementation?)
  plus preliminary timing numbers; a controlled final timing campaign has
  not run yet
- Source: `kbs/second-revision-science` (`NOT_MERGED`)
- Status: equivalence check `COMPLETE_DIAGNOSTIC`; controlled timing
  `PENDING`
- Evidence level: equivalence check `SUPPORTING_EVIDENCE`; timing numbers
  `IMPLEMENTATION_ONLY` until a controlled run exists
- Limitations: smoke-scale timing explicitly should not be cited as a
  final controlled result

## 9. Held-out cross-family replay (corrected)

- Question: does the method still outperform baselines under a corrected,
  non-contaminated, held-one-family-out evaluation protocol?
- Scope: in progress; an earlier fair-window comparison was found to have
  train/test overlap and is explicitly excluded from citation
- Source: `kbs/second-revision-science` (`NOT_MERGED`)
- Status: `COMPLETE_PARTIAL_SCOPE`
- Evidence level: intended as `PRIMARY_EVIDENCE`, currently
  `SUPPORTING_EVIDENCE` only until complete
- Limitations: this is the single highest-priority open item for making
  the method's head-to-head comparison citable without caveats

---

## Notes

- No experiment above exists solely to satisfy an external reviewer; each
  answers a scientific question stated in its own row.
- Historical/superseded material (an earlier, smaller continuation-policy
  exploration that predates experiment 7's causal formulation) is kept for
  provenance in the development branch's internal documentation and is not
  listed here as current evidence.
