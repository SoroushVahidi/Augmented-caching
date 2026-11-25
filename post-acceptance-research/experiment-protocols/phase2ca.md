# Wulver Phase 2C-A counterfactual supervision

The fixed development population is the seven digest-pinned Phase 2B traces. Each has a 50,000-request observed window, with the first 10,000 requests used for warm-up. Eligible decisions are full-cache misses in positions [10,000, 50,000) on an LRU trajectory. Caches are fully associative, equal-size and mandatory-admission. This campaign uses no confirmation workload and fits no learner.

## Sampling and roles

Balance the primary 20,000, 50,000 and 100,000 group budgets across the 21 family-capacity cells, redistributing any exhausted cell's share. Within cells, form eight chronological strata of 5,000 requests. Allocate at least one draw to each nonempty stratum where the budget permits and approximate proportional allocation for the remaining groups. In each stratum, take prefixes from one deterministic seeded permutation. Record the eligible, selected and fixed design inclusion probabilities by budget. Report sample counts and inverse-inclusion-probability estimates. No label influences inclusion.

The family roles are the original seven five-train/one-validation/one-test assignments. Every candidate from a decision stays in its original group. The fixed-suffix and resolved observation end are explicit. If a future temporal split shares a trace, admit a training group only when the appropriate label observation end falls strictly before its boundary, and reset sampling weights or report the restricted population. The fixed stratum guard is a convenience for splits at the stored boundaries. The main family splits have no within-trace role boundary.

The 256, 512 and 1024 strata cap the sample at 128 eligible groups per family-capacity cell. Tabulate occupancy and distinct-object availability, including cells with zero full-cache misses. Large-capacity diagnostics stay outside the primary budgets.

## Labels and diagnostics

Record future misses after the forced admission and baseline-relative signed advantage in the same oldest-first candidate order. Use eight fixed horizons, scaled request horizons with alpha 0.125, 0.25, 0.5, 1, 2 and 4, and distinct-object horizons with the same multipliers. Use one shared LRU continuation across these outputs and exact advantages. At observed trace end the counts are exact for the available finite suffix. A branch that reaches the trace boundary without coalescing has an unobserved continuation and is flagged. Record the prevented reference misses, the at-most-one additional miss and individual branch termination positions. Apply the optimal-set lower bound to all-candidate finite-horizon labels using the realized request length.

Tabulate all ties, pairwise ties, distinct labels, entropy, optimum fractions, random uniform optimum accuracy, LRU optimum rate, beneficial and harmful actions, attainable gain and disagreement with the exact suffix comparison. Distinguish requested and effective horizon lengths. Compare ratios and realized distinct turnover descriptively across matched decision groups and varying K. Distinct-object-normalized rows remain finite if the requested distinct threshold cannot be reached.

All shortlists include the LRU victim. Compare the existing Phase 2A causal mechanisms at K=2, 4, 8, 16 and 32, including 10 fixed uniform seeds. Time implementations on the common states, preserving group counts, coverage and design-weighted one-action gains. Positive, neutral and harmful use the signed exact LRU advantage and exclude the baseline from the action-rate denominator.

## Features and adequacy

F0 reuses November's ordered 26 columns at float64. F2 to F6 reuse the recurrent audit's causal resident, bounded ghost and request features. F3 uses the original 64-request decay plus a lazy 256-request decay with the same causal resident/ghost scope. Cleaning F1 uses only the established training families for each fold and drops exact constant and duplicate legacy columns on the bounded feature-analysis sample. Any subsequent model preprocessing must be refit only on its actual training split.

Feature information uses a bounded equally spaced sample of whole decisions per cell, up to 96 groups on primary cells and 32 on diagnostics. Report exact input aliases, conflicting outcomes, grouped variation, mixing across decisions and numerical rank. The plug-in class information estimate is descriptive and can be inflated by near-unique inputs. No independence across rows within a decision is assumed. Missingness uses the existing observation flags.

Report naturally occurring beneficial, harmful and neutral counts and profitable decision prevalence before any future enrichment. Provide fold-specific observed positives and design-based population estimates for each budget. Several hundred training positives and 50 to 100 validation positives are planning targets. Families with sparse evidence need separately proposed calibration. These diagnostics do not establish closed-loop gains.

## Execution and access

The bounded Wulver pilot must pass current-platform simulator, horizon, coalescence, candidate-order and archived feature checks before production. Independent family tasks reuse a single trace decode across capacities. The reducer requires a Slurm afterok dependency, producer IDs and complete matching partition receipts. Candidate-level records, partition endpoints and the archived parity fixture stay on restricted Wulver storage. The reducer prepares a compact allowlisted review archive with aggregate counts only. Upload after verified completion using the authorized private Drive account.
