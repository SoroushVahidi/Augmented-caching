# PE Publication-Grade Learned Closed-Loop Campaign — Final Report

Campaign: `pe_publication_learned_closed_loop_20260916`. Status: **COMPLETE_VALID**.
This was a **confirmatory, frozen-model evaluation** — no retraining, retuning,
model selection, feature change, or post-hoc adjustment occurred at any point
in or after this campaign. Preserve it as such.

## Frozen identity

| Field | Value |
|---|---|
| Model class | `HistGradientBoostingRegressor` |
| Model SHA256 | `8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6` |
| Protocol SHA256 | `897b8d42ebfcb7e401a5bbac5245186acba4608a7d3b4af23694d6ca1aeff4b1` |
| Split manifest SHA256 | `4783d96b1ecc242bf726943ba7f960605e3502ceb389cb9866ab17e93cf54b90` |
| Feature manifest SHA256 | `de7c1a2bb0a6f23a421635a82c53c1a9be78a36e2e5e62a744826d4fef9522bd` |
| Target / horizon | `y_loss`, H=16 |
| Source branch / HEAD | `experiment/pe-publication-learned-retrain-attempt2-20260915` @ `ab36cbaae83c93a8adb5d152a28314bd3dff8c08` |
| Campaign branch | `experiment/pe-publication-learned-closed-loop-20260916` |
| Embedded manifest `runner_commit` | `62ccfa252733b5d20d413af2cb74590d0af2f235` (stale, see `RUNNER_COMMIT_RECONCILIATION.json`) |
| Actual Wulver submission checkout | `9684b4dac6e107289738e8d3cdb801b04dd4ef78` (independently verified) |

## Slurm DAG (all terminal)

| Job | Role | State | Exit | Elapsed |
|---|---|---|---|---|
| 1291048 | production array (0-249) | 250/250 COMPLETED | 0:0 all | ~4 min max per task |
| 1291049 | validation | COMPLETED | 0:0 | 3s |
| 1291050 | aggregation | COMPLETED | 0:0 | 1s |

`validation_report.json`: `{"status": "PASS", "expected_tasks": 250, "completed_tasks": 250, "errors": []}`.
Zero non-empty stderr logs across all 250 tasks. Every result embeds the
correct, matching model and protocol SHA (verified across all 250 files, not
sampled).

## Design

5 families (`alibaba-block`, `metacdn`, `metakv`, `twemcache`, `wiki2018`) ×
2 capacities (32, 128) = **10 predetermined cells**, each scored on a fixed,
pre-registered held-out test interval. 6 policies: `learned`, `lru`, `mru`,
`sieve`, `lfu` (1 task/cell, deterministic) and `random` (20 seeds/cell) = 250
tasks total. No cells were added, removed, or reweighted after seeing results.

## Headline result (macro, all 10 cells; miss ratio, lower is better)

| Policy | Macro miss ratio | Micro (request-weighted) miss ratio |
|---|---|---|
| SIEVE | 0.70965 | 0.70981 |
| LRU | 0.71175 | 0.71191 |
| Random (seed mean) | 0.72285 | n/a* |
| **Learned** | **0.76290** | **0.76302** |
| LFU | 0.78245 | 0.78255 |
| MRU | 0.84839 | 0.84846 |

\* per-seed miss/scored-request counts were not retained in `random_seed_metrics.csv`,
only the mean miss ratio per seed; a request-weighted random figure is not
reconstructable from the retained artifact and is not reported.

**Excluding the two fully degenerate wiki2018 cells** (8 cells), the gap
widens, not narrows: learned macro 0.70363 vs LRU 0.63968, SIEVE 0.63706,
random 0.65356 — the frozen learned policy is not merely diluted by wiki2018,
it underperforms even on the cells where every policy has room to differ.

## Win/tie/loss (10 cells; a "tie" is exact floating-point equality — every
observed tie is a wiki2018 cell where all 6 policies score miss_ratio=1.0
exactly, not a near-tie elsewhere)

| Comparator | Learned wins | Ties | Learned losses |
|---|---|---|---|
| LRU | 1 | 2 | 7 |
| MRU | 8 | 2 | 0 |
| SIEVE | 1 | 2 | 7 |
| LFU | 5 | 2 | 3 |
| Random (mean) | 1 | 2 | 7 |

The single learned win against LRU/SIEVE/random is the same cell in all three
cases: **metakv/cap128** (learned 0.76497 vs LRU 0.76582, SIEVE 0.76656,
random-mean 0.76685 — a margin of 0.0009-0.0019). Full per-cell detail:
`final/learned_vs_baselines.csv`.

## Random-seed context (full detail: `final/learned_vs_random_distribution.csv`)

In the 7 loss cells, learned frequently loses to *every one* of the 20 random
seeds, not just the mean — e.g. twemcache/cap128: learned=0.3909, random
range [0.1586, 0.1709] (0/20 seeds worse than learned). In the one win cell
(metakv/cap128), learned beats all 20 random seeds (range [0.7663, 0.7675]).

## By family / by capacity (full detail: `final/by_family_summary.csv`,
`final/by_capacity_summary.csv`)

| Family | Learned | LRU | SIEVE | LFU | MRU | Random-mean |
|---|---|---|---|---|---|---|
| alibaba-block | 0.9792 | 0.9645 | 0.9538 | 0.9574 | 0.9988 | 0.9710 |
| metacdn | 0.6333 | 0.5472 | 0.5474 | 0.6186 | 0.8357 | 0.5698 |
| metakv | 0.7697 | 0.7660 | 0.7673 | 0.8090 | 0.8220 | 0.7682 |
| twemcache | 0.4323 | 0.2810 | 0.2798 | 0.5272 | 0.5855 | 0.3052 |
| wiki2018 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

metakv is the only family where learned is competitive with (marginally
ahead of) LRU/SIEVE/random at both capacities. twemcache shows the largest
absolute learned-vs-LRU/SIEVE gap (learned worse by ~0.15). wiki2018 is
degenerate for *every* policy at both capacities (miss ratio exactly 1.0),
consistent with the manuscript's existing characterization of this workload
as a mechanistically explained negative control.

## Interpretation (descriptive; no single causal mechanism established)

- **Does the frozen learned policy improve on LRU?** Generally no — 1 win, 7
  losses, 2 degenerate ties across the 10 predetermined cells.
- **On SIEVE?** Generally no — same 1/7/2 split.
- **On random?** Generally no — same 1/7/2 split (and the losses are often
  outside the full 20-seed random range, not just below the mean).
- **On MRU?** Yes, in every non-degenerate cell (8/8).
- **On LFU?** Mixed, with more wins than losses (5 wins, 3 losses, 2 ties).
- **Is performance homogeneous across families?** No — metakv is roughly at
  parity with the strong baselines; twemcache and alibaba-block show the
  learned policy clearly underperforming; wiki2018 is degenerate for all
  policies.
- **Does good offline predictive structure necessarily translate into
  closed-loop improvement?** No, not for this frozen model under this
  protocol. This is the central empirical fact this experiment establishes.

This experiment does not, by itself, identify *which* mechanism (tie-degenerate
supervision, action-underdetermined labels, continuation-policy mismatch,
distribution shift from the model's own actions, compounding cache-state
effects) is responsible for the gap between offline signal and closed-loop
outcome — several are plausible and consistent with this result, none is
established as *the* cause by this evidence alone. The correct, supported
statement is: **the frozen offline model does not translate into consistent
closed-loop gains under the held-out replay protocol** — not "tie degeneracy
causes the failure" or any other single-mechanism claim.

## Relating to the offline attempt-2 report

The frozen model was selected via `model_selection_protocol.json` on offline
validation-set regret/MAE criteria against `y_loss` (see
`analysis/pe_publication_learned_retrain_attempt2_20260915/`), not against
closed-loop miss ratio — the offline selection criterion and the closed-loop
outcome measured here are different objects by construction. Predicting a
finite-horizon counterfactual `y_loss` well enough to win an offline
regret/MAE comparison is not the same task as inducing, through a sequence of
eviction choices, a recursively-visited cache state that reduces misses over
an entire replay: the offline criterion is evaluated per-decision against a
fixed history, while the closed-loop outcome is path-dependent on every prior
choice the policy itself made. This is likely one of the paper's clearest
empirical lessons and should not be generalized beyond this one model class /
feature set / target / horizon combination.

## Limitations of this campaign specifically

- One model class (`HistGradientBoostingRegressor`), one feature set, one
  target (`y_loss`), one horizon (H=16), frozen before this evaluation.
- Two capacities (32, 128), not the release's full capacity range.
- Five families, one of which (wiki2018) is degenerate for every policy —
  it contributes no discriminating evidence in this campaign, by design of
  the trace itself, not a flaw in the evaluation.
- No architecture search, hyperparameter tuning, or alternative training
  target was evaluated in closed loop; this is a single confirmatory point,
  not an exhaustive study of learned-policy design space.
- This result should **not** be read as a general impossibility result for
  learned cache eviction — it is a negative/mixed result for one frozen
  configuration under one held-out protocol.

## No post-hoc tuning statement

No parameter, feature, target, split, cell, seed, tie-break rule, or warmup
choice was changed after any result in this campaign was observed. The model
was frozen (see `MODEL_SELECTION_FROZEN.json`,
`analysis/pe_publication_learned_retrain_attempt2_20260915/`) before this
closed-loop replay was designed or run. The protocol naming fix
(`cloudphysics`→`alibaba-block`, commit `62ccfa2`) and the sklearn runtime-
environment fix (commit `9684b4d`) were representational/infrastructure
corrections made *before* any task executed, verified byte-identical on the
scientific content (`cells_by_key`/`tasks`) both times — see
`RUNNER_COMMIT_RECONCILIATION.json`.

## Manuscript-ready claims (each traceable to an artifact below)

1. "A frozen H=16 HistGradientBoostingRegressor evaluated on a held-out
   closed-loop replay across 10 predetermined family×capacity cells did not
   consistently outperform LRU, SIEVE, or random eviction, losing in 7 of 10
   cells to each (with the same 2 cells degenerate ties in all three
   comparisons), while consistently outperforming MRU (8/8 non-degenerate
   cells) and winning more often than not against LFU (5 wins, 3 losses)."
   — source: `final/FINAL_SCIENTIFIC_SUMMARY.json` `wins_ties_losses`.
2. "Learned macro miss ratio 0.7629 (micro 0.7630) compares to LRU 0.7117,
   SIEVE 0.7097, random-mean 0.7228, LFU 0.7825, MRU 0.8484 across the same
   10 cells." — source: `final/FINAL_SCIENTIFIC_SUMMARY.json` `overall_summary`.
3. "This gap does not narrow when the degenerate wiki2018 cells are excluded
   (learned 0.7036 vs LRU 0.6397, SIEVE 0.6371, random 0.6536 on the
   remaining 8 cells)." — source: same file, `*_excl_wiki2018_8cell` fields.
4. "metakv/cap128 is the one cell where the learned policy is (marginally)
   best of the compared policies." — source: `final/learned_vs_baselines.csv`.

## Artifact map

- Raw per-task evidence: 250 `result.json` files under
  `/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916/`
  (SCRATCH; not copied into git — see `checksums.sha256` for integrity).
- Durable compact evidence (this directory, mirrored from
  `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916/`):
  `validation_report.json`, `campaign_summary.json`, `per_cell_metrics.csv`,
  `random_seed_metrics.csv`, `provenance.json`, `checksums.sha256`,
  `final/learned_vs_baselines.csv`, `final/by_family_summary.csv`,
  `final/by_capacity_summary.csv`, `final/learned_vs_random_distribution.csv`,
  `final/FINAL_SCIENTIFIC_SUMMARY.json`.
- Provenance: `RUNNER_COMMIT_RECONCILIATION.json`, `LAUNCH_RECORD.json`,
  `WULVER_RUNTIME_ENVIRONMENT.json`, `CLOSED_LOOP_PROTOCOL.json`,
  `CLOSED_LOOP_LEAKAGE_AUDIT.json`, `campaign_manifest.json`.

All derived tables and the checksum manifest were produced by a single
deterministic script (`derive_final.py`, run once on Wulver against the
already-completed raw outputs) that performed no replay, simulation,
retraining, or re-scoring — it only aggregates, tallies, and hashes existing
numbers. `checksums.sha256` was verified immediately after generation
(`sha256sum -c`, all 260 lines OK).
