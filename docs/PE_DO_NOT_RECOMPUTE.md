# PE (LAFC-Evict) — Do Not Recompute

Snapshot time: 2026-09-16 ~02:20 UTC. Each item below was individually
verified as complete and validated in this pass (or in the immediately
preceding QA pass on 2026-09-15) before being listed here. If you are
tempted to regenerate any of these, check the canonical artifact path first
— the cost of regenerating full-population or production-scale evidence is
substantial, and duplicate runs risk silent divergence from the numbers
already reported in the manuscript.

| Experiment | Status | Canonical artifact | Reason not to recompute | Exceptions |
|---|---|---|---|---|
| Tier-1 closed-loop production (LRU/MRU/random/SIEVE, 230 runs, 5 families × 2 capacities) | COMPLETE, VALIDATED, manuscript-used | `augmented-caching` repo, `analysis/closed_loop_tier1_evidence_20260914/` | Independently validated; provenance recorded (host `al-khwarizmi`); reported in `table_tier1_closed_loop` | None — if a bug is later found in the harness, treat as a new experiment with a new provenance record, do not silently overwrite this one |
| Full-population MRU continuation census (60/60 chunks, 2,363,286 decision-horizon records, capacities 32/64/128/256, horizons 4/8/16) | COMPLETE, VALIDATED, manuscript-used | `lafc-evict-dataset` repo, worktree `.claude/worktrees/continuation-mru-population-census-20260914`, raw output `analysis/continuation_policy_mru_population_census_20260914/outputs/20260914T042528Z_1a29e773a113/` (1.6G) | Complete-population enumeration, independently rechecked; raw output preserved separately from the manuscript branch | None |
| Continuation-sensitivity pilot (5,000 primary + 500 diagnostic pre-registered decisions, capacities 32/128) | COMPLETE, VALIDATED, manuscript-used | `augmented-caching` repo, branch `experiment/continuation-sensitivity-pilot-20260914` | Pre-registered design; validated end-to-end | None |
| Mechanistic / offline↔closed-loop linkage analysis | COMPLETE, VALIDATED, manuscript-used | `augmented-caching` repo, `analysis/closed_loop_mechanistic_analysis_20260914/` | Uses only existing trace-only and closed-loop evidence (no new simulation), explicitly disciplined about DIRECTLY OBSERVED vs. CONSISTENT WITH MECHANISM claims | None |
| Canonical H∈{4,8,16} candidate-label dataset (5–7 families depending on artifact, capacities 32/64/128/256) | COMPLETE, VALIDATED, manuscript-used | `paper/sigmod2027/results/candidate_label_stats/y_loss_summary.csv`; derived dataset `evict_value_v1_wulver_heavy_r1` | This is the existing basis the long-horizon (H=32/64/128) DAG extends, not replaces — regenerating it would not add anything the DAG isn't already doing at higher horizons | None |
| LFU policy implementation and unit tests | Implemented, tested, campaign NOT run | `augmented-caching` repo, branch `experiment/pe-tier2-closed-loop-integration-20260915`, `src/lafc/policies/lfu.py`, `tests/test_lfu.py` | The *code* is done and tested — do not reimplement it. The closed-loop *campaign* using this code has genuinely not been run yet and is a legitimate next step, not something to avoid | The campaign run itself is pending, not something to avoid — see `docs/PE_CURRENT_PROJECT_STATUS_AND_HANDOFF.md` section K |
| Cap32 long-horizon preflight + cap256 timing probe | COMPLETE (preflight); the cap256 probe is explicitly timing-only | `augmented-caching` repo, branch `experiment/pe-long-horizon-preflight-20260915` | Confirms the generator's correctness bounds at higher horizons before the production DAG was launched; re-running would only reproduce the same feasibility check | The cap256 timing probe's *numbers* should never be cited as a scientific result — it was never intended as one |

## Explicitly NOT on this list (do not treat as validated)

- **Long-horizon DAG (H=32/64/128), jobs 1288869/1288880/1288881** — ACTIVE
  as of this snapshot (17/20 production tasks done, validation/aggregation
  pending). Once it reaches COMPLETE_VALID, add it here with its canonical
  output path; do not add it preemptively.
- **Learned-model attempt 2** — test evaluation in progress as of this
  snapshot. Once its publication gate resolves to PASS, add the frozen model
  and its test metrics here; do not add it preemptively, and do not confuse
  it with attempt 1 (below).
- **Learned-model attempt 1**
  (`experiment/pe-publication-learned-retrain-20260915`, model sha256
  `c5bf9840f6be4b82903c82bbf1e3ace87c5b4656e4bed163eba84f2ee805611e`) —
  interrupted, no model-selection freeze, no test evaluation. This is
  preserved as historical evidence of what was tried, **not** as a reusable
  scientific result. Do not resume training from this checkpoint and
  present it as attempt 2's result, and do not report any number derived
  from it in the manuscript.
