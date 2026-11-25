# Experiment and evidence index

The [accepted KBS verification guide](../reviewer/START_HERE.md) and [accepted upload manifest](../../submission_kbs_r3_20261013/KBS_R3_SUBMISSION_MANIFEST_20261013.md) index the published evidence. The [historical experiment registry](../EXPERIMENT_REGISTRY.md) covers the earlier research program. Use the [methodology](METHODOLOGY.md) to determine whether rows from later experiments can be compared.

| Study | Evidence stage | Scope and conclusion |
|---|---|---|
| HALP and KBS matched comparisons | Published held-out comparison | Corrected cross-family comparison with causal HALP and calibrated controls. See the accepted verification guide for policy variants and exact result tables. |
| Stateless MLP and DeepSets | Retrospective replay | Finite-horizon supervision with matched candidate shortlists and capacity cells. Full comparisons and the original configuration are in the November preservation draft. |
| Earlier recurrent GRU/LSTM | Development and retrospective replay | History and scoring ablations. The preserved test and development rows use different populations. |
| Phase 2A/B | Forensics and exact LRU computation | Target ambiguity, shortlist coverage, exact-label parity, runtime improvement, and matched replay diagnostics. |
| Phase 2C-A | Group construction and feature analysis | Grouped supervision, held-family splits, feature aliasing, and platform parity. |
| Phase 2C-B | Retrospective replay | Larger grouped-supervision, gating, calibration and packed-inference comparisons. No advancement under the full joint rule. |
| A/B/C | Completed 21-cell retrospective replay | Optimal continuation, censoring-aware reuse, and training weights. No learned finalist passed the joint effectiveness and CPU criteria. |
| E/F | Completed retrospective replay and oracle analysis | Tie handling, bypass headroom, and heuristic admission. Conditional learned admission and combination stages did not qualify. |
| G | Completed screening and efficiency checks | Temporal and non-sequential candidates screened on validation. No full outer portfolio advanced. |
| I/J/M | Completed validation campaigns | Reuse forecasting, tie-aware targets, and continuation labels. Validation gains and label informativeness are retained as distinct findings. |
| D, H, K, L, M4 | Deferred or unlaunched | D was deferred in the bounded A/B/C matrix. M4 never advanced past its primary M gate. H, K and L have no verified executed campaign in this inventory. |

The [November preservation draft](https://github.com/SoroushVahidi/Augmented-caching/pull/53) provides source provenance, stage reproduction commands, model settings, and checksum-pinned aggregates for the November neural and Phase 2 studies. Its source and evidence are independently versioned. Later restricted receipts and source revisions are held in the project evidence archive. Use the per-campaign manifest and executed revision when deriving new summaries.

[Subsequent evidence and snapshot verification](RECONCILIATION.md) explain how the November report relates to later work.

The main CLI and archived runner paths are part of the reproduction record. The cache API and reference policies live in [src/lafc](../../src/lafc/). Training and replay entry points remain in [scripts](../../scripts/), with benchmark and scheduler notes under [docs](../README.md) and [slurm](../../slurm/).
