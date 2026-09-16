# LAFC-Evict / Performance Evaluation (PE) - Current Project Status and Handoff

Last verified: 2026-09-16 09:18 EDT.

This is the canonical PE handoff document for the `augmented-caching` scientific repository. It covers the PE line only. KBS worktrees and branches (`fairness`, `objective-ablation`, `halp`, `kbs-parallel`, `kbs-second-revision`, `cacheus`, `3l-cache`, and related KBS branches) are a separate publication line and must not be treated as garbage by a PE cleanup agent.

## Canonical Branch / HEAD Table

| Purpose | Repository | Branch | HEAD |
|---|---|---|---|
| Current PE manuscript | `/home/soroush/projects/lafc-evict-dataset/repo` | `polish/pe-results-independent-cleanup-20260915` | `aa535ae` |
| Current PE handoff/docs | `/home/soroush/projects/augmented-caching/repo` | `docs/pe-project-status-20260915` | updated by Query 3 final docs commit |
| Tier-2 LFU prepared branch | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-tier2-closed-loop-integration-20260915` | `4d2a9fd` |
| Long-horizon completed evidence | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-long-horizon-production-prep-20260915` | `cc8c1ad` |
| Learned attempt 1 historical archive | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-publication-learned-retrain-20260915` | `6c17c0b` |
| Learned attempt 2 canonical evidence | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-publication-learned-retrain-attempt2-20260915` | `ab36cba` |

## Current Compute State

- Active PE local processes: none found.
- Active PE tmux sessions: none found.
- Active PE Slurm jobs: none found in `squeue`.
- Long-horizon Wulver DAG: `COMPLETE_VALID`.
- Learned model attempt 2: `COMPLETE_VALID`, publication gate `PASS`.
- Tier-2 overnight/baseline production: `NOT_LAUNCHED`.

## Canonical Manuscript

- Repository: `/home/soroush/projects/lafc-evict-dataset/repo`
- Canonical branch: `polish/pe-results-independent-cleanup-20260915`
- Canonical HEAD: `aa535ae`
- PDF: `paper/performance_evaluation/latex/main.pdf`
- PDF page count: 47
- Manuscript safety rule for this handoff: do not add long-horizon numbers, learned-policy numbers, LFU numbers, or Wulver acknowledgment until the corresponding scientific findings are actually integrated.

The Wulver acknowledgment is intentionally absent for now. Add it only after Wulver-derived long-horizon results are incorporated into the manuscript.

## Completed PE Scientific Evidence

| Evidence | State | Canonical location |
|---|---|---|
| Tier-1 closed-loop production, LRU/MRU/random/SIEVE | COMPLETE_VALID | `analysis/closed_loop_tier1_evidence_20260914/` |
| Full-population MRU continuation census | COMPLETE_VALID | raw output in `lafc-evict-dataset` continuation census worktree; compact validation in manuscript repo |
| Mechanistic/offline-to-closed-loop linkage analysis | COMPLETE_VALID | `analysis/closed_loop_mechanistic_analysis_20260914/` |
| Continuation-sensitivity pilot | COMPLETE_VALID | `experiment/continuation-sensitivity-pilot-20260914` |
| Canonical H={4,8,16} candidate-label dataset | COMPLETE_VALID | `paper/sigmod2027/results/candidate_label_stats/y_loss_summary.csv`; `evict_value_v1_wulver_heavy_r1` |
| Long-horizon H={32,64,128} campaign | COMPLETE_VALID | `experiment/pe-long-horizon-production-prep-20260915`; durable PROJECT root `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1` |
| Learned model attempt 2 | COMPLETE_VALID; publication gate PASS | `experiment/pe-publication-learned-retrain-attempt2-20260915`; durable model backup `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/` |

## Prepared But Not Run

- Tier-2 LFU/classical baseline campaign: implementation, tests, manifest, aggregator, and launch guard are ready on `experiment/pe-tier2-closed-loop-integration-20260915`; production cells have not been run.
- Overnight Tier-2 baseline-capacity campaign: not launched.

## Historical / Failed / Superseded

- Learned attempt 1, `experiment/pe-publication-learned-retrain-20260915`: interrupted historical evidence; not publication-valid; do not use for final manuscript claims.
- First long-horizon launch `1288536_[0-19]`: failed due HOME quota. Downstream `1288547`/`1288548` cancelled. Superseded by scratch-backed successful DAG `1288869`/`1288880`/`1288881`.
- Cap256 probe `1288047`: timing-only, timed out, not scientific evidence.
- Manuscript template branch `manuscript/performance-evaluation-template-20260913`: historical superseded branch in the publication repository; do not use as current manuscript.

## Public Release Scope

- Public v0.3 preview: wiki2018-only.
- Full scientific five-family corpus: distinct from the fully redistributed public release.
- Reader-facing families: `alibaba-block`, `metacdn`, `metakv`, `twemcache`, `wiki2018`.
- Internal `cloudphysics` maps to reader-facing `alibaba-block`.
- Do not claim live HF/Zenodo/AWS state was verified unless a future agent actually performs that live verification.

## External Artifact Inventory

See `docs/PE_EXTERNAL_ARTIFACT_INVENTORY.md`.

## Ordered Next Scientific Steps

1. H16 canonical converter / existence audit. Reuse canonical H16; do not regenerate H16.
2. H16/H32/H64/H128 long-horizon comparative aggregation and interpretation.
3. Integrate validated long-horizon findings into the manuscript.
4. Publication-grade learned closed-loop evaluation using the frozen attempt2 model and predetermined leakage-safe evaluation cells.
5. Tier-2 classical baseline completion, including LFU as appropriate.
6. Final results-dependent manuscript integration: abstract, results, discussion, and conclusion.
7. Add Wulver acknowledgment once Wulver-derived long-horizon results are actually incorporated into the manuscript.

Do not execute these scientific steps as repository cleanup.

## Persistence Rule

- Long-running local PE computations must run in detached `tmux`.
- Long-running Wulver computations must run through saved `.sbatch` submissions.
- No scientific run may depend on an interactive terminal, agent session, chat session, or SSH connection remaining open.
