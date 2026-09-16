# LAFC-Evict / Performance Evaluation (PE) - Current Project Status and Handoff

Last verified: 2026-09-16 14:45 EDT.

This is the canonical PE handoff document for the `augmented-caching` scientific repository. It covers the PE line only. KBS worktrees and branches (`fairness`, `objective-ablation`, `halp`, `kbs-parallel`, `kbs-second-revision`, `cacheus`, `3l-cache`, and related KBS branches) are a separate publication line and must not be treated as garbage by a PE cleanup agent.

## Canonical Branch / HEAD Table

| Purpose | Repository | Branch | HEAD |
|---|---|---|---|
| Current PE manuscript | `/home/soroush/projects/lafc-evict-dataset/repo` | `manuscript/pe-learned-closed-loop-integration-20260916` | `8abf686` |
| Previous (superseded as "current") manuscript | `/home/soroush/projects/lafc-evict-dataset/repo` | `polish/pe-results-independent-cleanup-20260915` | `aa535ae` |
| Literature-positioning pass | `/home/soroush/projects/lafc-evict-dataset/repo` | `manuscript/pe-long-horizon-integration-20260916` | `999d23f` |
| Long-horizon manuscript integration | `/home/soroush/projects/lafc-evict-dataset/repo` | `manuscript/pe-long-horizon-integration-20260916` | `545a174` (superseded by `999d23f` on the same branch) |
| Current PE handoff/docs | `/home/soroush/projects/augmented-caching/repo` | `docs/pe-project-status-20260915` | this commit |
| Learned closed-loop campaign (COMPLETE_VALID) | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-publication-learned-closed-loop-20260916` | `30f178d` |
| Tier-2 LFU prepared branch | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-tier2-closed-loop-integration-20260915` | `4d2a9fd` |
| Long-horizon completed evidence | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-long-horizon-production-prep-20260915` | `cc8c1ad` |
| H16/H32/H64/H128 comparative analysis | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-h16-h128-comparative-integration-20260916` | `678adcb` |
| Learned attempt 1 historical archive | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-publication-learned-retrain-20260915` | `6c17c0b` |
| Learned attempt 2 canonical evidence | `/home/soroush/projects/augmented-caching/repo` | `experiment/pe-publication-learned-retrain-attempt2-20260915` | `ab36cba` |

## Current Compute State

- Active PE local processes: none found.
- Active PE tmux sessions: none found.
- Active PE Slurm jobs: none found in `squeue`.
- Long-horizon Wulver DAG: `COMPLETE_VALID`.
- H16/H32/H64/H128 comparative analysis: `COMPLETE_VALID`.
- Long-horizon manuscript integration: `COMPLETE`, on `manuscript/pe-long-horizon-integration-20260916`.
- Latest-literature positioning pass: `COMPLETE`, same branch at `999d23f`.
- Learned model attempt 2: `COMPLETE_VALID`, publication gate `PASS`.
- **Learned closed-loop campaign (jobs 1291048/1291049/1291050): `COMPLETE_VALID`.** 250/250 production tasks, validation PASS, aggregation COMPLETE. Negative/mixed result, preserved as-is (see Completed PE Scientific Evidence below) -- **do not retrain or resubmit this campaign.**
- Learned closed-loop manuscript integration: `COMPLETE`, on `manuscript/pe-learned-closed-loop-integration-20260916` at `8abf686` (not yet merged to any base branch).
- Tier-2 overnight/baseline production: `NOT_LAUNCHED` (see "Ordered Next Scientific Steps" -- reassessed below as optional, not required).

## Canonical Manuscript

- Repository: `/home/soroush/projects/lafc-evict-dataset/repo`
- Canonical branch: `manuscript/pe-learned-closed-loop-integration-20260916`
- Canonical HEAD: `8abf686`
- PDF: `paper/performance_evaluation/latex/main.pdf`
- Baseline polished PDF page count: 47
- Long-horizon integration PDF page count: 48
- Latest-literature positioning PDF page count: 49
- Learned closed-loop integration PDF page count: 53
- **Manuscript safety rule superseded**: learned-policy and matched LFU numbers ARE now integrated (Section "A Frozen Learned-Policy Closed-Loop Check"), sourced from the COMPLETE_VALID learned closed-loop campaign above. The negative/mixed result is preserved exactly as observed -- do not soften, hide, or re-run to get a different answer.

The Wulver acknowledgment is present on `manuscript/pe-long-horizon-integration-20260916` (and its descendant `manuscript/pe-learned-closed-loop-integration-20260916`) because validated Wulver-derived long-horizon and learned-closed-loop results are incorporated there.

`manuscript/pe-learned-closed-loop-integration-20260916` branches from `manuscript/pe-long-horizon-integration-20260916` at `999d23f` (after the literature-positioning pass) and has not been merged anywhere yet -- a human should review and merge when ready.

## Completed PE Scientific Evidence

| Evidence | State | Canonical location |
|---|---|---|
| Tier-1 closed-loop production, LRU/MRU/random/SIEVE | COMPLETE_VALID | `analysis/closed_loop_tier1_evidence_20260914/` |
| Full-population MRU continuation census | COMPLETE_VALID | raw output in `lafc-evict-dataset` continuation census worktree; compact validation in manuscript repo |
| Mechanistic/offline-to-closed-loop linkage analysis | COMPLETE_VALID | `analysis/closed_loop_mechanistic_analysis_20260914/` |
| Continuation-sensitivity pilot | COMPLETE_VALID | `experiment/continuation-sensitivity-pilot-20260914` |
| Canonical H={4,8,16} candidate-label dataset | COMPLETE_VALID | `paper/sigmod2027/results/candidate_label_stats/y_loss_summary.csv`; `evict_value_v1_wulver_heavy_r1` |
| Long-horizon H={32,64,128} campaign | COMPLETE_VALID | `experiment/pe-long-horizon-production-prep-20260915`; durable PROJECT root `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1` |
| H16/H32/H64/H128 comparative analysis | COMPLETE_VALID | `experiment/pe-h16-h128-comparative-integration-20260916` at `678adcb`; report `analysis/pe_h16_h128_comparative_20260916/PE_H16_H128_COMPARATIVE_REPORT.md` |
| Long-horizon manuscript integration | COMPLETE | `manuscript/pe-long-horizon-integration-20260916` at `545a174`; claim map `paper/performance_evaluation/LONG_HORIZON_CLAIM_TRACEABILITY.md` |
| Learned model attempt 2 | COMPLETE_VALID; publication gate PASS | `experiment/pe-publication-learned-retrain-attempt2-20260915`; durable model backup `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/` |
| Latest-literature positioning pass | COMPLETE | `manuscript/pe-long-horizon-integration-20260916` at `999d23f`; audit `paper/performance_evaluation/PE_LATEST_LITERATURE_POSITIONING_AUDIT.md` |
| Learned closed-loop campaign (learned/LRU/MRU/SIEVE/LFU/random, 10 cells, own held-out protocol) | COMPLETE_VALID | `experiment/pe-publication-learned-closed-loop-20260916` at `30f178d`; final report `analysis/pe_publication_learned_closed_loop_20260916/PE_LEARNED_CLOSED_LOOP_FINAL_REPORT.md`; durable PROJECT root `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916/` (+ `final/`); Slurm jobs 1291048/1291049/1291050 |
| Learned closed-loop manuscript integration | COMPLETE | `manuscript/pe-learned-closed-loop-integration-20260916` at `8abf686`; new Section "A Frozen Learned-Policy Closed-Loop Check" |

## Prepared But Not Run

- Tier-2 LFU/classical baseline campaign (under the **Tier-1** protocol, to extend the existing Tier-1 table with an LFU column comparable to its existing LRU/MRU/random/SIEVE entries): implementation, tests, manifest, aggregator, and launch guard are ready on `experiment/pe-tier2-closed-loop-integration-20260915`; production cells have not been run. **Reassessed 2026-09-16: this is now optional, not required.** The learned closed-loop campaign above already evaluated matched LFU (and LRU/MRU/SIEVE/random) in all 10 cells and that comparison is already integrated into the manuscript -- it uses a *different* held-out protocol than Tier-1, so it does not make a Tier-1-protocol LFU run scientifically redundant in a strict data-identity sense, but nothing in the current manuscript depends on a Tier-1-protocol LFU number. Run it only if a future author specifically wants LFU added to the existing Tier-1 table (Table 9); it is not blocking anything.
- Overnight Tier-2 baseline-capacity campaign: not launched (same reassessment applies).

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

1. ~~Publication-grade learned closed-loop evaluation using the frozen attempt2 model and predetermined leakage-safe evaluation cells.~~ **DONE 2026-09-16**: `experiment/pe-publication-learned-closed-loop-20260916` at `30f178d`, COMPLETE_VALID.
2. ~~Final results-dependent manuscript integration for the learned closed-loop outcome: abstract, results, discussion, limitations, conclusion.~~ **DONE 2026-09-16**: `manuscript/pe-learned-closed-loop-integration-20260916` at `8abf686`.
3. Tier-2 classical baseline completion (LFU under the Tier-1 protocol specifically): **optional**, not required for the current manuscript -- see "Prepared But Not Run" above.
4. Human review and merge of `manuscript/pe-learned-closed-loop-integration-20260916` into whatever the actual submission-target branch is.
5. Final manuscript QA pass (full clean build, full visual read-through, bibliography/reference check) before submission -- no further experiments are currently blocking this.

Do not execute these scientific steps as repository cleanup.

## Persistence Rule

- Long-running local PE computations must run in detached `tmux`.
- Long-running Wulver computations must run through saved `.sbatch` submissions.
- No scientific run may depend on an interactive terminal, agent session, chat session, or SSH connection remaining open.
