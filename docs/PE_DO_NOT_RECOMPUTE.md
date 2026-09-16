# PE (LAFC-Evict) - Do Not Recompute

Last verified: 2026-09-16 09:18 EDT.

This file lists PE evidence that is complete enough to preserve as canonical. Do not regenerate these artifacts merely to reproduce existing results. If a genuine bug is found later, create a new experiment with new provenance instead of overwriting these records.

| Experiment | State | Canonical artifact / branch | Do-not-recompute rule | Legitimate next work |
|---|---|---|---|---|
| Tier-1 closed-loop production: LRU/MRU/random/SIEVE, 230 runs | COMPLETE_VALID | `analysis/closed_loop_tier1_evidence_20260914/` | Do not rerun or overwrite. | Use existing compact validated summaries for manuscript context. |
| Full-population MRU continuation census: 60/60 chunks, 2,363,286 decision-horizon records, H={4,8,16} | COMPLETE_VALID | raw output under `lafc-evict-dataset` continuation census worktree; compact validation in manuscript repo | Do not rerun the census. | Reuse existing validation and summaries. |
| Continuation-sensitivity pilot | COMPLETE_VALID | `experiment/continuation-sensitivity-pilot-20260914` | Do not rerun under the same identity. | Treat any changed design as a new experiment. |
| Mechanistic / offline-to-closed-loop linkage analysis | COMPLETE_VALID | `analysis/closed_loop_mechanistic_analysis_20260914/` | Do not recompute the existing analysis. | Reuse with careful DIRECTLY OBSERVED vs CONSISTENT WITH MECHANISM language. |
| Canonical H={4,8,16} candidate-label dataset | COMPLETE_VALID | `paper/sigmod2027/results/candidate_label_stats/y_loss_summary.csv`; derived dataset `evict_value_v1_wulver_heavy_r1` | Do not regenerate H4/H8/H16. | First run an H16 existence/conversion audit, then reuse canonical H16 for comparison. |
| Long-horizon H={32,64,128} five-family x four-capacity campaign | COMPLETE_VALID | `experiment/pe-long-horizon-production-prep-20260915` at `cc8c1ad`; durable PROJECT root `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1` | DO NOT REGENERATE H32/H64/H128. Production job `1288869`, validation `1288880`, aggregation `1288881` are terminal and successful. | H16 canonical converter / existence audit, then H16/H32/H64/H128 comparative aggregation and interpretation. |
| Publication-grade learned model attempt 2: training, validation/model selection, frozen selected model, test evaluation, inference benchmark | COMPLETE_VALID; PUBLICATION GATE PASS | `experiment/pe-publication-learned-retrain-attempt2-20260915` at `ab36cba`; selected model SHA256 `8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6`; durable model backup `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/` | DO NOT RETRAIN merely to reproduce this run. Test was evaluated after model-selection freeze and was not used for model selection. | Future learned CLOSED-LOOP evaluation using the frozen attempt2 model and predetermined leakage-safe evaluation cells. |
| LFU policy implementation and unit tests | IMPLEMENTED_TESTED; production NOT_RUN | `experiment/pe-tier2-closed-loop-integration-20260915` at `4d2a9fd` | Do not reimplement LFU or its launch guard. | Tier-2 classical baseline production campaign remains a valid future run. |
| Cap32 long-horizon preflight and cap256 timing probe | PREFLIGHT_COMPLETE; TIMING_ONLY | `experiment/pe-long-horizon-preflight-20260915` | Do not cite the timing probe as scientific evidence. Do not rerun as a result. | Use only as historical feasibility/infrastructure context. |

## Explicitly Historical Or Invalid For Publication Claims

- Learned attempt 1, `experiment/pe-publication-learned-retrain-20260915` at `6c17c0b`, is interrupted historical evidence. It has no publication-valid model-selection freeze and no test evaluation. Do not use its model or numbers for final claims.
- First long-horizon production launch `1288536_[0-19]` failed due HOME quota. Downstream `1288547` and `1288548` were cancelled because that launch failed. The successful scratch-backed DAG supersedes these infrastructure failures.
- Cap256 probe `1288047` timed out and was timing-only.
