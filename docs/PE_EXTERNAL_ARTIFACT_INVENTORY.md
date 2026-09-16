# PE External Artifact Inventory

Last verified: 2026-09-16 09:18 EDT.

Git intentionally does not contain every scientific artifact. Large raw outputs and selected durable backups are tracked here by path, size, SHA where practical, and scientific status.

| Artifact | Status | Local path | Durable path | Size | SHA256 |
|---|---|---|---|---:|---|
| Attempt2 selected learned model | COMPLETE_VALID; publication gate PASS | `/home/soroush/projects/augmented-caching/worktrees/pe-publication-learned-retrain-attempt2-20260915/models/pe_publication_learned_retrain_attempt2_20260915/pe_evict_value_h16_hist_gb_attempt2_20260915.pkl` | `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/model/pe_evict_value_h16_hist_gb_attempt2_20260915.pkl` | 440981 | `8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6` |
| Attempt2 compact model provenance package | COMPLETE_VALID | `/home/soroush/projects/augmented-caching/worktrees/pe-publication-learned-retrain-attempt2-20260915/analysis/pe_publication_learned_retrain_attempt2_20260915/` | `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/provenance/` | compact | see individual JSON files |
| Long-horizon compact PROJECT copy | COMPLETE_VALID | `/home/soroush/projects/augmented-caching/worktrees/pe-long-horizon-production-prep/analysis/pe_long_horizon_production_v1/` | `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1/` | 86524 local compact bytes | checksum manifest SHA `fc67bcbddd8de33dfad2e020e8317cb29fad083abcead9b18ae44e1ca0b6d144` |
| Long-horizon raw outputs | COMPLETE_VALID; external to Git | none copied into Git | `/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1/raw/` and durable compact copy under `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1/` | external raw shards | governed by PROJECT `checksums.sha256` |
| MRU population census raw output | COMPLETE_VALID; manuscript-used compact summaries only | `/home/soroush/projects/lafc-evict-dataset/repo/.claude/worktrees/continuation-mru-population-census-20260914/analysis/continuation_policy_mru_population_census_20260914/outputs/20260914T042528Z_1a29e773a113` | local preserved worktree; not copied to augmented-caching Git | 1674761359 | not recomputed during Query 3 |

Do not duplicate massive raw outputs for cosmetic completeness. Preserve exact paths and checksum manifests, and copy only compact publication/provenance artifacts into Git unless a future artifact policy explicitly says otherwise.
