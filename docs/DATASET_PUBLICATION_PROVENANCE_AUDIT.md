# Dataset Publication Provenance Audit

Date: 2026-08-11

Scope: local preparation for a proposed `SoroushVahidi/lafc-evict-sample` v0.2 real-data preview. This audit uses local repository evidence only. It is not legal advice.

## Existing Publication Pipeline

The existing Hugging Face/publication scaffold is in `/home/soroush/lafc-evict-dataset`.

Key files:

| File | Role |
|---|---|
| `examples/tiny_candidate_rows.csv` | synthetic v0.1 input rows |
| `scripts/build_sample_release.py` | v0.1 synthetic sample release builder |
| `src/lafc_evict_dataset/release.py` | synthetic release implementation; writes Parquet, manifest, checksums, validation report, README |
| `scripts/export_lafc_evict_parquet.py` | legacy candidate-row CSV/Parquet export helper |
| `scripts/build_real_release.py` and `src/lafc_evict_dataset/real_release_build.py` | memory-safe real-release builder using DuckDB |
| `scripts/validate_release_schema.py`, `scripts/validate_real_release.py` | schema/release validation |
| `scripts/compute_release_checksums.py`, `scripts/repair_release_metadata.py` | checksum and metadata utilities |
| `scripts/publish_to_huggingface.py` | dry-run-first Hugging Face upload helper; not used for this audit |
| `dataset_card/DATASET_CARD.md`, `publication/HF_DATASET_CARD_TEMPLATE.md` | dataset card templates |
| `dataset_card/LICENSE_DATA.md`, `dataset_card/PROVENANCE.md` | data licensing/provenance policy |
| `manifests/source_family_registry.yaml` | machine-readable source-family governance registry |
| `manifests/lafc_evict_v0_1_open_families.json` | selected/excluded families for the earlier intended real open scope |

Reusable for v0.2:

- Parquet partitioning conventions.
- Manifest, checksum, validation, and dataset-card structure.
- Conservative family-governance pattern.
- Dry-run-first upload workflow.

Synthetic-only or needs extension:

- v0.1 release text says synthetic sample and not scientifically benchmarkable.
- `build_sample_release.py` is tied to `examples/tiny_candidate_rows.csv`.
- v0.2 needs real-derived sampling, object-ID pseudonymization, and a new provenance summary.

## Family Redistribution Status

Redistribution labels in this table are conservative local-evidence classifications for public release decisions.

| Family | Original source recorded locally | Source URL recorded locally | License/terms recorded locally | Attribution recorded locally | Redistribution status | Derived rows safe to publish? | Raw trace rows? | Blocker |
|---|---|---|---|---|---|---|---|---|
| `brightkite` | Brightkite / SNAP-derived trace family | `https://snap.stanford.edu/data/loc-brightkite.html` | `source_terms_and_privacy_require_review` in local registry | citation required | `UNCLEAR` | no, not for v0.2 | exclude | local governance marks `blocked_pending_review`; license/privacy review required |
| `citibike` | Citi Bike trip-data derived trace family | `https://citibikenyc.com/system-data` | `source_terms_require_review` in local registry | citation required | `UNCLEAR` | no, not for v0.2 | exclude | local governance marks `blocked_pending_review`; redistribution/privacy review required |
| `cloudphysics` | CloudPhysics / open cache trace collection block I/O family | `https://github.com/cacheMon/cache_dataset` | `open_trace_collection_needs_citation_review` | citation required | `UNCLEAR` | not until final review | exclude | specific source/provenance and attribution terms require confirmation |
| `metacdn` | MetaCDN trace family via open cache trace collection | `https://github.com/cacheMon/cache_dataset`; CMU mirror recorded in KBS docs | `open_trace_collection_needs_citation_review` | citation required | `UNCLEAR` | not until final review | exclude | final license/attribution review not recorded |
| `metakv` | MetaKV trace family via open cache trace collection | `https://github.com/cacheMon/cache_dataset`; CMU mirror recorded in KBS docs | `open_trace_collection_needs_citation_review` | citation required | `UNCLEAR` | not until final review | exclude | final license/attribution review not recorded |
| `twemcache` | Twitter cache trace / Twemcache open trace collection | `https://github.com/twitter/cache-trace`; CMU mirror recorded in KBS docs | `open_trace_collection_needs_citation_review` | citation required | `UNCLEAR` | not until final review | exclude | final license/attribution review not recorded |
| `wiki2018` | Wikimedia public pageviews derived proxy trace | `https://dumps.wikimedia.org/other/pageviews/` | `wikimedia_public_pageviews_with_attribution` in local registry | citation/attribution required | `ALLOWED_WITH_ATTRIBUTION` | yes for a pseudonymized preview, pending final review | exclude raw pageview rows | describe as pageview-derived proxy, not byte-for-byte CDN trace |

## Derived Dataset Semantics

### Cross-Family Evict-Value Dataset

Local source tree:

`/home/soroush/Augmented-caching-fairness/data/derived/evict_value_v1_cross_family_v1`

One row represents one candidate object in one cache eviction decision. The row records the trace/family/capacity/horizon, the decision identifier and time, the candidate object, feature values, and finite-horizon counterfactual eviction-loss labels.

Core columns observed:

- identifiers: `trace_name`, `trace_family`, `dataset_source`, `capacity`, `horizon`, `decision_id`, `decision_t`, `decision_chunk_id`, `candidate_page_id`, `split`;
- labels: `y_loss`, `y_value`;
- features: request/candidate bucket and confidence fields, recency/age, predictor and LRU scores, score/bucket/confidence gaps, cache summary statistics, predictor/LRU disagreement, recent request/hit rates.

The corrected cross-family tree uses folds keyed by held-out family. In a fold directory, the held-out family is excluded from the training rows for that fold. For the v0.2 preview, only `wiki2018` rows are sampled from a fold where `wiki2018` is a training family, not the held-out family.

### Objective-Ablation Scalar Dataset

Local source tree:

`/home/soroush/Augmented-caching-objective-ablation/data/derived/supervision_objective_ablation_v1`

One scalar row represents one candidate object in one cache eviction decision, with the same feature family as the evict-value rows plus multiple label/target columns for the objective-ablation study.

Core additional target columns observed:

- `eviction_loss_label`;
- `next_arrival_label_raw`;
- `next_arrival_label_censored`;
- `next_arrival_censored_flag`;
- `reuse_distance_label_raw`;
- `reuse_distance_label_censored`;
- `reuse_distance_censored_flag`.

The objective-ablation fold metadata records `held_out_family`, training families, validation family, horizon 4, capacities 32/64/128, and pairwise sampling settings. Pairwise rows were not available locally for `wiki2018` in the selected preview fold, so the v0.2 preview uses only scalar objective-ablation rows.

## Recommended Public Dataset Architecture

Recommendation: keep the small v0.2 preview in the existing `SoroushVahidi/lafc-evict-sample` repository, but keep it clearly labeled as a preview. Use separate configs/subsets inside that repo:

- `cross_family_evict_value_v1`;
- `objective_ablation_scalar`.

For a future full benchmark release, use a cleaner repository name such as `SoroushVahidi/lafc-evict`. The sample repository can remain the public preview/workflow-validation channel.

Rationale:

- v0.1 already established `lafc-evict-sample` as a small non-final publication target.
- v0.2 should preserve that history rather than silently converting it into the full benchmark.
- The full release will need larger storage, stricter governance, and likely a stable DOI.

## v0.2 Preview Design

The local preview builder is:

`scripts/analysis/prepare_hf_preview_v0_2.py`

Design choices:

- include only `wiki2018`;
- use deterministic SHA-256 sampling with seed `lafc-evict-sample-v0.2-preview-seed-20260811`;
- sample across capacities 32/64/128;
- sample from both early and late shards per capacity where available;
- pseudonymize `candidate_page_id` as `obj_<24 hex chars>`;
- omit raw trace rows and raw page titles;
- avoid machine-local source paths in release metadata;
- write Parquet under `analysis/huggingface_dataset_preview_v0_2/data/`;
- classify the package as `READY_AFTER_LICENSE_REVIEW`, not ready for upload.

## Release Readiness

Current classification: `READY_AFTER_LICENSE_REVIEW`.

Reason: a local preview can be prepared safely for review using pseudonymized Wiki-derived rows, but public upload should wait for explicit final approval of attribution/license wording and repository naming.
