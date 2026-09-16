# PE Publication-Grade Learned Model Attempt 2 Report

Generated from existing artifacts only. No metrics were recomputed for this report.

## Status

- Attempt: `pe_publication_learned_retrain_attempt2_20260915`
- Branch: `experiment/pe-publication-learned-retrain-attempt2-20260915`
- Base/training HEAD: `4d2a9fd18fd8c67b19dcccfea0b0769ad784f6d1`
- Final state: `COMPLETE_VALID`
- Publication gate: `PASS`
- Model selection frozen before test: yes, frozen at `2026-09-16T01:34:59.310323+00:00`
- Test evaluated after freeze: yes
- Test used for model selection: no
- Attempt1 model used: no

## Inputs

- Dataset manifest: `analysis/pe_publication_learned_retrain_attempt2_20260915/dataset_manifest.json`
- Dataset manifest SHA256: `268c72fe7a59f4327bd5df49fcdc6b7cbd3ecc183ad46974899202e17a038e1e`
- Feature manifest: `analysis/pe_publication_learned_retrain_attempt2_20260915/feature_names.json`
- Feature manifest SHA256: `de7c1a2bb0a6f23a421635a82c53c1a9be78a36e2e5e62a744826d4fef9522bd`
- Split manifest: `analysis/pe_publication_learned_retrain_attempt2_20260915/split_manifest.json`
- Split manifest SHA256: `4783d96b1ecc242bf726943ba7f960605e3502ceb389cb9866ab17e93cf54b90`
- Target: `y_loss`
- Horizon: `H=16`
- Families: `cloudphysics`, `metacdn`, `metakv`, `twemcache`, `wiki2018`
- Capacities: `32`, `64`, `128`, `256`

## Integrity Gates

- Feature leakage gate: `PASS`
- Split integrity: `PASS`
- Validation disjoint from training: `true`
- Test disjoint from training: `true`
- Test disjoint from validation: `true`
- No decision ID in more than one split: `true`
- No identical candidate key crosses splits: `true`
- No trace time overlap: `true`
- H16 only: `true`

## Model Selection

- Candidate models: `hist_gb`, `ridge`
- Selection rule:
  - minimize validation mean decision-level regret
  - break ties by non-tied mean regret
  - then optimal-set selection rate on non-tied decisions
  - then validation MAE
  - then inference cost
- Selected model: `hist_gb`
- Selected model class: `sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor`
- Selected model path: `models/pe_publication_learned_retrain_attempt2_20260915/pe_evict_value_h16_hist_gb_attempt2_20260915.pkl`
- Selected model SHA256: `8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6`
- Candidate `hist_gb` SHA256: `2fafbf89dd4baaf1b419d5bead4354c2065bd445ec0ac2af4c49e342a51f9ea8`
- Candidate `ridge` SHA256: `2ca6c0aafe5275a65f1317db2cb8dfaa624e652c43b2219b81b7d6e653a4cf89`

## Validation Metrics

Selected validation global metrics for `hist_gb`:

- Rows: `1440000`
- Decisions: `12000`
- Informative decisions: `5226`
- All-tied decisions: `6774`
- MAE: `2.6669223447850174`
- RMSE: `3.203083013146604`
- Mean regret: `0.06908333333333333`
- Median regret: `0.0`
- Non-tied mean regret: `0.1586299272866437`
- Optimal-set selection rate: `0.9318333333333333`
- Non-tied optimal-set selection rate: `0.8434749330271718`
- Strict pairwise accuracy: `0.4849276578757466`

## Test Metrics

Global test metrics:

- Rows: `17385568`
- Decisions: `147622`
- Informative decisions: `55657`
- All-tied decisions: `91965`
- MAE: `2.5816224770227745`
- RMSE: `3.081917755217113`
- Mean regret: `0.06240939697335086`
- Median regret: `0.0`
- Non-tied mean regret: `0.16553173904450474`
- Optimal-set selection rate: `0.9381596239042961`
- Non-tied optimal-set selection rate: `0.8359775050757318`
- Strict pairwise accuracy: `0.49162198321154754`

Per-family and per-capacity test metrics are preserved in `test_results.json`.

## Inference Benchmark

From `inference_benchmark.json`:

- cap32 estimated 50k replay: `43.14126225654036`
- cap32 feature time: `0.0007722457614727318`
- cap32 predict time: `0.00007656254805624485`
- cap32 selection time: `0.000014016935601830482`
- cap128 estimated 50k replay: `949.5195600902663`
- cap128 feature time: `0.018617366359103472`
- cap128 predict time: `0.0002773195831105113`
- cap128 selection time: `0.00009570525959134102`

## Reproducibility

- Clean-process reproducibility: `PASS`
- Clean-process sample MAE: `1.7905310181453413`
- Prediction digest: `53e1bc2921a43564af2eb13a8ccbe33284dab9b5377512a59c327a90f3dc37eb`
- Python: `3.12.3`
- sklearn: `1.8.0`
- numpy: `2.3.5`
- pandas: `3.0.2`

## Durable Model Backup

- Durable path: `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/`
- Remote selected model SHA256 verified: `8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6`
- Remote package includes selected model plus compact provenance markers.

## Handoff

This attempt is canonical learned-model evidence for the PE line. Do not retrain merely to reproduce this run. Future learned closed-loop evaluation should use the frozen selected model and the predetermined leakage-safe evaluation cells recorded in `future_closed_loop_manifest.json`.
