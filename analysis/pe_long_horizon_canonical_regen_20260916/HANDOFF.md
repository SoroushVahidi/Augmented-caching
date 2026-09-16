# PE long-horizon canonical regeneration handoff

Date: 2026-09-16

## State

Pre-submission STOP. The scientific source/provenance and pilot gates passed, but the submission/resource gate failed on host `al-khwarizmi` because `sbatch` is unavailable and `/mmfs1/scratch/ikoutis/sv96` is not mounted. No corrected production, validation, or aggregation jobs were submitted from this session.

## Why this supersedes the prior long-horizon campaign

The prior H32/H64/H128 campaign at `analysis/pe_long_horizon_production_v1` is preserved as historical provenance but is logically SUPERSEDED for manuscript evidence. Audit root cause: accidental stale/undercovered twemcache provenance/configuration divergence. The old campaign used a different source/provenance path and did not preserve the canonical H16 twemcache physical decision population.

Do not use `analysis/pe_long_horizon_production_v1` or `configs/pe_long_horizon_production` for manuscript results.

Prior campaign state recovered locally:
- production job: `1288536`
- validation job: `1288547`
- aggregation job: `1288548`
- relaunch record: `configs/pe_long_horizon_production/provenance/campaign_relaunch_record.json`
- old scratch run root: `/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1`
- old preserved output summary: `analysis/pe_long_horizon_production_v1`

## Canonical H16 configuration matched

Canonical H16 came from `slurm/evict_value_v1_wulver_heavy_train.sbatch` / `scripts/experiments/canonical/build_evict_value_dataset_wulver_v1.py` with:
- manifest: `analysis/wulver_trace_manifest_full.csv`
- manifest SHA256: `68b37b3511baffd253a159eea96d862753ffe85ba6610a8bf8e649b186da5e22`
- capacities: `32,64,128,256`
- canonical horizons: `4,8,16`
- corrected horizons to generate: `32,64,128`
- split mode: `trace_chunk`
- chunk size: `4096`
- split seed: `7`
- split train/val pct: `70/15`
- history window: `64`
- max requests per trace: `50000`
- max rows per shard: `500000`
- H16 regenerated: `false`

Canonical H16 decision authority:
`/home/soroush/projects/lafc-evict-dataset/repo/release/lafc-evict-v0.1-open-current-contract-preserved/data/decision_view/decision_view.parquet`

## Corrected campaign namespace

- namespace: `pe_long_horizon_canonical_regen_20260916`
- manifest: `configs/pe_long_horizon_canonical_regen_20260916/manifest.json`
- manifest SHA256: `84d268de8cbf82f4bef58aa9003ac324788a430ae0b5460d0297f0a7c36de624`
- run root: `/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_canonical_regen_20260916`
- physical tasks: `20`
- logical family x capacity x horizon cells: `60`
- production array: `0-19%10`
- expected candidate rows if completed: `277995072`
- expected candidate rows per horizon: `92665024`

## Corrected trace sources

All trace names are the canonical manifest trace names used for trace-chunk split hashing.

| family | trace_name | resolved source | rows | SHA256 |
| --- | --- | --- | ---: | --- |
| cloudphysics | `cloudphysics_alibaba_block_head_50k` | `/home/soroush/projects/augmented-caching/repo/data/processed/cloudphysics/trace.jsonl` | 50000 | `fedb3d31ee3c1dd847d25f967392cdf0fdec1e9c1fbb6808772e281bc8d4fb57` |
| metacdn | `metacdn_cdn_202303_head_50k` | `/home/soroush/projects/augmented-caching/repo/data/processed/metacdn/trace.jsonl` | 50000 | `7301f02e1373cb7e06f1ed2c417a8ebe16581319354780f811c8d032391f4e73` |
| metakv | `metakv_kvcache_202206_head_50k` | `/home/soroush/projects/augmented-caching/repo/data/processed/metakv/trace.jsonl` | 50000 | `4c229e841d546cd57a2edb6e3ddc53461ff4dc03d366e9d24e4239a89db7e0de` |
| twemcache | `twemcache_cluster26_sample100_50k` | `/home/soroush/projects/augmented-caching/repo/data/processed/twemcache/trace.jsonl` | 50000 | `62df27062ce2595c843fd91e01d26c9783ff8f165f6b26e983696eda7ada0bb9` |
| wiki2018 | `wiki2018_pageviews_en_50k` | `/home/soroush/projects/augmented-caching/repo/data/processed/wiki2018/trace.jsonl` | 50000 | `3813084bebfcf463ac22a685494b46aea45aded55d314411c9fbb058b5d67608` |

Twemcache full-source proof:
- canonical H16 twemcache candidate rows across all capacities: `14776736`
- canonical H4/H8/H16 twemcache candidate rows: `44330208`
- superseded long-horizon twemcache candidate rows: `10591392`

## Pilot result

Preflight report: `configs/pe_long_horizon_canonical_regen_20260916/provenance/preflight_report.json`

Pilot scope: twemcache and metakv, capacities 32 and 128, H32 decision enumeration compared to canonical H16 physical decision keys.

Result:
- twemcache cap32: generated `37849`, canonical `37849`, unexpected `0`, missing `0`, split identity `true`
- twemcache cap128: generated `31195`, canonical `31195`, unexpected `0`, missing `0`, split identity `true`
- metakv cap32: generated `38032`, canonical `38032`, unexpected `0`, missing `0`, split identity `true`
- metakv cap128: generated `37856`, canonical `37856`, unexpected `0`, missing `0`, split identity `true`

## Submission DAG

No job IDs exist yet for the corrected campaign. Submit only on a Wulver/SLURM host after rerunning preflight and receiving `preflight_status: PASS`.

Commands:

```bash
python scripts/pe_long_horizon_canonical_regen_20260916/build_campaign.py preflight
PROD=$(sbatch --parsable slurm/pe_long_horizon_canonical_regen_20260916_production_TEMPLATE.sbatch)
VAL=$(sbatch --parsable --dependency=afterok:${PROD} slurm/pe_long_horizon_canonical_regen_20260916_validation_TEMPLATE.sbatch)
AGG=$(sbatch --parsable --dependency=afterok:${VAL} slurm/pe_long_horizon_canonical_regen_20260916_aggregation_TEMPLATE.sbatch)
printf 'prod=%s val=%s agg=%s\n' "$PROD" "$VAL" "$AGG"
```

Dependency graph: production array -> afterok validation array -> afterok aggregation.

Monitoring command after submission:

```bash
squeue -j ${PROD},${VAL},${AGG}
```

## Validation criteria after completion

Validation must confirm:
- exact key uniqueness
- exact expected family/capacity/horizon coverage
- H32/H64/H128 decision populations identical to each other
- each H32/H64/H128 population matches canonical H16 exactly for all 20 family x capacity cells
- candidate row count equals decision count times capacity
- no missing/extra decisions
- no duplicate candidates
- generated metric summaries complete
- superseded long-horizon outputs are never mixed into corrected summaries

## Current branch/HEAD

- repo: `/home/soroush/projects/augmented-caching/worktrees/pe-h16-h128-comparative-integration-20260916`
- branch: `experiment/pe-long-horizon-canonical-regen-20260916`
- HEAD: `678adcbb8b6933522aa41e3ea18b400cc647f28a`
- upstream: none configured

## Explicit warnings

- Manuscript repository must remain untouched.
- H16 must not be regenerated.
- Submit corrected production no more than once.
- Do not promote stale long-horizon numerical artifacts into manuscript evidence.
- The existing Figure 4 visual-only dirty files are unrelated and must not be included in a campaign commit.
