# PE Long-Horizon Production Completion Report

Generated from existing Slurm accounting and durable PROJECT artifacts only. No long-horizon data was regenerated and no scientific interpretation is made here.

## Campaign Status

- Campaign: `pe_long_horizon_production_v1`
- Branch: `experiment/pe-long-horizon-production-prep-20260915`
- Worktree HEAD at preservation start: `7601a316e6e605052dc5d970d6b233180cbf4e34`
- Overall state: `COMPLETE_VALID`
- Production job: `1288869`
- Validation job: `1288880`
- Aggregation job: `1288881`
- Production array tasks: `20/20 COMPLETED`
- Production failures: `0`
- Validation: `COMPLETED`, exit `0:0`
- Aggregation: `COMPLETED`, exit `0:0`

## Scope

- Families: `cloudphysics`, `metacdn`, `metakv`, `twemcache`, `wiki2018`
- Capacities: `32`, `64`, `128`, `256`
- Production horizons: `32`, `64`, `128`
- Logical cells: `60`
- Physical production tasks: `20`
- H16 regenerated: `false`
- Shared-horizon optimization used: `false`
- Excluded families: `brightkite`, `citibike`
- Excluded production horizons: `4`, `8`, `16`

## Artifact Roots

- SCRATCH root: `/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1`
- Durable PROJECT root: `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1`
- Local compact evidence root: `analysis/pe_long_horizon_production_v1`

## Compact Evidence Preserved In Git

- `analysis/pe_long_horizon_production_v1/manifests/campaign_manifest.json`
- `analysis/pe_long_horizon_production_v1/provenance/campaign_relaunch_record.json`
- `analysis/pe_long_horizon_production_v1/checksums.sha256`
- `analysis/pe_long_horizon_production_v1/summaries/table_A_family_capacity_horizon.csv`
- `analysis/pe_long_horizon_production_v1/summaries/table_B_horizon_pooled.json`
- `analysis/pe_long_horizon_production_v1/summaries/table_C_h16_deltas.json`
- `analysis/pe_long_horizon_production_v1/summaries/table_C_stepwise_deltas.json`

The durable PROJECT campaign monitor log remains external at `/mmfs1/project/ikoutis/sv96/lafc-evict/pe_long_horizon_production_v1/logs/campaign_monitor.log`. Local `logs/` paths are intentionally ignored by repository policy and were not force-added.

## Verification

- Durable PROJECT `COMPLETE.json` count: `20`
- Durable PROJECT `validated_summary.json` count: `20`
- `table_A_family_capacity_horizon.csv`: `60` data records plus header, covering `20` cells for each of horizons `32`, `64`, and `128`
- Checksum manifest: `checksums.sha256`, `87` lines, copied from durable PROJECT root
- Compact preserved artifact SHA256 values matched the durable PROJECT copies at preservation time.

## Historical Infrastructure Failures

These failures are preserved as historical infrastructure context and must not be confused with the successful scientific campaign:

- First production launch `1288536_[0-19]`: all `20` array tasks failed due HOME quota (`OSError: [Errno 122] Disk quota exceeded`).
- First downstream validation `1288547`: cancelled after failed first production launch.
- First downstream aggregation `1288548`: cancelled after failed first production launch.
- Probe `1288047`: timing-only cap256 probe, timed out; not reused as production data.

## Handoff

The H32/H64/H128 generation, validation, aggregation, compact PROJECT copy, checksums, and provenance are terminal. Do not regenerate this campaign. The next scientific step is H16 canonical existence/conversion audit and H16/H32/H64/H128 comparative integration using canonical H16 without regenerating H16.
