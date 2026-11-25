# Reproducibility

## Rebuild the report from preserved aggregates

Run from the repository root with Python 3, Matplotlib 3.10, and Tectonic with its standard LaTeX bundle. The build reads the public result tables and never generates observations.

```sh
sha256sum -c post-acceptance-research/MANIFEST.sha256
python3 post-acceptance-research/report-source/figures.py
cd post-acceptance-research/report-source
mkdir -p build
SOURCE_DATE_EPOCH=1795608000 tectonic --outdir build report.tex
```

The editable figure generator consumes the committed Phase 2B timing grid and the Phase 2C-B policy table. Font, TeX and Matplotlib versions may affect rendered PDF bytes. The manifest pins the preserved renderings. `build/report.pdf` is the regenerated report.

## Source and original research entry points

These commands record how the completed experiments were run. They require authorized copies of the digest-pinned historical inputs, frozen checkpoint or stage artifacts as applicable, and the original workload boundaries. The completed scientific campaign is closed. The public package supports verification of the documented aggregates without those inputs.

| Stage | Implementation and configuration |
|---|---|
| November stateless neural | [Data and legacy features](../experiments/neural_eviction_20261121/data.py), [protocol](../experiments/neural_eviction_20261121/protocol.json), [campaign configuration](../experiments/neural_eviction_20261121/campaign.json), [models](../experiments/neural_eviction_20261121/models.py), [training](../experiments/neural_eviction_20261121/training.py), and [replay](../experiments/neural_eviction_20261121/replay.py) |
| Recurrent Phase 1 | [Causal features](../experiments/recurrent_neural_eviction/features.py), [models](../experiments/recurrent_neural_eviction/network.py), [sequences](../experiments/recurrent_neural_eviction/sequences.py), [campaign](../experiments/recurrent_neural_eviction/campaign.py), and [simulation](../experiments/recurrent_neural_eviction/simulation.py) |
| Phase 2A | [Forensic runner](../experiments/cache_supervision_forensics_phase2a/run.py), [public census](../experiments/cache_supervision_forensics_phase2a/public_census.py), [paired LRU and forced MIN](../experiments/cache_supervision_forensics_phase2a/math_core.py) |
| Phase 2B | [Shared exact labeler](../experiments/cache_supervision_phase2b/exact_lru.py), [comparators](../experiments/cache_supervision_phase2b/verify.py), [diagnostics](../experiments/cache_supervision_phase2b/diagnostics.py), [matched replays](../experiments/cache_supervision_phase2b/replay_experiment.py) |
| Phase 2C-A | [Grouped label and feature generation](../experiments/cache_supervision_phase2ca/campaign.py), [reducer](../experiments/cache_supervision_phase2ca/reduce.py) |
| Phase 2C-B | [Input verifier](../experiments/cache_supervision_phase2cb/assets.py), [training and calibration](../experiments/cache_supervision_phase2cb/learning.py), [online feature builder and replay](../experiments/cache_supervision_phase2cb/deploy.py), [stage runner](../experiments/cache_supervision_phase2cb/run.py), [reducer](../experiments/cache_supervision_phase2cb/reducer.py) |

Original entry points (arguments are placeholders for authorized inputs and separate output directories):

```sh
# November, with the original Torch and scikit-learn environment
python3 experiments/neural_eviction_20261121/run.py all \
  --config experiments/neural_eviction_20261121/campaign.json \
  --trace-root "$TRACE_ROOT" --original-model-root "$KBS_MODEL_ROOT" \
  --out "$NEURAL_OUT" --device cuda

# Recurrent Phase 1, with the historical sibling worktree layout
python3 experiments/recurrent_neural_eviction/campaign.py auto --out "$RECURRENT_OUT" \
  --device cuda --aggregation

# Phase 2A, from the experiment directory with its pinned upstream artifacts
cd experiments/cache_supervision_forensics_phase2a
python3 run.py --workers 6
python3 public_census.py --out runs/phase2a --workers 5
cd ../..

# Phase 2B, with the Phase 2A groups and frozen November source artifacts
python3 experiments/cache_supervision_phase2b/replay_experiment.py --stage students --workers 7
python3 experiments/cache_supervision_phase2b/replay_experiment.py --stage neural --workers 7

# Phase 2C-A, with per-family execution and reducer evidence
python3 experiments/cache_supervision_phase2ca/campaign.py plan "$TRACE_ROOT" "$CA_OUT"
python3 experiments/cache_supervision_phase2ca/campaign.py produce "$TRACE_ROOT" "$CA_OUT" --family "$HISTORICAL_FAMILY"
python3 experiments/cache_supervision_phase2ca/reduce.py "$CA_OUT"

# Phase 2C-B, after staging inputs and verifying the expected manifests
python3 experiments/cache_supervision_phase2cb/run.py verify "$CB_OUT" "$CA_OUT" "$PHASE2B_INPUT" "$TRACE_ROOT"
python3 experiments/cache_supervision_phase2cb/run.py "$STAGE" "$CB_OUT" "$CA_OUT" "$PHASE2B_INPUT" "$TRACE_ROOT" --family "$HISTORICAL_FAMILY"
python3 experiments/cache_supervision_phase2cb/reducer.py "$CB_OUT"
```

Run the original Phase 2C-B stages in order A, B, C, D across the seven declared outer folds. Use the [runtime and experiment settings](phase2cb-configuration.json), single-threaded numerical libraries, and the appropriate producer, input, stage-validation and reducer receipts. Several early runners assume the original sibling worktree and output layout. Phase 2A supports explicit `PHASE2A_*` source overrides enumerated in [sources.py](../experiments/cache_supervision_forensics_phase2a/sources.py). Phase 2B requires both pre-generated labels and existing checkpoint manifests. Phase 2C-A and 2C-B reducers require original receipt and output contracts. A standalone clone has the method code and metrics. It does not have all licensed inputs or original runtime evidence. [Access terms](LICENSING.md) · [Provenance](PROVENANCE.md)

## Table conventions

- Positive relative miss reduction is favorable. The policy CSV uses fractions and the report formats percentages. Whole-request CPU ratios for new Phase 2C-B portfolios use the common instrumented LRU path. Retrieved timing uses its source harness.
- The low-budget 1,176 rows predate the three exactly nested Phase 2C-A budgets. Model fits are counted as components, with two-part fits storing both components in one file.
- Blank advancement fields denote comparisons absent or inapplicable for that row. The benchmark CSV has no missing-as-zero convention. Omitted comparator/horizon pairs were absent from the verified table.
- Diagnostic shortlist and horizon tables reflect fixed-state opportunity. Closed-loop misses reflect each policy's cache trajectory. Timing components overlap when one is nested in another.
