# evict_value_v1_fair_v1 — frozen training protocol

**This protocol is frozen before training and before any evaluation
result on the canonical traces is observed.** See
`configs/evict_value_v1_fair_training_protocol.json` for the
machine-readable version; this document explains the reasoning. Any
change after seeing canonical-trace evaluation results requires a new
protocol ID and documented justification (see "Amendment policy" below).

## Why this protocol exists

`models/evict_value_wulver_v1_best.pkl` (the model behind the manuscript's
headline end-to-end table) was trained on chunk-level splits drawn from
the same 7 traces and same `[0, 50000)` request range later used to score
it — a confirmed train/test overlap (`analysis/reviewer_fairness/
evict_value_v1_overlap_audit.{json,md}`). This protocol trains a
replacement, `evict_value_v1_fair_v1`, with **zero** request-level or
trace-level overlap with the canonical evaluation range.

## Disjoint training corpus (Section 3 of the fairness task)

Candidates considered, in the task's stated preference order:

1. **Separate trace instances from the same workload families** — not
   available; each family here has exactly one canonical source file.
2. **Non-overlapping time window from the same source dataset** — checked
   directly: every one of the 7 raw source files
   (`data/raw/{family}/...`) has strictly more than 50,000 usable records
   (twemcache/metacdn/cloudphysics/metakv raw files: ~100,000 rows each;
   citibike: ~1.9M rows; brightkite: ~4.7M rows; wiki2018: ~100,000 rows).
   **Selected.**
3. Non-overlapping shards/clusters — not needed, (2) sufficed for all 7.
4. Separate workload families — not needed.
5. Prefix-only `[0,10000)` training — explicitly **not** used; a strictly
   better option (2) was available for every family, so the weakest option
   was not taken merely because it would have been easiest.

**Verified, not assumed**, that this later time range is genuinely
disjoint and genuinely a continuation of the same processing: every
family's dataset parser (`src/lafc/datasets/*.py`, unmodified) takes the
first `limit` parsed records in file order (`if limit is not None and
len(records) >= limit: break`). `scripts/build_evict_value_v1_fair_corpus.py`
re-parses each raw source with `limit=100000` and **asserts** that the
first 50,000 reparsed records exactly equal the existing canonical
`data/processed/{family}/trace.jsonl` before extracting
`records[50000:100000)` — this passed for all 7 families (see
`data/processed_fair_v1/PROVENANCE.json`). This proves the disjoint slice
comes from the same source, processed the same way, and cannot contain any
record that appears in the canonical `[0, 50000)` evaluation range.

Output: `data/processed_fair_v1/{family}/trace.jsonl`, ~50,000 records per
family (49,999 for metakv/metacdn, whose raw source has 99,999 lines).

## Candidate-dataset construction

Reuses the existing, unmodified `trace_chunk` split machinery
(`src/lafc/evict_value_wulver_v1.py:assign_split`) — **but applied only
within `data/processed_fair_v1/`**, so the 70/15/15 train/val/test split
happens entirely inside the disjoint `[50000, 100000)` region. No chunk of
any kind is ever drawn from `[0, 50000)`. This makes the "no training
label crosses the score boundary" requirement trivial to satisfy: the
score boundary (`10000`) is a property of the canonical traces, which this
corpus never touches at all.

**Scope reduction, documented, not hidden:** capacities `[32, 64, 128]`
(matching evaluation; skips 256) and horizon `[4]` only (not the original
`{4,8,16}` sweep). Reusing `horizon=4` is a modeling-prior choice, not
leaked test information: the original ablation's finding that shorter
horizons produce better labels is a statement about the label-construction
mechanism in general, not about the content of the 7 canonical traces
specifically, and it was not re-derived by looking at canonical-trace
performance in this session. This scope reduction was made **before**
building the dataset, for tractability within this session's time budget,
and is recorded here rather than silently narrowing the search after the
fact.

## Model selection

Reuses `scripts/train_evict_value_wulver_v1.py` **unmodified** — the same
selection methodology already used to select the contaminated `heavy_r1`
model, now applied to the disjoint corpus instead. It fits all three
candidate families (`ridge`, `random_forest`, `hist_gb`) on
`data/processed_fair_v1` training rows and selects by minimum
`val_mean_regret` (mean regret vs. oracle) on the `data/processed_fair_v1`
**validation** split — never on the canonical traces, never on the
`data/processed_fair_v1` internal test split (that split is a sanity
check, not used for selection). Tie-break: lower `val_mae`, then lower
`val_rmse`. Seed fixed at 0. The full comparison table (all 3 families,
all attempted configurations) is preserved, not just the winner.

**Output-path safety**: the script hardcodes its selected-best artifact's
filename to `evict_value_wulver_v1_best.pkl` inside whatever
`--models-dir` is passed. To guarantee zero risk of overwriting the
canonical `models/evict_value_wulver_v1_best.pkl`, this run points
`--models-dir` at a separate staging directory
(`models/fair_v1_staging/`); the winning artifact is copied to
`models/evict_value_v1_fair_v1.pkl` only after the run completes, as a
distinct final step, not by the training script itself.

## Evaluation (kept separate from training, run afterward)

The frozen, selected model is evaluated on the **canonical** traces
(`data/processed/`, never `data/processed_fair_v1/`) using this fairness
campaign's common protocol: history `[0, 10000)` (cache/state warm-up
only, no model retraining), scored suffix `[10000, 50000)`, capacities
32/64/128, model frozen throughout (`model_frozen_during_test: true`,
`online_adaptation_during_test: false` — `evict_value_v1` has no online
training loop at all, matching its original design). Uses the existing
`scripts/experiments/run_reviewer_fairness.py --policy evict_value_v1
--evict-value-model models/evict_value_v1_fair_v1.pkl`.

## Amendment policy

This protocol must not be edited after any canonical-trace evaluation
result is observed. A change after that point requires a new protocol ID
(`evict_value_v1_fair_v2`, etc.) and an explicit reason recorded in a new
section of this file, never a silent edit to the frozen configuration
above.
