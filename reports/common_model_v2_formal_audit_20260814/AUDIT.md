# Common-Model Objective Control V2 — Formal Audit

- Status: `COMPLETE` / `INTEGRITY_PASS` / `ARCHIVED`
- Date: 2026-08-14

## Campaign identity

- experiment: Common-Model Objective Control V2
- source commit: `4e9298d08ecee248d14d41b9ef5952d1ce4eead4`
- Slurm parent job: `1176758`
- job name: `common-v2`
- Wulver output path:
  `/mmfs1/project/ikoutis/sv96/common_model_v2_src_4e9298d/analysis/common_model_objective_control_wulver_v2/`
- local immutable archive path:
  `/home/soroush/kbs_wulver_archives/common_model_v2_1176758_20260814/`

## Scheduler health

- 21/21 array tasks completed
- all tasks `ExitCode 0:0`
- reducer completed and wrote campaign-level files
- Slurm stdout/stderr: 21 `.out` + 21 `.err` under `slurm/logs/` on Wulver and in the local archive (`slurm/logs/`)

## Integrity

- integrity audit: `PASS`
- completed units: 21
- expected rows: 84
- observed rows: 84
- unique keys: 84
- no duplicate family/capacity/objective rows
- no NaN/Inf in `misses`, `miss_ratio`, `validation_mean_regret`
- trace SHA-256s match the audit manifest per unit
- byte-level hash verification (local archive vs Wulver): `PASS`
  - 150/150 analysis files + 47/47 supporting provenance files, zero missing / zero extra / zero mismatch
- reducer status: `COMPLETE`

## Objective aggregate table

Recomputed independently from the verified `summary.csv` (84 rows; 21 units x 4 objectives).

| Objective | total misses | macro mean miss ratio | mean cell rank | rank-1 cells | rank-4 cells | mean validation regret |
|---|---|---|---|---|---|---|
| objective_eviction_loss | 571,976 | 0.68092 | 2.000 | 7/21 | 3/21 | 0.05238 |
| objective_pairwise | 577,339 | 0.68731 | **1.571** | **13/21** | 0/21 | **0.01190** |
| objective_reuse_distance | 615,850 | 0.73316 | 2.952 | 1/21 | 4/21 | 0.08452 |
| objective_next_arrival | 627,392 | 0.74689 | 3.476 | 0/21 | 14/21 | 0.05595 |

Ordering by total misses (ascending):

```
objective_eviction_loss < objective_pairwise < objective_reuse_distance < objective_next_arrival
```

Ordering by macro mean miss ratio:

```
objective_eviction_loss < objective_pairwise < objective_reuse_distance < objective_next_arrival
```

Head-to-head per-cell wins (A beats B / ties / A loses):

| A vs B | A wins | ties | A loses |
|---|---|---|---|
| eviction_loss vs next_arrival | 15 | 3 | 3 |
| eviction_loss vs reuse_distance | 13 | 4 | 4 |
| eviction_loss vs pairwise | 4 | 3 | 14 |
| pairwise vs next_arrival | 17 | 3 | 1 |
| pairwise vs reuse_distance | 17 | 3 | 1 |
| reuse_distance vs next_arrival | 17 | 3 | 1 |

## Corrected scientific interpretation

This section supersedes any earlier audit prose. The earlier audit-output claimed
an ordering `eviction < pairwise < reuse < next` while simultaneously describing
`objective_eviction_loss` as "materially worse" and claimed this strengthened an
objective-causality hypothesis. Those two statements are mutually inconsistent;
the recomputation below resolves the contradiction using only the verified
`summary.csv`, `completion_manifest.json`, and `integrity_audit.json`.

1. **Best by total misses.** `objective_eviction_loss` is nominally best
   (571,976 misses), but its margin over `objective_pairwise` (577,339) is only
   ~0.94%. These two are effectively tied at the aggregate level.

2. **Worst by total misses.** `objective_next_arrival` (627,392) is worst, and is
   also the least consistent: 0/21 rank-1 cells and 14/21 rank-4 cells.

3. **Valid V2 pairwise rank.** `objective_pairwise` is second by aggregate totals
   (~0.9% behind eviction_loss) but is the **best by per-cell rank** (mean 1.571,
   13/21 rank-1, never last) and by validation regret (0.0119, roughly 4.4x better
   than eviction_loss). On a per-cell basis pairwise wins 14 of 21 cells against
   eviction_loss.

4. **eviction_loss vs alternatives.** `objective_eviction_loss` performs
   **better than or tied with** every alternative at the aggregate level, and is
   clearly better than `objective_next_arrival` and `objective_reuse_distance`
   (beat each in 13-15 of 21 cells). It is not "materially worse" than anything.
   The prior statement that eviction_loss was "materially worse" is **false**.

5. **Objective-causality hypothesis.** The matched common-model V2 result does
   **not** support the hypothesis that the eviction-value training objective
   itself is responsible for poor performance. In a matched common model
   (identical folds, windows, features, architecture, seed), training on
   `objective_eviction_loss` performs at least as well as the alternatives.
   Whatever performance deficit the full `evict_value_v1` pipeline exhibits is
   therefore **not attributable to the eviction-loss training objective per se**
   in this matched control. This **weakens** (and is inconsistent with) the
   stronger "objective-causality" claim. No causal claim is asserted beyond the
   matched common-model setting.

6. **V1 scalar continuity.** V2 reproduces V1 exactly for the three scalar
   objectives (`objective_eviction_loss`, `objective_next_arrival`,
   `objective_reuse_distance`) on the regression unit `brightkite_cap32`
   (`SCALAR_V1_V2_EXACT_EQUIVALENCE`: keys, misses, miss ratio, validation
   regret, trace hash, seed, model arrays, model SHA-256). Conclusions that rest
   only on V1 scalar behavior of these three objectives remain valid.

7. **V1 pairwise invalid and superseded.** V1 `objective_pairwise` is
   `INVALID_FOR_FINAL_OBJECTIVE_COMPARISON` because the V1 runner discarded
   `label_i_preferred` and trained candidate-ID ordering pressure instead of the
   intended pairwise target. V2 corrects the orientation
   (`PAIRWISE_V2_SEMANTICS_VERIFIED`). V1 pairwise must not be used as evidence.

## Limitations

- Aggregate totals weight families/capacities by their differing request counts;
  per-cell rank and validation regret weight each of the 21 units equally.
- The aggregate eviction-vs-pairwise gap (~0.9% of total misses; 0.006 macro-mean
  miss ratio) is small; the two are effectively tied by totals while pairwise
  dominates per-cell and by validation regret.
- No statistical significance test was run; this is a descriptive recomputation
  of the verified campaign output.
- Model binary artifacts (`objective_*.npz`) are preserved only in the local
  immutable archive, not in Git; their SHA-256 values are recorded per row in
  `summary.csv`.
- This audit does not alter the manuscript or rebuttal conclusions; manuscript
  integration remains a separate explicit step.

## Relationship to manuscript hypotheses

The manuscript hypothesis that the eviction-value **training objective** drives
poor learned-caching performance is not supported by this matched common-model
control. The V2 result is consistent with the performance deficit originating
elsewhere (e.g., deployment/scoring path, model family, or training dynamics)
rather than the choice of the eviction-loss objective label itself. The V2
control is now the valid common-model comparison; V1 is superseded.

## Provenance chain

```
Wulver production campaign (job 1176758, commit 4e9298d)
  -> byte-verified local immutable archive
     /home/soroush/kbs_wulver_archives/common_model_v2_1176758_20260814/
  -> curated Git subset
     analysis/common_model_objective_control_wulver_v2/  (JSON + CSV only)
     reports/common_model_v2_formal_audit_20260814/
```

- Wulver -> archive: rsync + SHA-256 (150 + 47 files, all match).
- Archive -> Git: only `summary.csv`, `completion_manifest.json`,
  `integrity_audit.json`, and per-unit `summary.json` / `metadata.json` /
  `config_snapshot.json`; no binaries, no Slurm logs, no local archive directory
  inside the Git working tree.

## Machine-readable support

- `aggregate_recheck.json` in this directory: independent recomputation of
  aggregates, orderings, per-cell ranks, and head-to-head results.