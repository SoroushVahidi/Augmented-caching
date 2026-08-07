# Reviewer 1 Revision Roadmap

Permanent tracking document for the four Reviewer 1 major concerns raised on
`evict_value_v1`. This is a living status record, not reviewer-response prose
— manuscript text is written separately, later, once each concern's
experiments are complete.

Companion machine-readable file: `configs/reviewer_revision_roadmap.json`
(same content, structured for scripts/CI to consume).

Status vocabulary used throughout:

| Status | Meaning |
|---|---|
| `NOT_STARTED` | No work begun. |
| `IMPLEMENTED` | Code/protocol exists but the run(s) have not been (fully) executed. |
| `RUNNING` | A job is currently executing. |
| `PARTIAL` | Some but not all planned rows/artifacts exist. |
| `COMPLETE` | All planned rows/artifacts exist and pass integrity checks. |
| `BLOCKED` | Cannot proceed until an external condition changes. |
| `REJECTED` | Considered and explicitly not pursued (with reason recorded). |
| `SUPERSEDED` | Replaced by a later protocol version. |

Last updated: 2026-08-07 (audit of concurrently running jobs at that
timestamp; see each concern's "current jobs" field — these move quickly and
must be re-checked before being cited in the manuscript).

---

## Concern 1 — Comparison with existing learned cache-replacement methods

**Reviewer concern (paraphrase):** the manuscript does not compare
`evict_value_v1` against enough existing learned cache-replacement methods,
and the comparisons that do exist are not clearly fair (different
resources/windows/seeds).

**Scientific question:** under an identical, frozen evaluation protocol
(same traces, same score window, same capacities, same denominator), how does
`evict_value_v1` perform against LRB, 3L-Cache, CACHEUS, HALP, and the
original manuscript's lightweight baselines?

**Response strategy:** build one frozen fair-comparison protocol
(`reviewer_fairness_v1` / `reviewer_fairness_cross_family_v1`) that (a) gives
every external learned baseline its official/faithful implementation under a
common scoring window, and (b) retrains `evict_value_v1` itself under a
leave-one-family-out cross-family protocol so its own training-family
contamination is removed before the head-to-head comparison.

**Experiments:**
- `analysis/reviewer_fairness/policy_comparison_lrb.csv` — LRB fair-window comparison. **COMPLETE.**
- `analysis/reviewer_fairness/policy_comparison_three_l_cache.csv` (+ `/home/soroush/Augmented-caching-3l-cache/analysis/external_learned_baselines/three_l_cache/`) — 3L-Cache, primary batch_size=4096, six-setting grid sensitivity-only. **COMPLETE.**
- `analysis/reviewer_fairness/policy_comparison_cacheus.csv` (+ `/home/soroush/Augmented-caching-cacheus/analysis/external_learned_baselines/cacheus/`) — CACHEUS, official author code, upstream seed 123 preserved. **COMPLETE.**
- `analysis/reviewer_fairness/policy_comparison_halp.csv` — HALP fair-window comparison. **COMPLETE** (evaluation-resource equivalence: yes; training-resource equivalence with `evict_value_v1`: no — documented caveat, not resolved).
- All original manuscript baselines (LRU, SIEVE, FIFO-Reinsertion, blind_oracle_lru_combiner, REST v1, Trust-and-Doubt, Predictive Marker, Offline Belady as oracle-only) — 21/21 fair-window rows each. **COMPLETE.**
- `evict_value_v1_cross_family_v1` corrected retraining (leave-one-family-out, 7 folds × 4 objective-agnostic model families) — tmux `evict_cross_family_resume`, memory-bounded resumed pipeline. **RUNNING** (1/7 folds fully complete — `brightkite`: dataset+train+model+sha256 `b81e43d3...`, winner=`hist_gb` by val mean regret; fold 2/7 `citibike` stage-1 dataset build in progress; 1/7 final models complete).
- Cross-family held-out evaluation (7 families × 3 capacities = 21 rows) — **NOT_STARTED** (blocked on model completion, 6 folds remaining).
- `analysis/reviewer_fairness_v5/fairness_certificate.{json,md}` — **PARTIAL**, accurately reflects `evict_value_v1_cross_family_v1` as `NOT_RUN` pending training completion; all other policies already certified.
- `analysis/reviewer_fairness_v5/primary_comparison.csv` / `oracle_comparison.csv` — **NOT_STARTED** (blocked on certificate completion).
- **Legacy canonical LRB (PID 113981, archival only, does not block the primary comparison):** running ~22.5h at 100% CPU with an empty log and empty output directory. Read-only diagnostic (2026-08-07) found this **not** to be stuck — classified `ACTIVE_COMPUTE_LIKELY_VALID`. Evidence: CPU time actively accumulates (+11s over an 11s wall-clock window), `/proc/io` byte counters are unchanged over that window (in-memory compute, not blocked I/O), `lib_lightgbm.so` is loaded as expected. The script (`scripts/experiments/run_lrb_external_baseline.py`) evaluates `evict_value_v1` at the full 50,000-request budget for all 7 traces × 3 capacities at the same 75–316 ms/eviction-decision cost measured in Concern 4 — a ~25–35h total runtime is arithmetically plausible from that alone — and it writes its CSV/JSON/MD outputs only once, at the very end (no incremental output), with stdout block-buffered on the non-TTY log redirect (explains the empty log independent of progress). Recommendation: **leave running**; no evidence of a stall.

**Theory/motivation additions:** none required; this is purely an empirical
fairness-protocol concern.

**Current jobs/tmux sessions:** `evict_cross_family_resume` (memory-bounded,
`--memory-guard-gb 45`, repaired after an earlier OOM kill; repair verified
implementation-only, see `docs/evict_cross_family_oom_diagnosis.md`).

**Artifacts/output paths:** `analysis/reviewer_fairness/`,
`analysis/reviewer_fairness_cross_family_v1/`,
`analysis/reviewer_fairness_v5/`, `models/cross_family_v1_staging/`,
`data/derived/evict_value_v1_cross_family_v1/`.

**Blockers:** cross-family training wall-clock (7 folds × 3 model families
each); nothing scientific.

**Completion criteria:** all 7 cross-family models trained and audited
(no held-out-family leakage into train/val), model registry frozen
(`MODEL_SELECTION_FROZEN=true`), 21-row held-out evaluation complete and
clean, fairness certificate shows `evict_value_v1_cross_family_v1: PASS`,
`primary_comparison.csv`/`oracle_comparison.csv` built and machine-checked.

**Manuscript sections likely to change:** related-work/comparison table,
end-to-end results section, limitations (HALP training-resource caveat).

**Reviewer-response evidence needed:** the frozen `primary_comparison.csv`
plus the fairness certificate, and (once authorized) the frozen paired
statistical analysis over it.

---

## Concern 2 — Justification for the finite-horizon eviction-loss objective

**Reviewer concern (paraphrase):** it is not clear why finite-horizon
eviction loss is the right supervision target, versus other plausible
targets (e.g. predicting next-arrival time, reuse distance, or a pairwise
preference between candidates).

**Scientific question:** holding candidate generation, features, model
family, folds, capacities, and evaluation protocol fixed, does the
finite-horizon eviction-loss target outperform the alternative supervision
targets under the same leave-one-family-out held-out evaluation?

**Response strategy:** add a motivating explanation/example for why
eviction-loss supervision is decision-aligned in a way the alternatives are
not, then empirically test that claim with a frozen 4-objective ablation
that otherwise matches Concern 1's cross-family protocol.

**Experiments:**
- Motivating example added to the manuscript discussion (`git log`:
  `1dec44f docs: add motivating example for eviction-loss supervision`,
  `df0f0b9 fix: correct motivating example to match actual eviction-loss
  semantics`). **COMPLETE** (prose only; not gated on experiments).
- `configs/supervision_objective_ablation_v1.json` /
  `docs/supervision_objective_ablation_protocol.md` — frozen protocol,
  4 objectives (finite-horizon eviction loss, next-arrival prediction,
  reuse-distance prediction, pairwise preference) × 7 held-out families.
  **IMPLEMENTED.**
- `objective_ablation_pipeline` tmux — dataset build + training for all
  4 objectives × 7 folds = 28 models. **RUNNING** (5/7 folds complete —
  brightkite, citibike, cloudphysics, metacdn, metakv — i.e. 20/28 models;
  fold 6/7, `twemcache`, mid-dataset-build; `wiki2018` not yet started).
- Model registry freeze (`analysis/supervision_objective_ablation_v1/model_registry.json`) — **NOT_STARTED** (blocked on all 28 models).
- Held-out evaluation (4 objectives × 7 families × 3 capacities = 84 rows) — **NOT_STARTED** (blocked on registry freeze).
- Same-example / fairness alignment audit (`same_example_audit.json`, `fairness_audit.json`) — **NOT_STARTED**.

**Theory/motivation additions:** motivating example (done); no further
theory planned pending the ablation's outcome.

**Current jobs/tmux sessions:** `objective_ablation_pipeline`.

**Artifacts/output paths:** `data/derived/supervision_objective_ablation_v1/`,
`models/supervision_objective_ablation_v1/`,
`analysis/supervision_objective_ablation_v1/`.

**Blockers:** wall-clock only (4 remaining folds × dataset build + 4-objective
training each).

**Completion criteria:** 28/28 models complete and audited, registry frozen,
84/84 held-out rows complete and clean, same-example/fairness audits pass.

**Manuscript sections likely to change:** the supervision-target motivation
subsection, a new ablation table/figure, limitations if the ablation is
inconclusive or mixed.

**Reviewer-response evidence needed:** the frozen 84-row comparison plus
same-example audit confirming the four objectives were compared on
identical candidate/state populations.

---

## Concern 3 — Offline/online gap and the distribution-shift explanation

**Reviewer concern (paraphrase):** there is an unexplained gap between
strong offline (held-out) learning quality and weak online (deployed) cache
performance; the manuscript's proposed distribution-shift explanation is
speculative and untested.

**Scientific question:** is there measurable train/deployment state shift;
does it cause trajectory divergence and prediction-quality degradation; does
on-policy (DAgger-style) state collection reduce that shift; and if so, does
reducing the shift actually improve held-out miss performance?

**Response strategy:** implement a reduced two-condition ablation
(`OFF_POLICY_LRU` vs `DAGGER_ITER1`, the top two of five originally
prioritized conditions, reduction explicitly frozen before launch under
compute-budget authorization) across all 7 folds × 3 capacities, with
paired state-shift and trajectory-divergence diagnostics, then run the
predeclared statistical analysis only once complete.

**Experiments:**
- `configs/distribution_shift_ablation_v1.json` /
  `docs/distribution_shift_ablation_protocol.md` — frozen protocol,
  reduced matrix (`DAGGER_ITER2`, `LEARNED_CONTINUATION`, `MIXED_50_50`
  explicitly deferred to a v2). **IMPLEMENTED.**
- Original 9-hour pass (`distribution_shift_9h`) — stopped cleanly at its
  wall-clock budget with 12/42 primary rows (2/7 folds: brightkite,
  citibike). **STOPPED CLEANLY AT BUDGET (partial), not a failure.**
- 2-hour resumed continuation (`distribution_shift_2h_continue`) — resumed
  correctly (skipped already-complete folds), currently evaluating fold 3/7
  (`cloudphysics`). **RUNNING**, 17/42 as of last check. The process has now
  run ~4.5h wall time, well past its own `--max-wall-hours 2` budget, but
  continues to produce valid rows — the budget check evidently only fires
  between folds (same behavior as the original 9h pass).
- State-shift diagnostics (`state_shift_metrics.csv`) and trajectory
  divergence (`trajectory_divergence.csv`) — coverage tracks the primary
  rows exactly. **PARTIAL**, integrity clean so far.
- Frozen statistical analysis (paired test vs `OFF_POLICY_LRU`, Spearman
  correlations of shift-vs-miss-degradation) — **NOT_STARTED** (explicitly
  gated on full 42/42 completion; must not be run on partial data).

**Theory/motivation additions:** the compounding-distribution-shift account
is already in the manuscript discussion as a hypothesis (`main.tex`,
`subsec:discussion_analysis`); this ablation is designed to confirm, refute,
or partially support it — not to motivate new theory.

**Current jobs/tmux sessions:** `distribution_shift_2h_continue`.

**Artifacts/output paths:** `analysis/distribution_shift_ablation_v1/`.

**Blockers:** wall-clock only; per-fold cost (~4.5h average observed from
the 9h pass) means the 2h continuation's own budget will likely not be
enough to finish even one more fold, and a further resume will probably be
needed (not performed automatically — always a deliberate, audited step).

**Completion criteria:** 42/42 primary rows, matching diagnostic coverage,
zero failures/duplicates/NaNs, frozen paired statistical analysis executed
exactly once on the complete set.

**Manuscript sections likely to change:**
`subsec:discussion_analysis`/limitations (distribution-shift account),
possibly a new ablation subsection and table.

**Reviewer-response evidence needed:** the complete 42-row comparison, the
state-shift and trajectory diagnostics, and the frozen statistical
conclusion classified into one of: distribution-shift explanation
supported / shift exists but doesn't explain the gap / mechanism not
supported / mixed by workload-capacity.

---

## Concern 4 — Practical significance given added computational cost

**Reviewer concern (paraphrase):** the current implementation is
substantially slower per eviction decision than lightweight baselines
(manuscript reports 75.0/152.1/316.0 ms mean per eviction decision at
capacities 32/64/128 vs ~0.001–0.18 ms for LRU/FIFO-Reinsertion/SIEVE/REST,
four to five orders of magnitude), while not outperforming them on miss
ratio end-to-end — so it is unclear when, if ever, the added cost is
practically justified.

**Scientific question:** where does the overhead come from; how much of it
is avoidable research-prototype implementation cost vs. fundamental
algorithmic cost; can selective/hybrid invocation or candidate pruning
recover most of the quality at a fraction of the cost; and at what miss-cost
regime (if any) would the current method's cost be justified?

**Response strategy:** implement an exact-decision-preserving optimized
implementation, quality-preserving selective/pruning variants, and a
transparent break-even/miss-cost-sweep/Pareto analysis — all under a frozen
protocol — non-API only, before considering whether an API-backed
heterogeneous-cost case study would add anything beyond it.

**Experiments:**
- `configs/practical_significance_ablation_v1.json` /
  `docs/practical_significance_ablation_protocol.md` — frozen protocol.
  **IMPLEMENTED** (this session).
- `src/lafc/policies/evict_value_v1_optimized.py` — `cached_exact`,
  `vectorized_exact`, `vectorized_cached_exact` variants, each proven
  decision-identical to canonical on a controlled trace/model/seed.
  **IMPLEMENTED**, exact-equivalence tests passing.
- `src/lafc/policies/evict_value_v1_selective.py` — disagreement-based
  (LRU vs. shadow-SIEVE victim) and periodic selective invocation, plus
  top-k candidate pruning. **IMPLEMENTED.**
- `scripts/experiments/run_practical_significance_ablation.py` — modes
  `--profile/--exact-optimizations/--selective/--top-k/--weighted-cost/
  --break-even/--miss-cost-sweep/--pareto/--all/--resume`. **IMPLEMENTED.**
- Component-level profiling (bookkeeping / candidate construction / feature
  extraction / model prediction / ranking / other) — implemented, only run
  at uncontended-machine quality so far because Concerns 1–3's jobs were
  actively consuming CPU during this session. **IMPLEMENTED, timing
  campaign PREPARED_BUT_DEFERRED.**
- Exact-optimization equivalence + smoke-scale speedup — **PARTIAL**.
  Decision-equivalence confirmed by unit test; a smoke-scale run
  (`--max-requests 200`) across all 7 certified traces at capacities 32/64
  additionally shows ~22–30x speedup from batched model inference alone
  (`vectorized_exact`/`vectorized_cached_exact`) versus only ~1.0–1.08x from
  invariant-hoisting alone (`cached_exact`) at these small `k` — i.e. the
  dominant avoidable cost at k≤64 is unbatched per-candidate model calls,
  not the O(k²) redundant recomputation (which is expected to matter more
  as `k` grows toward 128). At capacity 128 specifically, the smoke run's
  200-request prefix was too short to reach eviction decisions in some
  traces, so those cells show no speedup number (not a failure, just no
  data at that scale). The `practical_significance_smoke` tmux session has
  since **completed and self-terminated cleanly** (no errors); all 9
  planned artifacts now exist under `analysis/practical_significance_ablation_v1/`
  at smoke scale. Final controlled numbers (all capacities, full request
  budget, idle machine) remain deferred.
- Selective invocation invocation-rate/quality outputs — **PARTIAL**
  (smoke scale, `selective_invocation.csv`, 42 rows, complete for the smoke run).
- Top-k victim-retention/quality-cost curve — **PARTIAL** (smoke scale,
  `topk_tradeoff.csv`, 77 rows).
- Model-complexity variants (ridge/random_forest/hist_gb, reusing the
  already-frozen h4 training grid, no new model search) — **PARTIAL**
  (smoke scale, `model_complexity_tradeoff.csv`, 63 rows).
- Break-even miss-cost analysis vs LRU/SIEVE/FIFO-Reinsertion — formula
  **IMPLEMENTED** and exercised against the certified fair-window miss
  counts (`break_even_miss_cost.csv`, 12 rows); final numeric conclusion
  depends on the deferred controlled timing campaign for the compute-cost
  term. **PARTIAL.**
- Miss-cost sweep (log-spaced synthetic grid) — **COMPLETE**
  (`miss_cost_sweep.csv`, 96 rows), runs independent of timing/machine load.
- Weighted/heterogeneous miss-cost analysis — audited: the canonical trace
  loader (`build_requests_from_lists`) assigns every `Page.weight = 1.0`
  uniformly; **no real heterogeneous cost metadata exists** in the current
  trace pipeline. Falls back to a predeclared, clearly-labeled synthetic
  sensitivity analysis only. **COMPLETE (synthetic)** (`weighted_cost.csv`, 84 rows).
- Quality-latency Pareto frontier — **PARTIAL**, populated from smoke-scale
  data (`pareto_frontier.csv`, 170 rows, 23 Pareto-efficient) pending the
  controlled campaign.
- Final controlled timing campaign — **BLOCKED**, deliberately deferred:
  Concerns 1–3's jobs (`evict_cross_family_resume`,
  `objective_ablation_pipeline`, `distribution_shift_2h_continue`) plus the
  archival legacy LRB process remain actively using the CPU (load average
  ~4.2/20 cores at last check); launching reviewer-facing timing evidence
  under that load would be scientifically invalid.

**API decision:** re-evaluated at the end of the smoke-scale work —
**API_CASE_STUDY_NOT_NEEDED**. The non-API evidence (measured batching
speedups, formalized break-even model, synthetic weighted-cost sensitivity,
confirmed absence of real heterogeneous-cost metadata) covers the
reviewer's question. Re-evaluate again once the controlled campaign
completes.

**Theory/motivation additions:** none planned yet; this concern is purely
about measurement, not new supervision theory.

**Current jobs/tmux sessions:** none active for Concern 4 (the
`practical_significance_smoke` tmux session completed and exited cleanly).

**Artifacts/output paths:** `analysis/practical_significance_ablation_v1/`.

**Blockers:** the final controlled timing campaign requires the machine to
be free of Concerns 1–3's active jobs; that is a scheduling constraint, not
a scientific one.

**Completion criteria:** controlled timing campaign complete under an idle
machine (median + variation reported, warm-up discarded); exact-optimization
speedups measured (not just proven equivalent); selective/top-k
quality-cost curves complete across the predeclared grids; break-even
numbers finalized against LRU/SIEVE/FIFO-Reinsertion and the strongest
certified lightweight baseline; Pareto frontier finalized; API decision
gate re-evaluated once the non-API picture is complete.

**Manuscript sections likely to change:**
`subsec:overhead_scalability`/practical-significance discussion, a new
break-even/Pareto figure, limitations.

**Reviewer-response evidence needed:** the complete
`analysis/practical_significance_ablation_v1/` artifact set collected under
a controlled, idle-machine run, plus the honest break-even conclusion
(including the explicit "no positive miss penalty makes the current policy
preferable" outcome if that is what the data shows).

---

## Do not forget

- Do not write the final reviewer response before each concern's
  experiments are complete — partial results must never be presented as
  conclusions.
- Do not choose hyperparameters, thresholds, or optimization variants based
  on held-out/test outcomes; selection must use validation data or
  predeclared fixed values only.
- Keep deployment, sensitivity, diagnostic, and primary-fair results in
  clearly separated artifacts — never blend them into one comparison table.
- Do not use API-backed experiments for Concern 4 unless the non-API
  evidence leaves a specific, named gap (see Concern 4's API decision gate
  each time it is re-evaluated).
- Negative/null findings (e.g. "no positive miss penalty makes the learned
  policy preferable," or "DAgger does not measurably reduce shift") must be
  reported honestly, not hidden or reframed as success.
- Re-audit "current jobs/tmux sessions" fields before citing them — these
  are snapshots and go stale within minutes on this campaign.

---

## Experiment dependency / recommended order

1. Let the C1 (`evict_cross_family_resume`), C2 (`objective_ablation_pipeline`),
   and C3 (`distribution_shift_2h_continue`) long-running jobs finish their
   remaining folds.
2. Perform C1/C2/C3 completion audits (leakage/isolation checks, row-count
   and integrity checks) once each reaches its full expected row count.
3. Run the C4 controlled timing campaign on an idle machine (no C1/C2/C3
   jobs active).
4. Run the final frozen statistical analyses for C1/C2/C3 (paired tests,
   correlations) only once each is complete — never on partial data.
5. Update the manuscript and reviewer response using the frozen, certified
   results.
6. Consider an API-backed case study for C4 only if the controlled non-API
   evidence leaves a specific, named gap.
