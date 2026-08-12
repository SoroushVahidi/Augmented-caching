# Next-Work Roadmap

Actionable, ordered roadmap for this branch. Companion to
[`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) (read that first for full
context). Each item states why it matters, its current status, what it
depends on, which machine it belongs on, its entry point, an expected cost,
a stopping rule, and what result would actually change the project's
scientific interpretation -- not just "do more experiments."

**2026-08-11 update:** several items below were reconciled against fresh
Wulver-side facts relayed by the user (not independently verified by this
workstation -- see
[`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)).
Some items that used to read "run this experiment" now read "sync and
review this already-completed Wulver result" instead -- check
[`WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](WULVER_TO_GITHUB_PROMOTION_QUEUE.md)
for the sync-priority order before assuming an item still needs to be
launched from scratch.

---

## Quick-glance operational summary

**NOW (this workstation, right now):**
- Leave the C0/C1/C2 production campaign running in tmux session
  `kbs_continuation_c0_c1_c2_production_20260811` (P2.5 below). Do not stop,
  signal, restart, or attach interactively.
- Do not launch a second heavy local experiment while it runs.
- No-compute work only (see below) is safe to do in parallel.

**WHEN LOCAL C0/C1/C2 FINISHES:**
- Run the integrity audit against the 21-unit manifest (63 policy rows, 21
  label-agreement rows, 21 pi2-training rows) before citing any outcome.
- Synthesize the C0-vs-C1-vs-C2 mechanism finding (P2.5's stopping rule).
- Feed the result into the Reviewer #3 / R2 Major 3 closure decision in
  `reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`.

**WHEN WULVER MAINTENANCE ENDS (not this workstation's task to trigger):**
- Let already-queued jobs (`1171965`-`1171967` LRB/3L-Cache/CACHEUS
  replication, horizon sweep `1169299`) auto-dispatch; do not resubmit them
  from here.
- Sync and audit the corrected held-out `evict_value_v1` 42/42 result (P1.1)
  and the completed controlled timing 420/420 result (P1.2).
- Audit the remaining horizon-sweep cells (18/35 pending) once complete.

**NO-COMPUTE (safe anytime, any machine state):**
- Dataset release/inventory prep (`analysis/huggingface_dataset_preview_v0_2/`
  review, without uploading).
- Reviewer evidence-table and manuscript-safe-summary preparation (see
  `analysis/kbs_reviewer_synthesis_prep_20260811/`).
- Documentation reconciliation, hygiene, and link-checking passes like this
  one.

**DO NOT RUN (already complete or intentionally not run under a stopping
rule -- relaunching would waste compute and contradict a recorded decision):**
- `100%` learning-curve fraction (P3.1) -- `STOP_SAMPLE_SIZE_HYPOTHESIS`.
- A duplicate objective-comparison ablation -- already `FINAL_VALIDATED`,
  84/84 rows.
- A duplicate controlled-timing campaign -- already `WULVER_ONLY_VALIDATED`,
  420/420 rows, `PROMOTE_NOW`.
- A duplicate reuse-tail diagnostic -- already `LOCAL_COMPLETE`, 21/21 cells.
- A duplicate target-degeneracy sweep beyond the one local cell without
  first checking whether the Wulver 21-cell result (job `1169513`) already
  covers it.
- A second local modern-baseline (LRB/3L-Cache/CACHEUS) campaign -- local
  controlled-window CSVs are already `LOCAL_EXACT_PROTOCOL_VALIDATED`
  (with caveats noted per policy); Wulver jobs are pending replication only.
- A local horizon-sensitivity sweep duplicate -- the base sweep is
  Wulver-side (`RUNNING`, job `1169299`); do not start a second copy
  locally.

## P0 -- completed closeout / do not relaunch

### P0.1 -- Local 50% learning-curve closeout

- **Status:** `COMPLETE`. The final `wiki2018|0.5` resume completed
  cleanly, and fraction `0.5` is audited at 7/7 families, 42/42 rows, all
  `status=ok`, duplicate-key count 0, NaN/Inf count 0, 7/7 fraction-0.5
  audit units, and 0 model SHA mismatches.
- **Synthesis:** `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`.
- **Scientific decision:** `STOP_SAMPLE_SIZE_HYPOTHESIS`. Within the tested
  `1%-50%` range, the sample-size explanation is not supported as the
  primary cause; this does not claim that more data can never help.
- **Action:** none. Do not launch `100%` as follow-up work for H1 under the
  current stopping rule.

### P0.2 -- Local reuse-tail horizon diagnostic closeout

- **Status:** `LOCAL_COMPLETE`. The local
  `reuse_tail_horizon_diagnostic_v1` run completed 21/21 family-capacity
  cells, with all seven families, capacities `32/64/128`, and horizons
  `1/2/4/8/16`.
- **Synthesis:** `docs/reuse_tail_horizon_diagnostic_v1_synthesis.md`.
- **Scientific decision:** H4/H11 are supported as a horizon-observability
  limitation: at H=4, `P(T>4 | resident)=0.9938544459677984` and
  `P(T>4 | resident, eventually reused)=0.9793302186526528`. This is not a
  causal excess-miss claim.
- **Action:** none. Do not relaunch this diagnostic unless the protocol is
  deliberately changed.

## P1 -- reviewer/publication blockers

### P1.1 -- Corrected held-out cross-family `evict_value_v1` replay

- **Why:** named the single highest-priority open item by the comparison-
  fairness audit; without it, no citable head-to-head comparison exists
  between `evict_value_v1` and modern learned baselines (LRB/3L/CACHEUS/HALP).
- **Status (updated 2026-08-11):** per user-relayed Wulver facts, this is
  now `WULVER_ONLY_VALIDATED` -- **COMPLETE, 42/42 rows**, SHA-256
  `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`. This
  item is **no longer "run it," it is "sync and review it"**. A 2026-08-11
  local evidence-prep pass validated the baseline side and created
  `analysis/kbs_r2_major1_evidence_prep_20260811/` plus the reusable
  comparison procedure `scripts/analysis/prepare_r2_major1_evidence.py`.
  Wulver jobs `1171965`-`1171967` remain pending because of maintenance, but
  the local controlled-window LRB/3L/CACHEUS rows already cover those
  baseline cells unless the missing Wulver config later proves materially
  different.
- **Dependency:** none blocking the sync/review itself; Wulver jobs
  `1171965`-`1171967` are replication/config-audit follow-up, not a local
  compute prerequisite.
- **Machine:** Wulver (already done); this workstation only needs to sync
  the result once access is available.
- **Entry point:** `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv`
  (Wulver path, not yet locally present).
- **Expected cost:** low, now that the compute itself is done -- just a
  sync + review pass.
- **Stopping rule:** integrity checks pass locally after sync (unique keys,
  no NaN/Inf, all `status=ok`, hash matches the value above), then
  `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...`
  writes the final matched comparison table.
- **What would change our interpretation:** if `evict_value_v1` beats the
  modern learned baselines once reviewed, the "target problem" narrative
  would need to be reconciled with a competitive result; if it still loses
  under this clean comparison, it strengthens the current negative-result
  narrative without the lingering unfairness caveat.

### P1.2 -- Controlled timing / practical-significance campaign

- **Why:** current local timing numbers are `SMOKE_ONLY` and explicitly
  marked not-final by their own artifact; any practical-deployment claim
  needs a real controlled measurement.
- **Status (updated 2026-08-11):** per user-relayed Wulver facts, this is
  now `WULVER_ONLY_VALIDATED` -- **COMPLETE**, Wulver job `1171758`,
  420/420 rows (7 families x 3 capacities x 4 policies x 5 repetitions).
  Mean per-request runtime: LRU 4.68us, FIFO-Reinsertion 5.17us, SIEVE
  9.52us, HALP-causal 870.66us (~186x LRU). This item is now `PROMOTE_NOW`
  per `WULVER_TO_GITHUB_PROMOTION_QUEUE.md` #2 -- **no local run needed**,
  just sync.
- **Dependency:** none; already complete on Wulver.
- **Machine:** Wulver (already done); sync to this workstation when access
  is available.
- **Entry point:** Wulver job `1171758` output (path not yet given
  locally).
- **Expected cost:** low -- sync only.
- **Stopping rule:** synced result passes the same integrity bar as any
  other promoted artifact (row count, no duplicate keys, provenance intact).
- **What would change our interpretation:** the result is already in --
  HALP-causal's ~186x overhead vs. LRU is a real, citable finding for any
  deployability discussion; it does not bear on H1-H11 (the target-
  formulation question is separate from computational cost). Modern
  LRB/3L/CACHEUS timing is not included in this 4-policy campaign and may
  need a separate pass if required.

## P2 -- highest-information mechanistic experiments

### P2.1 -- Multi-cell replication of target-degeneracy + exact-target-oracle

- **Why:** H2 and H3 currently rest on exactly one cell (brightkite, cap
  64, H=4) locally -- the single most important generalization gap in the
  local evidence base.
- **Status (updated 2026-08-11):** **target-degeneracy replication is
  already done on Wulver** (job `1169513`, 21/21 cells, unique-winner
  fraction = 0 across all of them) -- per `WULVER_TO_GITHUB_PROMOTION_QUEUE.md`
  #3 this is `NEEDS_REVIEW` (sync + locate the driver source) rather than
  something to re-run locally. **Exact-target-oracle replication remains
  genuinely `NOT_STARTED`** anywhere (no Wulver fact was given for this
  specific diagnostic beyond the original single cell) -- this is now the
  narrower, still-open half of this item.
- **Dependency:** none for the oracle-diagnostic replication; existing
  tooling, no code changes needed.
- **Machine:** local, for the still-open exact-target-oracle replication.
- **Entry point:** `scripts/experiments/run_exact_target_oracle_diagnostic.py`,
  re-run across the remaining 6 families x 3 capacities.
- **Expected cost:** medium (one cell already took non-trivial wall time;
  18 more cells at similar cost).
- **Stopping rule:** a majority of cells reported, with consistent
  direction (or a clearly characterized split by family/capacity).
- **What would change our interpretation:** the degeneracy side already
  points toward H3 generalizing (see the Wulver result above); if the
  oracle-diagnostic replication instead shows the learned model *not*
  beating the exact target oracle in most other cells, that would qualify
  (not overturn) the current H2/H3 reading -- the "model departure from
  target is net-beneficial" finding would need to be shown cell-specific
  rather than general.

### P2.2 -- `P(T > H | resident)` reuse-time-tail diagnostic

- **Why:** the literature-motivated, dimensionally direct quantity for
  horizon adequacy (see the hypothesis map's "Refined horizon-adequacy
  framing" section) -- more principled than the `H/C` ratio, and computable
  from data that already exists.
- **Status:** `LOCAL_COMPLETE`. The run in
  `analysis/reuse_tail_horizon_diagnostic_v1/` completed 21/21
  family-capacity cells and passed integrity. Synthesis:
  `docs/reuse_tail_horizon_diagnostic_v1_synthesis.md`.
- **Dependency:** none remaining for the resident-candidate diagnostic.
- **Machine:** local.
- **Entry point:** `scripts/experiments/run_reuse_tail_horizon_diagnostic.py`
  and `src/lafc/reuse_tail_horizon.py`.
- **Expected cost:** complete; no further local cost.
- **Stopping rule:** satisfied for the non-causal resident-candidate pass.
- **What changed our interpretation:** the high H=4 tail strengthens H4/H11
  as an observability limitation. It does not establish that reuse after H
  causes an avoidable miss; causal excess-miss attribution remains a
  separate, unimplemented question.

### P2.3 -- `H/C` and capacity-scaling diagnostic sweep

- **Why:** tests H10 directly; currently only run at one fixed capacity.
- **Status:** `EMPIRICALLY_STRENGTHENED_BUT_NOT_PROVEN_AS_LAW`. The
  Wulver-relayed broad degeneracy result and the local reuse-tail diagnostic
  both worsen with capacity at H=4, but this is still descriptive evidence,
  not a derived `H/C` scaling law.
- **Dependency:** P2.1's degeneracy script, re-run at fixed `H=4` across
  `C in {32,64,128}` (cheapest first pass; the learning-curve/objective-
  ablation CSVs already sweep capacity but don't carry the needed
  resolution metrics).
- **Machine:** local.
- **Entry point:** existing degeneracy script with `--capacity` swept.
- **Expected cost:** low (reuses existing tooling).
- **Stopping rule:** at least two capacities compared using
  `mean_optimal_set_fraction` / `target_entropy_bits`.
- **What would change our interpretation:** would either support or
  disfavor treating `H/C` as a useful covariate -- explicitly not assumed
  to be a law either way (see hypothesis map guardrail).

### P2.4 -- Strict-preference reversal diagnostic (H9)

- **Why:** the one audited cell is too degenerate to have strict (non-tied)
  preferences to test reversal on; need a cell where strict preferences are
  common.
- **Status:** `UNTESTED` as framed.
- **Dependency:** P2.1/P2.3's replication sweep, to first find a
  less-degenerate cell.
- **Machine:** local.
- **Entry point:** existing degeneracy script + a new reversal-rate
  measurement at H=8/H=16 conditioned on strict H=4 preference.
- **Expected cost:** low once a suitable cell is identified.
- **Stopping rule:** reversal rate measured on at least one cell with a
  non-trivial strict-preference fraction.
- **What would change our interpretation:** a high reversal rate would
  support H9 (short-horizon strict preferences are often wrong later); a
  low reversal rate (<10%) disfavors it per its own stopping rule.

### P2.5 -- Full-scale continuation-policy causal ablation (C1/C2)

- **Why:** H5's decisive test; currently only smoke-scale
  (`decision_count=3`).
- **Status (updated 2026-08-11):** `PRODUCTION_RUNNING_LOCAL_TMUX`.
  Local production runner is active in tmux session
  `kbs_continuation_c0_c1_c2_production_20260811` from source SHA
  `a813617f36822f793b0e48b0ee3e6009d56ee324` and config SHA-256
  `7556e120ead3b3e8a8c6d85ef7f800f2e8f1f1cb37800bde57b14d1a194d8670`.
  It emits C0 LRU, C1 frozen `pi1`, and C2 trained from frozen-`pi1`
  continuation labels; writes atomic `(held_out_family, capacity)` units;
  rebuilds aggregate CSVs from completed units; and runs with `--resume`,
  serial thread caps, and an 8-hour wall-time guard. No full-scale scientific
  result exists until final integrity passes.
- **Dependency:** monitor/resume only; do not relaunch a duplicate session. The old
  `build_rollout_candidate_rows_v2(..., reference_model=...)` path remains
  irrelevant to the frozen protocol and should not be used.
- **Machine:** local production runner is executing conservatively in tmux.
- **Entry point:** `scripts/experiments/run_continuation_policy_causal_ablation.py`
  with `configs/continuation_policy_causal_ablation_production_v1.json`.
- **Expected cost:** high for the full 7-family scale run.
- **Stopping rule:** all 7 families produce a comparison between
  LRU-continuation and frozen-`pi1`-continuation labels on matched
  decision/candidate examples.
- **What would change our interpretation:** if the more internally-
  consistent continuation assumption improves downstream misses over the
  current fixed-LRU-continuation labels, H5 gains real support; if it does
  not, H5 should be deprioritized per its own stopping rule.

## P3 -- completeness enhancements

### P3.1 -- `100%` fraction of the learning-curve campaign

- **Status:** `INTENTIONALLY_NOT_RUN_DUE_STOPPING_RULE`.
- **Reason:** the 50% campaign completed its intended H1 stopping-rule
  scope. The audited `1%-50%` curve shows no material monotonic downstream
  improvement, and pairwise remains flat/worse; H1 is now `DISFAVORED`
  within the tested range.
- **Action:** remove from active required work. Do not launch `100%` unless
  a future protocol change explicitly reopens the sample-size question.

### P3.2 -- Historical-tail diagnostic

- **Why:** H8's decisive test; a finite-horizon target implicitly assumes
  zero terminal value beyond `H`.
- **Status (updated 2026-08-11):** **already complete on Wulver** (job
  `1169665`): H=8 resolves ~24.6% of H=4-tied decisions, H=16 resolves
  ~38.7%, history-linear tie-breaking produces only tiny gains, leakage
  audit passed. This is `WULVER_ONLY_VALIDATED`, not implemented anywhere
  locally (confirmed absent by a fresh grep) -- item is now "locate and
  sync the source + result," not "design and implement from scratch."
- **Dependency:** none for reviewing the result; locating the source
  requires Wulver access.
- **Machine:** Wulver (already done); sync when access is available.
- **Entry point:** Wulver job `1169665` output/source (path not yet given
  locally).
- **Expected cost:** low once synced -- the compute is already done.
- **Stopping rule:** synced result passes the same integrity bar as any
  other promoted diagnostic.
- **What would change our interpretation:** the result is already in and
  is **weak support** for the horizon/tail concern (tie resolution, not a
  downstream miss-ratio improvement) -- do not read it as a policy win.

## P4 -- optional / deferred research

### P4.1 -- LRB / 3L-Cache / CACHEUS exact-protocol replication under the corrected held-out split

- **Why:** ensures the already-`FINAL_VALIDATED` (original-protocol)
  baseline comparison also holds under the corrected cross-family split
  used for P1.1, not just the original protocol.
- **Status (updated 2026-08-11):** `LOCAL_COMPLETE` for the local
  controlled-window CSVs, plus `WULVER_PENDING` for the independently
  submitted Wulver copies. Fresh local audit found
  `analysis/reviewer_fairness/policy_comparison_{three_l_cache,lrb,cacheus}.csv`
  complete at 42 rows per policy, 21 primary controlled-window rows per
  policy, all seven families, capacities `32/64/128`, all `ok`, no
  duplicates, and no NaN/Inf. Wulver jobs `1171965`-`1171967` remain
  pending because of maintenance, not failure.
- **Dependency:** none for local use of the audited CSVs. Sync the Wulver
  copies later for independent replication, or if the missing
  `configs/reviewer_fairness_exact_protocol_modern_20260811.json` proves to
  contain an additional constraint not recorded in local docs/source.
- **Machine:** local artifact is complete; Wulver remains queued.
- **Entry point:** local CSVs in `analysis/reviewer_fairness/`; Wulver jobs
  `1171965`-`1171967` later for replication.
- **Expected cost:** no local rerun needed.
- **Stopping rule:** results synced and integrity-checked, ideally reviewed
  alongside P1.1 as one coherent primary comparison table.
- **What would change our interpretation:** a material change in relative
  baseline ranking under the corrected split would be an important caveat
  on the existing `FINAL_VALIDATED` (original-protocol) comparison.

### P4.2 -- Reconcile the two independently-evolved revision-status tools

- **Why:** `main`'s `scripts/revision_status.py` / `revision_readiness.py`
  and this branch's `scripts/validation/revision_status.py` /
  `revision_readiness.py` are parallel, independently-evolved
  implementations of the same read-only status tool (different paths,
  colliding test filenames, same underlying constants) -- not a simple
  copy-forward case. See `DEVELOPMENT_STATUS.md` section 12 and the
  promotion-audit findings for detail.
- **Status:** `DEFERRED` -- requires a deliberate human merge decision, not
  an automated copy.
- **Dependency:** none blocking other work; purely a tooling-hygiene item.
- **Machine:** local.
- **Entry point:** manual review of both implementations side by side.
- **Expected cost:** low-medium.
- **Stopping rule:** one canonical implementation exists, or an explicit
  decision to keep both with non-colliding names/paths.
- **What would change our interpretation:** none -- pure tooling hygiene,
  no scientific content.

### P4.3 -- Fallback/uncertainty-aware selection experiment (H7)

- **Why:** H7 is currently `UNTESTED`; only indirect evidence exists.
- **Status:** `UNTESTED`, listed only as a future diagnostic candidate.
- **Dependency:** none blocking, but lower priority than P2.1-P2.5 since
  adjacent evidence (the exact deterministic oracle also loses to LRU)
  already mildly weakens this as the *dominant* explanation.
- **Machine:** local.
- **Entry point:** a margin-gated or uncertainty-aware selection variant of
  the existing deployed policy, or a direct measurement of predicted-value
  margins vs. decision correctness.
- **Expected cost:** medium.
- **Stopping rule:** would be disfavored if predicted-value margins are
  typically large yet misses remain high.
- **What would change our interpretation:** would only become a primary
  explanation if margins are shown to be small/unstable at most decisions.
