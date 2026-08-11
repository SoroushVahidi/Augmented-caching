# Next-Work Roadmap

Actionable, ordered roadmap for this branch. Companion to
[`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) (read that first for full
context). Each item states why it matters, its current status, what it
depends on, which machine it belongs on, its entry point, an expected cost,
a stopping rule, and what result would actually change the project's
scientific interpretation -- not just "do more experiments."

---

## P0 -- currently running / must finish first

### P0.1 -- Let the `50%` learning-curve fraction finish

- **Why:** in-flight evidence for H1 (insufficient training data); the
  campaign is the decisive test named in the hypothesis map for that
  hypothesis.
- **Status:** `RUNNING` -- 5/7 families complete as of the last check in
  `DEVELOPMENT_STATUS.md` section 7, `twemcache` in progress, `wiki2018`
  remaining.
- **Dependency:** none -- already launched.
- **Machine:** local workstation (`al-khwarizmi`), tmux session
  `kbs_learning_curve_50pct_20260810`.
- **Entry point:** none to launch -- this is a **do-not-touch** item. If it
  has clean-stopped at its `10h` wall-time budget before finishing
  `wiki2018`, the resume command is
  `python3 scripts/experiments/run_supervision_objective_learning_curve.py --resume --fractions 0.5 --max-wall-hours 10 --config configs/supervision_objective_learning_curve_v1.json --out-dir analysis/supervision_objective_learning_curve_v1 --models-dir models/supervision_objective_learning_curve_v1`
  -- but do not run this automatically; confirm via `tmux ls` and
  `campaign_state.json` first, and only resume as a deliberate, explicit
  next action, not as part of an unrelated task.
- **Expected cost:** ~1-2.5 more hours of wall time for `wiki2018` at
  observed per-family rates (~87-141 min/family at this fraction).
- **Stopping rule:** stop when `campaign_state.json` lists all 7 families
  complete for fraction `0.5` and the standard integrity checks pass
  (fraction 0.5 rows == 42, unique `(fraction,family,condition,capacity)`
  keys, all `status=ok`, unit audits == 7, model hashes verified, no
  duplicate rows, no malformed/NaN/Inf outside the known legitimate
  zero-pairs case, `<=25%` evidence unchanged/preserved).
- **What would change our interpretation:** a genuine, non-noise,
  monotonic improvement in downstream `miss_ratio` (not just offline fit
  metrics) between `25%` and `50%`/`100%` would reopen H1 (currently
  `DISFAVORED`). A continuation of the flat pattern already seen through
  `25%` would further support `DISFAVORED`.

## P1 -- reviewer/publication blockers

### P1.1 -- Corrected held-out cross-family `evict_value_v1` replay

- **Why:** named the single highest-priority open item by the comparison-
  fairness audit; without it, no citable head-to-head comparison exists
  between `evict_value_v1` and modern learned baselines (LRB/3L/CACHEUS/HALP).
- **Status:** `COMPLETE_PARTIAL_SCOPE`.
- **Dependency:** none blocking; independent of the learning-curve campaign.
- **Machine:** local for the protocol/code; full cross-family scale may need
  cluster time depending on runtime -- check current partial-run timings
  before assuming laptop-scale is sufficient.
- **Entry point:** `analysis/reviewer_fairness_cross_family_v1/` +
  `docs/reviewer_fairness_cross_family_v1.md` for the frozen protocol.
- **Expected cost:** high (explicitly rated "cost: High, reviewer_value:
  Very High" by the fairness audit).
- **Stopping rule:** all 7 held-out folds produce primary-eligible rows
  with no train/test overlap, matching the `primary_controlled_window`
  eligibility rules in `kbs_evidence_eligibility.md`.
- **What would change our interpretation:** if `evict_value_v1` beats the
  modern learned baselines once this is fixed, the "target problem"
  narrative would need to be reconciled with a competitive result; if it
  still loses under a clean comparison, it strengthens the current negative-
  result narrative without the lingering unfairness caveat.

### P1.2 -- Controlled timing / practical-significance campaign

- **Why:** current timing numbers are `SMOKE_ONLY` and explicitly marked
  not-final by their own artifact; any practical-deployment claim needs a
  real controlled measurement.
- **Status:** equivalence check `COMPLETE_DIAGNOSTIC`; timing `PENDING`.
- **Dependency:** none.
- **Machine:** local (controlled, low-noise environment preferred -- avoid
  running alongside the learning-curve worker to avoid CPU contention
  skewing timing).
- **Entry point:** `analysis/practical_significance_ablation_v1/`.
- **Expected cost:** low-medium (a few hours of controlled, low-contention
  wall time).
- **Stopping rule:** timing numbers collected under controlled, single-job
  conditions across the standard capacity/family grid.
- **What would change our interpretation:** a large real overhead would
  matter for any deployability claim, independent of the target-formulation
  question; does not by itself bear on H1-H11.

## P2 -- highest-information mechanistic experiments

### P2.1 -- Multi-cell replication of target-degeneracy + exact-target-oracle

- **Why:** H2 and H3 currently rest on exactly one cell (brightkite, cap
  64, H=4) -- the single most important generalization gap in the current
  evidence base.
- **Status:** `COMPLETE_DIAGNOSTIC` (one cell); replication `NOT_STARTED`.
- **Dependency:** none; existing tooling, no code changes needed.
- **Machine:** local.
- **Entry point:** the existing degeneracy/oracle scripts
  (`scripts/experiments/analyze_eviction_loss_target_degeneracy.py` and the
  exact-target-oracle runner), re-run across the remaining 6 families x 3
  capacities.
- **Expected cost:** medium (one cell already took non-trivial wall time;
  18 more cells at similar cost).
- **Stopping rule:** a majority of cells reported, with consistent
  direction (or a clearly characterized split by family/capacity).
- **What would change our interpretation:** if most other cells show
  materially *lower* tie fractions / higher target entropy than the
  brightkite cell, H3 would be disfavored as a general phenomenon per its
  own stated stopping rule.

### P2.2 -- `P(T > H | resident)` reuse-time-tail diagnostic

- **Why:** the literature-motivated, dimensionally direct quantity for
  horizon adequacy (see the hypothesis map's "Refined horizon-adequacy
  framing" section) -- more principled than the `H/C` ratio, and computable
  from data that already exists.
- **Status:** `NOT_STARTED` (new item from this pass).
- **Dependency:** none -- the exact-target-oracle's `learned_decisions.csv`
  already records `decision_id`, `request_t`, `chosen_candidate` per
  decision, sufficient to compute `T` by scanning the raw trace.
- **Machine:** local.
- **Entry point:** new small analysis script (not yet written) over
  existing decision logs + raw trace; no new replay engine required for
  this first pass.
- **Expected cost:** low (a few hours of implementation + fast batch
  computation over existing logs).
- **Stopping rule:** first pass computes `P(T>H)` bucketed distribution
  (e.g. `1-4, 5-8, 9-16, 17-32, >32, never reused`) for at least the
  existing brightkite cell; extend to other cells only if the first pass
  shows a meaningful fraction beyond `H`.
- **What would change our interpretation:** a high `P(T>H)` (most resident
  objects' next reuse falls outside the horizon) would strengthen H4/H11 as
  a plausible mechanism (not yet causal proof); a low `P(T>H)` would
  disfavor horizon truncation as primary, per H11's own stopping rule.

### P2.3 -- `H/C` and capacity-scaling diagnostic sweep

- **Why:** tests H10 directly; currently only run at one fixed capacity.
- **Status:** `UNTESTED`.
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
- **Status:** `IMPLEMENTATION_READY`, protocol frozen and sync-ready.
- **Dependency:** none code-wise; likely needs cluster-scale compute for
  the full 7-family campaign given the smoke run's cost profile.
- **Machine:** local implementation exists; full campaign likely needs
  Wulver per prior local docs (unverified this pass -- do not assume
  without rechecking).
- **Entry point:** `src/lafc/continuation_policy_ablation.py` +
  `configs/continuation_policy_causal_ablation_v1.json`.
- **Expected cost:** high (full 7-family scale).
- **Stopping rule:** all 7 families produce a comparison between
  LRU-continuation and frozen-`pi1`-continuation labels on matched
  decision/candidate examples.
- **What would change our interpretation:** if the more internally-
  consistent continuation assumption improves downstream misses over the
  current fixed-LRU-continuation labels, H5 gains real support; if it does
  not, H5 should be deprioritized per its own stopping rule.

## P3 -- completeness enhancements

### P3.1 -- `100%` fraction of the learning-curve campaign

- **Why:** completes H1's decisive test across the full data range.
- **Status:** `PENDING`.
- **Dependency:** P0.1 (the `50%` fraction) must finish and be audited
  first; do not launch `100%` before that, and do not launch it as a side
  effect of an unrelated task.
- **Machine:** local, or Wulver if wall-time cost is prohibitive locally
  (the `50%` fraction alone is costing most of a `10h` local budget).
- **Entry point:** same runner as P0.1, `--fractions 1.0`.
- **Expected cost:** high -- likely the single most expensive remaining
  local diagnostic given the `0.5` fraction's observed per-family cost.
- **Stopping rule:** same integrity checks as P0.1, scaled to `1.0`.
- **What would change our interpretation:** same as P0.1's -- only a real,
  monotonic downstream-miss improvement would reopen H1.

### P3.2 -- Historical-tail diagnostic

- **Why:** H8's decisive test; a finite-horizon target implicitly assumes
  zero terminal value beyond `H`.
- **Status:** `BLOCKED` locally -- not implemented in this worktree.
- **Dependency:** unclear whether Wulver-side work exists; per section 11
  of `DEVELOPMENT_STATUS.md`, this is `LAST_KNOWN_REMOTE_STATUS -- NOT
  RECHECKED IN THIS PASS`. Do not fabricate the missing implementation;
  check remote state first when explicitly authorized to do so.
- **Machine:** unknown until rechecked.
- **Entry point:** none locally yet.
- **Expected cost:** unknown.
- **Stopping rule:** n/a until a first diagnostic exists.
- **What would change our interpretation:** a dedicated `Q_H + V_tail_hat`
  comparison beating the plain finite-horizon target would support H8
  directly.

## P4 -- optional / deferred research

### P4.1 -- LRB / 3L-Cache / CACHEUS exact-protocol replication under the corrected held-out split

- **Why:** completeness -- ensures the already-`FINAL_VALIDATED` baseline
  comparison also holds under the corrected split from P1.1, not just the
  original protocol.
- **Status:** depends on P1.1 completing.
- **Dependency:** P1.1.
- **Machine:** local.
- **Entry point:** re-run the existing baseline harness against the
  corrected split once available.
- **Expected cost:** low, given the existing harness.
- **Stopping rule:** results consistent (or characterized as inconsistent)
  with the original-protocol comparison.
- **What would change our interpretation:** a material change in relative
  baseline ranking under the corrected split would be an important caveat
  on the existing `FINAL_VALIDATED` comparison.

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
