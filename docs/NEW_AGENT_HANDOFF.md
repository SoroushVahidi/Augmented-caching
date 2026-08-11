# New-Agent Handoff

Short capstone to [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) (read that
first -- it has the full project purpose, scientific question, evidence
table, and hypothesis interpretation). This file exists only to give a crisp
"what do I do right now" answer and a hard safety list, in one place.

**Last consolidated:** 2026-08-11 (continuation of the 2026-08-10 local
finalization/handoff pass). Re-verify git and tmux state yourself before
acting -- see the checklist in `DEVELOPMENT_STATUS.md` section 10.

## Immediate Next Actions

1. **Priority P0.** Machine: local workstation (`al-khwarizmi`). Do
   **nothing** -- let the running `50%` learning-curve tmux worker
   (`kbs_learning_curve_50pct_20260810`) finish on its own. Prerequisite:
   none. Stopping condition: `tmux ls` no longer shows the session, or
   `campaign_state.json` shows all 7 families complete for fraction `0.5`
   -- then run the integrity checks described in
   `DEVELOPMENT_STATUS.md` section 7 / `NEXT_STEPS.md` P0.1 before citing
   any `0.5` numbers.

2. **Priority P1.** Machine: local. If P0 has completed and passed
   integrity checks, audit the result and decide whether to launch `100%`
   (`NEXT_STEPS.md` P3.1) -- entry point:
   `python3 scripts/experiments/run_supervision_objective_learning_curve.py --resume --fractions 1.0 --max-wall-hours 10 --config configs/supervision_objective_learning_curve_v1.json --out-dir analysis/supervision_objective_learning_curve_v1 --models-dir models/supervision_objective_learning_curve_v1`.
   Prerequisite: P0 complete and audited, no other heavy job running.
   Stopping condition: same integrity checks as `50%`, scaled to `1.0`.

3. **Priority P1.** Machine: local (compute), local implementation already
   frozen. Replicate the exact-target-oracle and target-degeneracy
   diagnostics across the remaining 6 families x 3 capacities
   (`NEXT_STEPS.md` P2.1) -- entry point: the promoted
   `scripts/experiments/run_exact_target_oracle_diagnostic.py` and
   `scripts/experiments/analyze_eviction_loss_target_degeneracy.py` (now
   also runnable on `main` with `--no-learned`; use the full learned-model
   comparison on this branch, where the required model registry exists).
   Prerequisite: none blocking. Stopping condition: a majority of cells
   reported with a consistent direction, or a clearly characterized split.

4. **Priority P2.** Machine: local. Compute the new `P(T > H | resident)`
   reuse-time-tail diagnostic (`NEXT_STEPS.md` P2.2) from the existing
   `learned_decisions.csv` in `analysis/exact_target_oracle_diagnostic_v1/`
   plus the raw trace -- no new replay engine required for a first pass.
   Prerequisite: none. Stopping condition: a first bucketed `P(T>H)`
   distribution exists for at least the brightkite cell.

5. **Priority P1.** Machine: local for protocol/code, may need larger
   scale for full campaign. Advance the corrected held-out cross-family
   `evict_value_v1` replay (`NEXT_STEPS.md` P1.1) -- the single
   highest-priority open item per the comparison-fairness audit.
   Prerequisite: none blocking. Stopping condition: all 7 held-out folds
   produce primary-eligible rows with no train/test overlap.

For the full ranked P0-P4 roadmap beyond these five, see
[`NEXT_STEPS.md`](NEXT_STEPS.md).

## Do Not Do

- **Do not kill, signal, attach to, resume, or otherwise interact with**
  the `kbs_learning_curve_50pct_20260810` tmux worker unless it has
  already naturally exited and you have confirmed that via
  `campaign_state.json`, not just an empty-looking tmux pane (Python
  stdout buffering makes a live pane look empty -- this is expected, not a
  hang).
- **Do not launch the `100%` fraction or any other heavy experiment**
  without first confirming no other heavy job is running and the `50%`
  fraction has been audited.
- **Do not overwrite active or generated evidence** under `analysis/`,
  `models/`, or `logs/` -- these are gitignored but are evidence, not
  scratch space.
- **Do not infer current remote (Wulver) state from old snapshots** --
  every `LAST_KNOWN_REMOTE_STATUS` label in `DEVELOPMENT_STATUS.md` is
  unverified as of this pass (Wulver was not contacted). Recheck it
  directly when a task explicitly authorizes contacting Wulver; do not
  assume it is still accurate just because it is written down.
- **Do not treat any one-cell or one-family diagnostic as a universal
  finding.** H2, H3, H4 rest on one cell (brightkite, capacity 64, H=4);
  H5, H6 rest on one family (metacdn); H1's learning-curve result covers
  only the fractions completed so far. Replication is the shared
  precondition for upgrading any of these, per the hypothesis map's own
  stopping rules.
- **Do not use contaminated or superseded artifacts** as evidence -- see
  `DEVELOPMENT_STATUS.md` section 6 and
  [`RESULTS_AND_LIMITATIONS.md`](../../Augmented-caching-main/docs/RESULTS_AND_LIMITATIONS.md)
  section F (main) for the current list.
- **Do not conflate request-time reuse distance with classical stack /
  reuse distance.** `H` and `T` (`next_reuse_time_requests`) are counts of
  future *requests*; stack distance and footprint-style quantities count
  *distinct pages*. See the hypothesis map's "Refined horizon-adequacy
  framing" section for the exact distinction and why `H/C` is marked a
  coarse, unestablished covariate rather than a law.
- **Do not contact Wulver, SSH to Wulver, or query Slurm** unless the
  current task explicitly authorizes it.
- **Do not force-push, rewrite history, or run destructive git operations**
  on either branch without explicit user authorization for that specific
  action.
