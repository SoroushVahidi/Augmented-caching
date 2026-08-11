# New-Agent Handoff

Short capstone to [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) (read that
first -- it has the full project purpose, scientific question, evidence
table, and hypothesis interpretation). This file exists only to give a crisp
"what do I do right now" answer and a hard safety list, in one place.

**Last consolidated:** 2026-08-11 (local finalization/handoff pass, plus a
same-day reconciliation against fresh Wulver-side facts relayed by the
user -- see [`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)).
Re-verify git and tmux state yourself before acting -- see the checklist in
`DEVELOPMENT_STATUS.md` section 10.

## Immediate Next Actions

1. **Priority P0.** Machine: local workstation. Do **nothing** -- let the
   running `wiki2018|0.5` resume tmux worker
   (`kbs_learning_curve_50pct_wiki2018_resume_20260811`) finish on its own
   (it is restricted via `--held-out-families wiki2018` and cannot touch
   the other six already-complete families). Prerequisite: none. Stopping
   condition: `tmux ls` no longer shows the session, or
   `campaign_state.json` shows `wiki2018|0.5` complete -- then run the
   integrity checks described in `DEVELOPMENT_STATUS.md` section 7 /
   `NEXT_STEPS.md` P0.1 before citing the full `50%` fraction as complete.

2. **Priority P1.** Machine: none (Wulver already did this). Sync and
   review the corrected held-out cross-family `evict_value_v1` replay
   (42/42, SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`)
   -- see `WULVER_TO_GITHUB_PROMOTION_QUEUE.md` #1 for the exact review
   steps. Prerequisite: Wulver access to pull the artifact. Stopping
   condition: local integrity checks pass (hash match, unique keys, no
   NaN/Inf, all `status=ok`).

3. **Priority P1.** Machine: local. Fix the continuation-policy C0/C1/C2
   `reference_model=` interface mismatch (`NEXT_STEPS.md` P2.5) -- the
   production runner expects a keyword the protected source doesn't
   provide, and the existing draft only implements two of three needed
   conditions. This is currently the single largest concrete blocker for
   R2 Major 3 / R3's causal-explanation concern. Prerequisite: none.
   Stopping condition: the runner executes end-to-end on a small real
   example without the unexpected-keyword failure, and all three
   conditions are implemented.

4. **Priority P2.** Machine: local. Replicate the exact-target-oracle
   diagnostic across the remaining 6 families x 3 capacities
   (`NEXT_STEPS.md` P2.1) -- entry point:
   `scripts/experiments/run_exact_target_oracle_diagnostic.py`. (Target-
   degeneracy replication is already done on Wulver, 21/21 cells -- that
   half of this item is now a sync task, see `WULVER_TO_GITHUB_PROMOTION_QUEUE.md`
   #3, not a local re-run.) Prerequisite: none blocking. Stopping
   condition: a majority of cells reported with a consistent direction.

5. **Priority P2.** Machine: local. Compute the `P(T > H | resident)`
   reuse-time-tail diagnostic (`NEXT_STEPS.md` P2.2) from the existing
   `learned_decisions.csv` in `analysis/exact_target_oracle_diagnostic_v1/`
   plus the raw trace -- confirmed genuinely `NOT_STARTED` anywhere (fresh
   grep this pass found zero implementation). Prerequisite: none. Stopping
   condition: a first bucketed `P(T>H)` distribution exists for at least
   the brightkite cell.

For the full ranked P0-P4 roadmap beyond these five, see
[`NEXT_STEPS.md`](NEXT_STEPS.md); for Wulver-side sync priority, see
[`WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](WULVER_TO_GITHUB_PROMOTION_QUEUE.md).

## Do Not Do

- **Do not kill, signal, attach to, restart, or otherwise interact with**
  the `kbs_learning_curve_50pct_wiki2018_resume_20260811` tmux worker
  unless it has already naturally exited and you have confirmed that via
  `campaign_state.json`, not just an empty-looking tmux pane (Python
  stdout buffering makes a live pane look empty -- this is expected, not a
  hang).
- **Do not launch the `100%` fraction or any other heavy experiment**
  without first confirming no other heavy job is running and the full
  `50%` fraction (all 7 families) has been audited.
- **Do not overwrite active or generated evidence** under `analysis/`,
  `models/`, or `logs/` -- these are gitignored but are evidence, not
  scratch space.
- **Do not treat the Wulver-sourced facts in `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`
  as independently verified by this workstation.** They were relayed by
  the user from a separately audited Wulver session on 2026-08-11, not
  confirmed via direct Wulver contact from here. Re-verify job IDs/hashes
  directly when a task explicitly authorizes contacting Wulver, especially
  before citing them in anything manuscript-facing.
- **Do not assume the continuation-policy C0/C1/C2 runner is ready to
  launch** just because earlier docs say `IMPLEMENTATION_READY` -- there is
  a known `reference_model=` interface mismatch blocking production use;
  see `NEXT_STEPS.md` P2.5.
- **Do not conflate the original-protocol and exact-protocol LRB/3L-Cache/
  CACHEUS results.** The local `FINAL_VALIDATED` CSVs use the original
  `reviewer_fairness_v1` protocol; the Wulver jobs `1171965`-`1171967` are
  a separate re-run matched to the corrected cross-family split. Citing one
  as if it were the other misrepresents the comparison.
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
