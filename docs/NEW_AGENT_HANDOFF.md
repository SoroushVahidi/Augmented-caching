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

1. **Priority P1.** Machine: none (Wulver already did this). Sync and
   review the corrected held-out cross-family `evict_value_v1` replay
   (42/42, SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`)
   -- then rerun
   `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...` to
   materialize the final matched table. The baseline side is already locally
   validated in `analysis/kbs_r2_major1_evidence_prep_20260811/`.
   Prerequisite: Wulver access to pull the artifact. Stopping condition:
   local integrity checks pass (hash match, unique keys, no NaN/Inf, all
   `status=ok`) and the script writes the final comparison.

2. **Priority P1.** Machine: local. Build or sync and audit the
   continuation-policy C0/C1/C2 production runner (`NEXT_STEPS.md` P2.5).
   The local smoke path is `IMPLEMENTATION_REPAIRED_SMOKE_VALIDATED`, but
   no full campaign runner/resume/output-writer is present locally. The
   reported `reference_model=` unexpected-keyword defect is reproducible
   against the older v2 rollout kernel, but the offending production caller
   was not found in local source/history/worktrees. Prerequisite: none.
   Stopping condition: the runner executes end-to-end on a small real
   example without the unexpected-keyword failure, and all three
   conditions are implemented.

3. **Priority P2.** Machine: local. Replicate the exact-target-oracle
   diagnostic across the remaining 6 families x 3 capacities
   (`NEXT_STEPS.md` P2.1) -- entry point:
   `scripts/experiments/run_exact_target_oracle_diagnostic.py`. (Target-
   degeneracy replication is already done on Wulver, 21/21 cells -- that
   half of this item is now a sync task, see `WULVER_TO_GITHUB_PROMOTION_QUEUE.md`
   #3, not a local re-run.) Prerequisite: none blocking. Stopping
   condition: a majority of cells reported with a consistent direction.

4. **Priority P0.** Machine: local. Do not rerun the `P(T > H | resident)`
   reuse-tail diagnostic. It is `LOCAL_COMPLETE` at
   `analysis/reuse_tail_horizon_diagnostic_v1/` and synthesized in
   `docs/reuse_tail_horizon_diagnostic_v1_synthesis.md`. Only causal
   excess-miss attribution remains unimplemented.

For the full ranked P0-P4 roadmap beyond these five, see
[`NEXT_STEPS.md`](NEXT_STEPS.md); for Wulver-side sync priority, see
[`WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](WULVER_TO_GITHUB_PROMOTION_QUEUE.md).

## Do Not Do

- **Do not restart or otherwise interact with the old learning-curve tmux
  sessions.** The final `wiki2018|0.5` resume completed cleanly, and
  fraction `0.5` is audited at 7/7 families and 42/42 rows.
- **Do not launch the `100%` fraction for H1.** The predefined stopping
  decision is `STOP_SAMPLE_SIZE_HYPOTHESIS`; `100%` is intentionally not
  run under the current campaign scope, not an active missing task.
- **Do not overwrite active or generated evidence** under `analysis/`,
  `models/`, or `logs/` -- these are gitignored but are evidence, not
  scratch space.
- **Do not treat the Wulver-sourced facts in `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`
  as independently verified by this workstation.** They were relayed by
  the user from a separately audited Wulver session on 2026-08-11, not
  confirmed via direct Wulver contact from here. Re-verify job IDs/hashes
  directly when a task explicitly authorizes contacting Wulver, especially
  before citing them in anything manuscript-facing.
- **Do not assume the continuation-policy C0/C1/C2 experiment is ready to
  launch** just because the local smoke is repaired. Full execution still
  needs an audited production runner/resume/output path; see
  `NEXT_STEPS.md` P2.5.
- **Do not relaunch exact controlled-window LRB/3L-Cache/CACHEUS locally just
  because Wulver jobs `1171965`-`1171967` are pending.** The local
  `analysis/reviewer_fairness/policy_comparison_{lrb,three_l_cache,cacheus}.csv`
  files are complete and audited for the controlled-window rows:
  `LOCAL_EXACT_PROTOCOL_VALIDATED` for LRB,
  `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT` for 3L-Cache, and
  `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT` for CACHEUS. Sync
  Wulver later only for replication or if its missing config proves
  materially different.
- **Do not overgeneralize beyond each diagnostic's scope.** H2 and the
  original exact-target/tie-break H3/H4 diagnostics rest on one cell
  (brightkite, capacity 64, H=4), while the reuse-tail component of
  H4/H10/H11 now spans 21 family-capacity cells. H5 and H6 rest on one
  family (metacdn). H1 is broader but bounded: the completed `1%-50%`
  learning curve disfavors the sample-size explanation within that tested
  range; it is not a claim that more data can never help.
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
