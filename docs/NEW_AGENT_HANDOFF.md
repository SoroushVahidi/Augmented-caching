# New-Agent Handoff

Short capstone to [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) (read that
first -- it has the full project purpose, scientific question, evidence
table, and hypothesis interpretation). This file exists only to give a crisp
"what do I do right now" answer and a hard safety list, in one place. See
[`README.md`](README.md) for the full authoritative-status-doc hierarchy if
two documents appear to disagree.

**Repository / branch:** `/home/soroush/Augmented-caching-kbs-second-revision`
on branch `kbs/second-revision-science`, tracking
`origin/kbs/second-revision-science`. This is the canonical worktree for
current reviewer-science work; other local worktrees
(`Augmented-caching-main`, `-3l-cache`, `-cacheus`, `-fairness`, `-halp`,
`-kbs-parallel`, `-objective-ablation`) are feature-development or
historical, not entry points -- see `git worktree list`.

**Last consolidated:** 2026-08-13 -- **both remaining local heavy campaigns
(C0/C1/C2 continuation-policy causal ablation, distribution-shift ablation)
have completed and passed formal post-completion integrity audit.** There is
no local heavy compute running. Compact evidence:
`reports/kbs_final_evidence_20260813/`. Prior consolidation 2026-08-11
(local finalization/handoff pass, plus a same-day reconciliation against
fresh Wulver-side facts relayed by the user -- see
[`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)).
Re-verify git and tmux state yourself before acting -- see the checklist in
`DEVELOPMENT_STATUS.md` section 10.

## Active local compute -- none

As of 2026-08-13, neither the C0/C1/C2 continuation-policy campaign (tmux
session `kbs_continuation_c0_c1_c2_production_resume2_retry_20260812`) nor
the distribution-shift campaign (tmux session
`kbs_distribution_shift_completion_resume2_20260812`) is running; both tmux
sessions and worker processes have exited naturally. Both are
`FINAL_VALIDATED`:
- C0/C1/C2: `analysis/continuation_policy_causal_ablation_production_v1/`,
  models in `models/continuation_policy_causal_ablation_production_v1/`, log
  `logs/kbs_continuation_c0_c1_c2_production_resume2_retry_20260812.log`.
- Distribution-shift: `analysis/distribution_shift_ablation_v1/`, log
  `logs/kbs_distribution_shift_completion_resume2_20260812.log`.

Do not relaunch either. A third, unrelated tmux session,
`kbs_learning_curve_50pct_wiki2018_resume_20260811`, is also still present
from the already-completed 50% learning-curve campaign (see "Do Not Do"
below) -- it too is finished; leave it alone rather than killing it
speculatively.

Both campaign output directories remain gitignored (this is intentional and
should stay that way even though the campaigns are complete -- the raw
per-unit/per-fold trees are large and not meant to be tracked; the tracked,
durable record of their results is `reports/kbs_final_evidence_20260813/`).

## Immediate Next Actions

1. **Done (2026-08-13).** Priority P1 sync completed under explicit task
   authorization: the corrected held-out cross-family `evict_value_v1` replay
   (42/42, SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`,
   verified) and the 420/420 controlled timing campaign were both `rsync`'d
   from `login02:/mmfs1/project/ikoutis/sv96/Augmented-caching` to
   `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/`
   and `analysis/kbs_controlled_timing_20260810/` +
   `analysis/kbs_controlled_timing_final_analysis_20260811/` respectively.
   All required hashes matched (16/16 treatment, 13/13 timing) and an
   independent local re-audit passed every gate. See
   `reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md` and
   `controlled_timing_integrity.md`. Remaining next action: run
   `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...` to
   materialize the final matched table (baseline side already locally
   validated in `analysis/kbs_r2_major1_evidence_prep_20260811/`), preserving
   the caveat that no same-protocol comparison exists against
   LRU/SIEVE/FIFO/LRB/3L-Cache/HALP/CACHEUS.

2. **Done (2026-08-13).** The continuation-policy C0/C1/C2 production
   campaign (`NEXT_STEPS.md` P2.5) completed and passed its 21-unit
   integrity audit: 21/21 units, 63/63 policy rows, 21/21 label-agreement
   rows, 21/21 pi2-training rows, all `status=ok`. `FINAL_VALIDATED`;
   result C2 improves over C1 in 13/21 cells (H5 `PARTIALLY_SUPPORTED`).
   Do not rerun. See `reports/kbs_final_evidence_20260813/c0_integrity_summary.md`.

3. **Completed locally.** Exact-target-oracle replication, strict-preference/
   horizon, learned/exact agreement, C0/C1/C2 continuation-policy causal
   ablation, and distribution-shift ablation are all `FINAL_VALIDATED`;
   audit their existing artifacts but do not rerun them. Their generated
   outputs are in `analysis/exact_target_oracle_replication_v1/`,
   `analysis/strict_preference_horizon_diagnostic_v1/`,
   `analysis/learned_exact_target_agreement_v1/`,
   `analysis/continuation_policy_causal_ablation_production_v1/`, and
   `analysis/distribution_shift_ablation_v1/`. Compact evidence for the last
   two: `reports/kbs_final_evidence_20260813/`.

4. **Priority P0.** Machine: local. Do not rerun the `P(T > H | resident)`
   reuse-tail diagnostic. It is `LOCAL_COMPLETE` at
   `analysis/reuse_tail_horizon_diagnostic_v1/` and synthesized in
   `docs/reuse_tail_horizon_diagnostic_v1_synthesis.md`. Only causal
   excess-miss attribution remains unimplemented.

For the full ranked P0-P4 roadmap beyond these five, see
[`NEXT_STEPS.md`](NEXT_STEPS.md); for Wulver-side sync priority, see
[`WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](WULVER_TO_GITHUB_PROMOTION_QUEUE.md).

## Other local-only generated artifacts (not reviewer-blocking)

- `analysis/huggingface_dataset_preview_v0_2/` -- a ~134 MB gitignored
  staging artifact retained for provenance. It is not the public release
  source and must not be uploaded or treated as a second dataset repository.
  The authoritative current public release is maintained in the separate
  `/home/soroush/lafc-evict-dataset` repository: Hugging Face
  <https://huggingface.co/datasets/SoroushVahidi/lafc-evict> and Zenodo v0.2
  <https://doi.org/10.5281/zenodo.21895844>.
- `analysis/kbs_reviewer_synthesis_prep_20260811/` -- lightweight (~60 KB)
  reviewer-evidence synthesis notes (evidence table, manuscript-safe
  numerical summaries, a stale-reference scan that found no active stale
  references as of 2026-08-11). Tracked in git, same convention as
  `analysis/kbs_r2_major1_evidence_prep_20260811/`.

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
  as independently verified by this workstation**, with two exceptions:
  the corrected held-out `evict_value_v1` treatment and the controlled
  timing campaign were directly `rsync`'d from Wulver and independently
  hash- and structure-verified locally on 2026-08-13 under an explicit task
  authorization -- see `reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md`
  and `controlled_timing_integrity.md`. All other Wulver-sourced facts in
  that matrix were relayed by the user from a separately audited Wulver
  session on 2026-08-11, not confirmed via direct Wulver contact from here.
  Re-verify their job IDs/hashes directly when a task explicitly authorizes
  contacting Wulver, especially before citing them in anything
  manuscript-facing.
- **Continuation-policy C0/C1/C2 and distribution-shift are both complete
  and integrity-audited as of 2026-08-13.** Do not relaunch either. Cite
  outcomes only via `reports/kbs_final_evidence_20260813/` or the canonical
  raw CSVs -- do not cite the earlier partial/Wulver-merged checkpoints
  (18/42, 24/42) as current.
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
  H4/H10/H11 now spans 21 family-capacity cells. H5 and H6 now also each
  rest on the full 7-family/21-cell C0/C1/C2 and distribution-shift
  campaigns (`FINAL_VALIDATED` 2026-08-13) -- H5 is `PARTIALLY_SUPPORTED`
  (13/21 cells improve, one large counter-example), H6 is `DISFAVORED` as a
  shift-reduction-improves-performance story (do not claim distribution
  shift itself does not exist). H1 is broader but bounded: the completed
  `1%-50%` learning curve disfavors the sample-size explanation within that
  tested range; it is not a claim that more data can never help.
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
