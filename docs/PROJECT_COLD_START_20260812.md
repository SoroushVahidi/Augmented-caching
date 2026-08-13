# Cross-Project Cold-Start Handoff

**Observed:** 2026-08-12. This is a routing document, not a replacement for
the project-specific records linked below. Re-check live process and Git state
before acting.

**2026-08-13 update (KBS scope only):** the C0/C1/C2 continuation-policy and
distribution-shift campaigns referenced as "RUNNING" below have both since
completed and passed formal post-completion integrity audit
(`FINAL_VALIDATED`). See `docs/DEVELOPMENT_STATUS.md`,
`docs/NEXT_STEPS.md`, and `reports/kbs_final_evidence_20260813/` for the
current state; the "Active Local Compute" and "KBS Reviewer State" sections
below are stale for the KBS stream as of this update. Everything about the
SIGMOD/dataset-repo streams below is out of scope for this update and not
re-verified here.

## Project Boundary

There are four distinct streams:

| Stream | Owns | Canonical location |
|---|---|---|
| KBS second revision | KBS reviewer response, hypotheses, reviewer experiments, synthesis | `/home/soroush/Augmented-caching-kbs-second-revision` |
| Scientific source/data | source traces, generation code, derived supervision, experiment outputs | `/home/soroush/Augmented-caching` and registered worktrees |
| LAFC-Evict publication | dataset schemas, provenance, release manifests, publication tooling and version state | `/home/soroush/lafc-evict-dataset` |
| SIGMOD/PODS 2027 | `LAFC-Evict: A Large-Scale Counterfactual Benchmark for Learned Cache Eviction`, Research Round 3, Submission 919 | identified manuscript/review bundle under the dataset repo's `publication/bundles/sigmod2027_anonymous_review/` |

The KBS paper is **Decision-aligned eviction-value prediction for learning-augmented caching**. The SIGMOD manuscript is a separate dataset/benchmark manuscript. The same scientific data ecosystem supports both, but manuscript obligations, release payloads, and repositories are distinct. The submitted anonymous SIGMOD bundle is already identified; do not investigate its identity again and do not call it “Sigma.”

## Repository Snapshot

| Repository/worktree | Branch | HEAD | State |
|---|---|---|---|
| KBS second revision | `kbs/second-revision-science` | `12798d8` | clean, synchronized with `origin` |
| LAFC-Evict publication | `master` | `6137e20` | clean, synchronized with `origin` |
| Augmented-caching main worktree | `chore/repository-polish` | `ceb3670` | intentionally dirty/in progress; untouched by this handoff |
| Augmented-caching main branch worktree | `main` | `a01a60d` | clean, synchronized |
| 3L-Cache worktree | `feat/3l-cache-baseline` | `e351e70` | local feature work, untracked generated baseline outputs |
| CACHEUS worktree | `feat/cacheus-baseline` | `5a54b33` | local feature work, untracked generated baseline outputs |
| fairness worktree | `feat/reviewer-fairness-protocol` | `f221346` | dirty active/output work; do not clean speculatively |
| HALP worktree | `feat/halp-baseline` | `b32cb68` | clean local feature work |
| KBS parallel worktree | `kbs-revision-parallel-cleanup` | `9a1642a` | local logs/reports untracked |
| objective-ablation worktree | `feat/supervision-objective-ablation` | `3dce9d0` | dirty active/output work |

The feature worktrees have no configured upstream in this checkout unless
shown above. Do not reset, clean, or commit them as part of this handoff.

## Active Local Compute

**Superseded 2026-08-13:** the C0/C1/C2 continuation-policy campaign (tmux
session `kbs_continuation_c0_c1_c2_production_resume2_retry_20260812`) and
the distribution-shift campaign (tmux session
`kbs_distribution_shift_completion_resume2_20260812`) have both completed
naturally and passed formal integrity audit -- `FINAL_VALIDATED`, 21/21
units / 7/7 folds. Neither tmux session nor worker process remains. Do not
relaunch either. See `reports/kbs_final_evidence_20260813/`. The separate
learning-curve tmux session is also finished; do not relaunch or kill it.
No local heavy compute is currently active.

## Last-Known Wulver State

`LAST_KNOWN_WULVER_STATE -- REQUIRES_FRESH_WULVER_AUDIT_AFTER_MAINTENANCE`.
Maintenance was still active at the last independent check. At that time,
jobs `1169299` (horizon sensitivity), `1171965` (3L-Cache), `1171966` (LRB),
and `1171967` (CACHEUS) were queued/pending; horizon was **17/35 complete,
18 remaining**, and modern learned-baseline jobs were pending behind
maintenance. This is not a current Wulver claim. Do not contact Wulver,
SSH, use Slurm, submit, requeue, or cancel anything in this pass.

## KBS Reviewer State

- **R2 Major 1, modern baselines:** local/classical evidence exists and the corrected held-out `evict_value_v1` 42/42 result is Wulver-only and still needs local sync/integrity review before final synthesis. Exact LRB/3L-Cache/CACHEUS local rows are validated with their documented caveats; Wulver jobs are replication/provenance strengthening.
- **R2 Major 2, supervision objective:** `COMPLETE_VALIDATED`; 84/84 objective-ablation cells and the 50% learning-curve stopping decision are validated. Do not run the 100% curve.
- **R2 Major 3, offline/online mechanism (updated 2026-08-13):** C0/C1/C2 and
  distribution-shift are both `FINAL_VALIDATED` locally (21/21 units / 7/7
  folds, integrity audited). H5 (continuation mismatch) `PARTIALLY_SUPPORTED`;
  H6 (generic state-shift) `DISFAVORED`. Status: `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING`
  -- no local experiment remains; see `reports/kbs_final_evidence_20260813/`.
- **R2 Major 4, practical significance:** `SYNC_PENDING` from Wulver-relayed
  420/420 controlled timing rows; this is implementation timing, not a
  theoretical complexity proof, and the payload still needs local
  promotion/audit.
- **Reviewer #3 (updated 2026-08-13):** the causal continuation campaign is
  now complete and validated; final answer `PARTIALLY_SUPPORTED` /
  `REGIME_DEPENDENT`. Horizon sensitivity is last-known partial/queued
  (Wulver-side); fallback remains a limitation/future-work position unless
  explicitly required.

Detailed evidence remains in `DEVELOPMENT_STATUS.md`, `NEXT_STEPS.md`,
`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`, and the reviewer registry/coverage
documents.

## Publication State

The current public release is **LAFC-Evict v0.2 published preview**, owned by
`/home/soroush/lafc-evict-dataset`:

- Hugging Face: `SoroushVahidi/lafc-evict`
- Zenodo version DOI: `10.5281/zenodo.21895844`
- Zenodo concept DOI: `10.5281/zenodo.21895843`
- Public scope: Wiki2018, two configs, 4.8M rows, CC0, pseudonymized derived supervision.

The v0.3 Wiki2018-only candidate is **built and locally validated, not
uploaded**: `release/lafc-evict-v0.3-candidate/`, 22,356,992 total rows,
11,178,496 per config, two configs, approximately 70 MB. A separate
publication-readiness decision and approval is required before any upload.
v1.0 is planning-only and conditional on provenance clearance. The dataset
repository's JSON and Markdown state records are authoritative; do not use
KBS staging directories as release sources.

## Provenance and Data

`wiki2018` is the only family currently documented as
`CLEARED_FOR_PUBLIC_RELEASE`, with attribution/caveat. `cloudphysics`,
`metacdn`, `metakv`, and `twemcache` are **NOT CLEARED**. `citibike` and
`brightkite` are **BLOCKED_PENDING_REVIEW**, with additional privacy/
re-identification scrutiny for Brightkite. Never interpret
`eligible_pending_final_review` as public clearance.

Use the existing inventory and evidence pointers; do not rescan or delete
large trees. The previously inventoried relevant scientific/data assets are
approximately **322 GB**, including `heavy_r1` (~96 GB), `cross_family_v1`
(~93 GB), and `objective_ablation_v1` (~121 GB). Dataset preparation is not
disk cleanup. No large-tree deletion is authorized.

## Do Not Rerun

- 100% learning curve; the validated stopping rule is `STOP_SAMPLE_SIZE_HYPOTHESIS`.
- The completed 84-cell objective-ablation campaign.
- The completed 420-row controlled timing campaign.
- The completed reuse-tail diagnostic.
- Broad target-degeneracy analysis, existing HALP causal analysis, and already-completed distribution-shift cells.
- Exact-target oracle replication, strict-preference/horizon, and learned/exact agreement are final-validated and must not be rerun.
- A local duplicate of Wulver's horizon sweep or queued exact 3L/LRB/CACHEUS replication.
- Published v0.2 build, HF v0.2 publication, or Zenodo v0.2 deposition creation.
- Full expensive supervision generation solely to produce v0.3.

## Do Not Touch

Do not modify scientific result payloads, active workers, large release/data
trees, published v0.2 Parquet files, SIGMOD bundle contents, credentials, or
external hosting services. Do not upload v0.3. Do not contact Wulver or use
Slurm. The dataset repository is the only publication owner; the scientific
repository is the source/data owner, not the publication owner.

## Exact Next Actions

1. **Local compute: done.** C0/C1/C2 and distribution-shift both completed
   and passed formal integrity audit on 2026-08-13; see
   `reports/kbs_final_evidence_20260813/`. No further local monitoring or
   experimentation is needed for these two campaigns.
2. **Local evidence:** audit/sync the corrected held-out `evict_value_v1` result and other Wulver-complete artifacts only in an authorized Wulver-access session; do not rerun local duplicates.
3. **Wulver-dependent:** after maintenance is independently known to be over, audit the queued jobs and their provenance before any new submission or interpretation.
4. **Publication authorization:** separately decide whether the validated v0.3 candidate is publication-ready; no HF/Zenodo action is automatic.
5. **Provenance research:** resolve uncleared-family licensing/privacy status, then integrate only validated results into the KBS manuscript and reviewer response.

Before acting, a new agent should verify `git status`, branch/HEAD/upstream,
the live process list, the C0/C1/C2 manifest, the dataset publication JSON,
and the last-known Wulver marker. Historical relayed Wulver facts must not be
promoted to current facts without a fresh authorized audit.

## Source-of-Truth Map

| Topic | Authoritative location |
|---|---|
| Cross-project cold start | this file |
| KBS current status | `docs/DEVELOPMENT_STATUS.md` |
| KBS next actions | `docs/NEXT_STEPS.md` |
| KBS cold-start details | `docs/NEW_AGENT_HANDOFF.md` |
| KBS reviewer registry/coverage | `docs/reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md` and `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md` |
| KBS hypothesis state | `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` |
| Cross-environment evidence | `docs/CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md` |
| LAFC-Evict machine publication state | `/home/soroush/lafc-evict-dataset/publication/LAFC_EVICT_PUBLICATION_STATE.json` |
| LAFC-Evict human publication state | `/home/soroush/lafc-evict-dataset/docs/LAFC_EVICT_PUBLICATION_STATE.md` |
| Future release design/plan | dataset repo `docs/V0_3_V1_RELEASE_DESIGN.md` and `docs/V0_3_V1_BUILD_PLAN.md` |
| Family clearance | dataset repo `manifests/source_family_registry.yaml` |
| SIGMOD identified bundle | dataset repo `publication/bundles/sigmod2027_anonymous_review/` and its `publication/bundles/README.md` |
| Scientific source/data | `/home/soroush/Augmented-caching` |

## Cold-Start Answer

The KBS paper is the learning-augmented caching method revision; SIGMOD is
the separate LAFC-Evict benchmark manuscript; LAFC-Evict v0.2 is public; v0.3
is local-only; C0/C1/C2 and distribution-shift are both `FINAL_VALIDATED`
local experiments as of 2026-08-13 (no local KBS heavy compute remains
running); Wulver status is last-known partial/queued behind maintenance;
Major 2 and Major 3 are scientifically complete (synthesis pending), Major 4
is sync-pending with an implementation caveat, Major 1 remains sync-pending;
only Wiki2018 is cleared; and the next action is Wulver synchronization plus
manuscript/rebuttal synthesis, not a new local experiment or publication.
