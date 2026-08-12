# Cross-Project Cold-Start Handoff

**Observed:** 2026-08-12. This is a routing document, not a replacement for
the project-specific records linked below. Re-check live process and Git state
before acting.

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

The C0/C1/C2 continuation-policy campaign is **RUNNING** in tmux session
`kbs_continuation_c0_c1_c2_production_resume_20260812`, executing
`scripts/experiments/run_continuation_policy_causal_ablation.py` with
`--resume --max-wall-hours 8`. Read-only inspection observed **8/21 units
complete** and **24 policy rows**. It is not scientifically complete. Do not
stop, signal, attach interactively, restart, change its configuration, or
modify its outputs. The separate learning-curve tmux session is finished; do
not relaunch or kill it. Other active local workers are also out of scope.

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
- **R2 Major 3, offline/online mechanism:** C0/C1/C2 is running locally, 8/21 complete, with no final causal conclusion. Supporting diagnostics do not close this question.
- **R2 Major 4, practical significance:** `COMPLETE_WITH_CAVEAT` from Wulver-relayed 420/420 controlled timing rows; this is implementation timing, not a theoretical complexity proof, and the payload still needs local promotion/audit.
- **Reviewer #3:** mechanistic evidence is substantial, but the continuation campaign remains unresolved. Horizon sensitivity is last-known partial/queued; fallback remains a limitation/future-work position unless explicitly required.

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
- Exact-target oracle merely for duplication.
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

1. **Local compute:** passively monitor the current C0/C1/C2 campaign; when it naturally completes or stops, run integrity checks and synthesize it before making reviewer claims.
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
is local-only; C0/C1/C2 is the local experiment currently running; Wulver
status is last-known partial/queued behind maintenance; Major 2 is complete,
Major 4 is complete with an implementation caveat, while Major 1 and Major 3
remain open; only Wiki2018 is cleared; and the next action is passive local
campaign monitoring followed by integrity audit, not a new experiment or
publication.
