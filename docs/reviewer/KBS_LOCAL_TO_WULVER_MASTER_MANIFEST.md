# KBS Local-to-Wulver Master Transfer Manifest

Status: authoritative single manifest for what would need to move from this
local workstation (`al-khwarizmi`) to Wulver, and in what form. Supersedes
nothing existing -- `docs/reviewer/local_to_wulver_continuation_sync_manifest.md`
remains the detailed, protocol-specific manifest for the C1/C2 continuation
work specifically; this file is the whole-branch superset.

No transfer was performed while building this manifest. No Wulver contact was
made. This is planning material only.

Last updated: 2026-08-11. The local learning-curve `50%` campaign is now
`FINAL_50PCT_VALIDATED`; see Section C for the completed transfer status.

---

## A. Git source/config/test payload

Every file below is either new (`A`, relative to `origin/kbs/second-revision-science`)
or modified in a way classified in
[`KBS_LOCAL_WULVER_CONFLICT_MATRIX.md`](KBS_LOCAL_WULVER_CONFLICT_MATRIX.md).
These hashes are pinned as of 2026-08-10 -- recompute with `sha256sum` before
any actual transfer and treat a mismatch as "local payload changed, refresh
this manifest first."

| File | SHA256 | Conflict class |
|---|---|---|
| `configs/continuation_policy_causal_ablation_v1.json` | `881210c980b3d5615914823a5aa08922f02dd8e660f0dc9c7d7b180ef9ee2400` | NO_CONFLICT_EXPECTED |
| `configs/supervision_objective_learning_curve_v1.json` | `2cc93412c03c03ca9218ca67f441a2127fac95093b3186cc486517cc8b40c0b8` | NO_CONFLICT_EXPECTED |
| `scripts/experiments/analyze_eviction_loss_target_degeneracy.py` | `9e8fe8062fa8df6cc7bc332f297a214e15504d3a21e3d25ea7cad78172dae7dd` | NO_CONFLICT_EXPECTED |
| `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py` | `f44ff9974784debb836fbac0ad7e601c691b0f60ec416e47ace643fbcd00d87a` | NO_CONFLICT_EXPECTED |
| `scripts/experiments/run_exact_target_oracle_diagnostic.py` | `3a3e69cb43cc9876d35cf609c6a3cf00edf865bdac0a95716dd07cce0b32a054` | NO_CONFLICT_EXPECTED |
| `scripts/experiments/run_supervision_objective_learning_curve.py` | `fc73ae74dc63d92c469350a8a3d0759be487325d2527bbc3428ba9dfeb6d7cab` | NO_CONFLICT_EXPECTED |
| `scripts/validation/revision_readiness.py` | `3b006cbdc4546f96cc33b2d1e4086196472ca553da497c238143696947b6b14c` | NO_CONFLICT_EXPECTED |
| `scripts/validation/revision_status.py` | `28b4e0de6a5042237401b0ecf1c9b20ce3278434259ec91f5928a9775041e8de` | NO_CONFLICT_EXPECTED |
| `src/lafc/continuation_policy_ablation.py` | `a5bdc14ceb9269969f570860ac3a7e34ef75c9c3b4efd4dc1d1d0bd27b0b4701` | NO_CONFLICT_EXPECTED |
| `src/lafc/oracle_diagnostics.py` | `819d0ec493aaeaa32a204e0699cde907af6344bbcf3149fba7cfbc52ec372ce7` | NO_CONFLICT_EXPECTED |
| `src/lafc/reviewer_diagnostics.py` | `014f681cbdd753189136edd18eef8a6fd20dec82f66cb7b182b3d80e116dd600` | NO_CONFLICT_EXPECTED |
| `src/lafc/target_degeneracy.py` | `a0462432e5f6c87422f919943502f41e11ba5f82281bef4347963d132b2ec260` | NO_CONFLICT_EXPECTED |
| `tests/test_continuation_policy_ablation.py` | `c429a8fd957da005d0d0abdbe09ebaa980b4baab8a8407174b83653e805bca60` | NO_CONFLICT_EXPECTED |
| `tests/test_oracle_diagnostics.py` | `63440577121fa5f2019c27765d47c2b630f3f79bedbf3d204e2da674a7dbbf92` | NO_CONFLICT_EXPECTED |
| `tests/test_reviewer_diagnostics.py` | `49130ec64c11dbad78d84c3ea5513ad349f74b2ac7635d546c6ea10cbad259b8` | NO_CONFLICT_EXPECTED |
| `tests/test_revision_readiness.py` | `dccf4deb204c82c00dfadbffdde4a8db76ec509cc4b350388942a4091e39e284` | NO_CONFLICT_EXPECTED |
| `tests/test_revision_status.py` | `ef155d16ba56606c46b8611685a390b4a1320e4e14be614c73212b99f0f506c4` | NO_CONFLICT_EXPECTED |
| `tests/test_supervision_objective_learning_curve.py` | `b9c10568b9eacad296cbf736aca62489b3cc2605016526e1889cd6b6f322220c` | NO_CONFLICT_EXPECTED |
| `tests/test_target_degeneracy.py` | `0bc54401c668f381e0fe36203fc51f3b365192789d6a6df1251b6a1ec6ea70d5` | NO_CONFLICT_EXPECTED |
| `src/lafc/supervision_objective_ablation.py` | `f5c9fbc95e1df030f838a98aaa2f48233bc3f23c2efb588127fc7f5d2e30e118` | **NEEDS_SEMANTIC_REVIEW** -- refactor of existing shared code |
| `tests/test_supervision_objective_ablation.py` | `0d7db08942e9eabb6b89ca12a669744e7eb91d8c9c1a9fd003ee572b212ac12f` | KEEP_LOCAL_VERSION (additive) |
| `scripts/experiments/run_cross_family_heldout_eval.py` | `9cf8936667692b6b4e45b697c34078e8a9fc3220bec1932fa3f2ad60682d516a` | MANUAL_SECTION_MERGE (portability fix) |
| `scripts/experiments/run_evict_cross_family_pipeline.py` | `c67d0310ae770a89f559ac4d5276a6e66631d1db30346091e9790759a50c47cc` | MANUAL_SECTION_MERGE (portability fix) |
| `scripts/build_supervision_objective_ablation_dataset.py` | `5df2e646cf7ebaf11f288c375543f47df074dc53fd10a5abfaff669c06f1a810` | MANUAL_SECTION_MERGE (portability fix) |
| `scripts/experiments/resume_distribution_shift.py` | `2a7f38872ab01a00687b9f7e0846d24c1bcc4ce97fdaaf1d2d413ffb68566250` | MANUAL_SECTION_MERGE (portability fix) |
| `scripts/experiments/run_practical_significance_ablation.py` | `f3c49306c281ffe5e0d86a16dcaafb4ec7848c21aaa89dfb86c4ab5ea603c4fa` | MANUAL_SECTION_MERGE (portability fix) |
| `scripts/experiments/run_practical_significance_controlled.py` | `aac8d1677ed8de5c0277eda14cbf2389355b4746032059a5b90142653ba90fa9` | MANUAL_SECTION_MERGE (portability fix) |
| `configs/reviewer_fairness_protocol.json` | `ca2fa8c9d8b536fe254190baccdf84b349ee13bf7424e30be42459f88a642249` | MANUAL_SECTION_MERGE (portability fix) |
| `.gitignore` | `199b29bcb1caaec0274d670c630a38fe09f9f8c1749b4f1b357096690f77b014` | MANUAL_SECTION_MERGE |

Full per-file rationale: [`KBS_LOCAL_WULVER_CONFLICT_MATRIX.md`](KBS_LOCAL_WULVER_CONFLICT_MATRIX.md).

## B. Stable doc payload (living docs -- do not pin as integrity gates)

These docs are expected to keep changing as fractions/phases complete. List
them for transfer awareness; do not hash-gate them the way payload A is
hash-gated. Snapshot a hash only if explicitly freezing a point-in-time copy
for a specific handoff.

- `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` (new, this pass's predecessor)
- `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md` (new, this pass's predecessor)
- `docs/reviewer/KBS_LOCAL_WULVER_CONFLICT_MATRIX.md` (new, this pass)
- `docs/reviewer/KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md` (new, this pass -- self)
- `docs/kbs_second_revision_repository_state.md`
- `docs/reviewer/kbs_second_revision_artifact_map.md`
- `docs/reviewer_revision_roadmap.md`
- `docs/reviewer/kbs_negative_results_interpretation.md`
- `docs/reviewer/KBS_LOCAL_TO_WULVER_SYNC_STATUS.md`
- `docs/reviewer/local_to_wulver_continuation_sync_manifest.md` (protocol-specific; keep as the detailed C1/C2 sync doc)
- `docs/reviewer/kbs_evidence_eligibility.md`
- `docs/reviewer/kbs_comparison_fairness_audit.md`

## C. Generated results for rsync (not git payload)

Path + scientific status only -- no content hashes recorded here for
still-active outputs, per instructions.

| Path | Status | Transfer note |
|---|---|---|
| `analysis/supervision_objective_learning_curve_v1/` | `25%` `COMPLETE_VALID` (7/7, 42/42); `50%` `FINAL_50PCT_VALIDATED` (7/7, 42/42, all `status=ok`); synthesis at `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`; `100%` intentionally not run due `STOP_SAMPLE_SIZE_HYPOTHESIS` | Eligible for intentional transfer after audit; no active learning-curve writer |
| `models/supervision_objective_learning_curve_v1/` | mirrors the completed 50% analysis dir's fold state | eligible for intentional transfer after audit; do not commit model binaries |
| `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/` | one cell, `COMPLETE` | eligible for transfer now (static, not being written to) |
| `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/` | one cell, `COMPLETE` | eligible for transfer now |
| `analysis/distribution_shift_ablation_v1/` | `PARTIAL`, 24/42 rows, 4/7 families, `STOPPED_CLEANLY_PARTIAL` per `revision_status.py` | eligible for transfer now (not actively running); mark partial |
| `analysis/kbs_comparison_fairness_audit.json` | `COMPLETE` (as an audit) | eligible for transfer now |
| `analysis/practical_significance_ablation_v1/` | `SMOKE_ONLY` | eligible for transfer now; keep smoke-only caveat attached |
| `analysis/kbs_local_current_evidence_synthesis_20260810/` | `COMPLETE` (as a synthesis snapshot) | eligible for transfer now; timestamped snapshot, will not be updated in place |
| `analysis/continuation_policy_light/` | `COMPLETE` (already git-tracked, historical/exploratory) | already in git payload via normal clone, not an rsync item |

## D. Large models -- transfer only if needed

| Path | Note |
|---|---|
| `models/supervision_objective_ablation_v1/` | 28-model frozen registry; needed on Wulver only if Wulver does not already have an equivalent frozen registry from its own run of the same protocol -- verify before transferring, do not duplicate a frozen artifact set |
| `models/supervision_objective_learning_curve_v1/` | see hold in Section C; per-fraction, per-family, per-condition `.pkl` files, not large individually but numerous |

## E. Machine-local / do not transfer

- `logs/` (tmux/launcher scratch logs, machine- and session-specific)
- `.venv*/` (per-worktree virtualenvs)
- `__pycache__/`, `*.pyc`, `.pytest_cache/`
- active tmux session state itself (not a file)
- `/tmp` smoke artifacts (per the continuation sync manifest's existing out-of-scope list)

## F. Conflict-risk / manual merge

See [`KBS_LOCAL_WULVER_CONFLICT_MATRIX.md`](KBS_LOCAL_WULVER_CONFLICT_MATRIX.md)
for the full per-file table. Summary of what needs a human/agent to actually
diff against a Wulver checkout before merging (not a blind copy):

1. `src/lafc/supervision_objective_ablation.py` -- highest risk, refactored shared kernel.
2. Seven small path-portability fixes (`run_cross_family_heldout_eval.py`,
   `run_evict_cross_family_pipeline.py`, `configs/reviewer_fairness_protocol.json`,
   `scripts/build_supervision_objective_ablation_dataset.py`,
   `scripts/experiments/resume_distribution_shift.py`,
   `run_practical_significance_ablation.py`,
   `run_practical_significance_controlled.py`) -- low effort each, but still
   textual merges against a shared file, not new-file additions.
3. `.gitignore` and the general orientation docs (`README.md`, `docs/README.md`,
   `analysis/README.md`, `scripts/README.md`, `docs/repo_map.md`, etc.) --
   expected to have independently drifted on both sides.

Everything else in Section A is a new file and should apply cleanly.
