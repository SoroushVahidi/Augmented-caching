# KBS Local-Wulver Conflict Matrix

Status: authoritative future semantic-consolidation checklist.

This file classifies every source/config/test file touched by the local
`kbs/second-revision-science` branch (26 commits ahead of
`origin/kbs/second-revision-science` as of 2026-08-10) by expected conflict
risk against Wulver's independent work. It was built **without contacting
Wulver** -- classifications below reason from (a) whether the file is newly
added (`A`, lowest risk) vs. modified (`M`, higher risk), (b) whether the
diff to a modified file is purely additive/refactor-with-tests vs. behavior
-changing, and (c) what local sync docs already say about Wulver-side
parallel work. Treat every `NEEDS_SEMANTIC_REVIEW` row as unresolved until
an actual Wulver-side diff is inspected.

Last updated: 2026-08-10, HEAD `3cabb25015411d1d5c14558e17450584ff0ef212`
(before this pass's own commits) / current HEAD after this pass -- see
`docs/kbs_second_revision_repository_state.md` for the live pointer.

## New files (status `A` relative to origin) -- lowest conflict risk

New files cannot conflict except on path collision (Wulver independently
creating a file at the same path with different content). No evidence of
that exists in local sync docs.

| File | Classification | Notes |
|---|---|---|
| `configs/continuation_policy_causal_ablation_v1.json` | `NO_CONFLICT_EXPECTED` | new, frozen protocol config |
| `configs/supervision_objective_learning_curve_v1.json` | `NO_CONFLICT_EXPECTED` | new, frozen protocol config |
| `scripts/experiments/analyze_eviction_loss_target_degeneracy.py` | `NO_CONFLICT_EXPECTED` | new diagnostic |
| `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py` | `NO_CONFLICT_EXPECTED` | new smoke runner |
| `scripts/experiments/run_exact_target_oracle_diagnostic.py` | `NO_CONFLICT_EXPECTED` | new diagnostic |
| `scripts/experiments/run_supervision_objective_learning_curve.py` | `NO_CONFLICT_EXPECTED` | new experiment runner |
| `scripts/validation/revision_readiness.py` | `NO_CONFLICT_EXPECTED` | new, local-status-only tooling |
| `scripts/validation/revision_status.py` | `NO_CONFLICT_EXPECTED` | new, local-status-only tooling |
| `src/lafc/continuation_policy_ablation.py` | `NO_CONFLICT_EXPECTED` | new; does not reuse or modify `evict_value_v2_rollout.py` (see Source-of-Truth note below) |
| `src/lafc/oracle_diagnostics.py` | `NO_CONFLICT_EXPECTED` | new; imports from `supervision_objective_ablation.py` (see that file's row) |
| `src/lafc/reviewer_diagnostics.py` | `NO_CONFLICT_EXPECTED` | new; `build_nested_fraction_subsets` is local-only tooling |
| `src/lafc/target_degeneracy.py` | `NO_CONFLICT_EXPECTED` | new, standalone math module, no shared-kernel dependency |
| `tests/test_continuation_policy_ablation.py` | `NO_CONFLICT_EXPECTED` | new |
| `tests/test_oracle_diagnostics.py` | `NO_CONFLICT_EXPECTED` | new |
| `tests/test_reviewer_diagnostics.py` | `NO_CONFLICT_EXPECTED` | new |
| `tests/test_revision_readiness.py` | `NO_CONFLICT_EXPECTED` | new |
| `tests/test_revision_status.py` | `NO_CONFLICT_EXPECTED` | new |
| `tests/test_supervision_objective_learning_curve.py` | `NO_CONFLICT_EXPECTED` | new |
| `tests/test_target_degeneracy.py` | `NO_CONFLICT_EXPECTED` | new |
| All new `docs/reviewer/*.md`, `docs/kbs_second_revision_repository_state.md` | `NO_CONFLICT_EXPECTED` | new docs; see Master Manifest for living-doc handling |

## Modified files (status `M` relative to origin) -- higher scrutiny

| File | Classification | Notes |
|---|---|---|
| `src/lafc/supervision_objective_ablation.py` | **`NEEDS_SEMANTIC_REVIEW`** | Highest-risk file in the local-ahead set. `+200/-63` lines: adds `build_candidate_rows_for_full_cache_state` (the shared target-construction kernel oracle/degeneracy diagnostics now depend on) by **refactoring existing per-decision candidate-row logic out of an existing function**, not just appending new code. Purely additive test coverage (`test_selected_decision_filter_returns_exact_subset`) and all 12 existing tests in `tests/test_supervision_objective_ablation.py` still pass locally, so the refactor is regression-safe *against local tests* -- but if Wulver has independently modified this same file/function, this needs a real diff-level manual merge, not an automatic one. |
| `scripts/experiments/run_cross_family_heldout_eval.py` | `MANUAL_SECTION_MERGE` (low effort) | 4-line diff: removes a hardcoded `/home/soroush/Augmented-caching` default for `--data-read-root` (portability fix). If Wulver has its own machine-local default in this file, both sides need reconciling to a portable default, not a blind overwrite. |
| `scripts/experiments/run_evict_cross_family_pipeline.py` | `MANUAL_SECTION_MERGE` (low effort) | same portability fix pattern as above |
| `configs/reviewer_fairness_protocol.json` | `MANUAL_SECTION_MERGE` (low effort) | 1-line diff: removes a hardcoded `/home/soroush/Augmented-caching-fairness` worktree path |
| `scripts/build_supervision_objective_ablation_dataset.py` | `MANUAL_SECTION_MERGE` (low effort) | small portability fix, same family as above |
| `scripts/experiments/resume_distribution_shift.py` | `MANUAL_SECTION_MERGE` (low effort) | small portability fix |
| `scripts/experiments/run_practical_significance_ablation.py` | `MANUAL_SECTION_MERGE` (low effort) | small portability fix |
| `scripts/experiments/run_practical_significance_controlled.py` | `MANUAL_SECTION_MERGE` (low effort) | small portability fix |
| `tests/test_supervision_objective_ablation.py` | `KEEP_LOCAL_VERSION` (purely additive) | one new test function appended; no existing assertions changed |
| `.gitignore` | `MANUAL_SECTION_MERGE` (low effort, append-only in practice) | local additions are all narrow, new, path-specific ignore rules (learning-curve/oracle/degeneracy/synthesis output dirs); unlikely to remove any rule Wulver depends on, but a textual 3-way merge is still safest since `.gitignore` is a single shared file |
| `README.md`, `docs/README.md`, `analysis/README.md`, `scripts/README.md`, `docs/repo_map.md`, `docs/kbs_manuscript_workflow.md`, `docs/evict_value_v1_kbs_canonical_artifacts.md`, `docs/wulver_heavy_evict_value_experiment.md`, `docs/reviewer_revision_roadmap.md` | `MANUAL_SECTION_MERGE` | living/orientation docs, expected to have drifted on both sides; textual merge, not a hash-integrity concern |

## Special-attention files (explicitly requested)

- `src/lafc/evict_value_v2_rollout.py`: **not touched** by any of the 26
  local-ahead commits (confirmed via `git diff --name-status`). Classified
  `NO_CONFLICT_EXPECTED` from the local side -- we made zero edits. Whether
  Wulver has changed it independently is unknown (not contacted). If Wulver
  *has* changed it, this becomes `NEEDS_SEMANTIC_REVIEW` at sync time, not
  before. Source-of-truth note: this file's `simulate_rollout_misses`/
  `_choose_victim` "label-continuation policy" concept is a **different,
  older, exploratory** mechanism from the new `continuation_policy_ablation.py`
  C1/C2 causal protocol -- see the cross-reference added to
  `docs/reviewer/kbs_negative_results_interpretation.md` 9.7 in this pass.
  Only `_next_use_distance` from this file is reused by the current shared
  kernel (`supervision_objective_ablation.py`); the continuation/rollout
  logic itself is not shared.
- Held-out evaluation runner/tests
  (`run_cross_family_heldout_eval.py`, `run_evict_cross_family_pipeline.py`):
  see rows above -- portability-only changes, low-effort manual merge.
- Objective-ablation tests (`tests/test_supervision_objective_ablation.py`):
  see row above -- purely additive.
- `.gitignore`: see row above.
- Reviewer/status docs: see the doc rows above and the Master Manifest's
  living-docs section; do not pin hashes on these as sync integrity gates.
- Continuation-related code: `src/lafc/continuation_policy_ablation.py`
  (new, no conflict expected) vs. `src/lafc/evict_value_v2_rollout.py`
  (untouched, historical) -- see the dedicated note above; these are two
  different mechanisms, not one file with two conflicting versions.

## Source-of-Truth Audit Summary (see also the hypothesis/coverage maps)

No `AMBIGUOUS_SOURCE_OF_TRUTH` case was found where two files both claim
authoritative status for the *same* current protocol. One naming-adjacency
risk was found and documented (not code-level duplication): three different
modules use the word "continuation" for three different concepts
(`evict_value_v2_rollout.py` label-rollout continuation choice, historical/
exploratory; `distribution_shift_ablation.py` state-generation/trajectory
axis, already self-documented as distinct in
`docs/distribution_shift_ablation_protocol.md` Section 1; and
`continuation_policy_ablation.py`'s new C1/C2 policy-iteration
continuation, the current frozen protocol). A cross-reference was added to
`kbs_negative_results_interpretation.md` 9.7 in this pass rather than
touching any of the three implementations.

Shared-kernel reuse verified as intended (not duplicated): `oracle_diagnostics.py`
and `target_degeneracy.py`'s caller both route target construction through
`build_candidate_rows_for_full_cache_state` in `supervision_objective_ablation.py`;
`run_exact_target_oracle_diagnostic.py`, `analyze_eviction_loss_target_degeneracy.py`,
and `run_supervision_objective_learning_curve.py` all import the same
canonical score-window constants (`HISTORY_START=0, SCORE_START=10000,
SCORE_END=50000`) from `lafc.experiments.reviewer_fairness_common`, matching
the primary reviewer-fairness protocol's window exactly.
