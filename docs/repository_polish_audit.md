# Repository Polish Audit

## 1. Repository and Worktree Map

| Path | Branch | HEAD | Status | Active Experiment |
| :--- | :--- | :--- | :--- | :--- |
| `/home/soroush/Augmented-caching` | `main` | `a60e582` | Dirty | Legacy LRB (PID 113981) |
| `/home/soroush/Augmented-caching-3l-cache` | `feat/3l-cache-baseline` | `e351e70` | Clean | 3L-Cache baseline |
| `/home/soroush/Augmented-caching-cacheus` | `feat/cacheus-baseline` | `5a54b33` | Clean | Cacheus baseline |
| `/home/soroush/Augmented-caching-fairness` | `feat/reviewer-fairness-protocol` | `f221346` | Dirty | Cross-family (PID 1269456) |
| `/home/soroush/Augmented-caching-halp` | `feat/halp-baseline` | `b32cb68` | Clean | HALP baseline |
| `/home/soroush/Augmented-caching-kbs-parallel` | `kbs-revision-parallel-cleanup` | `9a1642a` | Dirty | - |
| `/home/soroush/Augmented-caching-objective-ablation` | `feat/supervision-objective-ablation` | `3dce9d0` | Dirty | Objective ablation (PID 3458550) |

## 2. Active Job Protected Paths

The following paths **MUST NOT** be modified while experiments are running:

- `/home/soroush/Augmented-caching/analysis/external_learned_baselines/lrb`
- `/home/soroush/Augmented-caching-fairness/data/derived/evict_value_v1_cross_family_v1`
- `/home/soroush/Augmented-caching-fairness/models/cross_family_v1_staging`
- `/home/soroush/Augmented-caching-fairness/analysis/reviewer_fairness_cross_family_v1`
- `/home/soroush/Augmented-caching-objective-ablation/data/derived/supervision_objective_ablation_v1`
- `/home/soroush/Augmented-caching-objective-ablation/analysis/supervision_objective_ablation_v1`
- `/home/soroush/Augmented-caching/data` (read access)

## 3. Repository Organization Findings

- **Cluttered Root**: The root directory contains multiple manuscript packaging folders (`submission_kbs_revision_docx`, `submission_kbs_revision_final`), zip files, and multiple data/result folders (`analysis`, `reports`, `tables`, `figures`, `logs`, `slurm`).
- **Misplaced Analysis Results**: `analysis/` root contains hundreds of one-off CSV/MD files that lack a subdirectory, making it difficult to distinguish between canonical and exploratory results.
- **Duplicate/Similar Scripts**: Multiple scripts in `scripts/` have similar names (e.g., `build_evict_pairwise_dataset_v1.py` vs `build_evict_value_pairwise_dataset.py`) with subtle implementation differences.
- **Source Package**: `src/lafc/` is generally well-structured, but several `evict_*.py` files live at the top level of the package instead of in submodules.

## 4. Git Hygiene Findings

- **Minimal .gitignore**: Previously lacked exclusions for virtual environments, IDE files, and temporary manuscript build artifacts. (Partially addressed in Query 1).
- **Untracked Artifacts**: Hundreds of untracked files in `analysis/`, `reports/`, and `submission_*` folders.
- **Dirty Worktrees**: Most active worktrees have uncommitted changes related to ongoing experiments.

## 5. Documentation Findings

- **Scattered Revision Docs**: Key documents for the reviewer revision (`reviewer_revision_roadmap.md`, `reviewer_next_stage_runbook.md`) are scattered across worktrees.
- **README Coverage**: Comprehensive, but refers to many "exploratory" or "internal" documents that may confuse new researchers.
- **Stale Links**: No major broken links found, but several documents overlap in scope.

## 6. Reproducibility Findings

- **Environment Dependencies**: `pyproject.toml` is the source of truth, but some environments lack `pulp`, causing test collection failures.
- **Test Suite**: 341 tests collected, but some baseline-related tests are skipped or fail collection in certain environments.
- **Seed Policy**: Most scripts support `--seed`, but it is not uniformly enforced across all exploratory runners.

## 7. Test/CI Findings

- **No Automated CI**: Repository relies entirely on local `pytest` execution.
- **Cross-Worktree Interference**: Pytest collection sometimes attempts to reach into adjacent worktrees if not properly scoped.

## 8. External Baseline/Provenance Findings

- **Honest Provenance**: `docs/baselines.md` provides clear citations and honest assessments of "faithfulness" for implemented policies.
- **Worktree Isolation**: Baselines like 3L-Cache and HALP are kept in separate worktrees, which helps with dependency isolation but complicates unified evaluation.

## 9. Artifact Policy Findings

- **Inconsistency**: There is no clear policy on which `analysis/` results should be tracked vs. ignored.
- **Provenance Preservation**: Provenance JSONs are generally produced but not always committed with the results.

## 10. Broken Link/Path Findings

- Internal relative paths in README and `repo_map.md` are currently valid.

## 11. Prioritized Cleanup Plan

| Severity | Task | Query |
| :--- | :--- | :--- |
| **P0** | Consolidate reviewer revision docs into `main` | Query 3 |
| **P0** | Fix dependency issues in `pyproject.toml` / environment | Query 3 |
| **P1** | Move manuscript packaging folders to `artifacts/` | Query 2 |
| **P1** | Group root-level `analysis/` files into subdirectories | Query 2 |
| **P2** | Normalize script names and remove truly obsolete ones | Query 2 |
| **P2** | Enhance `.gitignore` for all worktrees | Query 2 |
| **P3** | Refactor top-level `lafc/evict_*.py` into submodules | Query 2 |

---

## Proposed Scopes for Queries 2-4

### QUERY 2: Repository Structure & Git Hygiene
- Move manuscript artifacts to `artifacts/`.
- Organize `analysis/` and `reports/` into logical subdirectories.
- Clean up `scripts/` and normalize naming.
- Synchronize `.gitignore` across all worktrees.
- (Optional) Refactor `src/lafc/` top-level modules.

### QUERY 3: Documentation, Reproducibility & Provenance
- Consolidate all reviewer-revision documents into `docs/reviewer/`.
- Update `pyproject.toml` and provide a clear `environment.yml` or setup guide.
- Overhaul README to focus on canonical results while clearly labeling exploratory ones.
- Establish a clear artifact and provenance commitment policy.

### QUERY 4: Final Validation & Release Readiness
- Final test suite execution across all families.
- CI workflow draft (e.g., GitHub Actions).
- Final branch integration plan for merged baselines.
- Final release checklist and repository freeze.

## Overall Repository Polish Status

**IMPORTANT REPRODUCIBILITY/HYGIENE ISSUES FOUND**

The repository contains high-quality scientific work, but the organizational clutter and scattered documentation pose a risk to long-term reproducibility and ease of use for new researchers.
