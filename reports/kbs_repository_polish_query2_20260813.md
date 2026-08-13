# KBS repository polish query 2 report

## SUMMARY

Query 2 converted the repository's reviewer-facing documentation from a set of
competing historical gateways into one public entry path:

- [README.md](../README.md)
- [docs/reviewer/START_HERE.md](../docs/reviewer/START_HERE.md)
- [docs/reviewer/RESULT_VERIFICATION.md](../docs/reviewer/RESULT_VERIFICATION.md)
- [docs/reviewer/REPRODUCTION_MATRIX.md](../docs/reviewer/REPRODUCTION_MATRIX.md)

No scientific result, experiment code, running output tree, manuscript
conclusion, or large result directory was modified.

## README CHANGES

The README now states within the first screen:

- the paper title and research problem;
- that the work is a controlled study of a decision-aligned eviction target,
  not a universal cache-policy superiority claim;
- the current negative headline conclusion;
- the curated evidence package;
- the primary reviewer navigation and reproduction/verification documents.

It also adds concise sections for primary results, reproducibility and
verification, baselines, datasets/workloads, evidence classes, setup/checks,
citation, and contact.

## REVIEWER NAVIGATION

Created [docs/reviewer/START_HERE.md](../docs/reviewer/START_HERE.md) as the
main technical navigation page. It maps reviewer questions to manuscript
sections, artifacts, generating scripts, validation files, and status labels.

## PRIMARY VS HISTORICAL EVIDENCE

README and reviewer docs now distinguish:

- `PRIMARY`: corrected matched second-revision evidence in
  `reports/kbs_final_evidence_20260813/`;
- `SUPPORTING`: mechanistic diagnostics and contextual timing;
- `HISTORICAL`: older single-split or contaminated exploratory results;
- `RUNNING`: the two acceptance-risk controls currently in progress.

`analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` is called
out as contaminated/historical and ineligible for primary claims.

## BASELINE PROVENANCE CHANGES

[docs/baselines.md](../docs/baselines.md) now defines explicit categories:

- `OFFICIAL_UPSTREAM_WRAPPED`
- `INDEPENDENT_REIMPLEMENTATION`
- `ADAPTATION`
- `RECONSTRUCTION`
- `NATIVE_IMPLEMENTATION`

LRB and 3L-Cache no longer use stronger "faithful port" style language in the
main fidelity summaries. HALP is identified as a reconstruction/adaptation,
not official code. CACHEUS remains correctly described as official upstream
source executed through a wrapper/adaptor.

## DATASET PROVENANCE CHANGES

README now names all seven families and distinguishes:

- BrightKite and Citi Bike as non-cache event streams transformed into request
  sequences;
- CloudPhysics, MetaCDN, MetaKV, and Twemcache as storage/cache-derived
  sources;
- Wikimedia as a pageview-derived proxy.

It also discloses that raw data are not necessarily redistributed and upstream
licensing/access constraints remain.

## RESULT-VERIFICATION DOCUMENTATION

Created [docs/reviewer/RESULT_VERIFICATION.md](../docs/reviewer/RESULT_VERIFICATION.md).
It indexes result schemas, expected cardinalities, duplicate-key checks,
held-out-family leakage checks, deterministic regression tests, hashes,
model-selection separation, recomputed summaries, baseline tests, and known
verification gaps.

## REPRODUCTION MATRIX

Created [docs/reviewer/REPRODUCTION_MATRIX.md](../docs/reviewer/REPRODUCTION_MATRIX.md).
It maps main results to scripts, inputs, outputs, configs/seeds, approximate
cost, validation, and availability. The two active acceptance-risk controls are
marked `RUNNING_NOT_YET_PRIMARY`.

## STALE DOCS RETIRED/UPDATED

- [CANONICAL_KBS_SUBMISSION.md](../CANONICAL_KBS_SUBMISSION.md) is now a
  historical redirect/stub.
- [docs/kbs_manuscript_workflow.md](../docs/kbs_manuscript_workflow.md) is now
  a concise redirect.
- [docs/README.md](../docs/README.md) is now a categorized index.
- [docs/NEXT_STEPS.md](../docs/NEXT_STEPS.md),
  [docs/reviewer/KBS_POST_COMPLETION_HANDOFF.md](../docs/reviewer/KBS_POST_COMPLETION_HANDOFF.md),
  [docs/reviewer_revision_roadmap.md](../docs/reviewer_revision_roadmap.md),
  and [docs/reviewer/kbs_evidence_eligibility.md](../docs/reviewer/kbs_evidence_eligibility.md)
  were updated to separate the completed prior campaign from the two running
  acceptance-risk controls.
- [reports/README.md](../reports/README.md) now points to the current curated
  evidence package.

## ABSOLUTE PATH CLEANUP

Reviewer-facing docs were checked for workstation-specific paths. Historical
examples in [docs/reviewer/KBS_POST_COMPLETION_HANDOFF.md](../docs/reviewer/KBS_POST_COMPLETION_HANDOFF.md)
now use `REPO_ROOT=/path/to/Augmented-caching`, and conflict-matrix examples
use `$REPO_ROOT`-style wording. Historical provenance logs were not rewritten.

## KNOWN GAPS STILL OPEN

- Some old model binaries are unavailable, so byte-level reverification of
  every historical model artifact is not possible.
- LRB and 3L-Cache are independent reimplementations/adaptations, not official
  runtime executions.
- HALP is a supporting reconstruction/adaptation because no official public
  implementation exists.
- Raw datasets are not necessarily redistributed; upstream access and license
  constraints remain.
- CACHEUS official-source execution depends on an external clone that may need
  to be fetched locally before reruns.
- Repository-level CI is not documented as a validation guarantee.
- The two running acceptance-risk controls are not yet audited or integrated.

## ITEMS DEFERRED TO QUERY 3

- `CITATION.cff` creation.
- Script/config portability changes, including any remaining hardcoded paths in
  executable code.
- Any code-level documentation metadata changes designated for Query 3.

## ITEMS DEFERRED TO QUERY 4

- Physical relocation or quarantine of
  `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`.
- Any broader result-tree reorganization after reviewer-facing warnings are in
  place.
