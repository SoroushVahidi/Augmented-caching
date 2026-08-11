# R2 Major 1 evidence preparation package

This directory contains compact, non-duplicative evidence summaries for
Reviewer #2 Major Comment 1. The original generated CSVs are not copied;
they are referenced by path and SHA-256 in the JSON/CSV manifests here.

- `baseline_integrity.json`: row counts, hashes, protocol checks, and
  primary baseline summaries.
- `baseline_provenance.json`: runner, commit, branch, learning mode, seed,
  and caveats for LRB, 3L-Cache, and CACHEUS.
- `baseline_protocol_comparison.csv`: compact protocol-equivalence table.
- `baseline_summary.csv`: primary controlled-window baseline metrics.
- `trace_manifest.json`: durable reconstruction of the seven trace
  identities from local files and stored provenance hashes.
- `fairness_statement.md`: publication-facing fairness wording.
- `fairness_certificate_status.json`: certificate coverage and caveats.
- `treatment_status.json`: corrected evict_value_v1 local availability.
- `FINAL_COMPARISON_PENDING.md`: deterministic procedure for final synthesis
  once the verified corrected treatment CSV is synchronized.
- `reviewer_response_version_a.md` and `reviewer_response_version_b.md`:
  pending/internal and manuscript-ready response drafts.
