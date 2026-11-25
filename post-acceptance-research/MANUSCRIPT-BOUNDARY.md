# Accepted manuscript and November research

## Accepted manuscript

Soroush Vahidi, “Decision-aligned eviction-value prediction for learning-augmented caching,” *Knowledge-Based Systems*, volume 356, article 117353. Acceptance was November 15, 2026. The DOI is dated 2026 and Crossref assigns the article to the January 2027 issue. [Publisher DOI](https://doi.org/10.1016/j.knosys.2026.117353) · [Crossref record](https://api.crossref.org/works/10.1016/j.knosys.2026.117353)

| Record | Verified identity |
|---|---|
| Audited R3 manuscript and package source | `b7c2dd4707a387fd2e76b84b7a05a67df00c61e3` |
| Source worktree successor | `565fc1b` corrected the response files and retained identical manuscript TeX |
| Integrated October package baseline | `3f3782c` on `main`, after the final R3 upload-file PR |
| Consolidation base | `7d57a742f97f4ad40f85774cbe3a5ebdfd269a97`, November README update only since the October baseline |
| Packaged R3 manuscript PDF SHA-256 | `6857ff08f66bebfc990d8e5bbed03347ba427898f6d7282d6430c7615a18eaed` |
| Packaged R3 manuscript TeX SHA-256 | `2e4240255e869b51bdd4923192a526dcde52a7ee225a4ee03d06c308e16dc16e` |

The audited R3 SHA is the repository source identified by the submission manifest. The deposited publisher version has its own publication record. See the [October R3 package](../submission_kbs_r3_20261013/) and its [source and hash manifest](../submission_kbs_r3_20261013/KBS_R3_SUBMISSION_MANIFEST_20261013.md). The seven-file archive contains the revised PDF, editable LaTeX bundle and submission support files. The [R2 reviewer package](../submission_kbs_revision_final/) and [verification guide](../docs/reviewer/START_HERE.md) retain the longer review trail, reproduction matrix and result verification.

The [project simulator and original policies](../src/lafc/), [reviewer comparison and evidence summaries](../reports/kbs_final_evidence_20260813/), [Common-Model V2 audit](../reports/common_model_v2_formal_audit_20260814/AUDIT.md) and [tie-aware oracle audit](../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md) document the primary experimental claims. The original deployment pipeline underperformed the matched baselines. The accepted study includes the objective comparison, continuation and DAgger-style controls, timing, and workload-specific analysis. The [dataset acquisition guide](../docs/datasets.md), [reproduction matrix](../docs/reviewer/REPRODUCTION_MATRIX.md) and [citation metadata](../CITATION.cff) remain the manuscript entry points. LAFC-Evict is a related five-family release with its own protocol and per-family licensing.

## Subsequent research

November experiments, forensic analyses, labeler proofs and the controlled Phase 2 campaign are indexed in the [provenance inventory](PROVENANCE.md). Some controls are historical counts and others are new replays of existing frozen checkpoints. Each retains its provenance. The report, numbers, and experimental code in this package are separate from the accepted article and have not been peer-reviewed as part of it.
