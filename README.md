# Decision-aligned eviction-value prediction for learning-augmented caching

Public repository: <https://github.com/SoroushVahidi/Augmented-caching>

This repository contains code and verified experimental artifacts for the
Knowledge-Based Systems manuscript

**"Decision-aligned eviction-value prediction for learning-augmented caching"**

The paper studies a narrow question: whether a candidate-level eviction target
that estimates finite-horizon downstream miss cost is enough to make a useful
online cache-replacement policy. It is a controlled study of that target, not a
claim that `evict_value_v1` is a universally superior cache policy.

**Headline result.** Under the corrected held-out / matched second-revision
protocol, `evict_value_v1` does **not** outperform LRU, FIFO-Reinsertion,
SIEVE, LRB, 3L-Cache, CACHEUS, or HALP.

## Knowledge-Based Systems revision materials

For reviewers of the revised manuscript:

| Material | Location |
|---|---|
| Revised manuscript (PDF) | [submission_kbs_revision_final/01_Revised_Manuscript.pdf](submission_kbs_revision_final/01_Revised_Manuscript.pdf) |
| Response to reviewers | [submission_kbs_revision_final/02_Response_to_Reviewers.md](submission_kbs_revision_final/02_Response_to_Reviewers.md) |
| Reviewer verification guide | [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md) |
| Reproduction matrix | [docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md) |
| Result verification | [docs/reviewer/RESULT_VERIFICATION.md](docs/reviewer/RESULT_VERIFICATION.md) |

LaTeX source for the PDF is in
[submission_kbs_revision_final/07_LaTeX_Source/](submission_kbs_revision_final/07_LaTeX_Source/).

**Primary vs historical evidence**

- **PRIMARY / FINAL:** corrected held-out matched comparison; matched
  workload-specific Table 4 sources; Common-Model V2; tie-aware exact-target
  oracle; continuation C0/C1/C2 control; DAgger negative control; controlled
  timing campaign. See [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md).
- **HISTORICAL / NON-PRIMARY:** older single-split leaky evaluation; superseded
  common-model V1 pairwise result; exploratory Wulver `heavy_r1` manuscript
  workflow docs; other internal notes remaining in `docs/`. Do not treat those
  as current comparative evidence.

Common-Model V2 and the tie-aware oracle are **final audited controls** and are
reported in the revised manuscript.

## Primary evidence (compact)

- Matched learned-baseline comparison:
  [reports/kbs_final_evidence_20260813/major1_reviewer_summary.md](reports/kbs_final_evidence_20260813/major1_reviewer_summary.md)
- Common-Model V2 audit:
  [reports/common_model_v2_formal_audit_20260814/AUDIT.md](reports/common_model_v2_formal_audit_20260814/AUDIT.md)
- Tie-aware exact-target oracle audit:
  [reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md)
- Continuation / DAgger:
  [reports/kbs_final_evidence_20260813/c0_continuation_summary.csv](reports/kbs_final_evidence_20260813/c0_continuation_summary.csv),
  [reports/kbs_final_evidence_20260813/distribution_shift_summary.csv](reports/kbs_final_evidence_20260813/distribution_shift_summary.csv)
- Controlled timing:
  [reports/kbs_final_evidence_20260813/controlled_timing_summary.csv](reports/kbs_final_evidence_20260813/controlled_timing_summary.csv)

## Software in this repository

Simulator and policies live under `src/lafc/`. Install with:

```bash
pip install -e ".[dev]"
```

Quick baseline check:

```bash
python -m lafc.runner.run_policy \
  --policy lru \
  --trace data/example_unweighted.json \
  --capacity 3
```

Full scientific campaigns are HPC-scale. Reviewers can verify the published
numerical claims from the committed summaries above without rerunning those
jobs; see [docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md).

## Citation

```bibtex
@unpublished{vahidi2026decisionaligned,
  title  = {Decision-aligned eviction-value prediction for learning-augmented caching},
  author = {Soroush Vahidi},
  note   = {Manuscript submitted to Knowledge-Based Systems},
  year   = {2026}
}
```

## Contact

Soroush Vahidi, Ying Wu College of Computing, New Jersey Institute of
Technology. Contact email: `sv96@njit.edu`.
