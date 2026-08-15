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

## Canonical KBS revision materials

Start here: [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md)

| Material | Location |
|---|---|
| Revised manuscript (PDF) | [submission_kbs_revision_final/01_Revised_Manuscript.pdf](submission_kbs_revision_final/01_Revised_Manuscript.pdf) |
| Response to reviewers | [submission_kbs_revision_final/02_Response_to_Reviewers.md](submission_kbs_revision_final/02_Response_to_Reviewers.md) |
| LaTeX source | [submission_kbs_revision_final/07_LaTeX_Source/](submission_kbs_revision_final/07_LaTeX_Source/) |
| Reviewer verification guide | [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md) |
| Reproduction matrix | [docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md) |
| Result verification | [docs/reviewer/RESULT_VERIFICATION.md](docs/reviewer/RESULT_VERIFICATION.md) |

Folder index: [submission_kbs_revision_final/README.md](submission_kbs_revision_final/README.md).

**Do not use** files under [historical/](historical/) (including the old
“robust” zip and earlier manuscript copies) as current submission evidence.

**Primary vs historical evidence**

- **PRIMARY / FINAL:** corrected leave-one-family-out matched comparison
  (LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, FIFO-Reinsertion);
  workload-specific Table 4; full-pipeline objective comparison;
  Common-Model V2; tie-aware exact-target oracle; continuation C0/C1/C2;
  DAgger negative control; controlled timing.
  See [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md).
- **HISTORICAL / NON-PRIMARY:** older single-split leaky evaluation;
  superseded common-model V1; exploratory Wulver `heavy_r1` workflow docs;
  old manuscript/ZIP/DOCX packaging under `historical/`.

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

Published numerical claims can be checked from those committed summaries
without rerunning HPC jobs.

## Related public dataset

Related public dataset: LAFC-Evict provides derived cache-eviction
supervision data associated with this research program. The manuscript's
reported experiments use the exact source traces and audited artifacts
identified in the reviewer verification guide.

- LAFC-Evict (v0.3, wiki2018-only derived candidate rows):
  <https://huggingface.co/datasets/SoroushVahidi/lafc-evict>

This Hugging Face release is **not** claimed to be the exact payload of every
manuscript experiment (seven families, matched replay, Common-Model V2, and
related controls). Exact provenance is in
[docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md) and
[docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md).

## Data provenance

Upstream traces (BrightKite, Citi Bike, Wiki2018 pageview-derived inputs,
CloudPhysics, MetaCDN/MetaKV, Twemcache, SPEC CPU2006, and others) are
**third-party**. This project does not claim ownership of those sources.
Ingestion notes: [docs/datasets.md](docs/datasets.md),
[data/raw/README.md](data/raw/README.md).

Committed in-repo: example traces, processed summaries, and audited result
tables. Full raw traces and HPC campaign trees are generally external.
Project-generated artifacts are candidate features/labels, replay logs, and
the published comparison CSVs.

## Software in this repository

Simulator and policies live under `src/lafc/`. License: [LICENSE](LICENSE)
(MIT). Independent reimplementations and official wrappers are disclosed in
the manuscript (§3.4.4) and reviewer docs; they are not presented as the
original authors’ code.

Install:

```bash
pip install -e ".[dev]"
```

Quick LRU smoke check:

```bash
python -m lafc.runner.run_policy \
  --policy lru \
  --trace data/example_unweighted.json \
  --capacity 3
```

Lightweight tests (no HPC; SIEVE does not need extra extras):

```bash
pytest tests/test_sieve.py -q
```

LRB unit tests need the optional extra: `pip install -e ".[dev,lrb]"`.

Full scientific campaigns are HPC-scale. See
[docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md).

## Citation

```bibtex
@unpublished{vahidi2026decisionaligned,
  title  = {Decision-aligned eviction-value prediction for learning-augmented caching},
  author = {Soroush Vahidi},
  note   = {Manuscript submitted to Knowledge-Based Systems},
  year   = {2026}
}
```

See also [CITATION.cff](CITATION.cff).

## Contact

Soroush Vahidi, Ying Wu College of Computing, New Jersey Institute of
Technology. Contact email: `sv96@njit.edu`.
