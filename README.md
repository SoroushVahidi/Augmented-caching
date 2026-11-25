# Decision-Aligned Eviction-Value Prediction for Learning-Augmented Caching

**Published in *Knowledge-Based Systems* (2026).** Sole author: Soroush Vahidi (NJIT).

**In plain words.** When a cache is full and a new item arrives, the cache has to evict something. Classic rules like LRU ignore *what each eviction will cost later*. This project asks a narrow, testable question: if we train a model to predict, for each candidate, the extra misses its eviction would cause over a finite future horizon (a "decision-aligned" target), does that make a good online eviction policy? The repository contains the cache simulator, the learned policy (`evict_value_v1`), reimplementations and wrappers of classical and learned baselines (LRU, FIFO-Reinsertion, SIEVE, LRB, 3L-Cache, CACHEUS, HALP), and the audited result tables behind the paper. The honest headline is reported up front: under the corrected held-out protocol, the learned policy does **not** beat these baselines. The paper is a controlled study of the target, not a claim of a new best cache policy.

## Paper

- **Published version:** S. Vahidi. *Decision-aligned eviction-value prediction for learning-augmented caching.* Knowledge-Based Systems, article 117353, 2026. [doi:10.1016/j.knosys.2026.117353](https://doi.org/10.1016/j.knosys.2026.117353)
- **Preprint** (earlier title: "...for robust learning-augmented caching"): SSRN, 2026. [ssrn.com/abstract=6636732](https://ssrn.com/abstract=6636732) · [doi:10.2139/ssrn.6636732](https://doi.org/10.2139/ssrn.6636732)

```bibtex
@article{vahidi2026decisionaligned,
  author  = {Vahidi, Soroush},
  title   = {Decision-aligned eviction-value prediction for learning-augmented caching},
  journal = {Knowledge-Based Systems},
  pages   = {117353},
  year    = {2026},
  doi     = {10.1016/j.knosys.2026.117353}
}
```

## At a glance

| | |
|---|---|
| Simulator and policies | `src/lafc/` (installable package `lafc`) |
| Quick smoke test | `pip install -e ".[dev]"`, then `python -m lafc.runner.run_policy --policy lru --trace data/example_unweighted.json --capacity 3` |
| Reviewer and verification guide | [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md) |
| Companion dataset | [LAFC-Evict](https://github.com/SoroushVahidi/lafc-evict-dataset) on [Hugging Face](https://huggingface.co/datasets/SoroushVahidi/lafc-evict) and the [AWS Registry of Open Data](https://registry.opendata.aws/lafc-evict/) |
| Research and evidence index | [Experiment inventory](docs/research/README.md) · [Methodology](docs/research/METHODOLOGY.md) |
| Tutorial | [lafc-evict-tutorial](https://github.com/SoroushVahidi/lafc-evict-tutorial) |

---

The [contribution guide](CONTRIBUTING.md) documents the base install, test and review workflow.

Public repository: <https://github.com/SoroushVahidi/Augmented-caching>

This repository contains code and verified experimental artifacts for the
Knowledge-Based Systems article

**"Decision-aligned eviction-value prediction for learning-augmented caching"**

The paper studies a narrow question: whether a candidate-level eviction target
that estimates finite-horizon downstream miss cost is enough to make a useful
online cache-replacement policy. It is a controlled study of that target, not a
claim that `evict_value_v1` is a universally superior cache policy.

**Headline result.** Under the corrected held-out / matched second-revision
protocol, `evict_value_v1` does **not** outperform LRU, FIFO-Reinsertion,
SIEVE, LRB, 3L-Cache, CACHEUS, or HALP.

## Canonical KBS revision materials

> The final (third-round, accepted) upload package is in
> [submission_kbs_r3_20261013/](submission_kbs_r3_20261013/) (manifest:
> `KBS_R3_SUBMISSION_MANIFEST_20261013.md`). The table below indexes the
> second-revision materials, which hold the reviewer-facing evidence.

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

Final convenience submission archive:
[submission_kbs_revision_final/package/KBS_second_revision_UPLOAD_CORRECTED.zip](submission_kbs_revision_final/package/KBS_second_revision_UPLOAD_CORRECTED.zip)

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

- LAFC-Evict v1.0 (five trace families): Hugging Face branch
  [`v1.0`](https://huggingface.co/datasets/SoroushVahidi/lafc-evict/tree/v1.0)
  and AWS Open Data `s3://lafc-evict-open-data/v1.0/`. The Hugging Face `main`
  branch still holds the older v0.3 (wiki2018-only) release.
- Dataset repository and paper: [lafc-evict-dataset](https://github.com/SoroushVahidi/lafc-evict-dataset)

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
@article{vahidi2026decisionaligned,
  author  = {Vahidi, Soroush},
  title   = {Decision-aligned eviction-value prediction for learning-augmented caching},
  journal = {Knowledge-Based Systems},
  pages   = {117353},
  year    = {2026},
  doi     = {10.1016/j.knosys.2026.117353}
}
```

(`CITATION.cff` should also be updated from "submitted" to the published article.)

See also [CITATION.cff](CITATION.cff).

## Contact

Soroush Vahidi, Ying Wu College of Computing, New Jersey Institute of
Technology. Contact email: `sv96@njit.edu`.
