# Decision-aligned eviction-value prediction for learning-augmented caching

This repository supports the Knowledge-Based Systems manuscript
**"Decision-aligned eviction-value prediction for learning-augmented caching"**
by Soroush Vahidi.

The paper studies a narrow question: whether a candidate-level eviction target
that directly estimates finite-horizon downstream miss cost is enough to make a
useful online cache replacement policy. It is a controlled study of a
decision-aligned eviction target, not a claim that `evict_value_v1` is a
universally superior cache policy.

Current headline conclusion: under the corrected matched second-revision
evaluation, `evict_value_v1` does **not** outperform LRU, FIFO-Reinsertion,
SIEVE, LRB, 3L-Cache, CACHEUS, or HALP. The repository preserves the negative
result, the generating scripts, provenance records, validation checks, and
known caveats so the evidence can be inspected independently.

## Start Here

- Reviewer guide: [docs/reviewer/START_HERE.md](docs/reviewer/START_HERE.md)
- Curated evidence package: [reports/kbs_final_evidence_20260813/](reports/kbs_final_evidence_20260813/)
- Manuscript source: [submission_kbs_revision_final/07_LaTeX_Source/main.tex](submission_kbs_revision_final/07_LaTeX_Source/main.tex)
- Experiment registry: [docs/reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md](docs/reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md)
- Reproduction matrix: [docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md)
- Result verification: [docs/reviewer/RESULT_VERIFICATION.md](docs/reviewer/RESULT_VERIFICATION.md)

## Primary Results

The reviewer-facing evidence package is
[reports/kbs_final_evidence_20260813/](reports/kbs_final_evidence_20260813/).
It summarizes validated results while leaving the raw campaign directories as
the canonical scientific record.

Headline findings, matching the manuscript:

- The corrected matched comparison uses the same seven traces, capacities
  32/64/128, history prefix `[0,10000)`, scored suffix `[10000,50000)`, and
  hit/miss accounting for all compared policies.
- `evict_value_v1` loses on a clear majority of matched cells against every
  baseline in the primary comparison.
- A direct objective ablation finds the finite-horizon eviction-loss target is
  the worst or tied-worst objective among eviction loss, next arrival, reuse
  distance, and pairwise preference under the tested protocol.
- Mechanistic diagnostics point primarily to target degeneracy and horizon
  truncation; continuation mismatch is partial and regime-dependent, while the
  tested generic DAgger-style state-shift reduction does not improve misses.
- Controlled timing shows HALP-causal is much slower than lightweight
  baselines; `evict_value_v1` has a separate single-run timing measurement and
  is not part of that controlled four-policy timing table.

## Reproducibility And Verification

The repository includes machine-readable manifests, integrity checks,
deterministic regression checks, baseline provenance records, curated result
summaries, and tests so the reported results can be independently inspected.

Key audit locations:

- Experiment manifests and fold configs:
  [configs/fair_cross_family_v1/folds/](configs/fair_cross_family_v1/folds/),
  [configs/reviewer_fairness_protocol.json](configs/reviewer_fairness_protocol.json),
  [configs/supervision_objective_ablation_v1.json](configs/supervision_objective_ablation_v1.json),
  [configs/distribution_shift_ablation_v1.json](configs/distribution_shift_ablation_v1.json),
  [configs/continuation_policy_causal_ablation_production_v1.json](configs/continuation_policy_causal_ablation_production_v1.json).
- Row-count, duplicate-key, schema, hash, and leakage checks:
  [reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md](reports/kbs_final_evidence_20260813/heldout_treatment_integrity.md),
  [reports/kbs_final_evidence_20260813/c0_integrity_summary.md](reports/kbs_final_evidence_20260813/c0_integrity_summary.md),
  [reports/kbs_final_evidence_20260813/distribution_integrity_summary.md](reports/kbs_final_evidence_20260813/distribution_integrity_summary.md),
  [reports/kbs_final_evidence_20260813/controlled_timing_integrity.md](reports/kbs_final_evidence_20260813/controlled_timing_integrity.md).
- Held-out-family and model-selection separation:
  [docs/reviewer_fairness_cross_family_v1.md](docs/reviewer_fairness_cross_family_v1.md),
  [reports/kbs_final_evidence_20260813/heldout_treatment_provenance.md](reports/kbs_final_evidence_20260813/heldout_treatment_provenance.md).
- Deterministic regression and baseline tests:
  [tests/test_lrb.py](tests/test_lrb.py),
  [tests/test_three_l_cache.py](tests/test_three_l_cache.py),
  [tests/test_cacheus.py](tests/test_cacheus.py),
  [tests/test_halp.py](tests/test_halp.py),
  [tests/test_sieve.py](tests/test_sieve.py),
  [tests/test_exact_target_oracle_replication.py](tests/test_exact_target_oracle_replication.py),
  [tests/test_continuation_policy_ablation.py](tests/test_continuation_policy_ablation.py).
- Detailed verification guide:
  [docs/reviewer/RESULT_VERIFICATION.md](docs/reviewer/RESULT_VERIFICATION.md).

## Baselines

The primary matched comparison names seven baselines:

- **LRU**, **FIFO-Reinsertion**, and **SIEVE**: native classical/lightweight
  implementations.
- **LRB**: documented independent reimplementation under matched evaluation
  inputs, with unit-size and simulator-contract adaptations disclosed.
- **3L-Cache**: documented independent reimplementation/adaptation, with
  disclosed batch-size and unit-size differences.
- **CACHEUS**: official upstream source executed through a wrapper/adaptor as
  documented.
- **HALP**: supporting reconstruction/adaptation from the paper and official
  blog; no public official HALP implementation is available.

See [docs/baselines.md](docs/baselines.md),
[docs/lrb_method_spec.md](docs/lrb_method_spec.md),
[docs/three_l_cache_method_spec.md](docs/three_l_cache_method_spec.md),
[docs/cacheus_method_spec.md](docs/cacheus_method_spec.md),
[docs/cacheus_provenance.md](docs/cacheus_provenance.md),
[docs/halp_method_spec.md](docs/halp_method_spec.md), and
[docs/halp_provenance.md](docs/halp_provenance.md).

## Datasets And Workloads

The seven evaluated workload families are heterogeneous and should not be
described as a single class of production cache traces.

| Family | Repository interpretation |
|---|---|
| BrightKite | Location check-in event stream transformed into a request sequence. |
| Citi Bike | Bike-share trip event stream transformed into a request sequence. |
| CloudPhysics | Storage/block-I/O-derived cache workload. |
| MetaCDN | Cache-derived Meta CDN workload. |
| MetaKV | Cache-derived Meta key-value workload. |
| Twemcache | Twitter production cache trace. |
| Wikimedia | Pageview-derived request proxy, not a native CDN trace. |

Raw data are not necessarily redistributed in this repository. Some datasets
must be obtained from upstream sources by the user, and licensing or access
constraints remain those of the original providers. See
[docs/datasets.md](docs/datasets.md),
[docs/datasets_wulver_trace_acquisition.md](docs/datasets_wulver_trace_acquisition.md),
and [reports/kbs_verified_literature_20260813.md](reports/kbs_verified_literature_20260813.md).

## Primary, Supporting, Historical, Audited

- **PRIMARY**: corrected matched evaluation and curated second-revision
  evidence in [reports/kbs_final_evidence_20260813/](reports/kbs_final_evidence_20260813/).
- **SUPPORTING**: mechanistic diagnostics and timing/contextual analyses used
  to explain, qualify, or bound the primary result.
- **HISTORICAL**: older single-split or contaminated exploratory evaluation
  retained for provenance. In particular,
  `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` is
  contaminated/historical and must not be used as primary evidence.
- **AUDITED_NOT_YET_MANUSCRIPT**: two acceptance-risk controls are
  scientifically complete and audited, but are not yet integrated into the
  manuscript: common-model objective control V2
  ([reports/common_model_v2_formal_audit_20260814/AUDIT.md](reports/common_model_v2_formal_audit_20260814/AUDIT.md))
  and the tie-aware exact-target oracle
  ([reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md)).

## Setup And Checks

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
git diff --check
```

The full scientific campaigns are more expensive than unit tests. Many main
results can be inspected from committed summaries without rerunning HPC-scale
jobs; see [docs/reviewer/REPRODUCTION_MATRIX.md](docs/reviewer/REPRODUCTION_MATRIX.md)
for approximate cost and availability.

## Repository Layout

- `src/lafc/`: simulator, policies, feature builders, and experiment support.
- `scripts/`: experiment, analysis, dataset, and validation entry points.
- `tests/`: deterministic unit and regression tests.
- `configs/`: frozen experiment configs and fold manifests.
- `docs/`: protocols, provenance, reviewer maps, and internal notes.
- `reports/`: curated evidence packages and submission-support audits.
- `analysis/`: generated outputs and small tracked audits.

More detail: [docs/repo_map.md](docs/repo_map.md).

## Citation

Citation metadata will be finalized with the revised manuscript. Current
manuscript metadata:

```bibtex
@unpublished{vahidi2026decisionaligned,
  title  = {Decision-aligned eviction-value prediction for learning-augmented caching},
  author = {Soroush Vahidi},
  note   = {Manuscript submitted to Knowledge-Based Systems},
  year   = {2026}
}
```

No DOI is assigned in this repository.

## Contact

Soroush Vahidi, Ying Wu College of Computing, New Jersey Institute of
Technology. Contact email: `sv96@njit.edu`.
