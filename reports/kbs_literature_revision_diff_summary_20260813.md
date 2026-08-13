# Verified literature and provenance revision summary

Date: 2026-08-13  
Branch: `kbs/second-revision-science`

## References added

- S3-FIFO, GL-Cache, LAH/S4-FIFO, Merlin, MAT, LODL, and El Balghiti et al.'s predict-then-optimize generalization theory were added only after checking an official proceedings, publisher, or author-hosted primary source.
- The official Twitter cache-trace repository was added as the provenance source for the Twemcache-family trace.
- MAT is explicitly classified as an arXiv preprint; no later peer-reviewed version was used.

## References rejected or not used

- SCION was not added because a primary source and publication status could not be verified sufficiently for this revision.
- No unverified literature performance numbers were added.
- No first-of-kind claim was added for candidate-level learning or decision-focused caching.
- No broad claim that modern systems universally move machine learning off the hot path was added.

## Manuscript changes

- Related Work now organizes prior work by prediction target, preference/action supervision, learning granularity, execution cost, heuristic/control-plane design, and decision-focused learning.
- Novelty wording distinguishes structural decision alignment from decision informativeness and deployment efficiency.
- Practical significance now connects candidate filtering, group-level learning, sparse/control-plane learning, and simple heuristic baselines to the measured cost of the present method without claiming deployment readiness.
- The contribution and conclusion wording remains conservative pending the common-model objective control and tie-aware exact-oracle experiments.
- A conceptual, explicitly non-theorem connection to non-unique optima/degeneracy terminology is included through El Balghiti et al.

## Workload provenance changes

- The seven evaluated families are named explicitly.
- BrightKite and Citi Bike are identified as non-cache event streams transformed into request sequences.
- CloudPhysics, MetaCDN, MetaKV, and Twemcache are identified as cache/storage-derived sources under the repository's documented preprocessing.
- Wikimedia is identified as a pageview-derived proxy rather than a native Wikimedia CDN trace.

## Threats-to-validity changes

The manuscript now states the domain mismatch, 50,000-request windows, unit-size paging abstraction, nonuniform baseline fidelity, descriptive 21-cell comparison scope, and fixed learned-training seeds.

## Experiment-dependent passages intentionally left conservative

- The common-model control is not used to finalize the objective-only interpretation; the current text describes the original ablation accurately and leaves the stronger causal interpretation pending.
- The tie-aware oracle is not used to generalize the deterministic exact-oracle result to all tie policies; the current text explicitly identifies the sensitivity analysis as pending.
