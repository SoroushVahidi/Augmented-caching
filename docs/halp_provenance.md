# HALP Provenance and Licensing Document

## 1. Implementation origin

The HALP implementation in this repository (`src/lafc/policies/halp.py`,
`src/lafc/halp_model.py`, and `src/lafc/halp_features.py`) is an
**independent reimplementation of the published algorithmic description**,
adapted to this repository's unit-size, offline-replay paging setting. No
Google or YouTube source code was consulted, copied, translated, or
structurally mirrored, because **no such code is public** (see §2).

## 2. Authoritative sources consulted

- **Paper** (official paper only, primary source): Zhenyu Song, Kevin Chen,
  Nikhil Sarda, Deniz Altınbüken, Eugene Brevdo, Jimmy Coleman, Xiao Ju,
  Pawel Jurczyk, Richard Schooler, Ramki Gummadi. "HALP: Heuristic Aided
  Learned Preference Eviction Policy for YouTube Content Delivery Network."
  20th USENIX Symposium on Networked Systems Design and Implementation
  (NSDI '23), 2023. https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu
- **Google Research blog post** (author-affiliated official artifact,
  corroborating primary source): "Preference learning with automated
  feedback for cache eviction,"
  https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/
  — this was the source that disambiguated the reward-model architecture
  (two-layer MLP, not a linear scorer) and the continuous-online training
  schedule.
- **Official code / artifact**: searched and **not found**. HALP is
  reported as running in YouTube CDN production since early 2022; the
  paper is a deployment report, and no simulator, reference
  implementation, pretrained model, or artifact-evaluation package was
  located on the USENIX conference page, the Google Research publication
  page, author pages, or general code-hosting search, as of 2026-08-06.
- **Third-party implementations**: none identified in libCacheSim or any
  other credible public cache simulator as of this check.

## 3. Corrected claims (self-audit)

Two real discrepancies were found and corrected while preparing this
baseline, both recorded here so they remain auditable, per this task's
requirement not to approximate HALP and label it faithful without evidence.

### 3a. Model family: linear scorer → two-layer MLP

An earlier draft of this repository's HALP documentation and
`tests/fixtures/halp_reference/parity_evidence.md` described the reward
model as a linear scorer (`R(x) = wᵀx`, implemented via
`sklearn.linear_model.LogisticRegression`) and characterized this as a
faithful reproduction of "the paper's decision flow." Cross-checking the
official Google Research blog post (not consulted when that draft was
written) shows the production reward model is a **"light-weight two-layer
multilayer perceptron (MLP)."** A linear scorer and a two-layer MLP are
different model families with different expressive power — this was a real
discrepancy, not a stylistic one.

### 3b. A first MLP fix was itself mathematically invalid

The first attempted fix swapped in `sklearn.neural_network.MLPClassifier`,
trained the same way the linear model had been: on symmetrized
feature-*difference* vectors (`x_A - x_B`), with the resulting classifier
then queried on raw per-candidate features at inference time. This reuses a
trick that is exact for a *linear* reward function
(`R(x_A) - R(x_B) = R(x_A - x_B)` when `R` is linear) but is **not valid
for a nonlinear MLP** — `R(x_A) - R(x_B) != R(x_A - x_B)` in general. Using
it anyway silently produced an internally inconsistent scoring function:
confirmed empirically by `lbfgs failed to converge` warnings and
non-ranking-consistent behavior on a real smoke-test trace. This was caught
before the baseline was evaluated on any trace used for reviewer-facing
results.

The fix: `src/lafc/halp_model.py` now hand-rolls the shared-weight forward
pass `R(x) = W2 . relu(W1 x + b1)` and backpropagates the correct
Bradley-Terry / RankNet pairwise loss `-log(sigmoid(R(x_A) - R(x_B)))`
through both branches jointly, via deterministic full-batch gradient
descent. This is the standard, correct way to train a shared-weight
pairwise preference network and is consistent with the official blog's
description of a preference-comparison-trained MLP.

### 3c. Documentation and evidence updated accordingly

- `docs/halp_method_spec.md` documents both corrections above and every
  other deviation from the primary sources with an explicit fidelity
  classification, rather than asserting unqualified faithfulness.
- `tests/fixtures/halp_reference/parity_evidence.md` claims only
  **decision-semantic parity** (candidate generation direction,
  preference-label direction, victim-selection direction, and leakage
  isolation), not model-architecture or production-score parity, which
  cannot be verified without access to the closed production system.

## 4. License compatibility

The original HALP implementation runs inside Google's proprietary,
closed-source YouTube CDN production stack; no source code has been
released under any license, so there is no third-party license to be
compatible or incompatible with. This repository's `halp.py`,
`halp_model.py`, and `halp_features.py` are original code written from the
published paper and blog description, using this repository's own coding
conventions and this repository's already-present dependency
(`scikit-learn`, a core dependency of this repository — no new dependency
was added). They are covered by this repository's own license terms in the
same way as any other original module in `src/lafc/`.

No definitive legal conclusion is asserted beyond the above; this is not
legal advice.

## 5. Files consulted vs. files copied

| Source | Consulted for | Copied? | Translated? |
|---|---|---|---|
| USENIX NSDI '23 paper (song-zhenyu) | algorithmic description, evaluation metrics, deployment results | No | No |
| Google Research blog post | reward-model architecture, training schedule, candidate-selection mechanism | No | No |
| Official code | — | N/A (does not exist) | N/A |

All code in `src/lafc/policies/halp.py`, `src/lafc/halp_model.py`, and
`src/lafc/halp_features.py` is original, written to match the *described
behavior* in the two sources above, not derived from or structurally
mirroring any released source file.
