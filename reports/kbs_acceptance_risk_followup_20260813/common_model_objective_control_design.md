# Common-model objective control v1

## Scientific question

When candidate representation and scorer capacity are held fixed, does the
relative weakness of finite-horizon eviction-loss supervision persist against
reuse-distance, next-arrival, and pairwise preference supervision?

## Frozen design

- Protocol: the existing leave-one-family-out folds in
  `configs/fair_cross_family_v1/folds/`, with history `[0,10000)` and score
  `[10000,50000)`, capacities 32/64/128, and the seven held-out families.
- Features: exactly `EVICT_VALUE_V1_FEATURE_COLUMNS`, computed by the existing
  `iter_multi_label_candidate_rows` kernel. No feature selection or statistic
  is fit across families.
- Scorer: one shared two-layer ReLU candidate scorer, 8 hidden units, no output
  bias, feature standardization fit on training families only, seed 0,
  full-batch gradient descent, learning rate 0.02, 40 epochs, L2 penalty
  `1e-4`. The same architecture and capacity are used for every objective.
- Scalar objectives: MSE on the existing eviction-loss, censored next-arrival,
  and censored reuse-distance labels, with the existing min/max deployment
  directions. Pairwise objective: the same scorer trained with a RankNet /
  Bradley--Terry loss on the existing next-arrival-ordering pairs.
- Training examples: deterministic first 80 full-cache-miss decisions per
  training family and capacity; validation uses the first 40 decisions of the
  designated validation family. These counts are frozen before results and
  are applied identically to all objectives. Pairwise candidates are sampled
  deterministically to a maximum of 256 pairs per decision.
- One fixed seed (`0`) is used initially; this is a targeted control, not a
  variance campaign.
- Expected evaluation rows: 7 families × 4 objectives × 3 capacities = 84.

## Controls and interpretation

Held fixed: feature schema, scorer architecture/capacity, folds, example
selection, horizon H=4, windows, capacities, request budgets, trace hashes,
optimizer, epochs, seed, and deployment replay. Only target/loss construction
and the corresponding min/max inference direction differ. Held-out families
are never used for fitting, normalization, validation, or model selection.

If eviction-loss remains clearly inferior, the target-level interpretation is
strengthened. If differences shrink, the manuscript must attribute part of the
old contrast to model/representation interaction. If eviction-loss becomes
competitive, the old objective-ablation interpretation must be corrected.
