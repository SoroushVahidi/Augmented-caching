# Reviewer Fairness — statistical analysis plan

Frozen **before** viewing any `evict_value_v1_fair_v1` result on the
canonical traces. See `configs/reviewer_fairness_statistics.json` for the
machine-readable version.

## Experimental unit

One **(trace family, capacity)** instance — 7 traces × 3 capacities = 21
paired instances per policy. Requests *within* one instance are
sequentially dependent (cache and, for learned policies, model state
carries forward across the replay); they are never treated as independent
replicates for any statistical claim. This matches
`docs/reviewer_fairness_protocol.md`'s existing framing and is restated
here explicitly because it is the single most common way caching papers
overstate statistical power.

## Design

Paired comparison: every policy is scored on the identical 21 instances
under the identical `primary_controlled_window` protocol (same trace
hashes, same history/score boundary, same capacity semantics), so
per-instance differences are meaningful without an independence
assumption across instances that requests-within-instance do not satisfy.

## Reported quantities (minimum)

- Mean and median miss ratio across the 21 instances.
- Per-instance absolute and percentage miss-count difference vs. a
  reference policy (LRU, unless another reference is explicitly named).
- Wins/ties/losses per instance.
- 95% bootstrap confidence interval (10,000 resamples over the 21 paired
  instances) on the mean miss-ratio difference.

## Inferential test (if used)

Paired, two-sided **Wilcoxon signed-rank test** on the 21 per-instance
miss-ratio differences. Chosen over a paired t-test because it does not
assume the differences are normally distributed across heterogeneous
trace families (brightkite/citibike/wiki2018/twemcache/metakv/metacdn/
cloudphysics have very different locality regimes — the existing manuscript
already treats wiki2018 as degenerate). **Holm-Bonferroni** correction
across every pairwise comparison reported together in one table.
Significance threshold 0.05. Effect size: matched-pairs rank-biserial
correlation, always reported alongside the p-value, never the p-value
alone.

## Explicit prohibitions

- Never treat the 40,000 scored requests inside one instance as
  independent samples for a test — the unit is the instance, not the
  request.
- Never select which policies or instances to report after viewing
  results.
- Never revise the test, correction, or threshold after seeing
  `evict_value_v1_fair_v1`'s canonical-trace results — a revision at that
  point requires a new plan version and a documented reason, mirroring
  the amendment policy in `docs/evict_value_v1_fair_training_protocol.md`.
- Never pool `deployment_full_stream` and `primary_controlled_window` rows
  into one statistic — they answer different questions
  (`docs/reviewer_fairness_protocol.md` section 2) and must be reported
  separately.

## Status

**Plan only.** Not yet applied to any result table in this session — the
corrected `evict_value_v1_fair_v1` model was still training/being
evaluated when this plan was frozen (see final report). Applying this
plan is the next task, once every policy's primary rows are complete.
