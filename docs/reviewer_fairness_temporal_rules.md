# Temporal Eligibility Rules for Training-Data Selection

Frozen **before** any corrected training data was accepted as final. See
`configs/reviewer_fairness_temporal_rules.json` for the machine-readable
version.

## Frozen rule

> For a deployment-style primary comparison, model training and
> hyperparameter selection must not use observations chronologically later
> than the final test period, unless the study explicitly adopts a
> non-temporal transfer-learning protocol and discloses it.

## Classifications

| Code | Description | Eligible for deployment-style primary? |
|---|---|---|
| A | Earlier same-family data | Yes |
| B | Independent same-time shard, no object/event overlap | Yes |
| C | Different-family training (predeclared, not post-hoc) | Yes (with caveat) |
| D | Later same-family data (chronologically after test) | **No** — transfer-learning framing only, disclosed |
| E | Same-trace prefix, labels censored before boundary | Yes |
| F | Unverifiable or interleaved order (raw index ≠ time) | **No** — requires re-sorting or a different source first |
| G | No temporal axis in the source at all | **No** as a temporal claim — content-disjoint partition only, disclosed |

## Result of applying these rules to `evict_value_v1_fair_v1`

| Family | Classification | Eligible? |
|---|---|---|
| twemcache | D (later same-family) | No |
| metacdn | D (later same-family) | No |
| cloudphysics | D (later same-family) | No |
| brightkite | F (unverifiable/interleaved) | No |
| citibike | F (unverifiable/interleaved) | No |
| wiki2018 | G (no temporal axis) | No (not a temporal claim either way) |
| metakv | F (unverifiable) | No |

**0 of 7 families are eligible under the fair_v1 extraction strategy as
constructed.** This is a stronger and more precise conclusion than "fair_v1
uses future data" — for 4 of 7 families the problem isn't direction, it's
that no direction can currently be established at all.

## What would make each family eligible

- **twemcache, metacdn, cloudphysics** (classification D): the fix is
  mechanical — these families' `[0,50000)`/`[50000,100000)` split is
  *verified* chronological and contiguous, so **swapping** which half is
  training and which is test converts them directly from D to A
  (`[0,50000)` becomes training, genuinely earlier; `[50000,100000)`
  becomes the new canonical test). No re-extraction needed — both trace
  files already exist and are already verified disjoint.
  the same test-window change to `[50000,100000)` for a valid comparison
  (see final report for why this was not executed in this session).
- **brightkite, citibike** (classification F): requires re-parsing the raw
  source, sorting the resulting records by `timestamp` (already present
  in the canonical schema for both families), and *then* defining an
  earlier-training / later-test split on the sorted sequence — not a
  simple raw-row-position slice. Not yet implemented.
- **wiki2018, metakv** (classification G): no timestamp exists in the
  source for either. The only honest options are (a) a disclosed
  non-temporal partition (classification G, with explicit caveat language
  in any reviewer-facing table) or (b) sourcing an alternative,
  genuinely-timestamped dataset for these two families specifically. No
  alternative local source was found (checked `data/raw/{family}/
  manifest.json` for both — each lists exactly one file, no independent
  shards/clusters available).
