# Publication-facing fairness statement

The modern learned/adaptive baselines are evaluated under the same
controlled-window replay protocol as the corrected treatment evaluation:
seven trace families, capacities 32/64/128, object-slot capacity, unit
objects, and object misses over the primary scored suffix [10000,50000)
after each policy processes the common history prefix [0,10000). LRB and
3L-Cache are online repository reimplementations that adapt only from their
own in-trace stream; CACHEUS is represented as an official-source wrapper
with a separate provenance caveat because the external clone is not
vendored and is not currently live-verifiable in this worktree. The
offline evict_value_v1 treatment is conceptually different: its corrected
version uses leave-one-family-out training and a frozen model before the
held-out-family replay.
