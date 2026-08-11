# Reviewer #2 Major Comment 1 response draft - internal pending version

We have completed the baseline-side audit for the modern learned/adaptive
comparisons. LRB, 3L-Cache, and CACHEUS each have 42-row local controlled
window result files with 21 primary rows over the seven families and
capacities 32/64/128. The baseline rows use the common primary window
[10000,50000) after an identical history prefix [0,10000), object-slot
capacity, unit objects, and object-miss metrics. LRB and 3L-Cache are
online/adaptive reimplementations that learn only from their own in-trace
stream; CACHEUS uses the official-source wrapper, with provenance caveats
recorded separately.

The corrected evict_value_v1 treatment artifact remains
WULVER_ONLY_VALIDATED locally, so the final matched numerical synthesis is
pending synchronization of the verified CSV. No additional baseline
experiment is scientifically required for this comment; the pending Wulver
baseline jobs would serve as independent replication/provenance
strengthening.
