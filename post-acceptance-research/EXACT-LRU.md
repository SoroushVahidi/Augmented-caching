# Exact forced-eviction labels under LRU continuation

The [implementation](../experiments/cache_supervision_phase2b/exact_lru.py) accepts a full, ordered, unit-size, fully associative cache, an absent incoming object, and a future request sequence. Residents are unique and ordered oldest first. Admission is mandatory. The reference evicts the oldest and each alternative evicts one other initial resident. All branches then use deterministic LRU. Counts start after admission.

For candidate x, let Mₓ(h) be its continuation misses through h requests. Define Aₓ(h) = Mₗᵣᵤ(h) − Mₓ(h). Append the incoming object to the pre-admission resident order and treat the resulting virtual stack as an LRU cache with C+1 slots. Its C most recent entries are the reference cache and its oldest entry is the boundary. Store the reference order in a hash-indexed ordered structure, the boundary in a scalar, and a map from active victims to their output positions.

Maintain a shared count B of reference misses that hit the current boundary, starting at zero. An active alternative x equals the virtual stack with x deleted, in order, and has advantage B. The invariant holds at admission because x is absent from the alternative and the boundary remains present.

| Future request | Shared transition | Alternative resolution |
|---|---|---|
| Hit reference on y | Move y to most recent | If y is active, it misses, evicts the boundary, and coalesces at B−1 |
| Hit boundary | Increment B and reference misses, move old reference LRU to boundary, append request | Resolve the new boundary if active, at updated B |
| Miss both | Increment reference misses, move old reference LRU to boundary, append request | Resolve the new boundary if active, at unchanged B |

For a shared hit on an object other than x, moving the object commutes with deletion. On a boundary hit all active branches hit while the reference misses. On a miss outside the virtual stack, an active branch drops its old boundary. In either miss case the new deletion form persists, or resolves to the reference if x is the new boundary. Once ordered caches match at the same cursor, every future miss increment matches. Store resolved advantages individually and emit B for active branches. The reference advantage is zero. Recover absolute labels by subtracting each advantage from the shared reference miss count.

## Bounds

- **Additional misses.** Active advantages equal B, and resolved values are B or B−1 at resolution. B never decreases and is nonnegative. Hence every alternative adds at most one miss. The bound is attained with residents (a,b), incoming z and suffix (b), by forcing eviction of b.
- **Optimal-set size.** Each request resolves at most one alternative. For H<C−1 an active branch remains and attains the maximal final B. If B=0 the reference also attains B, so at most H alternatives are suboptimal. If B>0, the last boundary hit resolves either no alternative or one at final B. At most H−1 alternatives are suboptimal, and the reference adds one. For H≥C−1, existence of an optimum gives the bound. It is tight: request min(H,C−1) distinct nonbaseline initial residents, then repeat the final request if needed. Exactly the requested residents incur one extra miss when evicted. Capacity one has a single action.
- **Coalescence.** After C distinct future objects have appeared, all orders are the last C distinct future objects in recency order. Advantage is constant thereafter. Repeating the incoming object shows that no request-count bound alone forces coalescence for C≥2.
- **Complexity.** One scan for H requests, initialization and one output vector take expected O(C+H) time under standard hash assumptions, with O(C) core auxiliary state. Each explicit snapshot adds O(C) output time and retained output space, plus horizon bookkeeping. The optional distinct-object diagnostics retain a set of seen objects for the requested observation range and incur additional memory in that mode.
- **Finite traces.** Truncated snapshots are exact on the observed prefix. Absolute misses beyond joint resolution still require scanning the reference. Relative labels can stop at resolution. An unresolved branch at trace end has no established infinite-suffix label.

## Benefit and risk

The lower bound and integrality give A = max(A, 0) − 1{A = −1}. The conditional expected advantage is expected nonnegative benefit minus the probability of a net harmful action. Under the event accounting, gross prevented misses and the at-most-one branch-only miss are also recorded separately. They can both be positive on the same trajectory and their difference recovers A. The Phase 2C-B learner estimates the gross components: the shared boundary hits up to each branch endpoint and the indicator of a branch-only miss. It subtracts the estimated probability of a branch-only miss from the predicted gross benefit. These results concern one forced action with an LRU continuation. Learned continuations and repeated interventions require separate analysis.

## Recorded validation

The Phase 2B verification recorded 26,530 exhaustive, 10,000 randomized and 44 structured cases, with zero unexplained mismatches. All 1,176 primary and 504 enriched archived groups aligned. The earlier paired comparator has a horizon-zero edge case, and the compiled exhaustive and authoritative repository simulators agreed there. Recorded median speedups were 19.7× over compiled exhaustive replay, 24.4× over compiled Phase 2A paired replay and 205.7× over the repository Python simulator. The compiled exhaustive replay won in some small configurations, including C=32 and H=4. See the [benchmark CSV](aggregate-results/phase2b-labeler-benchmark.csv) and the [report](REPORT.pdf) for timing scope, memory and the derivation. Classical LRU stack theory provides background. Publication novelty has not been established.
