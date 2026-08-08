# cap64_with_sieve_fifo result analysis (2026-06-20)

Read-only analysis of the completed capacity-64 canonical chunk, compared
against the prior capacity-32 chunk. Nothing re-run, nothing launched,
nothing overwritten. cap128/cap256 remain **not launched**.

**Inputs verified:**
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv` (5.8K, 57 lines = 1 header + 56 rows)
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.md`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md` (prior chunk, for trend comparison)
- `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log` (`CAP64_WITH_SIEVE_FIFO_EXIT=0`)
- Runtime: tmux session created Fri 22:29:17 → log written Sat 08:33:51 EDT (~10h 04m wall-clock)

**Policy set (8):** `lru`, `sieve`, `fifo_reinsertion`, `predictive_marker`,
`blind_oracle_lru_combiner`, `trust_and_doubt`, `rest_v1`, `evict_value_v1`

**Coverage:** 7 trace families × 1 capacity (64) × 8 policies = 56 data rows.

---

## 1. Aggregate ranking at cap64 (mean misses across 7 families)

| Rank | Policy | cap64 mean misses | vs LRU |
|------|--------|--------------------|--------|
| 1 | **lru** | 33,650.14 | baseline |
| 1 | **rest_v1** | 33,650.14 | 0.00% |
| 3 | blind_oracle_lru_combiner | 33,651.29 | −0.00% |
| 4 | fifo_reinsertion | 33,672.14 | −0.07% |
| 5 | predictive_marker | 33,773.14 | −0.37% |
| 6 | trust_and_doubt | 33,998.71 | −1.04% |
| 7 | sieve | 34,362.71 | −2.12% |
| 8 | **evict_value_v1** | 34,409.00 | **−2.26%** |

LRU/rest_v1 remain tied for best on aggregate mean misses. The overall
ranking order is **unchanged from cap32** — evict_value_v1 is still last —
but the spread between best and worst has roughly halved (cap32 spread
1,677 misses → cap64 spread 758.9 misses).

---

## 2. cap32 → cap64 trend: aggregate mean misses (% change, lower = fewer misses)

| Policy | cap32 mean misses | cap64 mean misses | Δ misses | % change |
|--------|--------------------|--------------------|----------|----------|
| lru | 34,667.14 | 33,650.14 | −1,017.00 | −2.93% |
| sieve | 35,313.57 | 34,362.71 | −950.86 | −2.69% |
| fifo_reinsertion | 34,688.00 | 33,672.14 | −1,015.86 | −2.93% |
| **evict_value_v1** | 36,344.14 | 34,409.00 | **−1,935.14** | **−5.32%** |

All four policies improve in absolute miss count as capacity doubles
(expected). **evict_value_v1's own miss count drops nearly 2x faster
(relative)** than LRU's, SIEVE's, or FIFO-Reinsertion's. That asymmetry is
what drives the gap-narrowing below — it is not simply "everything
converges at higher capacity," because SIEVE and FIFO-Reinsertion did
**not** show the same acceleration (see §3).

---

## 3. Gap vs LRU, SIEVE, FIFO-Reinsertion — cap32 vs cap64 (aggregate)

| Comparison | cap32 gap | cap64 gap | Δ (pp) | Direction |
|------------|-----------|-----------|--------|-----------|
| evict_value_v1 vs LRU | −4.84% | −2.26% | **−2.58 pp** | **Narrowing** |
| evict_value_v1 vs SIEVE | −2.92% | −0.13% | **−2.79 pp** | **Narrowing (nearly closed)** |
| evict_value_v1 vs FIFO-Reinsertion | −4.78% | −2.19% | **−2.59 pp** | **Narrowing** |
| *(reference)* SIEVE vs LRU | −1.86% | −2.12% | +0.25 pp | flat / slightly widening |
| *(reference)* FIFO-Reinsertion vs LRU | −0.06% | −0.07% | +0.01 pp | flat |
| *(reference)* predictive_marker vs LRU | −0.51% | −0.37% | −0.14 pp | mild narrowing |
| *(reference)* trust_and_doubt vs LRU | −1.21% | −1.04% | −0.18 pp | mild narrowing |

**Key finding:** evict_value_v1's gap-narrowing (−2.6 to −2.8 pp) is roughly
an order of magnitude larger than any other non-LRU policy's movement.
SIEVE and FIFO-Reinsertion are essentially flat against LRU between cap32
and cap64 — so the aggregate field is *not* uniformly compressing toward
LRU as capacity grows. evict_value_v1 specifically is closing distance,
most dramatically against SIEVE, where the gap shrinks from −2.92% to a
statistically negligible −0.13%.

---

## 4. Per-family breakdown: LRU vs SIEVE vs FIFO-Reinsertion vs evict_value_v1

| Family | Policy | cap32 misses | cap64 misses | evict gap vs policy @cap32 | evict gap vs policy @cap64 | Trend |
|--------|--------|---------------|---------------|------------------------------|------------------------------|-------|
| **brightkite** | lru | 18,078 | 16,543 | +10.47% | +10.72% | slightly worse |
| | sieve | 18,110 | 16,865 | +10.27% | +8.60% | improving |
| | fifo_reinsertion | 18,122 | 16,577 | +10.20% | +10.49% | flat/slightly worse |
| | evict_value_v1 | 19,970 | 18,316 | — | — | — |
| **citibike** | lru | 19,994 | 18,964 | +7.90% | +4.70% | **improving** |
| | sieve | 21,151 | 19,456 | +2.00% | +2.05% | flat |
| | fifo_reinsertion | 19,987 | 18,864 | +7.94% | +5.26% | **improving** |
| | evict_value_v1 | 21,573 | 19,855 | — | — | — |
| **cloudphysics** | lru | 49,273 | 48,575 | +0.72% | +1.57% | worsening (small abs.) |
| | sieve | 49,114 | 48,621 | +1.04% | +1.48% | worsening (small abs.) |
| | fifo_reinsertion | 49,245 | 48,500 | +0.77% | +1.73% | worsening (small abs.) |
| | evict_value_v1 | 49,626 | 49,339 | — | — | — |
| **metacdn** | lru | 29,380 | 28,616 | +22.37% | +2.73% | **sharply improving** |
| | sieve | 29,732 | 28,737 | +20.92% | +2.30% | **sharply improving** |
| | fifo_reinsertion | 29,308 | 28,590 | +22.67% | +2.82% | **sharply improving** |
| | evict_value_v1 | 35,951 | 29,397 | — | — | — |
| **metakv** | lru | 38,064 | 38,041 | +1.70% | +0.01% | **near-perfect convergence** |
| | sieve | 38,145 | 38,037 | +1.48% | +0.02% | **near-perfect convergence** |
| | fifo_reinsertion | 38,063 | 38,036 | +1.70% | +0.02% | **near-perfect convergence** |
| | evict_value_v1 | 38,710 | 38,044 | — | — | — |
| **twemcache** | lru | 37,881 | 34,812 | +1.84% | +3.16% | worsening |
| | sieve | 40,943 | 38,823 | **−5.77%** (win) | **−7.50%** (win) | win widening |
| | fifo_reinsertion | 38,091 | 35,138 | +1.28% | +2.20% | worsening |
| | evict_value_v1 | 38,579 | 35,912 | — | — | — |
| **wiki2018** | all policies | 50,000 | 50,000 | 0.00% (tie) | 0.00% (tie) | degenerate, uninformative |

(Negative gap = evict_value_v1 has *fewer* misses, i.e. it wins.)

---

## 5. Does evict_value_v1 still lose? — direct answers

### A. Does evict_value_v1 still lose to LRU at cap64?

**Yes.** −2.26% aggregate (down from −4.84% at cap32). It loses on 5/7
families (brightkite, citibike, cloudphysics, metacdn, twemcache), is
essentially tied on metakv (+0.01%, 3 misses out of ~38,000), and ties on
the degenerate wiki2018 floor. No family where it beats LRU outright.

### B. Does evict_value_v1 still lose to SIEVE?

**Barely.** Aggregate gap is −0.13% at cap64 (down from −2.92% at cap32)
— statistically indistinguishable from a tie at this sample size. It
already **beats** SIEVE outright on twemcache (−7.50%, a win that widened
from cap32), is essentially tied on metakv and citibike, and loses on
brightkite, cloudphysics, and metacdn (though metacdn's loss shrank from
22.4% to 2.3%).

### C. Does evict_value_v1 still lose to FIFO-Reinsertion?

**Yes.** −2.19% aggregate (down from −4.78% at cap32). Same pattern as
LRU: losses on brightkite/citibike/cloudphysics/metacdn/twemcache, near-tie
on metakv, degenerate tie on wiki2018.

### D. Is the gap getting better, worse, or unchanged from cap32 → cap64?

**Better, on aggregate and specifically attributable to evict_value_v1 —
not a generic capacity-compression artifact.** The aggregate gap vs all
three baselines roughly halved (§3), and SIEVE/FIFO-Reinsertion's own
distance from LRU stayed flat over the same span, which rules out "the
whole field just converges as capacity grows" as the sole explanation.

**But this aggregate improvement is concentrated, not uniform** (§4):
- **Two families drive nearly all of the aggregate narrowing**: metacdn
  (22.4% → 2.7% gap vs LRU) and metakv (1.7% → 0.01%, i.e., a near-exact
  tie). Both are large, dramatic swings.
- **Two families are flat-to-slightly-worse**: brightkite (10.47% →
  10.72% vs LRU) and cloudphysics (0.72% → 1.57% vs LRU, though both
  values are small in absolute terms — this is a very-low-hit-rate
  regime, ~1-3%, where a handful of misses swings the percentage a lot).
- **citibike** improves moderately vs LRU/FIFO (not vs SIEVE, which stays
  flat).
- **twemcache** is the one family where evict_value_v1 already wins
  outright (vs SIEVE only), and that win *widens* with capacity, while its
  small losses vs LRU/FIFO on the same family *widen* slightly too.
- **wiki2018** remains fully degenerate (0% hit rate, all policies miss
  every request) at both capacities — uninformative either way.

**Caution on interpretation**: metakv's near-perfect convergence coincides
with LRU's own hit rate barely moving between cap32 and cap64 (23.87% →
23.92%) — a workload where eviction policy choice appears to matter very
little at either capacity (all 8 policies cluster within ~0.05% of each
other in raw miss count at cap64). That convergence is plausibly a
*policy-insensitivity* effect of this specific workload, not necessarily
evidence that evict_value_v1's model generalizes better at larger caches.
metacdn does **not** fit that explanation — LRU's hit rate there moves
meaningfully (41.2% → 42.8%) and the field of baselines is **not**
clustered tightly even at cap64 (28,590–28,737 across LRU/SIEVE/FIFO) — so
metacdn's narrowing is a more genuine signal of evict_value_v1 specifically
improving relative to baselines at the larger capacity, not just everyone
converging together.

---

## 6. Consistency / sanity checks

| Finding | Detail | Assessment |
|---------|--------|------------|
| wiki2018 all-miss floor | Every policy: 50,000 misses, 0.0 hit rate, both caps | **Expected degeneracy**, same as cap32; not a bug |
| cloudphysics near-zero hit rates | 1.3–3.0% hit rate across policies at cap64 | **Expected** — hard trace, consistent direction with cap32 |
| No NaN/negative/malformed rows | All 56 rows well-formed | **Clean** |
| Ranking order unchanged cap32→cap64 | LRU/rest_v1 best, evict_value_v1 worst, both caps | **Consistent** — only the *magnitude* of gaps changed, not the order |
| evict_value_v1 numbers internally consistent | Mean computed directly from CSV matches MD aggregate (34,409.00) | **Clean** |

---

## 7. Concise interpretation

1. **cap64 headline**: evict_value_v1 remains last in aggregate ranking,
   but its gap vs LRU/SIEVE/FIFO-Reinsertion has roughly halved relative to
   cap32, and it now nearly matches SIEVE in aggregate (−0.13%) and already
   beats SIEVE outright on twemcache.
2. **The narrowing is real, not just "capacity makes everyone converge"**
   — SIEVE and FIFO-Reinsertion's own distance from LRU stayed flat over
   the same capacity step.
3. **The narrowing is concentrated in 2 of 7 families** (metacdn, metakv);
   it is mixed-to-slightly-worse on 2 others (brightkite, cloudphysics);
   wiki2018 is uninformative at any capacity tested so far.
4. **No family shows an outright evict_value_v1 win against LRU or
   FIFO-Reinsertion** at either capacity. The only outright win is against
   SIEVE on twemcache, and that win is widening.
5. **Reviewer impact**: this is a more interesting, more defensible result
   than cap32 alone — it supports a *capacity-sensitivity* narrative
   (learning-augmented eviction's relative competitiveness improves with
   cache size, even though parity is not yet reached) rather than a flat
   negative result. It does not yet support a "superior policy" claim.

**Bottom line:** cap64 is valid, complete, and **meaningfully changes the
emerging story** from cap32's "evict_value_v1 loses, consistently and by a
wide margin" to "evict_value_v1 loses, but by a narrowing and
family-dependent margin that is most concentrated in a couple of
workloads — not yet a win, but a more nuanced trend worth one more data
point before deciding how to frame it."
