# cap32_with_sieve_fifo result analysis (2026-06-19)

Read-only analysis of the completed corrected capacity-32 canonical chunk.
Nothing re-run, nothing launched, nothing overwritten.

**Inputs verified:**
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv` (5.8K, 57 lines = 1 header + 56 rows)
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.md`
- `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log` (`CAP32_WITH_SIEVE_FIFO_EXIT=0`)
- Runtime: Fri 14:42:27 → Fri 19:58:48 EDT (~5h 16m)

**Policy set (8):** `lru`, `sieve`, `fifo_reinsertion`, `predictive_marker`,
`blind_oracle_lru_combiner`, `trust_and_doubt`, `rest_v1`, `evict_value_v1`

**Coverage:** 7 trace families × 1 capacity (32) × 8 policies = 56 data rows.

---

## 1. Per-trace ranking (misses; lower is better)

| Rank | brightkite | citibike | cloudphysics | metacdn | metakv | twemcache | wiki2018 |
|------|------------|----------|--------------|---------|--------|-----------|----------|
| 1 | lru/rest_v1 (18,078) | fifo_reinsertion (19,987) | **sieve (49,114)** | fifo_reinsertion (29,308) | fifo_reinsertion (38,063) | lru/rest_v1 (37,881) | all tie (50,000) |
| 2 | blind_oracle_lru_combiner (18,081) | lru/rest_v1 (19,994) | fifo_reinsertion (49,245) | lru/rest_v1 (29,380) | lru/rest_v1 (38,064) | blind_oracle_lru_combiner (37,884) | — |
| 3 | sieve (18,110) | blind_oracle_lru_combiner (19,995) | lru/rest_v1/trust_and_doubt (49,273) | blind_oracle_lru_combiner (29,381) | blind_oracle_lru_combiner (38,065) | fifo_reinsertion (38,091) | — |
| 4 | fifo_reinsertion (18,122) | predictive_marker (20,169) | blind_oracle_lru_combiner (49,274) | predictive_marker (29,506) | predictive_marker (38,105) | predictive_marker (38,260) | — |
| 5 | predictive_marker (18,483) | trust_and_doubt (20,813) | predictive_marker (49,384) | sieve (29,732) | sieve (38,145) | evict_value_v1 (38,579) | — |
| 6 | trust_and_doubt (18,772) | sieve (21,151) | evict_value_v1 (49,626) | trust_and_doubt (29,810) | trust_and_doubt (38,167) | trust_and_doubt (38,774) | — |
| 7 | evict_value_v1 (19,970) | evict_value_v1 (21,573) | — | evict_value_v1 (35,951) | evict_value_v1 (38,710) | sieve (40,943) | — |
| 8 | — | — | — | — | — | — | — |

**Per-trace winners:** LRU or rest_v1 on 3 families; fifo_reinsertion on 3;
sieve on 1 (cloudphysics); wiki2018 is a full tie (degenerate floor).

---

## 2. Aggregate ranking (mean misses across 7 families)

| Rank | Policy | Mean misses | vs LRU |
|------|--------|-------------|--------|
| 1 | **lru** | 34,667.14 | baseline |
| 1 | **rest_v1** | 34,667.14 | 0.00% |
| 3 | blind_oracle_lru_combiner | 34,668.57 | −0.00% |
| 4 | fifo_reinsertion | 34,688.00 | −0.06% |
| 5 | predictive_marker | 34,843.86 | −0.51% |
| 6 | trust_and_doubt | 35,087.00 | −1.21% |
| 7 | sieve | 35,313.57 | −1.86% |
| 8 | **evict_value_v1** | 36,344.14 | **−4.84%** |

LRU and rest_v1 are tied for best on aggregate mean misses.

---

## 3. Head-to-head comparisons (aggregate)

| Comparison | Base | Challenger | Δ misses | % (positive = challenger better) | Winner |
|------------|------|------------|----------|----------------------------------|--------|
| SIEVE vs LRU | 34,667 | 35,314 | +647 | −1.86% | **LRU** |
| FIFO-Reinsertion vs LRU | 34,667 | 34,688 | +21 | −0.06% | **LRU** (marginally) |
| SIEVE vs FIFO-Reinsertion | 34,688 | 35,314 | +626 | −1.80% | **FIFO-Reinsertion** |
| evict_value_v1 vs LRU | 34,667 | 36,344 | +1,677 | −4.84% | **LRU** |
| evict_value_v1 vs SIEVE | 35,314 | 36,344 | +1,031 | −2.92% | **SIEVE** |
| evict_value_v1 vs FIFO-Reinsertion | 34,688 | 36,344 | +1,656 | −4.77% | **FIFO-Reinsertion** |

---

## 4. evict_value_v1 — wins and competitiveness

| Trace family | evict misses | Best other | Gap | Outcome |
|--------------|-------------|------------|-----|---------|
| brightkite | 19,970 | 18,078 (lru) | +1,892 (+10.5%) | **Loss** |
| citibike | 21,573 | 19,987 (fifo) | +1,586 (+7.9%) | **Loss** |
| cloudphysics | 49,626 | 49,114 (sieve) | +512 (+1.0%) | **Loss** (near sieve) |
| metacdn | 35,951 | 29,308 (fifo) | +6,643 (+22.7%) | **Loss** (largest gap) |
| metakv | 38,710 | 38,063 (fifo) | +647 (+1.7%) | **Loss** |
| twemcache | 38,579 | 37,881 (lru) | +698 (+1.8%) | **Loss** |
| wiki2018 | 50,000 | 50,000 (all) | 0 | **Tie** (degenerate) |

**Strict wins:** none.  
**Competitive traces:** cloudphysics (+1.0% vs best), metakv (+1.7%), twemcache (+1.8%) — but still losses. metacdn is a clear outlier (+22.7%). wiki2018 provides no discriminatory signal (capacity 32 vs 50,000 unique pages).

---

## 5. Suspicious or degenerate output

| Finding | Detail | Assessment |
|---------|--------|------------|
| wiki2018 all-miss floor | Every policy: 50,000 misses, 0.0 hit rate | **Expected degeneracy** — capacity 32 with 50,000 unique pages; not a bug |
| cloudphysics near-zero hit rates | All policies 0.7–1.8% hit rate | **Expected** — hard trace at cap32; matches prior cap32 chunk |
| trust_and_doubt small drift vs old cap32 | 6/42 common-policy rows differ by 1–73 misses (see §6) | **Minor inconsistency** — investigate before citing trust_and_doubt across runs; does not affect evict_value_v1/LRU/SIEVE/FIFO conclusions |
| No policy returned NaN/negative hit rates | All 56 rows well-formed | **Clean** |
| No implausible ordering | SIEVE smoke (496 vs LRU 503 on 1k requests) directionally consistent with cap32 on most families | **Plausible** |

---

## 6. Consistency with previous cap32 run

Compared common policies between:
- **New:** `..._cap32_with_sieve_fifo.csv` (8 policies)
- **Old:** `..._cap32.csv` (7 policies, no sieve/fifo, includes blind_oracle)

**Common policies checked (6):** `lru`, `predictive_marker`, `blind_oracle_lru_combiner`, `trust_and_doubt`, `rest_v1`, `evict_value_v1` → 42 rows.

| Result | Count |
|--------|-------|
| Exact match | **36 / 42** (86%) |
| Mismatch (trust_and_doubt only) | **6 / 42** |

**Policies with 100% exact match (7 families each):** lru, predictive_marker, blind_oracle_lru_combiner, rest_v1, evict_value_v1 — **35/35 rows identical**.

**trust_and_doubt mismatches:**

| Family | Old | New | Δ |
|--------|-----|-----|---|
| brightkite | 18,845 | 18,772 | −73 |
| citibike | 20,856 | 20,813 | −43 |
| twemcache | 38,827 | 38,774 | −53 |
| metakv | 38,165 | 38,167 | +2 |
| metacdn | 29,811 | 29,810 | −1 |
| cloudphysics | 49,275 | 49,273 | −2 |

**Interpretation:** Adding SIEVE/FIFO did **not** alter the five core comparison policies (LRU, predictive_marker, blind_oracle_lru_combiner, rest_v1, evict_value_v1). The evict_value_v1 cap32 numbers are **bit-for-bit identical** to the prior no-SIEVE run. trust_and_doubt shows small drift (≤0.4% relative), worth noting but not blocking the main scientific conclusions.

**Aggregate evict_value_v1 vs LRU unchanged:** −4.84% on both runs.

---

## 7. Scientific decision — continue full sweep?

### Q1. Does evict_value_v1 remain worse than LRU on cap32?

**Yes.** −4.84% mean misses; loses on 6/7 families; ties only on the degenerate wiki2018 floor. Identical to the prior no-SIEVE cap32 chunk for evict_value_v1.

### Q2. Does it remain worse than SIEVE and FIFO-Reinsertion?

**Yes.** −2.92% vs SIEVE; −4.77% vs FIFO-Reinsertion on aggregate. Loses to both on every non-degenerate family.

### Q3. Is there any workload where it is competitive?

**Marginally on three families** (cloudphysics, metakv, twemcache: within ~2% of best), but **never wins**. metacdn is a large loss (+22.7%). wiki2018 is uninformative.

### Q4. Does cap32 support continuing cap64/cap128/cap256?

**Partially.** cap32 is necessary evidence (SIEVE/FIFO now measured; reviewer baseline gap partially closed) but **does not motivate a performance claim**. Higher capacities *could* help a learning policy, but:
- the validation pass (1 trace, cap256) already showed evict_value_v1 worse than LRU;
- cap32 at the smallest capacity already shows a consistent loss pattern;
- there is no family where evict_value_v1 leads at cap32 to suggest upside at larger capacities.

Continuing is justified for **reviewer completeness** (multi-capacity Table 3) and **honest negative reporting**, not because cap32 signals likely improvement.

### Q5. Reframe before more compute?

**Strong case for parallel reframing work now.** cap32 confirms the validation-sanity conclusion: the method does not beat LRU/SIEVE/FIFO-Reinsertion at capacity 32. A revision that leads with "decision-aligned target + rigorous end-to-end audit revealing when learning does not help" is more defensible than waiting for three more ~5h chunks that may extend the same pattern.

### Q6. If continuing, same 8-policy set for cap64?

**Yes, if any further capacity chunk is run** — use the same 8-policy set (`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`) for consistency with this canonical cap32 chunk and Reviewer #3's SIEVE/FIFO request. Do **not** reintroduce `blind_oracle` into the canonical set (known bad numbers, not in TABLE3_POLICIES).

---

## 8. Concise interpretation

1. **Best policy on cap32:** LRU ≡ rest_v1 (tied); fifo_reinsertion essentially tied (−0.06%).
2. **SIEVE:** Modestly worse than LRU overall (−1.86%); wins only on cloudphysics; worst on twemcache (−8.1% vs LRU).
3. **FIFO-Reinsertion:** Near-LRU overall; beats LRU on citibike, metacdn, metakv; slightly worse on brightkite/twemcache.
4. **evict_value_v1:** Clear aggregate loss vs all three (−4.84% vs LRU, −2.92% vs SIEVE, −4.77% vs FIFO-Reinsertion). No strict wins.
5. **Reproducibility:** Core policies (especially evict_value_v1) match the prior cap32 run exactly; trust_and_doubt has minor drift.
6. **Reviewer impact:** R3-Issue3/Rec2 partially addressable for cap32 (SIEVE/FIFO numbers now exist); R3-Issue1/Rec1 still need cap64/128/256 for a complete answer.

**Bottom line:** The corrected cap32 chunk is **valid, complete, and scientifically informative** — but it reinforces a **negative** end-to-end finding rather than overturning it.
