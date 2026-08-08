# KBS post-cap128 decision template

Status: TEMPLATE — fill in once `kbs_full_policy_comparison_cap128_with_sieve_fifo` completes
and its output has been verified with `scripts/paper/verify_kbs_policy_chunks.py`.

Do not fill in any section from partial/in-progress output. Wait for:

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.{csv,md}` to exist
- `CAP128_WITH_SIEVE_FIFO_EXIT=0` in `logs/kbs_full_policy_comparison/cap128_with_sieve_fifo.log`
- `verify_kbs_policy_chunks.py` to PASS on cap32+cap64+cap128 together

## 1. cap128 completion status

- [ ] Job exited with code 0 (no traceback, no partial CSV)
- [ ] CSV row count matches expectation (7 traces x 8 policies = 56 rows + header)
- [ ] `verify_kbs_policy_chunks.py --inputs ..._cap32_with_sieve_fifo.csv ..._cap64_with_sieve_fifo.csv ..._cap128_with_sieve_fifo.csv --expected-capacities 32,64,128 --expected-policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1` passes
- Completion timestamp: _fill in_
- Total wall-clock runtime: _fill in_

## 2. cap128 headline result

- Mean misses by policy at capacity 128 (from `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv` after rerunning `build_kbs_policy_trend_artifacts.py` with cap32+cap64+cap128):
  - _fill in table or summary_
- Headline one-line takeaway: _fill in_

## 3. cap32 -> cap64 -> cap128 trend

- Direction of `evict_value_v1` mean misses across capacities (improving / worsening / flat relative to LRU): _fill in_
- Direction of SIEVE and FIFO-Reinsertion baselines across the same capacities: _fill in_
- Does the cap32->cap64 trend (see `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`, generated 2026-06-20) continue, flatten, or reverse at cap128? _fill in_
- Per-trace-family ranking changes at cap128 vs cap32/cap64: _fill in_

## 4. evict_value_v1 gap vs LRU / SIEVE / FIFO-Reinsertion

| capacity | vs LRU (%) | vs SIEVE (%) | vs FIFO-Reinsertion (%) |
|---|---|---|---|
| 32 | +4.84 | +2.92 | +4.77 |
| 64 | +2.26 | +0.13 | +2.19 |
| 128 | _fill in_ | _fill in_ | _fill in_ |

(cap32/cap64 values copied from `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`; recompute after rerunning the trend script with cap128 included, in case of rounding drift.)

## 5. Does cap128 confirm improvement, stall, or reversal?

At cap32 and cap64, `evict_value_v1` trails LRU/SIEVE/FIFO-Reinsertion on mean misses, with the
gap vs LRU narrowing from +4.84% (cap32) to +2.26% (cap64) and the gap vs SIEVE narrowing from
+2.92% to +0.13%. Classify cap128 against this trend:

- [ ] **Improvement continues** — gap vs LRU/SIEVE keeps narrowing (or evict_value_v1 overtakes a baseline)
- [ ] **Stalls** — gap roughly flat vs cap64 (within noise)
- [ ] **Reverses** — gap widens again at cap128

Evidence: _fill in_

## 6. Is cap256 worth running?

Decision inputs:

- If improvement continues or stalls near parity: _recommendation fill in_
- If reversal: _recommendation fill in_
- Marginal compute cost of cap256 given cap128 wall-clock time: _fill in_ (cap256 launch command is staged in `reports/kbs_cap256_launch_template.md`, **not yet run**)

Recommendation: _fill in — GO / NO-GO / GO with reduced trace set_

## 7. Recommended manuscript framing

- [ ] Report cap32/cap64/cap128 trend as evidence of capacity-dependent convergence (if gap narrows)
- [ ] Report cap32/cap64/cap128 as evidence of a fixed, capacity-independent gap (if flat)
- [ ] Report cap128 as a reversal / negative result requiring discussion of why (if gap widens)
- [ ] Note any trace families (e.g. `wiki2018`, which saturates at 50000/50000 misses across all policies at cap32/cap64 because the working set vastly exceeds capacity) that should be flagged as uninformative for trend purposes at this capacity range

Suggested manuscript sentence (fill in once data is available): _fill in_

---
_Generated as a fill-in template while cap128_with_sieve_fifo was still running (2026-06-20). Do not treat any unfilled section as a result._
