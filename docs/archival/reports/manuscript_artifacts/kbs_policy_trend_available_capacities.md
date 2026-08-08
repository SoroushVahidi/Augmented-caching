# KBS policy trend — DRAFT / AVAILABLE CAPACITIES ONLY

**This is NOT the final canonical manuscript table.** Requested capacities: 32, 64, 128. Available capacities in inputs: 32, 64, 128.
All requested capacities are present in the inputs, but this script still only writes to the *_available_capacities filenames — promoting to a canonical table is a separate, explicit step.

Input chunks: analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv, analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv, analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv

## Per-capacity mean misses by policy

| policy | cap32 | cap64 | cap128 |
|---|---|---|---|
| blind_oracle_lru_combiner | 34668.6 | 33651.3 | 32496.9 |
| evict_value_v1 | 36344.1 | 34409.0 | 36316.6 |
| fifo_reinsertion | 34688.0 | 33672.1 | 32530.6 |
| lru | 34667.1 | 33650.1 | 32495.6 |
| predictive_marker | 34843.9 | 33773.1 | 32664.6 |
| rest_v1 | 34667.1 | 33650.1 | 32495.6 |
| sieve | 35313.6 | 34362.7 | 33279.3 |
| trust_and_doubt | 35087.0 | 33998.7 | 32928.9 |

## Per-capacity relative gap vs LRU (%)

| policy | cap32 | cap64 | cap128 |
|---|---|---|---|
| blind_oracle_lru_combiner | +0.00 | +0.00 | +0.00 |
| evict_value_v1 | +4.84 | +2.26 | +11.76 |
| fifo_reinsertion | +0.06 | +0.07 | +0.11 |
| lru | +0.00 | +0.00 | +0.00 |
| predictive_marker | +0.51 | +0.37 | +0.52 |
| rest_v1 | +0.00 | +0.00 | +0.00 |
| sieve | +1.86 | +2.12 | +2.41 |
| trust_and_doubt | +1.21 | +1.04 | +1.33 |

## evict_value_v1 gap vs LRU / SIEVE / FIFO-Reinsertion (%)

| capacity | vs lru | vs sieve | vs fifo_reinsertion |
|---|---|---|---|
| 32 | +4.84 | +2.92 | +4.77 |
| 64 | +2.26 | +0.13 | +2.19 |
| 128 | +11.76 | +9.13 | +11.64 |

## Per-trace-family ranking (1 = lowest mean misses) by capacity

### Capacity 32

| trace_family | ranked policies (best -> worst) |
|---|---|
| brightkite | lru(18078), rest_v1(18078), blind_oracle_lru_combiner(18081), sieve(18110), fifo_reinsertion(18122), predictive_marker(18483), trust_and_doubt(18772), evict_value_v1(19970) |
| citibike | fifo_reinsertion(19987), lru(19994), rest_v1(19994), blind_oracle_lru_combiner(19995), predictive_marker(20169), trust_and_doubt(20813), sieve(21151), evict_value_v1(21573) |
| cloudphysics | sieve(49114), fifo_reinsertion(49245), lru(49273), rest_v1(49273), trust_and_doubt(49273), blind_oracle_lru_combiner(49274), predictive_marker(49384), evict_value_v1(49626) |
| metacdn | fifo_reinsertion(29308), lru(29380), rest_v1(29380), blind_oracle_lru_combiner(29381), predictive_marker(29506), sieve(29732), trust_and_doubt(29810), evict_value_v1(35951) |
| metakv | fifo_reinsertion(38063), lru(38064), rest_v1(38064), blind_oracle_lru_combiner(38065), predictive_marker(38105), sieve(38145), trust_and_doubt(38167), evict_value_v1(38710) |
| twemcache | lru(37881), rest_v1(37881), blind_oracle_lru_combiner(37884), fifo_reinsertion(38091), predictive_marker(38260), evict_value_v1(38579), trust_and_doubt(38774), sieve(40943) |
| wiki2018 | blind_oracle_lru_combiner(50000), evict_value_v1(50000), fifo_reinsertion(50000), lru(50000), predictive_marker(50000), rest_v1(50000), sieve(50000), trust_and_doubt(50000) |

### Capacity 64

| trace_family | ranked policies (best -> worst) |
|---|---|
| brightkite | lru(16543), rest_v1(16543), blind_oracle_lru_combiner(16544), fifo_reinsertion(16577), predictive_marker(16800), sieve(16865), trust_and_doubt(17219), evict_value_v1(18316) |
| citibike | fifo_reinsertion(18864), lru(18964), rest_v1(18964), blind_oracle_lru_combiner(18965), predictive_marker(19025), trust_and_doubt(19379), sieve(19456), evict_value_v1(19855) |
| cloudphysics | fifo_reinsertion(48500), lru(48575), rest_v1(48575), blind_oracle_lru_combiner(48577), sieve(48621), trust_and_doubt(48695), predictive_marker(48867), evict_value_v1(49339) |
| metacdn | fifo_reinsertion(28590), lru(28616), rest_v1(28616), blind_oracle_lru_combiner(28617), predictive_marker(28649), sieve(28737), trust_and_doubt(28962), evict_value_v1(29397) |
| metakv | fifo_reinsertion(38036), sieve(38037), lru(38041), rest_v1(38041), blind_oracle_lru_combiner(38042), evict_value_v1(38044), predictive_marker(38047), trust_and_doubt(38081) |
| twemcache | lru(34812), rest_v1(34812), blind_oracle_lru_combiner(34814), predictive_marker(35024), fifo_reinsertion(35138), trust_and_doubt(35655), evict_value_v1(35912), sieve(38823) |
| wiki2018 | blind_oracle_lru_combiner(50000), evict_value_v1(50000), fifo_reinsertion(50000), lru(50000), predictive_marker(50000), rest_v1(50000), sieve(50000), trust_and_doubt(50000) |

### Capacity 128

| trace_family | ranked policies (best -> worst) |
|---|---|
| brightkite | lru(15479), rest_v1(15479), blind_oracle_lru_combiner(15480), fifo_reinsertion(15504), predictive_marker(15722), sieve(15861), trust_and_doubt(16140), evict_value_v1(23360) |
| citibike | fifo_reinsertion(17040), lru(17124), rest_v1(17124), blind_oracle_lru_combiner(17125), sieve(17341), predictive_marker(17345), trust_and_doubt(17575), evict_value_v1(24914) |
| cloudphysics | fifo_reinsertion(47650), lru(47748), rest_v1(47748), blind_oracle_lru_combiner(47752), trust_and_doubt(48002), sieve(48017), predictive_marker(48182), evict_value_v1(49230) |
| metacdn | predictive_marker(27748), fifo_reinsertion(27764), sieve(27797), lru(27811), rest_v1(27811), blind_oracle_lru_combiner(27812), trust_and_doubt(28101), evict_value_v1(32711) |
| metakv | sieve(37820), predictive_marker(37932), fifo_reinsertion(37945), trust_and_doubt(37959), lru(37984), rest_v1(37984), blind_oracle_lru_combiner(37985), evict_value_v1(38273) |
| twemcache | lru(31323), rest_v1(31323), blind_oracle_lru_combiner(31324), predictive_marker(31723), fifo_reinsertion(31811), trust_and_doubt(32725), evict_value_v1(35728), sieve(36119) |
| wiki2018 | blind_oracle_lru_combiner(50000), evict_value_v1(50000), fifo_reinsertion(50000), lru(50000), predictive_marker(50000), rest_v1(50000), sieve(50000), trust_and_doubt(50000) |

## Trend across available capacities (32 -> 64 -> 128)

| policy | cap32 | cap64 | cap128 | direction |
|---|---|---|---|---|
| blind_oracle_lru_combiner | 34668.6 | 33651.3 | 32496.9 | decreasing (improving) |
| evict_value_v1 | 36344.1 | 34409.0 | 36316.6 | decreasing (improving) |
| fifo_reinsertion | 34688.0 | 33672.1 | 32530.6 | decreasing (improving) |
| lru | 34667.1 | 33650.1 | 32495.6 | decreasing (improving) |
| predictive_marker | 34843.9 | 33773.1 | 32664.6 | decreasing (improving) |
| rest_v1 | 34667.1 | 33650.1 | 32495.6 | decreasing (improving) |
| sieve | 35313.6 | 34362.7 | 33279.3 | decreasing (improving) |
| trust_and_doubt | 35087.0 | 33998.7 | 32928.9 | decreasing (improving) |

---
_DRAFT / AVAILABLE CAPACITIES ONLY. Regenerate after each new capacity chunk completes and is verified._
