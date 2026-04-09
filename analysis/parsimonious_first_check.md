# Parsimonious Baseline First Check

Command:

- `python scripts/run_parsimonious_first_check.py`

| policy | cost | misses | hit_rate | queries_used | frac_miss_queried | frac_miss_fallback |
|---|---:|---:|---:|---:|---:|---:|
| lru | 6.00 | 6 | 40.00% | 0.0 | 0.00% | 0.00% |
| marker | 7.00 | 7 | 30.00% | 0.0 | 0.00% | 0.00% |
| predictive_marker | 5.00 | 5 | 50.00% | 0.0 | 0.00% | 0.00% |
| adaptive_query_b2 | 5.00 | 5 | 50.00% | 4.0 | 40.00% | 0.00% |
| adaptive_query_b4 | 5.00 | 5 | 50.00% | 6.0 | 40.00% | 0.00% |
