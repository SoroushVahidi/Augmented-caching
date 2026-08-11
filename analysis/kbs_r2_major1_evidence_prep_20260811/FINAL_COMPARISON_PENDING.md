# Final matched comparison status

The corrected treatment CSV was not found or not supplied with the expected
SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296` during this preparation pass.

Do not use `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`;
that file is the old contaminated/ineligible treatment-side result.

Once the verified CSV is synchronized, run:

```bash
python3 scripts/analysis/prepare_r2_major1_evidence.py \
  --treatment-csv analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv
```

The script will write `reviewer_ready_comparison.csv` and
`reviewer_ready_comparison.json` in this directory.
