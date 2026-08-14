# Recovery notes

The production worker exited after 21/21 units because campaign CSV writing
used only the first row's keys (`CSV_SCHEMA_UNION_BUG`).

Scientific units were not rerun. Campaign files were rebuilt from the 21
completed `units/*/summary.json` files via
`run_tie_aware_exact_target_oracle.py --aggregate-only`.

See `PROVENANCE.json` and `AUDIT.md`.
