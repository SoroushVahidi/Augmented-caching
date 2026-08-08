# FIFO-Reinsertion baseline audit (2026-06-19)

**Update 2026-06-19 (approved stop-and-relaunch pass).** The recommendation
in this audit has now been acted on: the obsolete `cap32_with_sieve` job
was stopped, and the corrected cap32 tmux job
`kbs_full_policy_comparison_cap32_with_sieve_fifo` is now running with
policy list
`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`.
cap64/cap128/cap256 remain not launched.

Original audit plus lightweight validation was done while
`kbs_full_policy_comparison_cap32_with_sieve` was running read-only on the
current local/cloud machine. This update records the later approved
stop-and-relaunch action.

## 1. What algorithm the repo currently implements

The repo currently implements `FIFOReinsertionPolicy` at:

- `src/lafc/policies/fifo_reinsertion.py`

Registry / runner wiring exists in both:

- `scripts/run_policy_comparison_wulver_v1.py`
- `src/lafc/runner/run_policy.py`

Dedicated tests exist at:

- `tests/test_fifo_reinsertion.py`

The implementation is a single-queue FIFO policy with one visited bit per
resident page:

1. hit: set `visited[pid] = True`, no reordering
2. full-cache miss: inspect the oldest/tail object
3. if visited: clear its bit and move it to the head
4. continue until an unvisited object is found, then evict it
5. insert the new page at the head with `visited = False`

This is exactly what the module docstring says and exactly what the tests
exercise.

## 2. Is it truly FIFO-Reinsertion or just a CLOCK/Second-Chance proxy?

It is scientifically defensible to call this implementation
FIFO-Reinsertion.

Reason:

- `reports/kbs_halp_fifo_source_verification.md` already established that
  the SIEVE NSDI'24 paper explicitly groups `Second Chance`, `CLOCK`, and
  `FIFO-Reinsertion` as different implementations of the same algorithmic
  family.
- The same paper defines the key contrast against SIEVE as:
  retained/visited objects are moved to the head in FIFO-Reinsertion,
  whereas SIEVE leaves them in place and advances a hand pointer.
- The repo implementation does exactly that retained-object reinsertion.

So this is not a misnamed unrelated heuristic. It is a CLOCK /
Second-Chance-family FIFO-Reinsertion implementation, which is exactly the
standard reading justified by the SIEVE/S3-FIFO source trail.

## 3. How it differs from SIEVE

The difference is specific and operational, not just naming:

- `fifo_reinsertion`: a visited survivor is physically moved to the head
- `sieve`: a visited survivor stays in place and the hand advances

Consequences:

- FIFO-Reinsertion mixes retained survivors back in with newly inserted
  pages
- SIEVE separates old retained survivors from new insertions and therefore
  preserves quick demotion better
- FIFO-Reinsertion needs no persistent hand state
- SIEVE needs a persistent hand pointer

This matches both the NSDI'24 paper and the code in
`src/lafc/policies/fifo_reinsertion.py` vs. `src/lafc/policies/sieve.py`.

## 4. Is the name `fifo_reinsertion` scientifically defensible?

Yes.

It would be misleading to relabel it as some novel proxy or ad hoc
surrogate. The current name is justified because:

- the implementation behavior matches the literature-backed definition
- the source-verification report already resolves the intended reviewer
  reading to this family
- the repo docstring explicitly anchors the name to Corbato + SIEVE + S3-FIFO

The most precise wording for manuscript/reviewer text is:

> FIFO-Reinsertion (CLOCK / Second-Chance family baseline)

That is better than renaming the code, because it preserves the requested
reviewer term while clarifying the exact family being instantiated.

## 5. Citation to use

Recommended citation stack:

1. `zhang2024sieve` for the modern terminology and the direct
   FIFO-Reinsertion-vs-SIEVE contrast
2. `yang2023s3fifo` for the broader FIFO-family framing
3. optional historical CLOCK / Second-Chance citation if desired

Practical recommendation:

- reviewer response / related-work text: cite `zhang2024sieve`
- if explaining the FIFO-family context or taxonomy in more detail: also
  cite `yang2023s3fifo`
- historical Corbato citation is optional, not necessary for closing
  Reviewer #3's request

## 6. Tests and lightweight validation

### 6.1 Targeted tests

Requested pytest slice:

```bash
pytest tests/ -k "fifo or reinsertion or second or clock or sieve or policy" -v
```

Result under bare system `pytest`: failed at collection with
`ModuleNotFoundError: No module named 'lafc'` because the system
interpreter is not the project environment.

Result under the existing project venv:

```bash
.venv_kbs_heavy_r1/bin/pytest tests/ -k "fifo or reinsertion or second or clock or sieve or policy" -v
```

- `39 passed, 260 deselected`
- no test failures

This is the meaningful result for repo readiness.

### 6.2 Tiny smoke test

To avoid overwriting the existing smoke files, this pass used fresh output
filenames:

- `analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke_audit.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke_audit.md`

Command:

```bash
.venv_kbs_heavy_r1/bin/python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-traces 1 \
  --max-requests-per-trace 1000 \
  --policies lru,sieve,fifo_reinsertion \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke_audit.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke_audit.md
```

Result:

- `lru`: 503 misses
- `sieve`: 496 misses
- `fifo_reinsertion`: 504 misses

Interpretation:

- end-to-end runner support is confirmed
- behavior is plausible
- this is not canonical evidence, only a tiny safety check

## 7. Does it appear in manuscript/policy tables?

Not yet.

Current manuscript-facing policy roster files:

- `tables/manuscript/table2_policy_roster.csv`
- `tables/manuscript/table2_policy_roster.tex`

currently include:

- `lru`
- `predictive_marker`
- `trust_and_doubt`
- `blind_oracle_lru_combiner`
- `rest_v1`
- `evict_value_v1`

They do **not** include either:

- `sieve`
- `fifo_reinsertion`

So if FIFO-Reinsertion is to be part of the final canonical comparison, a
Table 2 / manuscript-roster update is required.

## 8. Should FIFO-Reinsertion be included in final canonical comparisons?

Yes, recommended.

Reason:

- Reviewer #3 asked for it explicitly.
- It is already implemented, wired, tested, and tiny-smoke validated.
- It is a lightweight modern baseline in exactly the comparison family
  Reviewer #3 grouped with SIEVE.
- If cap64/cap128/cap256 are launched without it and you later decide it
  belongs in the paper, you will force another canonical rerun.

## 9. Would `cap32_with_sieve` need another rerun if FIFO-Reinsertion is included?

Yes.

Current running policy list is:

- `lru`
- `sieve`
- `predictive_marker`
- `blind_oracle_lru_combiner`
- `trust_and_doubt`
- `rest_v1`
- `evict_value_v1`

It does **not** include `fifo_reinsertion`.

Therefore, if the final manuscript-citable baseline set includes
FIFO-Reinsertion, the currently running `cap32_with_sieve` chunk will not
match that final policy set and cap32 will need a later rerun with both
SIEVE and FIFO-Reinsertion included.

## 10. Bottom line

- FIFO-Reinsertion exists in the repo
- exact file path: `src/lafc/policies/fifo_reinsertion.py`
- semantics: true FIFO-Reinsertion / CLOCK / Second-Chance family baseline
- name `fifo_reinsertion` is scientifically defensible
- targeted tests pass in the project venv
- tiny smoke test succeeds
- if you want a clean final canonical sweep before cap64, FIFO-Reinsertion
  should be added to the manuscript-citable policy set now, not later
