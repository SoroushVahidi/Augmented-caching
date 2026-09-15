# Predefined interpretation rules for long-horizon discriminativeness (Task 12)

Written *before* production results exist, so interpretation is not
retrofitted to whatever the numbers turn out to be. These are descriptive
classifications over the same `discriminativeness` metric already used in
the cap32 preflight (1 - all_tied_fraction) and computed per family x
capacity x horizon by `aggregate.py`'s Table A.

The cap32 preflight (pooled across 5 families, job 1287838/1287919) showed:
`H16=0.2999, H32=0.3257, H64=0.3330, H128=0.3332` -- i.e. most of the
H16->H128 gain was already realized by H64 in that one pooled, single-
capacity measurement. The rules below are stated generally, not tuned to
reproduce that specific number, and are meant to be applied per
family/capacity cell as well as pooled.

## Definitions

For a given family/capacity cell, let `D(H)` be discriminativeness at
horizon H. Define:
- `gain_total = D(128) - D(16)`
- `gain_to_32 = D(32) - D(16)`
- `gain_to_64 = D(64) - D(16)`
- `frac_realized_by_32 = gain_to_32 / gain_total` (undefined/NaN if `gain_total == 0`)
- `frac_realized_by_64 = gain_to_64 / gain_total`

## Classification rules (descriptive, applied per cell and pooled)

**SATURATING_BY_64** if, across most non-degenerate family/capacity cells,
`frac_realized_by_64` is close to 1 (i.e. the H64->H128 increment is small
relative to the H16->H64 increment) -- consistent with, but not proving,
the pattern already observed in the cap32 pooled preflight number.

**CONTINUES_IMPROVING** if, across important cells, `D(128)` is
substantially larger than `D(64)` -- i.e. `frac_realized_by_64` is
meaningfully below 1 and the H64->H128 increment is not small relative to
earlier increments.

**WORKLOAD_DEPENDENT** if the saturation pattern (SATURATING_BY_64 vs.
CONTINUES_IMPROVING) differs materially across families or capacities --
e.g. some family/capacity cells saturate by H64 while others keep gaining
meaningfully through H128. Given the cap32 preflight already pooled across
5 heterogeneous families (which the manuscript's own characterization
section documents as spanning discriminative to near-degenerate), workload
dependence is the single most likely of the three outcomes and should be
checked first, not assumed away by only looking at pooled numbers.

**Degenerate cells excluded from interpretation, not ignored in reporting**:
consistent with the manuscript's own treatment of wiki2018 as a
"mechanistically explained negative control" (near-zero discriminativeness
regardless of horizon), any cell with `D(16)` and `D(128)` both very close
to zero should be reported in Table A like every other cell, but excluded
from the saturation-pattern narrative the same way the manuscript already
excludes wiki2018 from its main discriminativeness discussion -- not
dropped from the data, just not used to argue for or against saturation.

## Explicit non-rules

- **No p-values, significance tests, or invented numeric thresholds
  (e.g. "saturating if gain < 0.01") are defined here.** The rules above are
  intentionally qualitative/relative ("close to 1", "substantially larger")
  because a single-probe, descriptive campaign does not license manufactured
  statistical precision. `aggregate.py` computes the continuous per-cell
  deltas (Task 7, Table C); a human should read those numbers against these
  qualitative rules, not have a script auto-declare a verdict.
- **This document does not predict which classification will hold.** The
  cap32 pooled number is suggestive of SATURATING_BY_64 but is (a) pooled
  across families rather than per-cell, (b) at a single capacity (32) only,
  and (c) not yet cross-checked against the cap256 probe -- exactly why
  production is needed before any RQ3 manuscript language is rewritten (see
  `docs/pe_long_horizon_manuscript_update_map.md`).
