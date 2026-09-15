# PE manuscript integration map (Task 11) -- read-only survey, no edits made

Source: `/home/soroush/projects/lafc-evict-dataset/repo/paper/performance_evaluation/latex/`
(the "LAFC-Evict" SIGMOD2027 benchmark paper -- confirmed as the correct PE
manuscript; see docs/pe_long_horizon_design_notes.md and the resolved
PE-scope finding in this session's chat history).

Every location below currently states or implies $H\in\{4,8,16\}$ / "H>16
untested" and would need updating once the long-horizon production result
lands. **None of these were edited as part of this task.**

| Location | Current statement | What changes |
|---|---|---|
| `main.tex` (abstract), lines 76-91 | "Across five trace families, four capacities, and three horizons..." / "this correlation is strongest at the longest of the three evaluated horizons ($H=16$ of $H\in\{4,8,16\}$), which is not evidence about horizons beyond 16." | Horizon count (three -> six, if H=32/64/128 are added alongside existing H=4/8/16) and the "not evidence about horizons beyond 16" disclaimer, once real H>16 evidence exists. |
| `sections/01_introduction.tex`, RQ3 (~line 50-52) | "among the three horizons this paper evaluates ($H\in\{4,8,16\}$), $H=16$ aligns offline and closed-loop evidence most strongly, though modestly; this paper does not evaluate $H>16$." | This is the exact sentence the long-horizon experiment is designed to answer -- direct rewrite once results exist, likely the single most important sentence in the whole paper to revisit. |
| `sections/03b_experimental_methodology.tex`, line 68 | "all five families, capacities 32/64/128/256, and horizons 4/8/16" | Horizon list extended to include 32/64/128 (methodology section describing what was run). |
| `sections/08e_continuation.tex`, line 43 | "...and horizons 4, 8, and 16, over the complete population" | Same -- describes the continuation-robustness census's horizon scope; likely out of scope for the long-horizon extension unless continuation robustness is re-checked at the new horizons too (not requested here). |
| `sections/09_limitations_ethics.tex`, lines 16-17 | "every horizon-dependent result is scoped to $H\in\{4,8,16\}$... $H>16$ remains untested." | This limitation statement is exactly what the long-horizon campaign resolves (partially or fully, depending on results) -- must not survive unchanged if H>16 data lands. |
| `sections/11_conclusion.tex`, lines 8-16 (RQ3 answer) | "...most strongly at the longest of the three horizons evaluated ($H=16$ of $H\in\{4,8,16\}$; horizons beyond 16 are untested) (RQ3, exploratory, $n=10$)." | Same sentence family as the introduction RQ3 answer -- needs the new horizon evidence folded in. |
| Figure 2 (discriminativeness), `scripts/figures/make_figure2_discriminativeness.py` line 30 | `HORIZONS = [4, 8, 16]`, reads `analysis/sigmod_target_discriminativeness_20260913/outputs/phase4_stratified_family_capacity_horizon.csv` | The figure-generation script's horizon axis is hardcoded; extending it requires both new production data AND a new/extended source CSV in the same `phase4_stratified_family_capacity_horizon.csv` schema (family, capacity, horizon, discriminativeness) -- the aggregator's `table_A_family_capacity_horizon.csv` (this package, Task 7) is schema-compatible and could feed this directly. |
| `latex/tables/table_regret_tie_stats.tex` | Candidate-row/y_loss/regret statistics computed at existing (H<=16) scope | Would need a parallel or extended table if H=32/64/128 regret/tie statistics are to be reported alongside the existing ones. |

## What this map does NOT do
- No manuscript text was changed.
- No claim is made yet about *what* the new numbers will show (saturating,
  continuing to improve, or workload-dependent) -- see
  `docs/pe_long_horizon_interpretation_rules.md` for the predefined,
  results-agnostic rules that will classify the outcome once production
  data exists.
- RQ3 is the load-bearing research question here; RQ1/RQ2/RQ4/RQ5 are not
  directly about horizon scope and are not flagged above unless they
  explicitly cite the $H\in\{4,8,16\}$ boundary.
