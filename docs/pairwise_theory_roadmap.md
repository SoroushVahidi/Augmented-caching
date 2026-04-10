# Pairwise theory roadmap for learning-augmented caching

## Scope and goal
This note is a **research roadmap**, not a proof. The purpose is to translate current empirical pairwise results into clean theorem targets for learning-augmented caching with ranking-style advice.

---

## A) Precise problem setup

### A.1 Online paging setting
- Universe of pages: \(\mathcal U\).
- Request sequence: \(\sigma = (r_1,\dots,r_T)\), \(r_t \in \mathcal U\).
- Cache capacity: \(k\).
- At time \(t\), let cache state before serving \(r_t\) be \(C_t\), \(|C_t|\le k\).
- On a miss with \(|C_t|=k\), algorithm must evict one page from candidate set
  \[
  S_t := C_t.
  \]

Belady-induced eviction ranking at decision \(t\):
- Let \(\tau_t(p)\) be next-arrival time of page \(p\) after \(t\), with \(+\infty\) if absent.
- Belady eviction choice is \(\arg\max_{p\in S_t} \tau_t(p)\).
- This induces a total preorder \(\prec_t^{\star}\) on \(S_t\) by descending \(\tau_t(\cdot)\) (tie-broken deterministically).

### A.2 Pairwise predictor and induced ordering
A pairwise predictor outputs scores
\[
\widehat P_t(i \succ j) \in [0,1], \qquad i,j\in S_t,
\]
interpreted as “\(i\) is the better **eviction** candidate than \(j\).”

From pairwise outputs, induce an eviction order \(\widehat\prec_t\), e.g. via Copeland/Borda wins:
\[
W_t(i) := \sum_{j\in S_t\setminus\{i\}} \widehat P_t(i\succ j),
\]
then choose eviction victim \(\hat v_t = \arg\max_i W_t(i)\) (deterministic tie-break).

### A.3 Candidate pairwise/inversion-style error notions

#### (i) Local inversion count at decision \(t\)
\[
I_t := \#\{(i,j): i\prec_t^{\star} j\text{ but } j\widehat\prec_t i\}.
\]
- Captures **ordering disagreement** within current candidate set.
- Normalized version: \(\bar I_t = I_t / \binom{|S_t|}{2}\).

#### (ii) Phase-wise inversion count
Partition requests into phases (e.g., marker-style phases) \(\Phi_1,\Phi_2,\dots\). Define
\[
I(\Phi_m) := \sum_{t\in\Phi_m} I_t,
\quad
\bar I(\Phi_m):=\frac{I(\Phi_m)}{\sum_{t\in\Phi_m}\binom{|S_t|}{2}}.
\]
- Captures burstiness / adversarial concentration of ranking mistakes.

#### (iii) Queried-pair cumulative inversion budget
For algorithms querying subset \(Q_t\subseteq S_t\times S_t\):
\[
I_Q := \sum_t \#\{(i,j)\in Q_t : \text{predicted orientation differs from }\prec_t^{\star}\}.
\]
- Matches partial-comparison or shortlist regimes.

### A.4 Comparison with scalar \(\ell_1\)-style next-arrival error
Let scalar-prediction setup have \(\hat\tau_t(p)\) and error \(\sum_{t,p}|\hat\tau_t(p)-\tau_t(p)|\).

What inversion notions capture better:
- Relative eviction ranking quality inside the decision set.
- Decision-local brittleness under candidate competition.
- Compatibility with pairwise-trained models that never predict calibrated \(\tau\)-values.

What inversion notions miss:
- Magnitude of timing error (small/large timing mistakes can induce same inversion count).
- Global calibration and confidence structure.
- Non-local dependence where one bad eviction cascades across many future times.

---

## B) Theory roadmap (in increasing difficulty)

### Target 1: Zero-inversion consistency (restricted)
**Candidate statement:** if \(I_t=0\) for all miss decisions under a fixed coupling and deterministic tie-breaking, then algorithm eviction choices match Belady on those queried candidate sets.

- Mostly supported by existing LA ideas: “perfect advice => oracle agreement” templates.
- New part: translating from pairwise relation to chosen victim requires aggregation-consistency assumptions (acyclic/tournament consistency or tie rules).

### Target 2: Rank-shift / local-cost lemma
**Candidate statement:** per decision excess cost proxy is bounded by function of local inversion profile (or top-rank displacement), possibly under shortlist size \(m\).

- Existing support: rank-based potential arguments in parsimonious/robust LA frameworks.
- New argument needed: map inversion mistakes to cache-state drift and future miss increments.

### Target 3: Competitive-style guarantee with fallback
For pairwise-aware algorithm with robust fallback (e.g., combiner/guard):
\[
\text{Misses} \le \alpha\,\text{OPT} + \beta\,\mathcal E_{\text{pair}} + \gamma,
\]
where \(\mathcal E_{\text{pair}}\) could be phase-normalized inversion mass.

- Existing support: robust fallback analyses, decomposition into base + correction terms.
- New argument needed: integrate inversion error with switching logic and prove bounded worst-case excess.

---

## C) Proof templates

### Direction 1: Parsimonious / rank-shift adaptation
- **Likely key lemma:** if eviction candidate is displaced by \(\Delta_t\) positions relative to Belady order, then one-step or short-window excess is bounded by \(f(\Delta_t)\) under restricted spacing assumptions.
- **Likely obstruction:** tiny displacement can still trigger long cascades without structural assumptions on reuse gaps.
- **Useful toy/counterexamples:** short sequences where one wrong eviction causes repeated immediate misses versus sequences where same inversion is harmless.

### Direction 2: Inversion-primary online template
- **Likely key lemma:** potential increase is charged to pairwise discordant edges; cumulative extra misses bounded by charged inversion mass plus fallback overhead.
- **Likely obstruction:** pairwise edges are local to changing candidate sets; need consistent coupling when cache states diverge from Belady.
- **Useful toy/counterexamples:** examples with same total inversion count but different temporal concentration (early vs late phase) causing very different regret.

---

## D) Concrete conjectures / theorem candidates

1. **Zero-local-inversion agreement (restricted coupling).**
   Under deterministic tie-breaking and consistent pairwise aggregation, if \(I_t=0\) at every miss decision on the realized trajectory, then the pairwise policy matches Belady eviction choices on that trajectory.

2. **Top-rank displacement bound under shortlist \(m\).**
   If shortlist policy always contains Belady victim and predictor induces at most one adjacent swap in shortlist order per decision, then per-decision local regret is bounded by a shortlist-dependent constant under a bounded-gap reuse assumption.

3. **Phase-normalized inversion excess bound (restricted class).**
   For traces with bounded phase volatility, excess misses over Belady are at most affine in \(\sum_m \bar I(\Phi_m)\cdot |\Phi_m|\).

4. **Fallback robustness with inversion-rate trigger.**
   A combiner that switches to robust fallback when rolling inversion proxy exceeds threshold admits bound:
   \(\text{Misses} \le \text{RobustBaseline} + O(B + R)\), where \(B\) is number of switches and \(R\) is cumulative above-threshold inversion mass.

5. **Negative conjecture (explicitly likely false without assumptions).**
   “One inversion per decision implies at most constant (+1) total extra misses” is false in general; cascading effects can make excess unbounded in horizon without spacing/phase assumptions.

---

## What is already supported vs genuinely new

### Already supported by LA-caching proof ecosystem
- Robust-fallback decomposition mindset.
- Potential-style charging and phase-based accounting.
- Perfect-advice consistency paradigms.

### Requires genuinely new arguments
- Converting pairwise inversion metrics into competitive guarantees.
- Handling dynamic candidate sets under state divergence.
- Characterizing when local ranking errors remain local vs cascade globally.

---

## Immediate next steps
1. Use `scripts/analyze_pairwise_inversion_examples.py` to curate positive/negative toy families.
2. Pick one restricted theorem regime (e.g., shortlist + bounded reuse-gap) and prove Target 1 + partial Target 2.
3. Only then attempt fallback-competitive statement (Target 3).
