# KBS-facing framing: “knowledge” and predictive context (repo-only)

This note aligns **terminology** with **what the codebase actually uses**, so the paper can describe `evict_value_v1` as a **knowledge-informed decision framework for sequential cache control under uncertainty** without over-claiming beyond committed artifacts.

---

## 1. What “knowledge” means here (artifact-backed)

- **Per-request signals:** Traces are ingested as `Request` objects with optional **metadata** (`bucket`, `confidence`) attached at prediction time (`lafc/evict_value_wulver_v1.py`, `_parse_jsonl_trace`). These are treated as **external predictive context**—the simulator does not invent them; they come from the processed trace / manifest pipeline.
- **Structured cache state:** Features combine **LRU order**, **per-page bucket/confidence**, **predictor vs. LRU scores** over the current candidate set, and **cache-level statistics** (`src/lafc/evict_value_features_v1.py`). That is **knowledge** in the sense of **summarized observable state** plus **hints** about future-related structure (buckets/confidence), not full lookahead.
- **Learned mapping:** Training supervises a model to approximate **counterfactual short-horizon miss counts** (`y_loss`) from that feature vector—turning raw signals into an **eviction-value estimate** used at decision time.

**Supported by repo artifacts today:** offline dataset summaries, `model_comparison_heavy_r1.csv`, `best_config_heavy_r1.json`, method schematic figure.

**Not supported by canonical KBS artifacts today:** multi-trace **end-to-end** ranking of `evict_value_v1` vs. baselines without `policy_comparison_heavy_r1.csv`.

---

## 2. Predictive context at decision time

- Represented as **numeric features**: request and candidate **buckets** and **confidence** values, **predictor/LRU disagreement**, gaps to “best” predictor/LRU victims, and **recent request/hit rates** for the candidate (`EVICT_VALUE_V1_FEATURE_COLUMNS`).
- At runtime, the policy updates **bucket/confidence maps** and **bounded deques** for recent activity (`EvictValueV1Policy` in `src/lafc/policies/evict_value_v1.py`).

This matches a **knowledge-based systems** narrative: decisions use **explicit structured inputs** (metadata + cache geometry + heuristic disagreements) fused by a learned scorer.

---

## 3. Why “online control under uncertainty” is accurate

- **Uncertainty:** The future is not known globally; the method uses a **finite lookahead teacher** (\(H\)) for labels and metadata-derived hints at runtime—not clairvoyant Belady unless a `blind_oracle` baseline is run separately for comparison.
- **Control:** Eviction is a **sequential decision** that changes physical cache state; the policy is a **closed-loop mapping** from state + request features to eviction choice.

---

## 4. What already supports the framing vs. what needs more experiments

| Narrative element | Supported now (repo) | Still thin / needs runs |
|-------------------|----------------------|-------------------------|
| Knowledge-rich feature design + offline supervised target | Yes (`evict_value_features_v1`, Wulver dataset builder) | Stronger ablation isolating feature groups would be new experiments |
| Multi-model offline selection (ridge/RF/HistGB) | Yes (`model_comparison_heavy_r1.csv`) | Alternative model families = new runs |
| End-to-end empirical story on 7 families | Needs `policy_comparison_heavy_r1.csv` | Run `heavy_eval` |
| Guard / fallback quality | Code exists (`evict_value_v1_guarded`) | No canonical `heavy_r1` artifact in builder’s `EVIDENCE_FILES` |
| Robustness across non-heavy drivers | Exploratory paths only | Explicit new campaigns |

---

## 5. Conservative one-line positioning (safe today)

> We formulate eviction as scoring candidates using **prediction-time metadata and cache-state features**, trained offline to match a **short-horizon counterfactual loss**, and evaluate the learned policy in **trace replay** alongside strong baselines—**subject to completing the canonical heavy\_r1 evaluation artifact** for full multi-trace numbers.
