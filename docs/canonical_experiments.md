# Canonical experiment guide

This document lists the exact command sequences to reproduce the manuscript's primary results and the reviewer-revision experiments.

## 1. Manuscript canonical results (KBS Wulver line)

These results form the core of the **Knowledge-Based Systems** submission. They are tagged as `heavy_r1` to distinguish them from exploratory smoke runs.

### A. Full pipeline (Build → Train → Evaluate)

To reproduce the main quantitative results from scratch:

1. **Ingest data**:
   ```bash
   python scripts/setup/prepare_all.py --dataset all
   ```

2. **Build training dataset**:
   ```bash
   python scripts/experiments/canonical/build_evict_value_dataset_wulver_v1.py --tag heavy_r1
   ```

3. **Train models**:
   ```bash
   python scripts/experiments/canonical/train_evict_value_wulver_v1.py --tag heavy_r1
   ```

4. **Evaluate policies**:
   ```bash
   python scripts/experiments/canonical/run_policy_comparison_wulver_v1.py --tag heavy_r1
   ```

5. **Generate manuscript artifacts**:
   ```bash
   python scripts/experiments/canonical/paper/build_kbs_main_manuscript_artifacts.py
   ```

### B. Artifact verification

After completion, the following artifacts must exist:
- `analysis/manuscript_canonical/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`
- `tables/manuscript/` (CSV and TeX snippets)
- `figures/manuscript/` (PDF and PNG)

## 2. Reviewer-revision experiments

These experiments address specific reviewer concerns and are managed in separate worktrees/branches.

### Concern 1: Cross-family training
Retrains `evict_value_v1` on held-out families to test generalization.
- **Runner**: `scripts/experiments/run_cross_family_heldout_eval.py`
- **Output**: `analysis/reviewer_revision/reviewer_fairness_cross_family_v1/`

### Concern 2: Supervision-objective ablation
Compares different loss functions (eviction loss, arrival loss, etc.).
- **Runner**: `scripts/experiments/run_supervision_objective_ablation.py`
- **Output**: `analysis/reviewer_revision/supervision_objective_ablation_v1/`

### Concern 3: Distribution shift
Evaluates policy robustness under dataset drift.
- **Runner**: `scripts/experiments/resume_distribution_shift.py`
- **Output**: `analysis/reviewer_revision/distribution_shift_ablation_v1/`

### Concern 4: Practical significance
Controlled timing benchmarks on an idle machine.
- **Runner**: `scripts/experiments/run_practical_significance_controlled.py`
- **Output**: `analysis/reviewer_revision/practical_significance_ablation_v1/`

## 3. External baselines (Canonical comparisons)

The primary external baseline is **LRB**. It is run across all canonical traces:

```bash
python scripts/experiments/run_lrb_external_baseline.py \
  --capacities 32,64,128 \
  --out-dir analysis/external_learned_baselines/lrb
```

## 4. Smoke tests (Lightweight verification)

For a quick end-to-end check without the full `heavy_r1` budget:

```bash
python scripts/experiments/exploratory/run_evict_value_v1_first_check.py
```

This runs a small-scale version of the pipeline on the `example_unweighted.json` trace.
