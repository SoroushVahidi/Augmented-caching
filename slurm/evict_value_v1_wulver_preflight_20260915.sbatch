#!/bin/bash
#SBATCH --job-name=lafc-preflight
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/preflight-%j.out
#SBATCH --error=slurm/logs/preflight-%j.err

set -euo pipefail

# Print info
echo "[$(date)] Starting preflight job on $(hostname)"
echo "Source Commit: ceb36705b59d0d16db55d0fc0bfb63a5c7a2a6a1"

# Navigate to repo
cd ~/lafc-work/Augmented-caching

# Set environment
source .venv/bin/activate
export PYTHONPATH=src

# Run dataset builder (note flat path on Wulver: scripts/build_evict_value_dataset_wulver_v1.py)
python scripts/build_evict_value_dataset_wulver_v1.py \
  --trace-glob "data/processed/metacdn/trace.jsonl" \
  --capacities "32" \
  --horizons "16,32,64,128" \
  --max-requests-per-trace 15000 \
  --split-mode trace_chunk \
  --chunk-size 4096 \
  --max-rows-per-shard 500000 \
  --split-seed 7 \
  --out-dir data/derived/evict_value_v1_wulver_preflight_20260915 \
  --overwrite

echo "Preflight dataset built successfully!"

# Run verification and stats printing
python -c "
import pandas as pd
import json
from pathlib import Path

out_dir = Path('data/derived/evict_value_v1_wulver_preflight_20260915')
print('Verifying output contents...')
shards = list(out_dir.glob('**/*.part*.csv'))
print(f'Found {len(shards)} parts.')

for shard in shards:
    df = pd.read_csv(shard)
    print(f'Shard {shard.name}: shape {df.shape}')
    
    # Calculate tie rate per horizon
    for h in (16, 32, 64, 128):
        h_df = df[df['horizon'] == h]
        if len(h_df) == 0:
            continue
            
        dec_groups = h_df.groupby('decision_id')
        total_decs = len(dec_groups)
        tied_decs = 0
        for name, group in dec_groups:
            if group['y_loss'].min() == group['y_loss'].max():
                tied_decs += 1
                
        tie_rate = tied_decs / total_decs if total_decs > 0 else 0.0
        print(f'  H={h}: decisions={total_decs}, tie_rate={tie_rate:.4f}')
"
