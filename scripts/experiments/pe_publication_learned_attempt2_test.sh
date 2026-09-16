#!/usr/bin/env bash
set -euo pipefail

WORKTREE="/home/soroush/projects/augmented-caching/worktrees/pe-publication-learned-retrain-attempt2-20260915"
OUT_DIR="$WORKTREE/analysis/pe_publication_learned_retrain_attempt2_20260915"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$WORKTREE"
export PYTHONPATH=".:src"
export OMP_NUM_THREADS=1
export PE_ATTEMPT2_TMUX_SESSION="${PE_ATTEMPT2_TMUX_SESSION:-pe-learned-attempt2-test}"

{
  echo "TEST_SCRIPT_START=$(date -Iseconds)"
  echo "WORKTREE=$WORKTREE"
  echo "BRANCH=$(git branch --show-current)"
  echo "HEAD=$(git rev-parse HEAD)"
  echo "PYTHON=$(command -v python)"
  python --version
  python scripts/experiments/pe_publication_learned_attempt2.py test
  echo "TEST_SCRIPT_END=$(date -Iseconds)"
} >> "$LOG_DIR/test.log" 2>&1
