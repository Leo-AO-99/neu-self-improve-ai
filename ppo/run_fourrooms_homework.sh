#!/usr/bin/env bash
set -euo pipefail

# Homework-focused default run (best config among all-three runs so far).
UPDATES="${1:-250}"
EPISODES_PER_UPDATE="${2:-8}"
SEED="${3:-42}"

python "ppo/poly_ppo.py" \
  --algo all \
  --env-id "MiniGrid-FourRooms-v0" \
  --updates "${UPDATES}" \
  --episodes-per-update "${EPISODES_PER_UPDATE}" \
  --seed "${SEED}" \
  --lr 1e-4 \
  --ppo-epochs 4 \
  --pretrain-rollouts 800 \
  --pretrain-epochs 8 \
  --shaping-coef 0.0 \
  --output "ppo/results_fourrooms_enhanced.json"

python "ppo/summarize_results.py" --input "ppo/results_fourrooms_enhanced.json"
