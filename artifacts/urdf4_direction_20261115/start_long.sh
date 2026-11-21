#!/usr/bin/env bash
set -euo pipefail
cd /home/nszb/gym
source unitree-rl/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python -u tools/urdf4_amp/train_cts_direction.py \
  --receipt artifacts/urdf4_direction_20261115/preflight/receipt.json \
  --max_iterations 30000 --num_envs 4096 --seed 1 \
  --headless --sim_device cuda:0 --rl_device cuda:0 "$@"
