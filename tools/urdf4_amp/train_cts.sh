#!/usr/bin/env bash
# Foreground launch; nothing starts until this script is explicitly executed.
set -euo pipefail
URDF4_CTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$URDF4_CTS_ROOT/unitree-rl/bin/activate"
cd "$URDF4_CTS_ROOT/unitree_rl_gym"
exec python legged_gym/scripts/train.py \
  --task urdf4_amp_cts_v1 \
  --num_envs 4096 --max_iterations 8000 \
  --run_name v2_style_cycle_seed1 --seed 1 \
  --headless --sim_device cuda:0 --rl_device cuda:0 "$@"
