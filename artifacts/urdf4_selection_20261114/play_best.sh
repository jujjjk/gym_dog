#!/usr/bin/env bash
set -euo pipefail
URDF4_PREVIEW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$URDF4_PREVIEW_ROOT/unitree-rl/bin/activate"
cd "$URDF4_PREVIEW_ROOT/unitree_rl_gym"
exec python legged_gym/scripts/play.py \
  --task urdf4_amp_cts_v1 --num_envs 1 \
  --load_run "$URDF4_PREVIEW_ROOT/unitree_rl_gym/logs/urdf4_amp_cts_gait_first/Nov13_21-24-21_gait_first_30000" \
  --checkpoint 17800 --sweep --duration_s 600 \
  --sim_device cuda:0 --rl_device cuda:0 "$@"
