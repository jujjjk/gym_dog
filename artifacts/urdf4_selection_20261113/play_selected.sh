#!/usr/bin/env bash
set -euo pipefail
URDF4_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$URDF4_REPO_ROOT/unitree_rl_gym"
if [[ -f "$URDF4_REPO_ROOT/unitree-rl/bin/activate" ]]; then
    source "$URDF4_REPO_ROOT/unitree-rl/bin/activate"
fi
python legged_gym/scripts/play.py \
    --task urdf4_omni_v1 \
    --load_run "$URDF4_REPO_ROOT/artifacts/urdf4_selection_20261113" \
    --checkpoint 5400 --num_envs 1 --duration_s 120 \
    --vx 0.3 --vy 0 --wz 0 \
    --sim_device cuda:0 --rl_device cuda:0 "$@"
