#!/usr/bin/env bash
# Stage 1: flat-ground standing, from scratch. About 600 iterations.
# This is a long run. Start it yourself; nothing here is automatic.
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${GYM_ROOT}/unitree_rl_gym"
# shellcheck disable=SC1091
source "${GYM_ROOT}/unitree-rl/bin/activate"
export PYTHONPATH="${GYM_ROOT}/isaacgym/python:${GYM_ROOT}/unitree_rl_gym"

python3 legged_gym/scripts/train.py \
    --task=dog_rs01_straight_stand \
    --headless \
    --num_envs="${NUM_ENVS:-4096}" \
    --max_iterations="${MAX_ITERATIONS:-600}" \
    --run_name="${RUN_NAME:-rs01_straight_stand}" \
    "$@"
