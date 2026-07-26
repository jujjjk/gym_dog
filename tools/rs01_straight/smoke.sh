#!/usr/bin/env bash
# 5-iteration smoke test for both stages. Confirms the training loop runs and
# that the reward signal is not clipped flat to zero. Deletes its own logs.
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${GYM_ROOT}/unitree_rl_gym"
# shellcheck disable=SC1091
source "${GYM_ROOT}/unitree-rl/bin/activate"
export PYTHONPATH="${GYM_ROOT}/isaacgym/python:${GYM_ROOT}/unitree_rl_gym"

for task in dog_rs01_straight_stand dog_rs01_straight_walk; do
    echo "=== ${task} ==="
    python3 legged_gym/scripts/train.py \
        --task="${task}" \
        --headless \
        --num_envs=256 \
        --max_iterations=5 \
        --run_name=smoke 2>&1 |
        grep -E "Mean reward|Mean episode length|Learning iteration"
done

find "${GYM_ROOT}/unitree_rl_gym/logs/rs01_straight" -maxdepth 1 -name '*_smoke' \
    -type d -exec rm -rf {} + 2>/dev/null || true
echo "Smoke logs removed."
