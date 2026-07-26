#!/usr/bin/env bash
# Stage 2: low-speed straight diagonal walking at 0.03-0.12 m/s, from scratch.
# About 3000 iterations. This is a long run; start it yourself.
#
# Optional warm start from the Stage 1 stand policy:
#   STAND_RUN=Jul25_18-00-00_rs01_straight_stand STAND_CHECKPOINT=600 \
#       ./tools/rs01_straight/train_straight.sh
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${GYM_ROOT}/unitree_rl_gym"
# shellcheck disable=SC1091
source "${GYM_ROOT}/unitree-rl/bin/activate"
export PYTHONPATH="${GYM_ROOT}/isaacgym/python:${GYM_ROOT}/unitree_rl_gym"

extra=()
if [[ -n "${STAND_RUN:-}" ]]; then
    # Both stages share the 52-wide observation and the 12 actions, so the
    # stand actor loads directly into the walk task.
    extra+=(--resume --load_run "${STAND_RUN}")
    if [[ -n "${STAND_CHECKPOINT:-}" ]]; then
        extra+=(--checkpoint "${STAND_CHECKPOINT}")
    fi
fi

python3 legged_gym/scripts/train.py \
    --task=dog_rs01_straight_walk \
    --headless \
    --num_envs="${NUM_ENVS:-4096}" \
    --max_iterations="${MAX_ITERATIONS:-3000}" \
    --run_name="${RUN_NAME:-rs01_straight_walk}" \
    "${extra[@]}" \
    "$@"
