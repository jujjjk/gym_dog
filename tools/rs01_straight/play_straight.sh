#!/usr/bin/env bash
# Play back a trained policy in the Isaac Gym viewer.
#
#   ./tools/rs01_straight/play_straight.sh                    # newest walk run
#   TASK=dog_rs01_straight_stand ./tools/rs01_straight/play_straight.sh
#   LOAD_RUN=Jul25_18-00-00_rs01_straight_walk CHECKPOINT=3000 \
#       ./tools/rs01_straight/play_straight.sh
#
# play.py pins the command for these tasks: 0.0 m/s for the stand task and
# 0.08 m/s for the walk task.
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${GYM_ROOT}/unitree_rl_gym"
# shellcheck disable=SC1091
source "${GYM_ROOT}/unitree-rl/bin/activate"
export PYTHONPATH="${GYM_ROOT}/isaacgym/python:${GYM_ROOT}/unitree_rl_gym"

extra=()
if [[ -n "${LOAD_RUN:-}" ]]; then
    extra+=(--load_run "${LOAD_RUN}")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    extra+=(--checkpoint "${CHECKPOINT}")
fi

python3 legged_gym/scripts/play.py \
    --task="${TASK:-dog_rs01_straight_walk}" \
    "${extra[@]}" \
    "$@"
