#!/usr/bin/env bash
# Export a trained policy to ONNX plus a deployment contract JSON.
# Needs no GPU and no Isaac Gym: the actor is rebuilt from the checkpoint.
#
#   ./tools/rs01_straight/export_onnx.sh                      # newest walk run
#   LOAD_RUN=Jul25_18-00-00_rs01_straight_walk CHECKPOINT=3000 \
#       ./tools/rs01_straight/export_onnx.sh
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${GYM_ROOT}/unitree_rl_gym"
# shellcheck disable=SC1091
source "${GYM_ROOT}/unitree-rl/bin/activate"
export PYTHONPATH="${GYM_ROOT}/unitree_rl_gym:${GYM_ROOT}/rsl_rl"

extra=()
if [[ -n "${LOAD_RUN:-}" ]]; then
    extra+=(--load_run "${LOAD_RUN}")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    extra+=(--checkpoint "${CHECKPOINT}")
fi

python3 legged_gym/scripts/export_dog_onnx.py \
    --task="${TASK:-dog_rs01_straight_walk}" \
    "${extra[@]}" \
    "$@"
