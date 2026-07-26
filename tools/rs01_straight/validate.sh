#!/usr/bin/env bash
# Stage 0: everything that must pass before training is worth starting.
# Safe to run repeatedly; it never writes to logs/.
set -euo pipefail

GYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${GYM_ROOT}/unitree_rl_gym"
# shellcheck disable=SC1091
source "${GYM_ROOT}/unitree-rl/bin/activate"
export PYTHONPATH="${GYM_ROOT}/isaacgym/python:${GYM_ROOT}/unitree_rl_gym"

echo "=== 1/3 static URDF and joint semantics audit (no GPU needed) ==="
python3 legged_gym/scripts/audit_dog_joints.py --task dog_rs01_straight_walk

echo
echo "=== 2/3 Isaac Gym runtime validation: stand ==="
python3 legged_gym/scripts/validate_dog_straight.py \
    --task dog_rs01_straight_stand --headless

echo
echo "=== 3/3 Isaac Gym runtime validation: straight walk ==="
python3 legged_gym/scripts/validate_dog_straight.py \
    --task dog_rs01_straight_walk --headless

echo
echo "Reports written to ${GYM_ROOT}/artifacts/rs01_straight/"
