#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/nszb/gym
GYM_ROOT="$ROOT/unitree_rl_gym"
PYTHON_ENV="$ROOT/unitree-rl/bin"
SEED_MODEL="$GYM_ROOT/logs/rough_fanfan_omni_desat_torque/Jul14_11-59-33_symmetric_transition_from_force_coord_5280/model_5530.pt"

GPU_INDEX="${GPU_INDEX:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-900}"

if [[ ! -f "$SEED_MODEL" ]]; then
    echo "missing 5530 seed: $SEED_MODEL" >&2
    exit 2
fi

export PATH="$PYTHON_ENV:$PATH"
export PYTHONPATH="$ROOT/isaacgym/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

cd "$GYM_ROOT"
exec python legged_gym/scripts/train.py \
    --task fanfan_omni_tilt_recovery_5530 \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --headless
