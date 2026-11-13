#!/usr/bin/env bash
# Launch urdf4_omni_v1 training in the background and follow its log.
#
#   bash legged_gym/scripts/train_urdf4_omni.sh --start            # GPU 0, seed 1, 5000 iterations
#   URDF4_GPU=1 URDF4_SEED=2 bash legged_gym/scripts/train_urdf4_omni.sh --start
#   URDF4_ITERATIONS=8000 bash legged_gym/scripts/train_urdf4_omni.sh --start
#
# Ctrl+C stops following the log, not the training.  Stop training with
#   pkill -f 'scripts/train.py.*--task=urdf4_omni_v1'
set -euo pipefail
[[ "${1:-}" == --start ]] || { sed -n '2,9p' "$0"; exit 1; }

URDF4_GPU=${URDF4_GPU:-0}
URDF4_SEED=${URDF4_SEED:-1}
URDF4_ITERATIONS=${URDF4_ITERATIONS:-5000}

cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
if pgrep -f "scripts/train.py.*--task=urdf4_omni_v1.*--sim_device=cuda:${URDF4_GPU}" >/dev/null; then
  echo "urdf4_omni_v1 is already training on GPU ${URDF4_GPU}; refusing a duplicate launch."; exit 1
fi

# Asset contract and nominal stand, then finite randomized rollout.
python legged_gym/scripts/check_urdf4_omni.py --headless \
  --sim_device="cuda:${URDF4_GPU}" --rl_device="cuda:${URDF4_GPU}" | grep -E '^\[(asset|stand|ok)\]|Error'
python legged_gym/scripts/check_urdf4_omni.py --headless --run_name=randomized \
  --sim_device="cuda:${URDF4_GPU}" --rl_device="cuda:${URDF4_GPU}" | grep -E '^\[(rollout|ok)\]|Error'

URDF4_STAMP=$(date +%Y%m%d_%H%M%S)
URDF4_OUT=/home/nszb/gym/artifacts/urdf4_omni_v1
mkdir -p "$URDF4_OUT"
URDF4_LOG="$URDF4_OUT/train_gpu${URDF4_GPU}_seed${URDF4_SEED}_${URDF4_STAMP}.log"
nohup python -u legged_gym/scripts/train.py \
  --task=urdf4_omni_v1 --headless --num_envs=4096 \
  --max_iterations="$URDF4_ITERATIONS" --seed="$URDF4_SEED" \
  --run_name="gpu${URDF4_GPU}_seed${URDF4_SEED}" \
  --sim_device="cuda:${URDF4_GPU}" --rl_device="cuda:${URDF4_GPU}" \
  </dev/null >"$URDF4_LOG" 2>&1 &
echo "PID=$!  log=$URDF4_LOG"
echo "TensorBoard: tensorboard --logdir /home/nszb/gym/unitree_rl_gym/logs/urdf4_omni_v1"
tail -F "$URDF4_LOG"
