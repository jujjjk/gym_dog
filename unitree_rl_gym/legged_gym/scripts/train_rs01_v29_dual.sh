#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_v29_dual.sh --start'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_v29_training_launch.lock
flock -n 9 || { echo 'V29 launcher already running'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_omni_v29_' >/dev/null; then
  echo 'V29 training already running; refusing duplicate launch.'; exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v28_curriculum/Oct04_14-35-37_v28_gpu0_seed16_20261004_143533
test -f "$RS01_BASE/model_32000.pt"
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_v29/long_$RS01_STAMP
mkdir -p "$RS01_OUT"
for RS01_GPU in 0 1; do
  RS01_SEED=$((16+RS01_GPU))
  RS01_SRC=$([ $RS01_GPU -eq 0 ] && echo "$RS01_BASE" \
    || echo /home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v28_curriculum/Oct04_14-35-37_v28_gpu1_seed17_20261004_143533)
  nohup python -u legged_gym/scripts/train.py \
    --task=rs01_omni_v29_mince --num_envs=4096 --max_iterations=3000 \
    --run_name="v29_gpu${RS01_GPU}_seed${RS01_SEED}_${RS01_STAMP}" --seed="$RS01_SEED" --resume \
    --load_run="$RS01_SRC" --checkpoint=32000 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    9>&- </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  echo "GPU $RS01_GPU seed=$RS01_SEED PID=$! log=$RS01_OUT/gpu${RS01_GPU}.log"
done
echo 'Mince cadence 3.0 Hz on V28 Stage0 envelope, no automatic speed expansion. Ctrl+C stops log viewing, not training.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
