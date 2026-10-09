#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_v30_dual.sh --start'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_v30_training_launch.lock
flock -n 9 || { echo 'V30 launcher already running'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_omni_v30_' >/dev/null; then
  echo 'V30 training already running; refusing duplicate launch.'; exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v29_mince/Oct04_16-52-45_v29_gpu0_seed16_20261004_165242
test -f "$RS01_BASE/model_34000.pt"
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_v30/long_$RS01_STAMP
mkdir -p "$RS01_OUT"
for RS01_GPU in 0 1; do
  RS01_SEED=$((16+RS01_GPU))
  RS01_SRC="$RS01_BASE"
  nohup python -u legged_gym/scripts/train.py \
    --task=rs01_omni_v30_clearance --num_envs=4096 --max_iterations=3000 \
    --run_name="v30_gpu${RS01_GPU}_seed${RS01_SEED}_${RS01_STAMP}" --seed="$RS01_SEED" --resume \
    --load_run="$RS01_SRC" --checkpoint=34000 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    9>&- </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  echo "GPU $RS01_GPU seed=$RS01_SEED PID=$! log=$RS01_OUT/gpu${RS01_GPU}.log"
done
echo 'V30 clearance 35/28mm; both seeds from A34000; unchanged RS01 actuator and command envelope. Ctrl+C stops log viewing, not training.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
