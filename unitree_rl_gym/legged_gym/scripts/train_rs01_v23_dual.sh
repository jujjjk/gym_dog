#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_v23_dual.sh --start'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_v23_training_launch.lock
flock -n 9 || { echo 'V23 launcher already running'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_omni_v23_' >/dev/null; then
  echo 'V23 training already running; refusing duplicate launch.'; exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v22_sensor/Sep22_09-57-44_v22_gpu1_seed16_20260922_095740
test -f "$RS01_BASE/model_23500.pt"
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_v23/long_$RS01_STAMP
mkdir -p "$RS01_OUT"
RS01_TASKS=(rs01_omni_v23_balance18 rs01_omni_v23_balance20)
for RS01_GPU in 0 1; do
  nohup python -u legged_gym/scripts/train.py \
    --task="${RS01_TASKS[$RS01_GPU]}" --num_envs=4096 --max_iterations=3000 \
    --run_name="v23_gpu${RS01_GPU}_seed16_${RS01_STAMP}" --seed=16 --resume \
    --load_run="$RS01_BASE" --checkpoint=23500 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    9>&- </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  echo "GPU $RS01_GPU PID=$! task=${RS01_TASKS[$RS01_GPU]} log=$RS01_OUT/gpu${RS01_GPU}.log"
done
echo 'Ctrl+C exits log viewing only; training continues.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
