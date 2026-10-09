#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_v25_dual.sh --start'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_v25_training_launch.lock
flock -n 9 || { echo 'V25 launcher already running'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_omni_v25_' >/dev/null; then
  echo 'V25 training already running; refusing duplicate launch.'; exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v24_support11/Oct03_22-18-18_v24_gpu1_seed16_20261003_221814
test -f "$RS01_BASE/model_29000.pt"
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_v25/long_$RS01_STAMP
mkdir -p "$RS01_OUT"
RS01_TASKS=(rs01_omni_v25_cadence20 rs01_omni_v25_cadence22)
for RS01_GPU in 0 1; do
  nohup python -u legged_gym/scripts/train.py \
    --task="${RS01_TASKS[$RS01_GPU]}" --num_envs=4096 --max_iterations=3000 \
    --run_name="v25_gpu${RS01_GPU}_seed16_${RS01_STAMP}" --seed=16 --resume \
    --load_run="$RS01_BASE" --checkpoint=29000 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    9>&- </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  echo "GPU $RS01_GPU PID=$! task=${RS01_TASKS[$RS01_GPU]} log=$RS01_OUT/gpu${RS01_GPU}.log"
done
echo 'Ctrl+C exits log viewing only; training continues. No new push/mass randomization.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
