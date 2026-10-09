#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_amp_dual.sh --start'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_amp_training_launch.lock
flock -n 9 || { echo 'AMP launcher already active'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_amp_style' >/dev/null; then
  echo 'AMP training already running'; exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v29_mince/Oct04_16-52-45_v29_gpu0_seed16_20261004_165242
test -f "$RS01_BASE/model_34000.pt"
test -f /home/nszb/gym/artifacts/rs01_reference/design_10s_20261005/manifest.json
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_amp/long_$RS01_STAMP
mkdir -p "$RS01_OUT"
for RS01_GPU in 0 1; do
  RS01_SEED=$((16+RS01_GPU))
  nohup python -u legged_gym/scripts/train.py --task=rs01_amp_style \
    --num_envs=4096 --max_iterations=3000 \
    --run_name="amp_gpu${RS01_GPU}_seed${RS01_SEED}_${RS01_STAMP}" --seed="$RS01_SEED" \
    --resume --load_run="$RS01_BASE" --checkpoint=34000 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    9>&- </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  echo "GPU=$RS01_GPU PID=$! log=$RS01_OUT/gpu${RS01_GPU}.log"
done
echo 'Experimental kinematic AMP style; real RS01 plant. Ctrl+C stops log viewing only.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
