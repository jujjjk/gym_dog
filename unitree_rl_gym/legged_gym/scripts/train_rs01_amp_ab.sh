#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start || "${1:-}" == --smoke ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_amp_ab.sh --start|--smoke'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_amp_ab_launch.lock
flock -n 9 || { echo 'AMP A/B launcher already active'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_amp_' >/dev/null; then
  echo 'AMP training already running'; exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_amp_style/Oct05_14-46-37_amp_gpu0_seed16_20261005_144633
test -f "$RS01_BASE/model_37000.pt"
test -f /home/nszb/gym/artifacts/rs01_reference/amp_retime_20261006/manifest.json
RS01_ENVS=4096; RS01_ITERS=3000; RS01_MODE=long
if [[ "$1" == --smoke ]]; then RS01_ENVS=256; RS01_ITERS=100; RS01_MODE=smoke; fi
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_amp/ab_${RS01_MODE}_${RS01_STAMP}
mkdir -p "$RS01_OUT"
RS01_PIDS=()
for RS01_GPU in 0 1; do
  RS01_TASK=rs01_amp_control
  if [[ "$RS01_GPU" == 1 ]]; then RS01_TASK=rs01_amp_retime; fi
  nohup python -u legged_gym/scripts/train.py --task="$RS01_TASK" \
    --num_envs="$RS01_ENVS" --max_iterations="$RS01_ITERS" \
    --run_name="${RS01_MODE}_gpu${RS01_GPU}_seed16_${RS01_STAMP}" --seed=16 \
    --resume --load_run="$RS01_BASE" --checkpoint=37000 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    9>&- </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  RS01_PIDS+=("$!")
  echo "GPU=$RS01_GPU PID=$! task=$RS01_TASK log=$RS01_OUT/gpu${RS01_GPU}.log"
done
if [[ "$RS01_MODE" == smoke ]]; then
  RS01_STATUS=0
  for RS01_PID in "${RS01_PIDS[@]}"; do wait "$RS01_PID" || RS01_STATUS=1; done
  exit "$RS01_STATUS"
fi
echo 'Experimental A/B, same seed and commands. Ctrl+C stops log viewing only.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
