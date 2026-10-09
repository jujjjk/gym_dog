#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == --start || "${1:-}" == --smoke ]] || { echo 'Usage: bash legged_gym/scripts/train_rs01_amp_scratch_ab.sh --start|--smoke'; exit 1; }
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
exec 9>/tmp/rs01_amp_scratch_ab.lock
flock -n 9 || { echo 'Scratch launcher already active'; exit 1; }
if pgrep -f 'scripts/train.py.*--task=rs01_amp_scratch_' >/dev/null; then
  echo 'Scratch training already running'; exit 1
fi
test -f /home/nszb/gym/artifacts/rs01_reference/amp_scratch_slow_20261007/manifest.json
RS01_ENVS=4096; RS01_ITERS=10000; RS01_MODE=long
if [[ "$1" == --smoke ]]; then RS01_ENVS=256; RS01_ITERS=100; RS01_MODE=smoke; fi
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_amp/scratch_${RS01_MODE}_${RS01_STAMP}
mkdir -p "$RS01_OUT"
RS01_PIDS=()
for RS01_GPU in 0 1; do
  RS01_TASK=rs01_amp_scratch_original
  if [[ "$RS01_GPU" == 1 ]]; then RS01_TASK=rs01_amp_scratch_slow; fi
  # Intentionally NO --resume / --load_run / --checkpoint. Both networks start new.
  nohup env OMP_NUM_THREADS=1 python -u legged_gym/scripts/train.py --task="$RS01_TASK" \
    --num_envs="$RS01_ENVS" --max_iterations="$RS01_ITERS" \
    --run_name="${RS01_MODE}_scratch_gpu${RS01_GPU}_seed16_${RS01_STAMP}" --seed=16 --headless \
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
echo 'Both groups start from scratch. Ctrl+C stops log viewing, not training.'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
