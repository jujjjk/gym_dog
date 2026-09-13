#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--start" ]]; then
  echo '手动启动两卡长训：bash legged_gym/scripts/train_rs01_v19_dual.sh --start'
  exit 1
fi
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
if pgrep -f 'scripts/train.py.*--task=rs01_omni_v19_' >/dev/null; then
  echo '已有 V19 训练进程，请勿重复启动。'
  exit 1
fi
RS01_BASE=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v18_balance18/Sep12_22-22-32_v18_A_clearance18_seed16_20260912_222228
test -f "$RS01_BASE/model_12000.pt"
RS01_STAMP=$(date +%Y%m%d_%H%M%S)
RS01_OUT=/home/nszb/gym/artifacts/rs01_v19_placement/long_$RS01_STAMP
mkdir -p "$RS01_OUT"
RS01_TASKS=(rs01_omni_v19_placement_soft rs01_omni_v19_placement_soft)
for RS01_GPU in 0 1; do
  RS01_SEED=$((16 + RS01_GPU))
  nohup python -u legged_gym/scripts/train.py \
    --task="${RS01_TASKS[$RS01_GPU]}" --num_envs=4096 --max_iterations=3000 \
    --run_name="v19_gpu${RS01_GPU}_seed${RS01_SEED}_${RS01_STAMP}" --seed="$RS01_SEED" --resume \
    --load_run="$RS01_BASE" --checkpoint=12000 --headless \
    --sim_device="cuda:$RS01_GPU" --rl_device="cuda:$RS01_GPU" \
    </dev/null >"$RS01_OUT/gpu${RS01_GPU}.log" 2>&1 &
  echo "GPU $RS01_GPU PID=$! task=${RS01_TASKS[$RS01_GPU]} log=$RS01_OUT/gpu${RS01_GPU}.log"
done
echo '以下显示实时日志；Ctrl+C只退出日志查看，不停止后台训练。'
tail -F "$RS01_OUT/gpu0.log" "$RS01_OUT/gpu1.log"
