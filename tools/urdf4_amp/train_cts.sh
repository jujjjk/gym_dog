#!/usr/bin/env bash
set -euo pipefail
URDF4_CTS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$URDF4_CTS_ROOT/unitree-rl/bin/activate"
URDF4_CTS_CHECKPOINT="$(python - "$URDF4_CTS_ROOT" <<'PY'
import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]);path=root/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/behavior_acceptance.json'
if not path.exists():raise SystemExit('行为验收尚未通过，拒绝启动长训练。先运行 evaluate_cts_behavior.py --approve。')
r=json.loads(path.read_text());checkpoint=Path(r['checkpoint'])
if not r.get('approved') or not checkpoint.exists() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=r['checkpoint_sha256']:
 raise SystemExit('行为验收回执或模型不匹配，拒绝启动长训练。')
print(checkpoint)
PY
)"
URDF4_CTS_RUN="$(dirname -- "$URDF4_CTS_CHECKPOINT")"
URDF4_CTS_ITER="$(basename -- "$URDF4_CTS_CHECKPOINT" .pt)"
URDF4_CTS_ITER="${URDF4_CTS_ITER#model_}"
cd "$URDF4_CTS_ROOT/unitree_rl_gym"
exec python legged_gym/scripts/train.py \
  --task urdf4_amp_cts_v1 --num_envs 4096 --max_iterations 8000 \
  --run_name v3_rsi_amp_seed1 --seed 1 --resume \
  --load_run "$URDF4_CTS_RUN" --checkpoint "$URDF4_CTS_ITER" \
  --headless --sim_device cuda:0 --rl_device cuda:0 "$@"
