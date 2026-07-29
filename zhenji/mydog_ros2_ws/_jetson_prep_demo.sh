#!/usr/bin/env bash
set -e
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "=== production entry ==="
python3 - <<'PY'
from pathlib import Path
p = Path('src/mydog_policy/launch/sim2real_realdata.launch.py')
print(p.read_text(encoding='utf-8'))
PY

echo "=== validate saturation_recovery model ==="
ros2 run mydog_policy mydog_validate_symmetric_transition_model \
  "$(ros2 pkg prefix mydog_policy)/share/mydog_policy/models/fanfan_saturation_recovery_best.onnx" \
  2>/dev/null || true

# Prefer package-specific validator if present
if ros2 pkg executables mydog_policy | grep -q mydog_validate; then
  ros2 pkg executables mydog_policy | grep -E 'validate|saturation|hardware|realdata' || true
fi

python3 - <<'PY'
import hashlib
from pathlib import Path
files = [
    'src/mydog_policy/resource/fanfan_saturation_recovery_best.onnx',
    'src/mydog_policy/resource/fanfan_hardware_balance_5530_best.onnx',
    'src/mydog_policy/resource/fanfan_symmetric_transition_5530.onnx',
]
for rel in files:
    p = Path('/home/jetson/mydog_ros2_ws') / rel
    if p.exists():
        print(p.name, hashlib.sha256(p.read_bytes()).hexdigest())
    else:
        print(p.name, 'MISSING')
PY

echo "=== check launch/executables ==="
ros2 pkg executables mydog_policy | grep -E 'saturation|hardware|realdata|tilt' || true
test -f /home/jetson/mydog_ros2_ws/demo_ladder_cmd.sh
chmod +x /home/jetson/mydog_ros2_ws/demo_ladder_cmd.sh
test -f /home/jetson/mydog_ros2_ws/DEMO_AFTERNOON.md
echo "DEMO_FILES_OK"
echo "=== motor service ==="
curl -s -m 2 http://127.0.0.1:8000/api/state >/dev/null && echo MOTOR_API_UP || echo MOTOR_API_DOWN
echo "=== PREP_DONE ==="
