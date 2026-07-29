#!/usr/bin/env bash
# Afternoon demo velocity ladder for symmetric-transition 5530 / realdata entry.
# Usage (after policy launch is READY):
#   bash /home/jetson/mydog_ros2_ws/demo_ladder_cmd.sh
# Env:
#   DEMO_HOLD_SEC=8   seconds per step (default 8)
#   DEMO_MAX_STEP=2   0=zero only, 1=+0.08, 2=+0.12, 3=+0.18
#   DEMO_RATE_HZ=50   cmd_vel publish rate

set -e
source /opt/ros/humble/setup.bash
source /home/jetson/mydog_ros2_ws/install/setup.bash

HOLD="${DEMO_HOLD_SEC:-8}"
MAX_STEP="${DEMO_MAX_STEP:-2}"
RATE="${DEMO_RATE_HZ:-50}"

# Use --times instead of timeout(1). Killing ros2 topic pub with SIGTERM
# often prints: "publisher's context is invalid" even when publish succeeded.
times_for_hold() {
  local sec="$1"
  python3 - <<PY
import math
print(max(1, int(math.ceil(float("${sec}") * float("${RATE}")))))
PY
}

pub_cmd() {
  local vx="$1"
  local label="$2"
  local sec="${3:-$HOLD}"
  local n
  n="$(times_for_hold "${sec}")"
  echo ""
  echo "=== [${label}] vx=${vx} for ${sec}s (~${n} msgs @ ${RATE}Hz) ==="
  ros2 topic pub -r "${RATE}" -t "${n}" -p 50 \
    /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: ${vx}, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
}

echo "Demo ladder: hold=${HOLD}s max_step=${MAX_STEP} rate=${RATE}Hz"
echo "Policy launch must already be running and past STARTUP_STAND READY."

pub_cmd 0.00 "step0_zero"
if [ "${MAX_STEP}" -ge 1 ]; then
  pub_cmd 0.08 "step1_creep"
fi
if [ "${MAX_STEP}" -ge 2 ]; then
  pub_cmd 0.12 "step2_demo"
fi
if [ "${MAX_STEP}" -ge 3 ]; then
  echo "WARNING: step3 is optional; only if 0.12 looked calm."
  pub_cmd 0.18 "step3_optional"
fi

echo ""
echo "=== returning to zero for 3s ==="
pub_cmd 0.00 "step_end_zero" 3
echo "DONE. Policy will hold stand after cmd timeout if no more /cmd_vel."
