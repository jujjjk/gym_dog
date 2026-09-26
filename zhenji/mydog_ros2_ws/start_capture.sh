#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /home/jetson/deploy_staging/b23500-observation-opt-20260923/install/setup.bash
export ROS_DOMAIN_ID=99
export PYTHONPATH=/usr/local/lib/python3.10/site-packages:${PYTHONPATH:-}
case "${1:-dry}" in
  dry) ENABLE_SEND=false; STAND_ONLY=true ;;
  live) ENABLE_SEND=true; STAND_ONLY=false ;;
  *) echo 'Usage: bash start_capture.sh [dry|live] [mount.yaml]' >&2; exit 2 ;;
esac
EXTRA=()
if [ -n "${2:-}" ]; then
  test -f "$2"
  EXTRA=(--params-file "$2")
fi
CAPTURE_DIR="/home/jetson/mydog_ros2_ws/log/B23500_observation_$(date +%Y%m%d_%H%M%S)_$$"
echo "Capture: $CAPTURE_DIR"
exec ros2 run mydog_policy mydog_capture23500_node --ros-args \
  -p enable_send:="$ENABLE_SEND" -p stand_only:="$STAND_ONLY" -p observation_pipeline_enabled:=true -p observation_timing_mode:=common_time -p fast_commands:=true -p continuous_commands:=true \
  -p motor_base_url:=http://127.0.0.1:8000 -p imu_port:=/dev/myimu \
  -p max_motor_age_ms:=80.0 -p max_imu_age_sec:=0.06 -p http_timeout_sec:=0.040 \
  -p max_abs_roll_rad:=0.45 -p max_abs_pitch_rad:=0.45 \
  -p startup_ready_error_rad:=0.12 -p startup_ready_hold_sec:=2.0 \
  -p hardware_torque_limit_nm:=14.0 -p continuous_torque_nm:=6.0 \
  -p thermal_derate_full_rms_nm:=8.0 -p thermal_rms_time_constant_sec:=2.0 \
  -p gyro_bias_calibration_sec:=5.0 -p walk_start_stable_sec:=1.0 \
  -p command_min_vx_mps:=0.001 -p command_max_vx_mps:=0.40 -p command_timeout_sec:=0.35 \
  -p capture_dir:="$CAPTURE_DIR" -p debug_csv_path:="${CAPTURE_DIR}_controller.csv" \
  "${EXTRA[@]}"
