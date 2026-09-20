# B18000 supported standing and explicit IMU calibration

The 2026-09-19 revision starts at measured joint positions, ramps to stand,
then requires the operator to support the robot with all feet bearing weight
before requesting gyro bias calibration. It does not redefine tilted posture
as level. Existing roll/pitch, torque/current, feedback and timing limits remain.

The old controller must be stopped with the robot physically supported before
starting this workspace. Ctrl+C/disarm does not disable motor power. No motor
startup or motion was performed during deployment verification.

For the isolated Jetson installation, in each NEW shell:

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/deploy_staging/b18000-calibration-v2-20260919/install/setup.bash
export ROS_DOMAIN_ID=99
ros2 pkg prefix mydog_policy
```

The prefix must refer to the isolated workspace above. Start only one controller:

```bash
mkdir -p /home/jetson/mydog_ros2_ws/log/manual_trials
ros2 launch mydog_policy rs01_model18000.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/manual_trials/b18000_fixed_$(date +%Y%m%d_%H%M%S).csv
```

This enables motors and ramps to stand, but cannot authorize walking before
calibration. With supported, stable standing and `mode=ready`, request:

```bash
ros2 service call /mydog/model18000/calibrate_imu std_srvs/srv/SetBool '{data: true}'
ros2 topic echo /mydog/model18000/status std_msgs/msg/String --field data
```

Hold still for five continuous seconds. Motion, frame gaps or feedback errors
reset sample collection. Require `calibration_state=complete`, `imu_calibrated=true`,
`mode=ready`, `walk_start_stable=true`, and `timing_ready=true` before a trial:

```bash
ros2 run mydog_policy mydog_model18000_command --march --seconds 3
```

High-rate sends now use the existing `/tmp/lingzu_motor_rt.sock` protocol, with
verified torque/current flags and feedback in the response. Initialization still
uses the verified HTTP configuration/enable handshake. There is no automatic
HTTP fallback or replay of a failed command. Socket preflight is read-only.
`rt_spi_ms`, `rt_server_ms`, and `send_dt_ms` distinguish server/device work from
client roundtrip cost. The 50Hz entry check is repeated during the ready-to-walk
gate, not only when arming. Hardware send frequency remains unvalidated until
an operator-controlled trial; offline tests and read-only timings are insufficient.

Calibration v2 checks the entire five-second joint-position span (maximum 0.02 rad per joint),
not instantaneous velocity telemetry. Stand-position error, level posture, gyro-motion,
IMU freshness, orientation-span and gyro variance/bias checks remain active. A failed
window is discarded. Status exposes calibration_reason and calibration_joint_span_rad.
Recorded encoder q/dq replay was validated with synthetic stationary IMU; it is not a
physical IMU calibration or walking acceptance test.
