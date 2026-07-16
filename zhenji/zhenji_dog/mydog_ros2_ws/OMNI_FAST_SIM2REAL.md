# Fanfan Omni Yaw-Clean Sim2Real

Target model:

```text
task: fanfan_omni_yaw_drift_clean
config_class: legged_gym.envs.fanfan.fanfan_omni_safe_config:FanfanOmniYawDriftCleanCfg
checkpoint: model_5100.pt
onnx: fanfan_yaw_clean_5100.onnx
sha256: c9c9621c97620100b9f61bb9c508bedad80d7edde3db03d8279218a1e6946cc8
```

The launch entry is still `sim2real_omni_fast.launch.py`, but its defaults now
target the yaw-clean model above.

## Build And Validate

On the robot computer:

```bash
cd /home/jetson/mydog_ros2_ws
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

MODEL=/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_yaw_clean_5100.onnx
ros2 run mydog_policy mydog_validate_omni_yaw_clean "$MODEL"
```

Expected output includes:

```text
OK task=FanfanOmniYawDriftCleanCfg sha256=c9c9621c97620100b9f61bb9c508bedad80d7edde3db03d8279218a1e6946cc8
```

## Dry Run

Use this first with the robot supported and motor sending disabled:

```bash
RUN=$(date +%Y%m%d_%H%M%S)
ros2 launch mydog_policy sim2real_omni_fast.launch.py \
  onnx_path:="$MODEL" \
  enable_send:=false \
  startup_stand_first:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/fanfan_yaw_clean_dry_${RUN}.csv
```

Then send a few preset commands from another terminal:

```bash
source /home/jetson/mydog_ros2_ws/install/setup.bash
ros2 run mydog_policy mydog_omni_yaw_clean_command stand --duration 2
ros2 run mydog_policy mydog_omni_yaw_clean_command forward_slow --duration 3
ros2 run mydog_policy mydog_omni_yaw_clean_command backward_slow --duration 3
ros2 run mydog_policy mydog_omni_yaw_clean_command left_slow --duration 3
ros2 run mydog_policy mydog_omni_yaw_clean_command turn_left_slow --duration 3
```

Check that all feedback is fresh, the loop is near 50 Hz, and the CSV is being
written.

## First Motor Send

Initial safety defaults:

```text
torque limit: 8 Nm
vx limit: +/-0.12 m/s
vy limit: +/-0.025 m/s
yaw limit: +/-0.25 rad/s
use_model_pd_gains: true
model_kp_scale: 1.0
model_kd_scale: 1.0
startup_stand_first: true
debug_csv: enabled
zero command stand protection: enabled
policy action command gate: enabled
policy action/gait full ratio: 0.5
policy action/gait max gate scale: 0.9
deployment gait phase period scale: 1.35
reset gait phase on command start: enabled
```

Start with the robot supported and a hardware stop ready:

```bash
RUN=$(date +%Y%m%d_%H%M%S)
ros2 launch mydog_policy sim2real_omni_fast.launch.py \
  onnx_path:="$MODEL" \
  enable_send:=true \
  startup_stand_first:=true \
  motor_torque_limit_nm:=8.0 \
  model_kp_scale:=1.0 \
  model_kd_scale:=1.0 \
  deployment_gait_phase_period_scale:=1.35 \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/fanfan_yaw_clean_air_${RUN}.csv
```

Do not publish walking commands until the log shows:

```text
[STARTUP_STAND][READY] ONNX default pose is stable.
```

Confirm the active budget parameters:

```bash
ros2 param get /mydog_policy_node motor_torque_limit_nm
ros2 param get /mydog_policy_node torque_limit_nm
ros2 param get /mydog_policy_node torque_safety_budget_nm
ros2 param get /mydog_policy_node expected_active_torque_budget_nm
ros2 param get /mydog_policy_node use_model_pd_gains
ros2 param get /mydog_policy_node debug_csv_path
```

The four torque budget values should all be `8.0`, `use_model_pd_gains` should
be `true`, and `debug_csv_path` must be non-empty.

Zero command protection keeps the robot at the ONNX default pose instead of
running `pure_rl` when:

```text
abs(cmd_x) < 0.01
abs(cmd_y) < 0.01
abs(cmd_wz) < 0.03
```

Small non-zero commands are gated by command size. For example, with
`vx=0.06` and the initial `vx` envelope of `0.12`, policy action and the ONNX
gait reference now reach the walking gate cap. The grounded test cap is `0.9`,
so `vx=0.12` is softened slightly but still has enough authority to move.
When a command starts from stand, the gait phase is reset so the swing pattern
does not begin from an arbitrary wall-clock phase. The gait phase period is
scaled from the model's `0.54 s` to about `0.73 s`, lowering the real-robot
step frequency from about `1.85 Hz` to `1.37 Hz`. The CSV appends:

```text
action_cmd_gate_scale
zero_cmd_stand_active
```

For the next test, `cmd=0` rows should still have `action_used` near zero,
`gait_ref_policy` near zero, and `torque_limited` close to zero. During walking,
if it still cannot move, inspect calf `final_limited_joint_mask`, measured
torque, and foot contact before raising the torque budget.

## Initial Presets

Run the slow presets first:

```bash
ros2 run mydog_policy mydog_omni_yaw_clean_command forward_slow --duration 3
ros2 run mydog_policy mydog_omni_yaw_clean_command backward_slow --duration 2
ros2 run mydog_policy mydog_omni_yaw_clean_command left_slow --duration 2
ros2 run mydog_policy mydog_omni_yaw_clean_command right_slow --duration 2
ros2 run mydog_policy mydog_omni_yaw_clean_command turn_left_slow --duration 2
ros2 run mydog_policy mydog_omni_yaw_clean_command turn_right_slow --duration 2
```

Stop immediately on wrong joint direction, stale feedback, high-frequency shake,
continuous limiting, fast temperature rise, or measured torque staying near the
8 Nm budget.
