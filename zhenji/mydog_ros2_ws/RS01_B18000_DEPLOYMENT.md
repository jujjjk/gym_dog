# B18000 guarded Sim2Real candidate

## Status and base revision

Latest upstream main `eb8f249` pulled by fast-forward, including the operator's
50Hz hardware-loop fix `5544692`. Existing `11.md` and old models remain untouched.
This package is adapted and offline-validated; NO hardware was commanded and physical
Sim2Real acceptance is still pending. Do not treat this as untethered walking approval.

Policy: V20 B/model18000,61observations/12actions,50Hz.
ONNX `src/mydog_policy/resource/B18000.onnx`, SHA256:
`1dd9e9e337fcba95a6f4fa1126894c36675af08933b5bdaf0d552835a6de4361`.
Source checkpoint SHA256 `0d6c90c84742c53d679cefa73fb3f409d5254b0fba8280313f59bb0fd71c8b57`.
Actual RS01 mass11.7317kg; thigh180mm/calf202.158mm; actual URDF hash
`48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4`.

## Deployment parity and remaining physical gaps

- Dedicated B contract rejects old A6850 and hash overrides.
- Training-matched sensor-only61D observation, previous clipped action, phase and heading.
- Conditional8deg inward hip target mapping BEFORE existing rate/acceleration limiter.
- Guarded sent target does not overwrite actor limiter history.
- Reuse upstream PD40/1,14Nm operating protection,6Nm continuous reference,
  thermal RMS full8Nm/tau2s, update next tick from max(safe demand, measured torque).
- No software imitation of the actuator's physical delay/FOPDT on real hardware.
- Reuse fixed asynchronous control/send/telemetry threads and shared exclusive lock.
- B arm requires40samples at approximately50Hz for control and successful sends:
  median18–22ms,p95<=30ms,range8–40ms and last successful send<=40ms old.
  Loss while walking disarms and uses inherited soft return to stand; no automatic re-arm.
  These are initial timing-test bounds, not certified hardware safety tolerances.
- Default dry-run and stand-only, explicit operator arm, existing command deadman and limits retained.
- Reception ages/counters do NOT establish acquisition-clock synchronization.
  Existing upstream motor/IMU transport envelope remains250ms; the timestamped IMU wrapper
  separately checks its actual frame freshness. Site measurements remain essential.
- Upstream hold-enabled behavior retained: **Ctrl+C, disarm or software fault is NOT motor
  disable/emergency power-off**. Have an independent physical emergency stop and tether.

## Local evidence

54offline tests passed (B contract/mapping/history/timing gate and upstream guarded controller
regressions). ROS constructor tests mocked all devices; missing vendor YbImuLib was replaced
with a test-only constructor that raises if used, not installed as a runtime substitute.
125s/6250same-state MuJoCo replay: observation/action/limited target max error0;
guarded target4.34e-7rad, thermal active limit2.04e-5Nm (float precision).
Previously4initial-phase125s MuJoCo sequences completed without falls/overspeed/observed flight.
No Jetson benchmark, real sensor synchronization or ground walking validation yet.
ROS colcon build/install passed in an isolated local build directory, including B ONNX/JSON
and the new entry points.1000synthetic-input ticks finite; local CPU core P95 approximately
0.47ms, not a Jetson benchmark or sensor/HTTP loop-frequency measurement.

An overlay archive is provided at `artifacts/rs01_b18000_deployment_20260914.tar.gz`.
It assumes the upstream base above. Inspect its paths and back up affected files first;
do not extract over unknown newer/local Jetson edits. It contains model and code only,
not vendor IMU libraries, test stubs, device credentials or a hardware auto-start script.

## Install (on Jetson, operator controlled)

Use the existing ROS Humble workspace containing this updated package. Preserve its vendor IMU
library, motor server and existing calibration; do not overwrite them with test stubs.
The package requires numpy,onnxruntime,requests,pyserial and the real YbImuLib in the ROS Python.
Do not use the IsaacGym Python3.8 environment for ROS Humble's Python3.10.

```bash
source /opt/ros/humble/setup.bash
# cd to the updated mydog_ros2_ws on Jetson
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash
ros2 run mydog_policy mydog_validate_model18000 \
  "$(ros2 pkg prefix mydog_policy)/share/mydog_policy/models/B18000.onnx"
```

1. Read feedback only (no motor sends):

```bash
ros2 launch mydog_policy rs01_model18000.launch.py \
  enable_send:=false stand_only:=true debug_csv_path:=/tmp/b18000_dry.csv
```

2. With tether, emergency stop and area clear, operator may explicitly start stand-only:

```bash
ros2 launch mydog_policy rs01_model18000.launch.py \
  enable_send:=true stand_only:=true debug_csv_path:=/tmp/b18000_stand.csv
```

3. Only after stand/calibration/feedback and actual timing are accepted, launch walk-capable mode
(still not automatically armed):

```bash
ros2 launch mydog_policy rs01_model18000.launch.py \
  enable_send:=true stand_only:=false debug_csv_path:=/tmp/b18000_trial.csv
```

Second terminal (source ROS/workspace first):

```bash
ros2 topic echo /mydog/model18000/status
# Explicit3s march trial, then automatic disarm/soft stand:
ros2 run mydog_policy mydog_model18000_command --march --seconds 3
# After inspecting the march log, a separate low-speed forward trial:
ros2 run mydog_policy mydog_model18000_command --vx 0.10 --seconds 3
# Explicit disarm, does not disable motor power:
ros2 service call /mydog/model18000/arm std_srvs/srv/SetBool '{data: false}'
```

Do not start multiple controller instances. Old A6850 and new B share the same lock.
Do not start all-direction trials immediately: verify timestamp progress,loop/send frequency,
posture,raw/guarded demand,temperature and operator-controlled stop behavior first.
Send trial CSV plus `.status.jsonl` back for actual Sim2Real evaluation.

## Rollback

Stop issuing B commands, disarm, and follow the site's physical power/stop procedure.
Old `rs01_model6850.launch.py`, policy/hash and its behavior remain available.
Never relabel B ONNX as `stand_only_6850.onnx` or launch it with the A core.
