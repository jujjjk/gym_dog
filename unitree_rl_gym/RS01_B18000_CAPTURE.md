# B18000 simulation observation capture

Implements the user's 61D observation checklist using the original V20 B18000 checkpoint in Isaac Gym. No environment, reward, model or actuator behavior changes. Runs one robot with nominal physics, no observation noise, no pushes or random commands. Task/weight fixed to the B18000-compatible V20B contract. Current base commit `5779387`, local V21 and unrelated edits preserved.

## Start

Every 5 s change movement with march between movements, visible playback and automatic CSV capture:

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python legged_gym/scripts/collect_rs01_b18000.py \
  --case sequence --seconds 5 --seed 20260925 \
  --sim_device cuda:0 --rl_device cuda:0
```

Fixed forward 0.15 m/s, 30 s, three captures:

```bash
python legged_gym/scripts/collect_rs01_b18000.py \
  --case forward --seconds 30 --repeats 3 --seed 20260925 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

Full suite (march, stand, forward/backward ±0.15 m/s, lateral ±0.10 m/s, yaw ±0.20 rad/s, combined 0.20/0.08/0.25), each 30 s × three repetitions:

```bash
python legged_gym/scripts/collect_rs01_b18000.py \
  --case suite --seconds 30 --repeats 3 --seed 20260925 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

`suite` resets posture per segment; ordinary fixed-case repeats and `sequence` are continuous. Thermal guard history is deliberately preserved across resets. For independent cold starts, launch separate processes. `stand` explicitly disables gait; zero-speed `march` keeps stepping. Sequence contains no static stand. Headless skips wall-clock pacing only, not physics steps or policy ticks. Ctrl+C closes the current CSV and marks summary failed/interrupted. Any environment reset or nonfinite observation/action stops collection; does not silently stitch reset trajectories together.

## Output and timing

Default output root `/home/nszb/gym/artifacts/rs01_b18000_capture/`; unique timestamp subdirectory per invocation, one CSV per segment/repetition, `metadata.json`, `summary.json`. 50 Hz, one row per policy step; no file overwritten. Anonymous twelve-channel fields use FL/FR/RL/RR × hip/thigh/calf, **not the ordering of examples in the checklist**. Also provides explicit leg/joint column names. Metadata records checkpoint/hash/config/observation layout and coordinate frames.

Each row pairs pre-step sensors and the EXACT 61D input used by the actor with its raw/clipped actions and resulting targets. Targets are tapped at the first torque evaluation, before terminal reset can erase them. The collector never calls the estimator or limiter a second time. `limited_target_q` is rate/acceleration-limited; `target_q` is the final thermal/torque-guarded SENT target. `target_rate` belongs to the pre-guard limiter, not the derivative of guard-projected targets. Feedback torque columns are the previous interval's final substep, not the future response to this row's action. `done_after_step` belongs to the following 20 ms interval. Initial/reset rows have `sensor_sample_valid=0` until the estimator has actually updated.

Synchronous simulation IMU/motor timestamps share simulation time; age/skew=0 is not evidence of real-robot synchronization. Raw 100 Hz independent IMU is not available in this model. Motor current, degrees-C temperature and hardware error codes are blank. Electromagnetic/applied torques and guard RMS proxy are provided; guard RMS is NOT temperature. Sensor gyro/orientation are simulator-derived signals, not a calibrated hardware IMU.

Ground truth includes world XYZ and world/body velocity separately. `est_vz` is intentionally zero in the existing estimator: use `gt_world_vz` and `gt_z` to study bounce. Estimated stance is distinct from physical force-based contact. Per-leg residual is the shared diagonal-pair planar disagreement in this strict-pair estimator. `foot_height` is FK body-frame Z; actual flat-ground clearance is `gt_foot_clearance`. All positions m, angles rad, speeds m/s or rad/s, torque Nm, timestamps s, explicit ages ms.

## Validation

Continuous-motion smoke: 15 segments × 1 s = 750 rows; suite smoke: 9 segments × 1 s = 450 rows. Both complete without resets/nonfinite output. Independent ONNX replay of recorded 61D input matches recorded raw action to maximum absolute error 1.91e-6 and 1.67e-6 respectively. CSV checks validate observation scalings, previous-action alignment, clipping, 20 ms timestamps, and guard demand against the SAME row's measured q/dq and limited target. This verifies logging alignment, not real-world accuracy.

Smoke folders:

- `/home/nszb/gym/artifacts/rs01_b18000_capture/20260921_192748_1789990068321714583`
- `/home/nszb/gym/artifacts/rs01_b18000_capture/20260921_192809_1789990089634572562`

Offline checker:

```bash
python legged_gym/scripts/check_rs01_capture.py /absolute/path/to/capture_directory
```

Rollback: simply use the original play script; these standalone capture/check scripts do not change training, deployment, actuator constraints or existing models. No hardware operation and no training started.
