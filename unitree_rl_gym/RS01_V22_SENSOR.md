# V22 causal sensor snapshot

## State / scope

Based on `5779387a25db9e2923682837f0d78f51e0b10770` and local V21. Preserve all earlier tasks, models, unrelated edits and the real RS01 actuator model. No long training, deployment, push or remote mutation performed. Existing URDF/plant geometry and limits are unchanged from the audited RS01 model (11.7317 kg, 180/202.158 mm links). Policy 50 Hz, physical feedback 2.5 ms; identified response clock, peak/thermal limits, bounded hip mapping and target limiter retained. New host guard reads the policy's measured motor snapshot, while internal physical motor PD still reads true physical feedback. No ideal actuator substitution.

## Evidence and changes

| Problem | Evidence | Change / verification |
|---|---|---|
| Perfect estimator inputs | Existing estimator consumed true q/dq/gyro | Independent sampled/delayed motor and IMU queues before SAME leg odometry |
| Stale substep IMU | Parent base_ang_vel/gravity only update at policy callback | Refresh actor root tensor and DOF tensor at every acquisition substep, derive body gyro from fresh root quaternion/world angular velocity |
| Hidden clean observation slots | Parent observation builders read physics tensors | Build 61D directly from sensor snapshot, no parent observation construction |
| Actor/reward command conflict | Sensor yaw could differ from truth; command also changes during transition | Cache actor-visible effective command at step entry, reward uses that exact target against post-step physics truth |
| Double-counted skew | Extra skew randomization would duplicate delay | Motor/IMU own acquisition times and transport delays; skew ONLY timestamp difference |
| Host guard sees unrealistically current joints | Deployment guard consumes measured q/dq | Project sent target using SAME snapshot q/dq and sampled torque; physical PD unchanged |

Files: `rs01_sensor_snapshot.py`, `rs01_omni_v22_env.py`, `rs01_omni_v22_config.py`, task registration, `audit_rs01_v22.py`, V22 tests and dual launcher. Wide evaluator adds task names only. No reward added, removed or reweighted. V21 coordination/clearance/width/roll/support objectives remain.

## Sensor contract

- 32-slot physics-tick queues (80 ms at 2.5 ms), independently phased motor 5 ms / IMU 10 ms acquisition. Rates are engineering priors, NOT asserted hardware timestamp measurements.
- Select newest acquired sample whose arrival time is no later than current policy time; hold between arrivals. Noise is sampled with acquisition, not re-sampled when observation is read. No future interpolation or old-episode packets.
- Transport delays episode-wise: motor 0–10 ms Normal / 0–15 ms Robust; IMU 10–30 / 10–40 ms, quantized to physical ticks. Ages include acquisition/sample-hold in addition to transport; therefore total age/skew can exceed nominal transport bounds. There is no extra skew parameter.
- q sigma .0005 rad, clip .0015; dq sigma .08 rad/s, clip .25/.30; gyro sigma .004 rad/s, clip .015; episode gyro bias ±.005 rad/s.
- Consistent residual IMU mounting rotation: Normal roll/pitch ±2°, Robust ±3°. Rotate gyro into the same residual frame as the measured quaternion; derive gravity and yaw from that quaternion. Quaternion stays normalized. Not a substitute for actual mounting calibration.
- B mixture: 25% clean, 56.25% Normal, 18.75% Robust (Robust 25% of non-clean). A disables all sensor corruption and reads current samples. Same seed16, starting A20500, budget, PPO and reward configs; RNG streams are not guaranteed paired bit-for-bit after randomization.
- Reset bootstraps a current measured snapshot and flushes only reset environments' queues; no fabricated delayed pre-reset samples. Guard thermal history persists.
- Estimator, joint/gyro/gravity slots, heading reference and correction all use one snapshot. Raw user command, previous clipped action and gait phase remain controller-known quantities, not delayed physical sensors. Zero slots remain zero; estimated velocity error slots recomputed from same estimated velocity/target.
- No additional generic final-observation noise. No yaw drift, history, dimension reduction, new reward or enlarged actuator range in this first experiment. No accumulated path estimate is used by the 61D actor.

## Validation

41 unit tests passed (existing V21 and related tests included). New tests check causal delays, derived skew, reset isolation, clean samples, residual rotation, fresh root gyro vs poisoned policy-rate cache, 61D layout, and unchanged plant/reward configs. `audit_rs01_v22.py` asserts each step's actor IMU/joint slots match the snapshot, actor target equals its transition's reward target exactly, and tracking reward compares that target with physics truth. Repeated observation reads do not re-run the estimator.

Both groups: 512 environments, 100 added training iterations, A20500→20600, seed16. Runs:

```
logs/rs01_omni_v22_clean/Sep21_21-20-31_v22_clean_smoke
logs/rs01_omni_v22_sensor/Sep21_21-20-33_v22_sensor_smoke
```

Audit logs in `../artifacts/rs01_v22/`: 6 commands (march, forward/backward ±.2 m/s, lateral +.2 m/s, yaw +.3 rad/s, combination .2/.08/.25), 12 s × 4 environments each, first 2 s excluded in aggregate stats, seed20260925. All finite and target-alignment checks passed.

| Training group / evaluation sensors | Resets | Max per-case flight ratio | Mean roll RMS ° | Mean per-case XY estimator RMSE m/s |
|---|---:|---:|---:|---:|
| A clean / clean | 0 | 0 | 1.47 | .0186 |
| A clean / Normal | 1 (lateral) | .0005 | 2.01 | .0381 |
| B sensor / clean | 0 | 0 | 1.43 | .0183 |
| B sensor / Normal | 0 | 0 | 1.70 | .0377 |

Normal audit max observed IMU age 32.5 ms, motor age 12.5 ms, signed skew 27.5 ms. Sensor B is not yet accurate: target forward .2 gives .238 m/s; lateral .2 gives .252 m/s. Raw PD peak 18.88 Nm in lateral audit is pre-actuator demand, not delivered motor torque. Do not claim estimator repaired itself: it is unchanged and RMSE barely moves; the first evidence concerns policy resilience. Original A20500 also failed a separate shorter 6 s/2-env Normal lateral probe (2 resets); that probe is not directly comparable to the table.

Additional wide smoke: 23 march/moving commands × 4 environments × 12 s, seed20260925, both 20600 checkpoints. A clean profile and B configured mixed sensor profile each completed with zero resets, zero observed flight and finite output (`clean_wide.log`, `sensor_wide.log`). This is not a claim of all-Robust acceptance or long-horizon Sim2Real readiness. Static stand is reported separately in raw wide logs, excluded from these moving totals.

## User-started training

```bash
cd /home/nszb/gym/unitree_rl_gym
bash legged_gym/scripts/train_rs01_v22_dual.sh --start
```

GPU0 clean control, GPU1 sensor mixture, both seed16 / 4096 envs / 3000 additional iterations from ORIGINAL A20500, not smoke weights. Fixed learning rate 1e-4, fresh optimizer inherited. Logs displayed automatically; Ctrl+C exits viewing only. Launcher refuses duplicate V22 training. Long training is an experiment, not authorization to deploy the eventual model.

## Remaining gates / rollback

Recovery of clean and perturbed velocity accuracy, both lateral/yaw signs and mixed switches, multi-seed/longer runs, torque/thermal behavior, world-Z bounce, and matching MuJoCo/deployment measured-command semantics remain required. Current V21 MuJoCo bridge does NOT implement this new sensor model or delayed host guard, so V22 weights are not automatically approved for that bridge or hardware. Calibrate fixed installation/synchronization faults instead of training around them. Keep B18000 and A20500 untouched; select their old task/bridge to roll back. Sensor robustness is not proof that the separate clean-simulation bounce problem is fixed.
