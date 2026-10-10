# Guard5 push / payload training

Based on current repository10f80e8 (Kimi's Guard5 deployment update), preserving
model32750, deployment code, existing dirty files and rejected lateral-cadence
experiment. This task derives from **Guard5**, NOT the slower-cadence pilot.
RS01 skill workflow: isolated task, unchanged real actuator, measured external
disturbances, smoke before authorized long training.

## Physical contract and change

Task `rs01_v22_guard5_robust`: original61D sensor-only actor /12 actions, rewards,
FL+RR/FR+RL clock, duty factor, 5-degree inward-target map and real RS01 plant remain
unchanged. Identified delay, torque guard, target-rate/acceleration limits and motor
14Nm operating cap are preserved;17Nm hardware capability and6Nm continuous
reference are not relaxed. No root-state/velocity overwrite is used for pushes.

Nominal URDF mass11.7317368kg, thigh .180m, calf-foot .202158m. Equivalent additional
trunk mass is sampled uniformly0–2kg at environment creation. About25% of worlds
remain nominal mass and receive no pushes. The existing simulator body-property
path recomputes inertia. Payload stays fixed within a world: this is **not** sudden
weight attachment, high-mounted cargo, moving payload or COM-offset randomization.
Real payload mass/placement still needs subsequent physical validation.

Other worlds receive random horizontal force directions, peak6–18N, duration
.10–.20s, using a smooth sin² envelope on every2.5ms physics substep. Equivalent
application point is .06m vertically above trunk COM (F plus r×F torque). Full
pulse impulse is .5×peak×duration, .3–1.8Ns before curriculum scaling. Initial
wait4–7s; subsequent quiet intervals6–10s after each pulse. First120 simulated
seconds/environment ramp force amplitude from50% to100%. For PPO24steps/update,
this is about250 updates, not120 wall seconds. Reset cancels any active pulse;
thermal history remains governed by the unchanged parent implementation.

Problems/evidence: user reports successful Guard5 real deployment and requests
push/load robustness. Existing training lacks these disturbances. Add physical
mass/force variation, NOT new conflicting rewards or relaxed actuator constraints.

## Verification

-20 phase/guard/robustness unit tests pass; exact config comparison verifies only
  payload/push settings change. Tests cover pulse integral/finite duration and
  reset clearing the selected environment's external wrench.
-100 PPO updates,512env, seed16 from32750 completed successfully in a separate run:
  `logs/rs01_v22_guard5_robust/Oct09_22-10-58_robust_smoke100_seed16/model_32850.pt`.
-Final logging patch separately exercised in3 updates; payload, push count, impulse
  and force-scale metrics appear in Episode logs. These metrics are NOT rewards.
-Full-strength evaluation bypassed the force ramp (scale1), NORMAL sensors,
  seed20261013,13 fixed commands×2env×20s. Commands up to .2m/s translation,
  .3rad/s yaw, diagonals±(.2,.1,0), combined±(.2,.08,.25), plus march.
  Both original32750 and smoke32850 passed: **0 resets,0 sampled flight,finite**.
 35 force pulses were delivered in each test.9/26 worlds were nominal in this
  sample. Simulator body properties verified total mass11.7317–13.7282kg,
  maximum added mass1.99645kg. Original rollout accumulated31.684Ns scalar impulse.
  Motor peak remained14Nm. Metrics are policy-rate samples, not full substep traces.

| Actual speed under load/push | Original32750 | Smoke32850 |
|---|---:|---:|
| Forward vx, command .2m/s | .1830 | .2032 |
| Left vy, command .2m/s | .1777 | .1479 |
| Right vy, command -.2m/s | -.1807 | -.1874 |

The smoke proves executable learning and continued motion under these bounded
disturbances, NOT universal improvement. Left tracking regressed; do not deploy
the smoke model. Long training starts from the preserved, deployed32750.

## Authorized long run

GPU0,4096 environments, seed16,5000 ADDITIONAL updates from32750, expected final
iteration37750. Experiment directory `logs/rs01_v22_guard5_robust`;
run name `robust_push_payload_seed16`. Unchanged fixed LR1e-4.

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python -u legged_gym/scripts/train.py \
  --task=rs01_v22_guard5_robust --num_envs=4096 --max_iterations=5000 \
  --run_name=robust_push_payload_seed16 --seed=16 --resume \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_phase_guard5/Oct09_15-13-52_guard5_from29750_seed16 \
  --checkpoint=32750 --headless --sim_device=cuda:0 --rl_device=cuda:0
```

Do not launch a second copy if the agent-started process is still running.
Live stdout: `/home/nszb/gym/artifacts/rs01_guard5_robust/train_long.log`.
Use `tail -f` for progress. TensorBoard:

```bash
source /home/nszb/gym/unitree-rl/bin/activate
tensorboard --logdir=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_guard5_robust --port=6006
```

Open localhost:6006. Follow Train/mean_episode_length, termination reward,
phase-contact quality and Episode/robust_force_scale, robust_pushes, robust_payload_kg,
robust_impulse_ns. Reward growth is not a deployment acceptance criterion.

## Remaining checks / rollback

Later selection must compare clean operation, mass-only, push-only and combined
stress, multiple seeds, direct command switches and MuJoCo/real-robot parity.
No heavy load, large shove, high-speed0.4m/s or untethered real-world robustness
claim is made by this short smoke. Do not change the deployed policy until selected
and physically validated. Rollback: original task/model32750, untouched.
