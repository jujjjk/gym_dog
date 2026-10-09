# RS01 stability-first, non-AMP, single-GPU task

## Repository and model

2026-10-08, branch `fix/rs01-heading-odom-soft-inhibit`, HEAD `4fee2ce`.
Existing dirty changes, old checkpoints and AMP experiments preserved. New task
`rs01_stable_phase` starts a fresh actor/critic, no old checkpoint or reference
motion. Runner is ordinary PPO, not AMP, residual-policy or symmetry runner.

Actual asset: `dog_urdf/urdf/dog_rs01.urdf`, mass11.7317368kg; thigh~.180m,
calf-to-foot~.202158m; default foot y=±.14725m, foot radius16mm. Inertia matrices
pass positive-eigenvalue and triangle checks; this is not hardware identification.
Equal vertical-load Jacobian estimates at default posture: four-foot hip/calf
~2.51/3.40Nm, two-foot~5.02/6.79Nm (ignores limb dynamics/unequal load).

## Physical execution contract

Inherited unchanged from V30: 61 sensor-derived observations,12 direct actions,
policy50Hz, physics400Hz; PD40/1; target scales .22rad/joint; original hip target
mapping; rates2/2.6/3.2rad/s and accelerations60/78/96rad/s²; measured per-motor
delay~39–56ms, response~19–38ms, gain/friction; host guard and thermal history.
Configured motor peak remains14Nm, continuous reference6Nm; URDF peak17Nm.
PD position demands at .05/.10/.20rad are2/4/8Nm before damping (1Nm per rad/s).
No ideal actuator, state projection, gain increase or velocity rewrite introduced.

## Evidence → design

| Issue | Evidence | New mechanism | Acceptance measurement |
|---|---|---|---|
| Inward support | User reports real-machine imbalance; older stance corridor allowed8–10cm | Per-foot gravity-aligned yaw-frame13–21cm support/late-landing corridor | Per-foot loaded inward fraction, body roll, resets |
| Phase not realized by feet | Phase already in61D observation | Single contact+25mm swing-height phase objective | FL+RR / FR+RL contact/lift timing |
| One foot stays planted | All-four timer cannot detect a single planted foot | Independent validated-lift timers | Longest contact and no-lift duration per foot |
| False lift via contact chatter | Force threshold alone can flicker | At least10mm clearance and40ms airborne before resetting timer | Unit test and recorded heights/contact |
| Reward conflicts | Long inherited reward lineage | Independent scales, one velocity task and one phase objective | Enabled scales and per-term contributions |

Contact uses the existing public force-threshold mask throughout. No action
mirroring or identical left/right load requirement; foot support width is not a
guarantee that COM lies safely within the instantaneous support region.

## Initial training envelope

- Fixed2Hz phase; duty.70. Each diagonal has .35s stance/.15s swing, with .10s
  four-foot overlap at each exchange. Phase continuous across command switches.
- Commands sampled every4s: march; forward.12m/s; reverse.10m/s; lateral±.06m/s;
  yaw±.25rad/s; forward diagonals(.08,±.04); combination(.08,.03,.15).
- Zero command explicitly means marching. No randomly sampled stand.
- No-lift cost begins after .45s. After3s actual startup time, >1.5s without a
  validated lift, >.5s continuous qualified air, or >.2s support inside9cm ends
  the episode. These are learning/failure conditions, NOT forced foot lifting.
  Startup timing does not use PPO's randomized episode-length counter.
- Non-AMP PPO, fixed learning rate3e-4, fixed action std.35; singleGPU0.
- No new push/mass perturbations yet. Sensor uncertainty remains inherited.

## Files and validation

- `legged_gym/envs/rs01_omni_v2/rs01_stable_phase.py`: independent config/rewards,
  phase targets, measured foot corridor, per-foot lift clocks and failure reasons.
- `legged_gym/envs/__init__.py`: task registration only.
- `legged_gym/scripts/train_rs01_stable.py`: scratch-only startup checks and exact
  config/source-hash snapshot in each run's `experiment_contract.json`.
- `legged_gym/scripts/play_rs01_stable.py`: ten directly switched commands, explicit
  command duration; logs widths, contact duration, height, body motion and torque.
- `tests/test_rs01_stable_phase.py`:4 tests; actuator tests4 and V22 sensor tests7
  also pass. All15 pass. Three- and12-iteration64env startup/reset smokes finish.

Smoke checkpoints are NOT stable walking candidates. Their resets are expected;
software/actuator validation is distinct from a learned-motion acceptance gate.
The seed20261008,checkpoint12,4env,40s direct-switch smoke is recorded at
`../artifacts/rs01_stable_phase/sequence_12_1791441908042084740.json`: finite=True,
52 resets, all52 identified as gait timeout; motor peak5.956Nm. This verifies the
stalled-foot failure path, NOT successful locomotion. GPU0 long run subsequently
started, PID627014, log `/tmp/rs01_stable_phase_gpu0.log`; startup checks passed and
PPO iterations advanced. GPU1 was not assigned a new job.
The new random-policy/12-step model is the reproducible scratch baseline; no claim
that an old deployment model was re-evaluated or its real-world failure diagnosed.

## Run and inspect

Training launch (authorized single GPU;10,000 iterations,4096env,seed16):

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
OMP_NUM_THREADS=1 python legged_gym/scripts/train_rs01_stable.py \
  --task=rs01_stable_phase --num_envs=4096 --max_iterations=10000 \
  --run_name=stable_phase_scratch_gpu0_seed16 --seed=16 --headless \
  --sim_device=cuda:0 --rl_device=cuda:0
```

Viewer once a checkpoint is selected (replace run/checkpoint explicitly):

```bash
python legged_gym/scripts/play_rs01_stable.py --task=rs01_stable_phase \
  --load_run=/absolute/run --checkpoint=1000 --num_envs=4 --duration_s=5 \
  --seed=20261008 --sim_device=cuda:0 --rl_device=cuda:0
```

This is new training, not permission for hardware deployment. Select checkpoints
by resets, diagonal alternation, inward support, body motion and torque, then run
30/60s fixed-command, multiple-seed, thermal and MuJoCo tests before Sim2Real.
Current viewer torque peaks are policy-rate samples, not exhaustive substep peaks.
The fixed2Hz clock must later be reflected in deployment metadata/control.
Rollback: choose an existing old task/checkpoint; no old task was overwritten.
