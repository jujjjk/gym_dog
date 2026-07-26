# dog_rs01_trot

This task trains the new `dog_urdf` robot for a phase-conditioned diagonal
walk. The legacy task key remains `dog_rs01_trot` and does not modify Go2.

## Coordinate and phase contract

Policy/URDF order is `FL, FR, RL, RR`, with `hip, thigh, calf` inside each
leg. The fixed diagonal-walk phase offsets are:

- phase 0.0: `FL + RR`
- phase 0.5: `FR + RL`

For a moving command, a two-foot swing is legal only when it is exactly
`FL + RR` or `FR + RL`. Same-side pace, front/rear bound, three-foot swing,
and full flight terminate an episode on their first 50 Hz sample after the
0.2 s startup grace period. A single airborne foot is tolerated so a small
diagonal touchdown offset does not cause a false reset.

The observation has 52 values and includes `sin(2*pi*phase)` and
`cos(2*pi*phase)`. At a zero velocity command the phase contact and swing
rewards are gated off in favor of four-foot standing, so the policy is not
rewarded for stepping in place.

During a non-zero command, four simultaneous contacts are penalized. Each
foot also accumulates continuous contact time and is penalized after 0.42 s.
The dynamic walk uses a 0.38 s cycle with diagonal phase offsets 0.0/0.5 and a
0.70 stance ratio. The active diagonal changes every 0.19 s, with about 76 ms
of four-foot overlap at each transfer. During that overlap the newly landed
pair must accept at least 42% of nominal robot weight and 58% of current foot
load before the old pair leaves, and both toes must pass the minimum support
test. The phase may wait at most four 20 ms control samples, so it covers the
measured actuator delay without turning into an indefinite static-balance step.
Complete flight terminates the episode on its first 50 Hz sample. Prolonged
four-foot contact remains penalized; locomotion penalties are disabled for a
zero-velocity stand command.

The oscillator handoff is contact-aware and a smooth secondary gate attenuates
a swing until the opposite diagonal carries load. At least one exact diagonal
must carry 30% of nominal weight, with at least 5 N on each of its two toes. A
93% diagonal action projection and explicit single-foot-swing penalty keep the
two swing feet coordinated while leaving a 7% residual for the real front/rear
actuator and load differences.

## CPG base controller

The nominal gait is generated before PPO by `rs01_cpg.py`:

- two coupled limit-cycle oscillators drive `FL+RR` and `FR+RL` at exactly
  180 degrees separation;
- the support foot moves rearward at a speed derived from the commanded body
  velocity, while the swing foot follows a fast 30 mm clearance trajectory;
- foot targets are converted to thigh/calf targets by analytic IK derived
  directly from the supplied `dog_rs01.urdf` link vectors;
- PPO supplies only a bounded joint residual for dynamic balance, load
  acceptance and the remaining real-motor asymmetry.

At the 0.15 m/s test command, zero residual reaches about 0.10--0.12 m/s across
the measured actuator population. The contact-aware clock runs around 2.28 Hz
when it must wait for load, versus the 2.63 Hz free-running oscillator. The
residual policy still has to reduce brief contact-threshold mismatches and the
measured 23.8% raw-torque clipping fraction before any hardware work. Target
rate/acceleration guards remain active, and delivered torque stays hard-clipped
to 14/16/17 Nm.

The reference phase is resolved through each runtime foot name, not an assumed
rigid-body index, so the contact reward and physical swing target remain
aligned even if an asset parser changes body ordering.

The initial base pose is `[0, 0, 0.316]` with identity quaternion. The Gym
height reward targets the supplied settled height `0.309475 m`; the supplied
MuJoCo settled reference `0.307120 m` is retained in the config as metadata.
The URDF inertial masses sum to `11.7317 kg`; training randomizes trunk payload
by `-0.30` to `+0.80 kg` around that model.

## Motor model and safety provenance

The model uses `rs01shujv/rs01_actuator_data_20260720.json`:

- real 50 Hz command rate; the 11.73 kg dog uses hip `60/1.0`, thigh
  `75/1.3`, and calf `80/1.4` PD gains. Fanfan is 7.24 kg and uses
  `60/0.6`, `70/0.8`, `70/0.8` respectively;
- per-motor FOPDT gain and time constant;
- measured effective Coulomb friction and reversal error band;
- closed-loop pure delay randomized over 38.6-55.0 ms;
- aggregate 5%-95% ranges for 0x11 and 0x41 because those motors were removed
  for repair and have no individual identification data.

Unknown rotor armature, viscous friction, static friction, pure gear backlash,
and torque-speed curve remain unset. The measured reversal band is not labelled
as mechanical backlash.

The RS01 manual specifies 6 Nm rated continuous torque, 17 Nm peak torque,
315 rpm no-load speed, and a -20 to 50 C operating environment. The current
training peak caps are 14/16/17 Nm for hip/thigh/calf, at or below the manual's
17 Nm peak; EMA/sustained-torque rewards still use 6 Nm for every joint. The
chosen working position envelope is narrower than the mechanical URDF range.

The manual's 6 Nm rating is stated with a 260 x 280 mm heat sink. PPO cannot
provide thermal protection. Real deployment must retain the motor fault/timeout
path, stop on any RS01 fault bit, monitor motor/driver temperature, and choose a
continuous torque below 6 Nm if the installed cooling has not been validated.
The manual's 103 C over-temperature fault is an emergency protection threshold,
not a normal operating target.

## Zero to stand

`stand_transition.py` is deterministic and is never optimized by PPO. It uses
a 4 s minimum-jerk position curve, smoothly ramps Kp/Kd, and validates every
sample against conservative position/rate/acceleration limits. Its output is
in URDF/policy coordinates; do not send it directly to CAN until the new
robot's motor signs and ordering have been verified joint by joint.

Preview it without touching hardware:

```bash
cd /home/nszb/gym/unitree_rl_gym
python -m legged_gym.envs.dog.stand_transition --duration 4.0
```

## Training

From the configured Isaac Gym Python environment:

Preview the CPG alone first. This does not load a policy checkpoint and always
uses a zero action residual:

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python3 legged_gym/scripts/play_cpg.py --task=dog_rs01_trot
```

Then train a new residual policy on top of that CPG. Do not resume the old
joint-reference policy because it learns to cancel a different base motion:

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python3 legged_gym/scripts/train.py --task=dog_rs01_trot \
  --run_name=rs01_cpg_residual_balance \
  --max_iterations=3000 --headless
```

The training actor samples exploration noise, so its viewer is not a valid
gait preview. Inspect the deterministic policy after a checkpoint is written:

```bash
python3 legged_gym/scripts/play.py --task=dog_rs01_trot --checkpoint=-1
```

The actor output layer starts near zero, so the first policy mean leaves the
CPG intact while exploration learns balance corrections. Training permits
4/3/2 invalid-contact samples early enough to observe complete cycles, then
tightens to one; normal learned-policy play/test is always strict. The CPG-only
viewer does not reset on brief non-diagonal threshold mismatches, but full
flight, body contact and joint safety termination remain active.

## RS01 straight-walking retraining: `dog_rs01_straight_stand` / `dog_rs01_straight_walk`

Two tasks trained **from scratch** for one narrow goal: hold a stable flat-ground
stance, then walk straight forward at 0.03–0.12 m/s with an FL+RR / FR+RL
diagonal gait, all inside a 10 N·m RS01 envelope. They do not resume any
existing checkpoint and log to a separate `logs/rs01_straight/` experiment.

Deliberately excluded: lateral motion, turning, rough terrain, height scanner,
camera, lidar, and real-motor id/sign/zero-offset mapping. Joint semantics come
straight from the URDF.

Config: `dog_rs01_straight_config.py`. Scripts:

* `legged_gym/scripts/audit_dog_joints.py` — static URDF/joint/static-torque
  audit, no GPU or Isaac Gym required.
* `legged_gym/scripts/validate_dog_straight.py` — Isaac Gym runtime check of the
  DOF order, default pose, contacts, torque envelope and reset reasons.
* `legged_gym/scripts/export_dog_onnx.py` — ONNX plus a deployment contract JSON,
  rebuilt from the checkpoint without Isaac Gym.

Wrapper scripts and the full manual run guide live in `tools/rs01_straight/` and
`artifacts/rs01_straight/README_RUN_MANUALLY.md`.

```bash
./tools/rs01_straight/validate.sh          # stage 0, always run this first
./tools/rs01_straight/smoke.sh             # 5 iterations, throwaway
./tools/rs01_straight/train_stand.sh       # stage 1, 600 iterations
./tools/rs01_straight/train_straight.sh    # stage 2, 3000 iterations
./tools/rs01_straight/play_straight.sh
LOAD_RUN=<run> CHECKPOINT=<n> ./tools/rs01_straight/export_onnx.sh
```
