# RS01 Go2-style straight task

Task name: `rs01_go2_straight`

This task is independent of the existing dog and fanfan tasks. Its learning
surface follows the repository's Go2 task:

- 50 observations: Go2's 48-D body/joint/action state plus gait phase sin/cos;
- 12 direct joint-position residual actions;
- flat ground and forward-only velocity commands;
- the compact Go2 reward set plus one phase-resolved foot-load error, with no
  CPG joint trajectory, fixed stride, or reference policy.

The robot-specific substitutions are:

- `dog_urdf/urdf/dog_rs01.urdf`;
- the supplied 0.316 m standing reset and joint angles;
- real controller gains and target rate/acceleration limits;
- per-motor response gain, time constant, observed closed-loop delay and
  Coulomb friction from `rs01shujv/rs01_actuator_data_20260720.json`;
- 6 N·m continuous rating retained as telemetry and a 17 N·m hard
  electromagnetic peak limit.

Contact/reset settings are scaled to the RS01 URDF instead of copied from Go2:
the 16 mm feet use a 3 mm contact offset, two physics substeps, 8/2 position
and velocity iterations, 0.25 m/s maximum depenetration velocity, and 0.003 rad
reset joint noise.  PPO uses fixed 0.25 action noise; the Go2 default 1.0 noise
caused immediate contact chatter and 17 N·m saturation on this geometry.

The walking reward gives four-foot contact a 0.12 s handoff grace period, then
ramps a prolonged-contact cost to full strength by 0.24 s.  This permits smooth
double support but prevents a stationary policy from collecting velocity/yaw
reward indefinitely.  Positive zero-yaw tracking was replaced by a light
yaw-rate cost.

The original 48-D policy repeatedly converged either to front/rear-paired
hopping or to static loading of one diagonal because it had no observable
cycle state.  The minimal correction is a 0.60 s phase exposed as sin/cos and
one composite support error. Continuous vertical-force distribution guides
unloading, while the shared 2 N contact mask requires scheduled swing feet to
actually leave contact. `FL+RR` and `FR+RL` are
half a cycle apart; a 0.65 stance ratio creates brief four-foot handoffs and no
scheduled flight.  The policy still directly supplies all 12 joint targets:
there is no CPG position reference or action compensation.  Raw PD demand
above 17 N·m and actual 17 N·m clipping are also penalized.

The composite error is converted to a positive Go2-style exponential tracking
reward (`sigma=0.10`).  Long four-foot standing still receives the full
prolonged-contact penalty, so it cannot collect a net support subsidy.

The first phase-conditioned pilot learned the correct load-transfer direction
but kept a third toe brushing the floor.  A small URDF-scaled swing target now
uses the actual foot rigid-body height: the 16 mm collision-sphere centre
follows a sinusoid up to 30 mm, i.e. only 14 mm of peak ground clearance.
There is also a narrow safety cost when both front feet or both rear feet are
airborne at once.  These terms do not generate joint targets; the policy still
outputs all 12 motor position commands directly.

The measured delay is intentionally named `observed_closed_loop_delay_s`. The
identification report states that it cannot be interpreted as pure
communication delay.

Train from scratch:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_straight \
  --headless \
  --run_name=rs01_phase_load_from_scratch
```

Play a checkpoint:

```bash
python3 legged_gym/scripts/play_rs01_go2_straight.py \
  --task=rs01_go2_straight \
  --load_run=<run-directory> \
  --checkpoint=<iteration>
```

## Rear-leg coordination polish

Task `rs01_go2_straight_rear_coord` is checkpoint-compatible with the 50-D
straight task and is intended only for conservative continuation from the
accepted model_550.  It keeps the gait clock, 12 direct actions, gains, action
scale, rate/acceleration limits, and measured RS01 actuator unchanged.

The polish adds contact-state disagreement inside `FL+RR` and `FR+RL`, plus an
extra clearance-shortfall signal applied only to scheduled `RL/RR` swing.  It
does not force front and rear foot heights to be identical.  PPO uses lower
fixed noise, a fresh low-rate optimizer, and a frozen model_550 executed-action
reference so the accepted front-leg motion cannot drift quickly.

Run only a short checkpointed pilot first:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_straight_rear_coord \
  --headless \
  --max_iterations=50 \
  --run_name=rear_coord_pilot_from550 \
  --resume \
  --load_run=Jul26_20-17-54_pilot_live_clearance_resume500 \
  --checkpoint=550
```

Do not treat the pilot as hardware-ready solely because its diagonal timing is
better; reject continuations that increase peak saturation or body yaw.

## Closed-loop straight-path polish

Task `rs01_go2_straight_path_polish` conservatively continues from the accepted
rear-coordinated model. It appends one wrapped desired-minus-current heading
scalar to the 50-D observation, giving a 51-D actor input. Actor and critic
checkpoint inputs are migrated from 50 to 51 dimensions; all learned weights
are retained and the new heading column starts at zero.

The heading term tracks a bounded restoring yaw rate instead of penalizing all
yaw rate. This distinction lets the robot turn back toward its initial heading
when it has drifted. It does not add global lateral position, change the gait
clock, alter the 12 direct actions, or modify the measured RS01 actuator.
Deployment must provide the same wrapped yaw relative to the heading captured
at locomotion start.

The first deterministic 30 s pilot at 0.35 m/s selected model_625 rather than
the final model_650: model_625 reduced lateral displacement from 5.03 m to
1.24 m and final heading error from 59.15 degrees to -6.67 degrees without
reducing the exact diagonal-contact rate. Since later updates crossed through
zero and over-corrected, fine continuation uses a `5e-5` fixed learning rate
and saves every five iterations.

The fine continuation was ranked with 16 parallel 30 s nominal rollouts.
Model_635 is the selected path checkpoint: mean path lateral RMS was 0.430 m,
mean absolute final lateral displacement was 0.872 m, exact desired-contact
matching was 67.88%, and 17 N.m motor saturation was 17.86%. Model_640 had a
slightly lower heading RMS but more lateral path error, so it was not selected.

Fine-polish from the selected checkpoint:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_straight_path_polish \
  --headless \
  --max_iterations=25 \
  --run_name=path_polish_fine_from625 \
  --resume \
  --load_run=Jul26_21-06-58_path_polish_pilot_from600 \
  --checkpoint=625
```

Play the selected fine-polish checkpoint:

```bash
python3 legged_gym/scripts/play_rs01_go2_straight.py \
  --task=rs01_go2_straight_path_polish \
  --load_run=Jul26_21-12-32_path_polish_fine_from625 \
  --checkpoint=635
```

## Kp40 actuator-feasibility continuation

Task `rs01_go2_straight_kp40` preserves the 51-D heading observation, diagonal
clock, 12 direct outputs, measured RS01 response/delay/friction model, and
17 N.m electromagnetic peak limit. It changes only the control experiment to
`Kp=40/40/40`, `Kd=1/1/1`, and an evidence-selected action scale of `0.18`.

Applying Kp40 directly to model_635 at action scale 0.14 reduced raw torque P95
from about 56 N.m to 14.6 N.m and peak saturation from about 17.9% to 3.2%,
but speed collapsed to 0.025 m/s. A scale sweep selected 0.18 as the smallest
useful continuation point: it produced 0.112 m/s, raw torque P95 32.2 N.m,
and 8.0% peak saturation before adaptation. Larger scales moved faster but
spent more time on the peak limit.

The continuation trains only at 0.18--0.28 m/s and uses a tighter velocity
tracking sigma so standing is not an attractive shortcut. Start from the
accepted 51-D model_635; this is not a from-scratch task:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_straight_kp40 \
  --headless \
  --max_iterations=100 \
  --run_name=kp40_adapt_from635_pilot \
  --resume \
  --load_run=Jul26_21-12-32_path_polish_fine_from625 \
  --checkpoint=635
```

## Kp40 conservative refinement

Task `rs01_go2_straight_kp40_polish` starts from the selected model_730. It
does not change Kp/Kd, action scale, command range, gait period, duty factor,
clearance, observations, or actuator dynamics. It lowers PPO learning rate to
`5e-5`, fixed exploration to `0.07`, and saves every five iterations.

The only reward changes are modest increases to raw-over-peak torque, actual
peak saturation, and diagonal pair synchronization. A stronger frozen
model_730 action reference limits regression. Run only 30 iterations first:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_straight_kp40_polish \
  --headless \
  --max_iterations=30 \
  --run_name=kp40_polish_from730 \
  --resume \
  --load_run=Jul27_09-51-57_kp40_adapt_from635_pilot \
  --checkpoint=730
```

Reject a continuation if deterministic speed falls below 0.20 m/s, path
lateral RMS exceeds 0.398 m, exact desired-contact matching falls below
56.64%, or 17 N.m saturation exceeds 11.03%.

## Sim2Sim bridge from model_730

The independent Sim2Sim tasks preserve the 51 observations, 12 direct outputs,
Kp40, gait phase, default standing state and measured RS01 actuator. The first
adaptation keeps Kd1; the measured calf repair below changes only calf Kd, and
the robust task inherits that selected repair contract.

Stage A reduces only calf target authority from 0.18 to 0.14 rad and limits
calf target rate/acceleration to 2.6 rad/s and 72 rad/s2. Hip and thigh
authority remain 0.18 rad. The model_730 warm-start still runs continuously in
Isaac Gym with zero flight, while raw-over-17 falls from about 11.0% to 8.3%;
speed/contact timing must be recovered by the short adaptation.

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_sim2sim_adapt \
  --headless \
  --max_iterations=60 \
  --run_name=sim2sim_adapt_from730 \
  --resume \
  --load_run=Jul27_09-51-57_kp40_adapt_from635_pilot \
  --checkpoint=730
```

Do not start Stage B merely because reward rises. First select a Stage-A
checkpoint that preserves zero flight/reset, restores at least 0.20 m/s and
56% exact contact matching, and improves MuJoCo survival beyond the model_730
5.9 s baseline.

The first Stage-A continuation reached speed/contact targets but did not lower
raw demand. Deterministic model_785 analysis showed that the four calf joints
hit the URDF 32.9867 rad/s velocity ceiling during swing: calf damping demand
was about 33 N.m while its position-error contribution was only 7--8 N.m.
Use the isolated calf-repair task before any randomization. It keeps Kp40,
changes only calf Kd from 1.0 to 0.55 based on a fixed-checkpoint sweep, and
adds direct calf-speed and pre-clamp action-saturation costs:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_sim2sim_calf_repair \
  --headless \
  --max_iterations=80 \
  --run_name=sim2sim_calf_repair_from785 \
  --resume \
  --load_run=Jul27_11-23-28_sim2sim_adapt_from730 \
  --checkpoint=785
```

The resume loader must report a fixed action standard deviation of 0.08; the
observation-adaptation path now reapplies the destination task's exploration
contract after loading the source checkpoint.

The 80-update calf-repair run selected `model_815`: under the nominal PhysX
contract it retained zero flight/reset, 66.7% exact contact matching and a
0.168 m 30-second path RMS. The remaining failure was physical calf velocity:
15.2% of calf samples still touched the 32.9867 rad/s URDF ceiling. A fixed
model_815 sweep showed that calf Kd=0.50 halves this to 7.86% while retaining
0.217 m/s, 65.18% exact contact and zero flight. Continue with the isolated
nominal task before adding randomization:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_sim2sim_kd050 \
  --headless \
  --max_iterations=40 \
  --run_name=sim2sim_kd050_from815 \
  --resume \
  --load_run=Jul27_12-10-20_sim2sim_calf_repair_from785 \
  --checkpoint=815
```

This stage keeps all action scales, target rate/acceleration limits, rewards,
phase, observations and RS01 actuator identification unchanged. Only calf Kd,
learning rate, exploration standard deviation and conservative continuation
strength change. Do not use Kd=0.45: the fixed-checkpoint ablation reduced
exact contact matching to 47.85%.

Stage B adds only narrow randomization after the Kd=0.50 checkpoint passes:
friction 0.85--1.15, base mass
+/-0.30 kg, response gain +/-5%, time constant +/-10%, motor friction +/-10%,
and a shared +/-1 physics-step delay offset. Continue it from the selected
Kd=0.50 run:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_sim2sim_robust \
  --headless \
  --max_iterations=100 \
  --run_name=sim2sim_robust_from_kd050 \
  --resume \
  --load_run=<selected-kd050-run> \
  --checkpoint=<selected-kd050-checkpoint>
```

## Matched MuJoCo short transfer

Task `rs01_go2_sim2sim_matched_transfer` is a conservative continuation of the
selected robust checkpoint. It preserves 51 observations, 12 direct actions,
Kp40/Kd0.50, the real standing pose, measured per-motor centre values, gait
timing, action scales and the 17 N.m electromagnetic peak. Its narrow
response/time/friction and +/-1-step delay perturbations are sampled
independently per joint so that the actor cannot rely on perfectly paired
left/right dynamics.

Run only 30 updates first and save every five:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_sim2sim_matched_transfer \
  --headless \
  --max_iterations=30 \
  --run_name=matched_mujoco_transfer_short_from870 \
  --resume \
  --load_run=Jul27_13-07-38_sim2sim_robust_from_kd050_840 \
  --checkpoint=870
```

PhysX reward is not the acceptance result. Export each retained checkpoint,
regenerate its new-machine MuJoCo scene, and require a complete 30-second run
with meaningful forward body velocity, bounded yaw, no fall/flight regression,
and no raw-torque or 17 N.m saturation regression.

## Runtime-parity fix and 52-D heading recovery

Isaac Gym's verified runtime DOF order for this URDF is `FL, FR, RL, RR`, with
hip/thigh/calf inside each leg. It differs from URDF declaration order. The
environment now refuses to run if that order changes, and schema-version-2
exports use the same authoritative list. Old schema-version-1 ONNX files must
not be used.

After fixing both joint routing and the MuJoCo body-velocity frame, the
untrained model_900 bridge changed from effectively zero body speed and 11.5%
exact contact to 0.222 m/s, 76.5% exact contact, zero flight and no 30-second
fall. The remaining failure is accumulated heading/lateral drift.

Task `rs01_go2_sim2sim_heading52` replaces the clipped scalar heading input
with `sin(error), cos(error)`. Its 51-to-52 checkpoint migration preserves the
first 50 columns, maps the old local heading weight to `2*sin(error)`, and
initializes `cos(error)` to zero. It adds only reset heading/yaw-rate recovery
coverage; motor, PD, action scale, target limits, gait and torque limits remain
unchanged.

First run a 30-update pilot from the selected model_900:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_sim2sim_heading52 \
  --headless \
  --max_iterations=30 \
  --run_name=heading52_recovery_from900 \
  --resume \
  --load_run=Jul27_14-22-47_matched_mujoco_transfer_short_from870 \
  --checkpoint=900
```

Reject the pilot unless matched MuJoCo retains at least 0.20 m/s, zero
fall/flight, raw P95 below 20 N.m and peak saturation below 10%, while reducing
30-second unwrapped yaw drift below the fixed-runtime baseline magnitude of
0.887 rad.

## Actor-side estimator parity after model_1850

Task `rs01_go2_estimator_parity` keeps the selected model_1850 policy
interface and locomotion contract unchanged: 54 observations, 12 direct
actions, Kp/Kd, action scales, gait period/duty, rewards and RS01 actuator
limits are inherited without retuning. The only intentional change is the
source of actor observations:

- body linear velocity is estimated from joint position/velocity and ideal
  body-frame IMU angular velocity with the same FK, stance selection,
  hysteresis and filtering as the Jetson node;
- the last two straight-path observations are integrated from that estimate
  in the latched heading frame;
- simulator root velocity and root path remain available to rewards and
  diagnostics, but are no longer visible to the actor.

The real node now latches heading/path zero only after the robot has completed
the stand ramp and remained quiet and supported for one second. This prevents
manual support or a moving stand transition from becoming the policy's path
origin.

Run a short continuation first; do not immediately launch a long run:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_estimator_parity \
  --headless \
  --max_iterations=100 \
  --run_name=estimator_parity_short_from1850 \
  --resume \
  --load_run=Jul29_11-08-16_path54_sim2sim_transfer_from1725 \
  --checkpoint=1850
```

Evaluate a retained checkpoint with the deterministic parity report:

```bash
RS01_EVAL_DURATION_S=30 RS01_EVAL_VX=0.23 \
python3 legged_gym/scripts/evaluate_rs01_estimator_parity.py \
  --task=rs01_go2_estimator_parity \
  --headless \
  --load_run=<estimator-parity-run> \
  --checkpoint=<checkpoint>
```

Accept continuation only if estimator error and true-root path error both
remain bounded; optimizing the internally estimated path alone is not an
acceptable result.
