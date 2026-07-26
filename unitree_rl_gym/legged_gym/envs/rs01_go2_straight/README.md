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
