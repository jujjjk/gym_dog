# RS01 video-inspired procedural reference examples

## Delivered scope

This is a deliberately expressive **kinematic style prototype**, not motion capture,
not MPC, not an AMP training implementation, and not an approved physical teacher.
The supplied 38.17s video informed appearance only. No measured video joint angles,
forces or metric clearances are claimed. Existing V29/V30 code and weights unchanged.
Repository baseline: fix/rs01-heading-odom-soft-inhibit, HEAD4fee2ce, dirty worktree retained.

RS01-specific: actual dog_rs01.urdf, 11.7317368323kg, thigh~180mm and calf~202.158mm,
16mm foot sphere, true joint axes/limits/asymmetric geometry via MuJoCo Jacobian IK.
Scene and actuator contract are checked against source URDF hashes.

## Design

- 2Hz, stance fraction .65: each leg supports325ms and swings175ms; double support
  occupies30% of the reference cycle. FL+RR / FR+RL offset half a cycle.
- Straight/backward/march foot clearance60mm, directional motion50mm.
- Root height290mm; nominal foot half-width180mm (360mm total width).
- World-fixed stance anchors, command-dependent SE(2) touchdown locations.
- Quintic horizontal interpolation and C2 vertical lift curve; no stance sliding
  in the prescribed reference. No demand for identical leg torques.
- Exact contact labels are **desired**, not measured. Kinematic base is prescribed.
- Ten independent4s clips at50Hz: march, forward.25m/s, backward-.18m/s,
  lateral±.12m/s, turns±.45rad/s, forward diagonals(.18,±.08,0), combined(.15,.06,.3).
  Clip boundaries are NOT physical transitions; do not concatenate across them.

## Commands

```bash
cd /home/nszb/gym
source /home/nszb/gym/unitree-rl/bin/activate
# Kinematic example preview; repeats the selected clip. Mouse rotates/zooms.
python mujoko/rs01_go2/reference_gait.py --motion forward --viewer
# All clips, fresh timestamped output directory:
python mujoko/rs01_go2/reference_gait.py --motion all
# Existing generated data, explicit unvalidated-style opt-in:
python mujoko/rs01_go2/read_reference.py artifacts/rs01_reference/design_v3 --allow-kinematic
# Real actuator open-loop diagnostic, not an expert controller:
python mujoko/rs01_go2/reference_gait.py --motion forward --physics
```

`--lift 70 --directional-lift 55` changes mm targets; `--half-width .18`,
`--height .29`, `--frequency 2`, `--duty .65` are adjustable. Unreachable IK fails
explicitly instead of silently clipping feet. Output directories cannot overwrite
existing recordings. Increasing parameters does not imply RS01 feasibility.

## Reading data

Each clip has CSV for plotting plus NPZ with timestamp, root quaternion(wxyz),
joint positions/velocities, foot positions in world and body frames, command,
leg phase, desired contact, IK error and transition-valid flags.
Joint/foot order: FL,FR,RL,RR; each leg hip,thigh,calf. SI units except metadata mm.
`amp_features`: 46 columns = q12,dq12,body feet12,body linear velocity3,
body angular velocity3,projected gravity3,root height1.
Not a replacement for the 61D policy observation. The AMP discriminator may use
these features; deployed actor must retain sensor-available observations.

```python
import numpy as np
d = np.load('/home/nszb/gym/artifacts/rs01_reference/design_v3/forward.npz')
q = d['joint_pos_rad']       # (200,12)
style = d['amp_features']    # (200,46)
valid = np.flatnonzero(d['transition_valid'][:-1])
pairs = np.concatenate([style[valid], style[valid+1]], axis=1)  # (199,92)
command = d['command'][valid]
```

`ReferenceDataset.sample()` balances clips and never joins their boundaries.
Normalisation and command conditioning must be added in a future AMP learner;
there is no claim of a stock AMP-loader-compatible format. Manifest explicitly
sets `amp_training_approved=false`; loader requires opting into style experiments.

## Evidence and limits

Four unit tests pass: diagonal/no-flight scheduling, fixed stance anchors and
continuous transitions, all-mode IK/finite46D features, no cross-clip sampling.
Ten clips total2000frames. Maximum IK residual<.001mm, sampled lift peaks59.85/49.88mm,
minimum signed lateral foot distance over all modes157mm. Those are reference
properties, NOT physically measured stability or load acceptance.

Physical replay uses existing V22Sim (B23500 ONNX supplies actuator metadata only;
policy inference is replaced), sensor snapshot, rate/accel limiter, thermal guard,
measured delay/FOPDT, physics-rate PD40/1. No ideal motor, forced root pose,
external stabilization, joint-state projection or automatic reset during replay.
17Nm is peak motor capability,6Nm continuous reference; host guard retains its
configured derating. Torque telemetry is sampled at policy boundaries, not all
physics substeps. At Kp40, position errors .05/.1/.2rad imply2/4/8Nm before damping.

Forward4s diagnostic (`physics_forward_v3/physical_replay.json`):

| Measure | Result |
|---|---:|
| Stop reason | completed, no observed flight/illegal ground contact |
| Reference forward displacement | +.995m |
| Actual forward displacement | -.192m |
| Max absolute roll/pitch | 3.86° /8.13° |
| Joint tracking RMSE | .169rad |
| Sampled raw PD peak | 12.46Nm |
| Actual joint speed peak | 10.21rad/s |
| Max foot clearance FL/FR/RL/RR | 26.6/26.2/.1/27.5mm |

**Physical teacher rejected.** Reference joint speeds up to7.59rad/s conflict with
current target-rate limits2/2.6/3.2rad/s. Delay and finite support compliance further
change realized contacts. No fall is not successful tracking. Physical recordings
are separated from reference clips and are not fed to the reference loader.

Next engineering stage: retain this visual design, resolve achievable trajectory
timing under target/actuator limits, then implement support-force/attitude feedback
(MPC/WBC or another validated closed-loop teacher). Only record accepted dynamic
rollouts as physical demonstrations. No long training or hardware deployment run.
Rollback: ignore this independent tool and continue using existing V29/V30 tasks.
