# Guard5 checkpoint selection — 2026-10-09

## Scope and provenance

Run: `rs01_v22_phase_guard5/Oct09_15-13-52_guard5_from29750_seed16`.
Training completed through32750 (3000 additional updates from29750). Evaluated
30500,31750,32450,32700,32750; this is a shortlist, not every saved checkpoint.
Early/mid checkpoints plus late high-reward checkpoints and the final model were
included. Reward was used for shortlisting only. No training, code/config changes,
export, deployment or checkpoint overwrite during selection; only this report/logs.
Repository `d2597ab`, `fix/rs01-heading-odom-soft-inhibit`, existing dirty files
preserved. Current uncommitted guard implementation was used.

Hardware/control unchanged from `../../../unitree_rl_gym/RS01_V22_PHASE_GUARD.md`:
11.7317kg RS01 URDF, identified real actuator, 50Hz policy, 400Hz physics,
configured motor peak14Nm (not the17Nm hardware capability). 5 degrees limits
requested inward targets, NOT actual joint state. Sensor/effective-command and
reward semantics unchanged. No parameter or PD experiment was in scope.

## Fixed-command screening

All candidates and original phase-lift29750 evaluated with the SAME seed20261010,
NORMAL sensor profile, no pushes/plant randomization, 13 commands ×2 environments
×12s. Metrics exclude initial2s; resets count the full interval. Commands: march;
forward/backward±.2m/s; lateral±.2m/s; yaw±.3rad/s; four diagonals±(.2,.1,0);
combined±(.2,.08,.25). Summary values are means of per-case metrics except peaks.

| Model | Resets | Roll RMS deg | Desired-contact match | vx/vy/wz RMSE (m/s,m/s,rad/s) | Raw torque peak Nm |
|---|---:|---:|---:|---|---:|
| Original29750 | 0 | 1.873 | .675 | .0371/.0389/.1155 | 16.85 |
| Guard30500 | 0 | **1.164** | **.710** | .0355/.0346/.1069 | **14.36** |
| Guard31750 | 0 | 1.280 | .704 | .0354/.0349/.1017 | 15.89 |
| Guard32450 | 0 | 1.262 | .708 | .0335/.0345/.1044 | 15.13 |
| Guard32700 | 0 | 1.306 | .708 | .0309/.0361/.0987 | 14.87 |
| Guard32750 | 0 | 1.288 | .707 | **.0310/.0348/.0972** | 15.96 |

All observations/rewards finite, no sampled flight in screening.32750 had no sampled
same-axle flight or speed-domain violations; mean per-case peak saturation fraction
.0000705 (0.00705%). These are policy-rate samples, not every physics substep.
Its vertical velocity RMS averaged .0470m/s; stability is not equivalent to a
perfectly motionless base. Contact-match ratio measures desired timing, not proof
of zero diagonal touchdown lag or perfect load sharing.

## Direct switching

Seed20261010, NORMAL sensors, 4 environments,13 actions switching every3 seconds,
39s/environment. No march inserted between movements; initial action is march.
Metrics include transitions. Original result is the preserved identical-protocol
`../rs01_v22_phase_lift_guard_probe.log` from the initial guard audit.

| Model | Resets / sampled flight | Mean roll / pitch RMS deg | Actual inward hip peak deg | Raw torque peak Nm |
|---|---|---|---:|---:|
| Original29750 | 0 / 0 | 1.784 / .947 | 19.050 | 15.240 |
| Guard30500 | 0 / 0 | 1.190 / 1.062 | **8.109** | 16.198 |
| Guard32750 | 0 / 0 | 1.277 / 1.088 | 8.544 | 16.889 |

Actual inward peak fell about55% for32750, but is not zero and exceeds the requested
5-degree limit because compliance, dynamics and contact are retained. Its stance
inside-fraction proxy was zero; this uses existing command-conditioned half-width
thresholds and is NOT a dynamic support margin. It cannot establish push tolerance.
Transition raw torque increased versus the baseline even though fixed-command
raw peak decreased; do not claim universal torque improvement.

## Actual tracking and foot geometry

Same fixed screening protocol above; signed actual velocities:

| Command | Original29750 | Guard30500 | Guard32750 |
|---|---:|---:|---:|
| Forward vx=.2m/s | .196 | .213 | .206 |
| Backward vx=-.2m/s | -.200 | -.197 | -.196 |
| Left vy=.2m/s | .189 | .161 | .175 |
| Right vy=-.2m/s | -.216 | -.188 | -.176 |
| Left yaw=.3rad/s | .271 | .289 | .292 |
| Right yaw=-.3rad/s | -.305 | -.300 | -.272 |
| Backward-left vy=.1m/s | .095 | .068 | .083 |

32750 left-turn mean front/rear spacing .2915/.2930m versus original .1895/.1944m;
right-turn .3005/.2990m versus .1944/.1972m. Forward spacing .2971/.3003m versus
.2525/.2557m. These are mean body-frame widths including all phases, not minimum
loaded-foot spacing. Adduction is substantially reduced but not proven eradicated.
Lateral commands remain undertracked by roughly12%; backward-left lateral tracking
is worse than the original on this seed. Average RMSE improvements do not imply
every action improved.

## Recommendation and limits

Overall demonstration choice:32750 for improved speed tracking with
reduced adduction and roll. Keep30500 as the lower-roll, lower-raw-torque alternative;
32750 is not strictly better on all metrics. It passed the appended longer and
stress tests below. No claim of sim2real readiness: MuJoCo/real controllers
must implement the new mapping identically, then validate disturbances, payload,
thermal behavior and more seeds. No deployment artifacts were produced.

Rollback/demo baseline remains `rs01_v22_phase_lift` with original29750, untouched.

## Longer independent-seed and faster-switch checks

32750, seed20261011, NORMAL sensors, fixed13 commands ×2env ×60s, first2s excluded
from metrics: **zero resets, zero sampled flight, finite observations/rewards**.
Mean per-case roll RMS1.207 degrees, contact match .722, vx/vy/wz RMSE
.0302/.0329/.1010. Raw torque peak17.462Nm is a request, not applied torque;
the configured motor cap remained14Nm. No long-duration thermal or disturbance
qualification is implied by60 seconds per environment.

32750 faster sequence: seed20261010,4env,13 actions×3s, all commands scaled1.5
(axis translation±.3m/s, yaw±.45rad/s): **zero resets, zero sampled flight, finite**.
Mean per-stage roll/pitch RMS1.317/1.365 degrees; peak actual inward hip9.060 degrees;
raw torque peak16.450Nm. Original29750 previously reset once at backward-left under
this same faster switching protocol (see phase-lift final selection report).
Actual forward vx=.2865m/s; left vy=.2202m/s; right vy=-.1901m/s versus requested
±.3m/s. Left/right also drift backward at vx=-.0554/-.0344m/s. Thus survival improved
but fast lateral tracking remains poor (roughly27%/37% under-speed); do not call
this fully accurate0.3m/s omnidirectional motion.0.4m/s was not tested here.

## Viewer (13 actions,3s each, normal speed)

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python /home/nszb/gym/artifacts/rs01_v22_phase_scratch/select_play.py \
  --task=rs01_v22_phase_guard5 \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_phase_guard5/Oct09_15-13-52_guard5_from29750_seed16 \
  --checkpoint=32750 --sequence --duration_s=3 --num_envs=1 \
  --speed_scale=1 --sensor_profile=normal --seed=20261010 \
  --sim_device=cuda:0 --rl_device=cuda:0
```
