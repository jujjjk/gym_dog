# V22 Guard5 lateral-motion pilot

## Scope / repository / hardware

Branch `fix/rs01-heading-odom-soft-inhibit`, base commit `d2597ab` with the existing
uncommitted Guard5 changes. User edits including `11.md` preserved. Baseline32750
and all previous tasks/checkpoints remain unchanged. No long training, push or
deployment. This is an experimental task, not a new default/best-model declaration.

Actual URDF rechecked: `dog_urdf/urdf/dog_rs01.urdf`, mass11.7317368323kg,
SHA256 `48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4`.
Thigh .180m, calf-foot .202158m; hardware/control audit otherwise unchanged from
`RS01_V22_PHASE_GUARD.md`. Real RS01 delay, feedback timing, rate/acceleration limits,
host thermal guard, 14Nm motor cap and friction retained. 17Nm hardware capability
and6Nm continuous reference are not relaxed. Kp/Kd40/1 unchanged: proportional
demands at .05/.10/.20rad error are2/4/8Nm before damping. No physical-state clamp.

## Diagnostic evidence / intervention matrix

Baseline run `rs01_v22_phase_guard5/Oct09_15-13-52_guard5_from29750_seed16`, model32750.
NORMAL sensors; nominal plant/no pushes; seed20261012; six commands×2 environments
×20s, metrics exclude initial2s. Commands: forward .3, lateral±.2 and±.3m/s,
yaw .45rad/s. Separate simulator processes; identical actor for interventions.

| At lateral command ±.3m/s | Original5° | Loosen inward to8° | Pure-lateral cadence2.1Hz | Lateral outward target9° |
|---|---|---|---|---|
| Actual left/right vy, m/s | .2342/-.2320 | .2156/-.2174 | .2591/-.2343 | .2031/-.1889 |
| Left/right unintended vx, m/s | -.0450/-.0267 | -.0243/-.0334 | .0062/.0068 | -.0016/.0014 |
| Left/right hip rate saturation | .5125/.5335 | .4840/.5075 | .4897/.5057 | .3850/.3656 |
| Left/right target-limiter gap, rad | .0938/.0964 | .1015/.1052 | .0859/.0807 | .0493/.0464 |
| Left/right roll RMS, degrees | 1.65/2.13 | 3.19/2.90 | 2.69/3.31 | 3.58/2.60 |

All these20s trials had zero resets. Neither loosening inward limits nor reducing
outward amplitude improves this fixed actor's lateral tracking. Slower cadence
improves phase matching (left .802→.876; right .829→.888, per-foot agreement),
reduces backward drift, but worsens roll without adaptation. This supports a short
cadence experiment, NOT a conclusion that slower cadence is universally superior.

Baseline hip action saturation is74.6–76.7% at fast lateral motion; rate saturation
counts samples at >=95% of the configured hip target-rate bound. Motor14Nm peak
saturation is essentially zero here, but **host protection is NOT inactive**:
12.5–13.1% of joint samples have a projected host PD request, with mean per-step
minimum host limits8.01/8.14Nm. Do not infer unlimited torque headroom from the
motor peak metric alone. Delay, target-rate limits and host guarding all remain
plausible contributors; no unique root cause is claimed.

The actor's mean vy estimate at fast lateral motion is roughly±.198m/s vs true
±.232–.234m/s, another imperfection. Effective commands remain approximately±.300;
the reward is not accidentally requesting slower motion. Estimator/control/reward
semantics are unchanged in this pilot; no simulator truth is inserted into actor.

Existing sampling: lateral20% of moving modes, high magnitude band30% of those.
This may limit exposure, but has not been proven causal. Sampling is deliberately
NOT changed simultaneously, nor are reward terms/weights changed.

## Implemented single-variable experiment

New task `rs01_v22_phase_lateral`, derived from Guard5. Only the gait frequency
function is changed. For fast pure lateral commands, cap at2.1Hz instead of2.5Hz.
With duty .72, nominal swing time increases .112→.133s. This does not eliminate
the identified motor delay or guarantee the actual foot follows that schedule.

Continuous blend:
`pure=clip(1-|vx|/.1-|wz|/.2,0,1)`;
`moving=clip((|vy|-.08)/.12,0,1)`;
`f=f_original+pure*moving*(min(f_original,2.1)-f_original)`.
Uses raw shared command, no contact/truth gating. March/straight/pure turn and the
tested±(.2,.1) diagonals keep their old frequency; low-vx/yaw mixed commands may
blend smoothly. Phase remains integrated continuously with the existing diagonal
offsets. No phase resets, action mirroring, new penalties or actuator changes.

Files: `legged_gym/envs/rs01_omni_v2/rs01_v22_phase_lateral.py`, registration in
`envs/__init__.py`, focused tests, and support in the existing `select_play.py`.
Read-only diagnostics: `../artifacts/rs01_v22_phase_guard/audit_lateral.py`.

Deployment warning: any actor trained on this task needs this same cadence rule
and the Guard5 target mapping in MuJoCo/real controller. Deployment has NOT been
updated; do not export as ordinary V22 or ordinary Guard5 and assume parity.

## Validation / short training protocol

24 unit tests passed:17 phase/guard/cadence tests +7 V22 sensor tests. Config/PPO
equality tests verify only cadence settings and experiment name change. Existing
61D/12-action contract and shared sensor-derived actor/reward target are preserved.

Paired short training:512 environments, seed16,100 updates each from baseline32750,
same optimizer, fixed LR1e-4, same original sampling and exploration. Each branch
saves32850, which means100 additional updates, not32850 new updates.

- Control: `logs/rs01_v22_phase_guard5/Oct09_17-47-54_lateral_ab100_seed16`.
- Candidate: `logs/rs01_v22_phase_lateral/Oct09_17-50-15_lateral_ab100_seed16`.

Compare the same diagnostic protocol after training, then check continuous switching
before promoting anything. This is a small pilot, not a multi-seed training study.

## Completed paired-pilot evaluation / decision

Both100-update checkpoints are finite with optimizer LR1e-4. Same diagnostic seed,
commands, sensor profile and20s window as above; all six cases had zero resets.

| At lateral±.3m/s | Original32750 | Original config +100 updates | Cadence candidate +100 updates |
|---|---|---|---|
| Left/right actual vy, m/s | .2342/-.2320 | .2412/-.2155 | .2512/-.2315 |
| Left/right unintended vx, m/s | -.0450/-.0267 | -.0150/-.0321 | -.0007/-.0102 |
| Left/right roll RMS, degrees | 1.65/2.13 | 1.40/2.21 | **2.88/2.91** |
| Left/right hip rate saturation | .5125/.5335 | .5149/.4832 | .4910/.4800 |
| Left/right host projection fraction | .1252/.1309 | .1181/.1275 | .1156/.1281 |

Candidate improves some tracking, but balance deteriorates versus BOTH the original
and equal-training control. Rate saturation remains substantial, and host protection
is still active. This is not evidence that simply reducing cadence solves lateral
performance. Do not promote the candidate or recommend long training this variant.

Candidate faster continuous switching: NORMAL sensors, seed20261010,4env,
13 actions×3s, scale1.5 (axis translation.3m/s, yaw.45rad/s), zero resets,
finite observations/rewards and zero sampled flight. However, compared with the
original32750 under the same protocol:

| Metric | Original32750 | Cadence32850 |
|---|---:|---:|
| Mean per-stage roll RMS, degrees | 1.317 | 1.597 |
| Peak actual inward hip, degrees | 9.060 | 11.140 |
| Raw torque peak, Nm | 16.450 | 17.193 |
| Left/right mean vy, m/s | .2202/-.1901 | .2298/-.1937 |

The switching test reinforces the rejection: modest speed gain does not justify
worse roll, inward excursion and raw demand. **Keep original32750 as best available**.
No claim of successful stability optimization or sim2real qualification. Additional
command sampling/trajectory shaping needs separate evidence; do not stack rewards
or weaken real actuator protection as a reaction to this pilot.

Raw logs: `../artifacts/rs01_v22_phase_guard/lateral_pilot/`. Reproduce diagnosis:

```bash
cd /home/nszb/gym
source /home/nszb/gym/unitree-rl/bin/activate
OMP_NUM_THREADS=1 python artifacts/rs01_v22_phase_guard/audit_lateral.py \
  --task=rs01_v22_phase_guard5 \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_phase_guard5/Oct09_15-13-52_guard5_from29750_seed16 \
  --checkpoint=32750 --duration_s=20 --eval_envs=2 --seed=20261012 \
  --headless --sim_device=cuda:0 --rl_device=cuda:0
```

The same command with `--inward_deg=8`, `--lateral_hz=2.1`, or
`--lateral_outward_deg=9` individually reproduces each intervention. For the trained
candidate use its task/run above and checkpoint32850, without intervention flags.

## Risks / rollback

Lower cadence can increase roll and support loading; direct intervention already
showed that tradeoff. Do not erase this evidence or select only velocity gains.
Keep baseline32750 unless the full tradeoff is acceptable. Rollback simply uses
`rs01_v22_phase_guard5` and original32750. No prior model/config was overwritten.
