# V20 controlled geometry A/B

## Repository and baseline

Branch `fix/rs01-heading-odom-soft-inhibit`, HEAD `3a0a7f4`; existing dirty files preserved.
Both branches resume V19 soft seed16 model15000:
`logs/rs01_omni_v19_placement_soft/Sep13_15-30-26_v19_gpu0_seed16_20260913_153022/model_15000.pt`.
Same training seed16 isolates the intervention; this is not a multi-seed robustness test.

Baseline nominal posture: 30s, measure6–30s, 4env/command, eval seed20260918.

| Command | RF ahead of LF (mm) | Hip action clipping | Geometry deduction / tracking per step |
|---|---:|---:|---:|
| March | 51.1 | 72.6% | .00929 / .25643 |
| Forward .2m/s | 20.3 | 71.0% | .00329 / .23206 |
| Reverse -.2m/s | 63.3 | 73.7% | .00875 / .24720 |

March thigh target FL/FR = -10.04/-19.04deg; actual -9.90/-19.98deg.
Target is already biased; this does not prove the actuator played no role in learning.

## Change and conflict audit

| Problem | Change | Acceptance |
|---|---|---|
| Persistent footprint bias/width shortfall | A doubles existing placement weight .5→1.0 | Reduced cycle-mean bias and increased width without resets/flight |
| Severe inward hip targets | B adds command-conditioned inward target compression to A | Reduced actual adduction; no loss of omnidirectional capability |

No new reward names or scales. A/B rewards identical, width target220mm unchanged.
Final weight is `.5 + straight_gate * .5`: only march/straight/reverse are strengthened.
Initial unconditional weight1.0 A smoke had one reset in left0.30m/s; rejected.
The final conditional variant is trained again from the original V19 baseline,
not from this rejected short-run checkpoint. This avoids strengthening width pressure on lateral/yaw.
Existing filtered fore/aft term remains conditional; normal opposite-phase swings are not forced equal.
No action mirror, equal-force constraint, phase change, or new contact/tracking gate.
Both retain raw action saturation penalty, unchanged previous-action observation semantics.
Reward improvement alone is NOT acceptance. More severe V19 geometry previously failed some tests,
so V20 must be re-evaluated instead of assumed better.

## B action contract

After `default + scale * clip(raw_action)`, before the existing rate/acceleration limiter:

`g = clip(1 - abs(raw_vy)/.08 - abs(raw_wz)/.20, 0, 1)`

Inward hip target multiplied by `1 - g*(1 - radians(8)/.22)`.
Left negative/right positive are inward according to actual RS01 URDF axes.
Outward targets, thigh/calf targets unchanged. Pure lateral/yaw commands outside the gate
recover the original mapping. Reverse and march use the straight limit.
This limits requested targets, NOT physical joint state; actual joints may exceed8deg.
Command changes remain smoothed by the existing rate/acceleration limiter.

Default target projection is identity for all pre-V20 tasks. Mapping lives in
`rs01_omni_v20_env.map_hip_targets`; B must use this exact mapping in MuJoCo and real deployment,
after action scaling and before target limiting. Bare actor ONNX with an old deployment
controller is NOT compatible. Deployment adapters are not changed or validated in this training task.

## Hardware and training unchanged

Actual `dog_urdf/urdf/dog_rs01.urdf`: 11.7317kg, thigh180mm/calf202.158mm, point-foot radius16mm.
Keep FL+RR/FR+RL diagonal gait, stance ratio .72 and18mm clearance.
Keep RS01 identified delay/response/friction, Kp40/Kd1, .22rad action scale,
rate2/2.6/3.2rad/s, acceleration60/78/96rad/s², policy20ms, PD2.5ms,
14Nm operating peak guard and6Nm continuous reference. No ideal-actuator substitution.
No physical torque, velocity, or thermal constraints relaxed.
Full previous command ranges retained. PPO fixed1e-4; optimizer freshly initialized on resume.

## Run and rollback

Unit checks: `python -m unittest discover -s tests -p 'test_rs01_omni_v*.py'`.
Manual long run: `bash legged_gym/scripts/train_rs01_v20_dual.sh --start`.
GPU0=A, GPU1=B,4096env,3000 additional iterations15000→18000, same seed16.
Launcher prints PIDs and tails both logs. Ctrl+C leaves background training running.
Existing checkpoints untouched; rollback by selecting V19 task and original model15000.
No long training is launched by the assistant.

## Final conditional smoke results

30 unit tests passed. Each branch:512env,100 PPO iterations, seed16, fresh optimizer,
model15000→15100. Runs:

- A: `logs/rs01_omni_v20_geometry/Sep13_18-27-05_v20_A_conditional_smoke`.
- B: `logs/rs01_omni_v20_bounded_hip/Sep13_18-27-07_v20_B_conditional_smoke`.

Nominal flat, no randomization, eval seed20260922,23march/motion cases×4env×10s:

| Metric | A | B |
|---|---:|---:|
| Resets | 0 | 0 |
| Flight observed at50Hz | 0 | 0 |
| Finite observations/rewards | Yes | Yes |
| Raw PD peak Nm (NOT applied torque) | 25.48 | 15.35 |
| Peak joint speed rad/s | 20.35 | 17.23 |

Posture12s, measure6–12s,4env, eval seed20260918:

| Command | A RF-ahead mm | B RF-ahead mm | A front width mm | B front width mm |
|---|---:|---:|---:|---:|
| March | 59.2 | 34.2 | 194.8 | 235.4 |
| Forward .2m/s | 16.5 | 11.9 | 205.0 | 235.6 |
| Reverse -.2m/s | 71.5 | 26.4 | 196.7 | 230.2 |

B actual inward hip means: march5.5/5.6/4.0/4.0deg,
forward6.2/4.9/4.0/3.7deg, reverse6.3/5.9/5.7/5.7deg (FL/FR/RL/RR).
B is the more promising intervention. A remains a comparison, not a gait-quality winner;
its raw torque peak and reverse bias need particular attention. Residual fore/aft bias remains.
These are smoke tests, not sustained thermal, multi-seed, MuJoCo or real acceptance.
Raw posture artifacts are in `artifacts/rs01_v20/A_final` and `B_final`.

Both final125s sequences (25commands,5s each,4env,seed20260922) completed with
0resets and finite observations/rewards. March separates movements; no random stand.
Sequence JSONs in `artifacts/rs01_v16_selection/`:

- A: `sequence_15100_20260922_1789295613076483967.json`.
- B: `sequence_15100_20260922_1789295629824177026.json`.

Training/evaluation logs archived to `artifacts/rs01_v20/logs/`.
Initial unconditional variant sequences were stopped after the failed A wide test;
only the final conditional sequences above are completed acceptance evidence.
