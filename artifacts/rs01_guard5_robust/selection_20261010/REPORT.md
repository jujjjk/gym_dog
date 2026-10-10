# Robust Guard5 selection — 2026-10-10

## Scope / repository / unchanged hardware

Completed training run: `rs01_v22_guard5_robust/Oct09_22-16-53_robust_push_payload_seed16`,
5000 added updates32750→37750. Five of101 saved checkpoints tested:34750,36100,
37400,37650,37750. Early/mid/late reward peaks and final checkpoint form a shortlist,
not exhaustive proof of the global best. Base commit10f80e8; user edits preserved.
No training, reward/control changes, export or deployment in this selection turn.
Only diagnostic logs, this report and a read-only demo script were added.

Physical/actuator contract remains the one in
`../../../unitree_rl_gym/RS01_GUARD5_ROBUST_TRAINING.md`: nominal11.7317kg RS01,
.180m thigh/.202158m calf-foot, real actuator delays/rate limits/host thermal guard,
14Nm motor operating cap, 5-degree inward TARGET compression. Sensor-only61D actor,
12 actions,50Hz policy,400Hz physics. No ideal actuator or physical-state clamp.

## Fixed combined-load/push screening

All five candidates: NORMAL sensors, seed20261013,13 fixed commands×2env×20s.
Commands: march, vx±.2,vy±.2m/s,yaw±.3rad/s, four diagonals±(.2,.1,0),
combined±(.2,.08,.25). **Metrics include startup**, unlike nominal evaluator below.
Payload sampled0–2kg, nominal mixture25%; actual seed produced9/26 nominal worlds,
maximum payload1.99645kg. Full-strength sin² force pulses6–18N,.1–.2s, no curriculum
ramp.35 pulses per run. Original32750 comparison reuses the identical-protocol
baseline in `../rs01_robust_baseline_probe.log`.

Numbers below are arithmetic means of per-command RMS, not worst-case values.
XY bias is the mean norm of each case's MEAN velocity minus command, not time RMSE.

| Model | Resets / sampled flight | Roll RMS deg | Pitch RMS deg | XY bias m/s | Raw torque peak Nm |
|---|---|---:|---:|---:|---:|
| Original32750 | 0 / 0 | 1.746 | 3.092 | .0367 | 16.04 |
| Robust34750 | 0 / 0 | 1.423 | 1.269 | .0190 | 16.35 |
| Robust36100 | 0 / 0 | 1.357 | 1.212 | .0213 | 16.90 |
| Robust37400 | 0 / 0 | 1.453 | 1.177 | .0221 | 16.10 |
| Robust37650 | 0 / 0 | 1.440 | 1.228 | .0208 | 16.88 |
| Robust37750 | 0 / 0 | 1.342 | 1.191 | .0200 | 16.03 |

All observations/rewards finite. Raw requests are not applied motor torque.
All candidates surviving this protocol does NOT prove arbitrarily strong shove
tolerance.37750 is slightly better than36100 in this particular loaded fixed test.

## Nominal regression checks

No added payload/push; NORMAL sensors, seed20261010,13 commands×2env×12s,
initial2s excluded from metrics. Evaluated with unchanged `rs01_v22_phase_guard5`
because action/observation/cadence are identical without disturbances. Original
baseline results are from the previous identical fixed-command selection.

| Model | Resets | Roll RMS deg | vx / vy / wz RMSE (m/s,m/s,rad/s) |
|---|---:|---:|---|
| Original32750 | 0 | 1.288 | .0310 / .0348 / .0972 |
| Robust34750 | 0 | 1.490 | .0385 / .0390 / .1181 |
| Robust36100 | 0 | 1.343 | .0361 / .0407 / .1159 |
| Robust37750 | 0 | 1.512 | .0363 / .0413 / .1208 |

36100 best preserves nominal balance among these three new candidates. It does
NOT beat original32750 in nominal tracking. Nominal36100 contact-clock match .700,
exact diagonal support fraction .531, zero sampled same-axle flight/total flight;
mean peak saturation fraction .000353. Do not equate timing match with perfectly
simultaneous diagonal touchdown or measured load symmetry.

## Continuous loaded switching / independent seed

New read-only viewer `../play_sequence.py`: seed20261014,4env,13 actions×3s=39s,
direct switches with no inserted march (initial march only). **All4 worlds eligible
for payload and full-strength pushes**, no nominal subset.20 pulses per run.

| Model | Resets | Mean roll / pitch RMS deg | Actual inward hip peak deg |
|---|---:|---|---:|
| Original32750 | 0 | 2.270 / 5.169 | 7.381 |
| Robust36100 | 0 | 1.325 / 1.153 | 10.541 |
| Robust37750 | 0 | 1.420 / 1.134 | 9.689 |

36100 has zero sampled flight; raw peak15.352Nm. Actual inward excursion increased,
despite better body stability. The5-degree policy target bound is not an actual
joint-state bound; this maximum also includes swing and does not by itself measure
the loaded support margin. Do not hide this tradeoff or claim adduction eradicated.

## Recommendation / remaining risks

36100 is the **balanced demonstration candidate**, keeping37750 as an
alternative for loaded fixed commands. Selection weighs nominal balance as well as
loaded stability, not training reward alone.37750 has slightly lower pitch/inward
peak in the switch test, but36100 has lower roll both there and in the nominal
screen. Old32750 remains the deployed fallback.
No new real-robot qualification or MuJoCo result is claimed. Push magnitude, payload
placement, estimator error, dynamic support margin and higher speeds need further
validation before replacing a deployed policy. Simulated equivalent trunk mass is
not eccentric/high-mounted cargo.

## Longer confirmation

36100, independent seed20261015,13 fixed commands×2env×60s, full-strength combined
payload/push profile as above: **0 resets,0 sampled flight,finite**,117 delivered
pushes; maximum sampled payload1.59053kg (not an explicit worst-case2kg test).
Mean per-command roll/pitch RMS1.379/1.103 degrees, raw torque peak16.626Nm.
This is60s per world, not a continuous multiminute thermal qualification.

## Demo:4 robots,3 seconds per action, full disturbances

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python /home/nszb/gym/artifacts/rs01_guard5_robust/play_sequence.py \
  --task=rs01_v22_guard5_robust \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_guard5_robust/Oct09_22-16-53_robust_push_payload_seed16 \
  --checkpoint=36100 --duration_s=3 --num_envs=4 \
  --disturbances --speed_scale=1 --seed=20261014 \
  --sim_device=cuda:0 --rl_device=cuda:0
```

Omit `--disturbances` to disable added payload AND forces for nominal viewing.
Viewer playback is paced at no faster than real time; headless tests remain fast.
No policy weights or training configs are changed by either viewer mode.
