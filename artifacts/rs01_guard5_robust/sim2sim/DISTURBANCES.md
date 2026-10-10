# MuJoCo payload + push evaluation (2026-10-10)

Repository and hardware baseline: see REPORT.md (10f80e8, existing dirty work preserved). Only Guard5 MuJoCo playback changed; no rewards, training, model weights or real deployment changed. RS01 control, 61D sensor observations and phase semantics are unchanged.

## Implementation and validation

- `mujoko/rs01_go2/guard_disturbances.py`: external world-frame wrench at trunk COM, equivalent application point 6 cm above COM; no velocity/pose overwrite. Random horizontal direction, peak 6–18 N, sin-squared pulse 0.1–0.2 s, initial delay 4–7 s, quiet gap 6–10 s. Full strength, no curriculum ramp during evaluation.
- `sim2sim_v22_guard.py`: optional --pushes, --payload_kg [0,2], --disturbance_seed. Defaults preserve nominal playback. JSON records realized events and impulse.
- Payload is equivalent mass distributed like the trunk, unchanged COM, inertia scaled by mass ratio. This is not off-center/high-mounted cargo, and is not guaranteed identical to PhysX's recomputed inertia. Scene mass 11.731736 kg becomes 13.731736 kg at +2 kg.
- MuJoCo mj_setConst must use scratch MjData, not live data; initialization asserts robot qpos/qvel unchanged. Early `push_payload2` and `baseline_push_payload2` files are INVALID implementation debugging runs, not policy failures. Use names ending `_valid` below.
- With perturbations disabled, `nominal_regression.csv` is byte-for-byte identical to previous `fast.csv`.
- Numeric pulse integral differs from analytic sum 0.5*peak*duration by <2e-6 Ns in tested runs. Candidate and baseline seed14 received identical events and impulse (4.014948 Ns). Source compilation passed; payload and unchanged-state assertions passed.

## Matched protocol and outcomes

Model36100 candidate, model32750 baseline. Same 13-stage direct-switch sequence and NORMAL sensors as REPORT.md. All command components scaled 1.5 (cardinal ±0.3 m/s, yaw ±0.45 rad/s). No resets or recovery. Seed denotes both sensor and disturbance seed.

| Model / perturbation | Seed | Duration | Pushes | Roll RMS deg | Illegal ground-contact frames | Sampled raw peak Nm |
|---|---:|---:|---:|---:|---:|---:|
| 36100 pushes, no payload | 20261014 | 39s, 3s/action | 4 | 1.473 | 0/1950 | 19.140 |
| 36100 pushes +2kg | 20261014 | 39s, 3s/action | 4 | 1.442 | 0/1950 | 18.148 |
| 32750 pushes +2kg | 20261014 | 39s, 3s/action | 4 | 3.167 | 532/1950 (27.28%) | 15.398 |
| 36100 pushes +2kg | 20261015 | 65s, 5s/action | 8 | 2.525 | 280/3250 (8.62%) | 16.279 |

All completed, finite, no fall/overspeed stop, no sampled flight. Sampled motor peak 14 Nm in all four. Raw requests above operating cap are not actual motor torque. Contact/torque records are 50 Hz, not substep extrema.

The longer candidate run's illegal contact frames occur in backward_right (74), combined (76), combined_reverse (130). Instrumented replay (`push_payload2_seed15_contacts`) is CSV-identical and identifies RL_calf_knee_collision and RR_calf_knee_collision: both rear knee collision geometries. These count non-foot collision geometries contacting the ground, not a force-thresholded body-impact severity measure. Passing the fall detector alone is insufficient.

Candidate 39s +2kg: final 2 seconds of left/right stages average vy +0.191/-0.194 m/s versus ±0.300 requested; unintended vx -0.084/-0.038 m/s. Baseline corresponding vy +0.152/-0.125 m/s. Thus loaded stability improves in this matched sample, but tracking remains weak.

## Decision

36100 is better than32750 in this matched loaded/pushed test, but NOT accepted as fully robust: seed15 has non-foot contact and lateral tracking is insufficient. Seed and action duration changed together, so the source of cross-run difference is not isolated. No thermal endurance, real cargo COM shift or strong human shove certified. Retain deployed32750 unchanged; further diagnosis should focus on sustained loaded backward/combined support before widening force ranges.

## Playback / rollback

Use the command in REPORT.md plus `--pushes --payload_kg 2 --disturbance_seed 20261014`. Terminal PUSH lines identify exact events. For the failing-contact longer sequence use `--interval 5 --sensor_seed 20261015 --disturbance_seed 20261015`. The visualization does not draw a payload mesh; its mass/inertia are physical.

Omit --pushes and use --payload_kg 0 to recover original nominal behavior. No long training was launched.
