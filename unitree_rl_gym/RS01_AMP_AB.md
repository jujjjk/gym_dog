# RS01 AMP A/B — 2026-10-06

Both groups warm-start actor/critic from AMP A37000, seed 16, same ten commands.
GPU 0: `rs01_amp_control`, original reference, 2 Hz / 0.65 duty / 60–50 mm.
GPU 1: `rs01_amp_retime`, revised reference, 1.5 Hz / 0.60 duty / 40–35 mm.
Each reference motion lasts 10 s at 50 Hz. No new reward terms or weight changes.
Actor observations, real RS01 actuator limits, delays and safety guards are unchanged.
Reward gait clock and reference timing are checked at startup.

A restores the original AMP discriminator. B initializes a new discriminator and
normalization only when the reference SHA differs; same-reference resumes restore it.
Both reset the PPO optimizer as in the existing training configuration.
This compares two training recipes, not a single-variable causal experiment.

Reference peak joint speed: 7.590 -> 4.560 rad/s.
Maximum reference joint-speed / target-rate-limit ratio: 2.568 -> 1.754.
The revised reference STILL exceeds limits: it is a kinematic style prototype,
not a physically validated teacher, MPC solution, or deployment qualification.
Longer training cannot guarantee exact imitation of this reference.

User-started 2-GPU training (3000 additional iterations from 37000):
```bash
cd /home/nszb/gym/unitree_rl_gym
bash legged_gym/scripts/train_rs01_amp_ab.sh --start
```
Logs appear under `artifacts/rs01_amp/ab_long_<timestamp>/gpu{0,1}.log`.
Ctrl+C stops log viewing only; training continues. Do not run duplicate launchers.

Smoke: `bash legged_gym/scripts/train_rs01_amp_ab.sh --smoke`
(256 environments / 100 iterations per group). Smoke success is not evidence
of better gait: compare tracking, resets, clearance, inward stance, body oscillation
and torque on the same fixed commands and continuous switches before promotion.

## Smoke results

Both runs completed 100 iterations (checkpoint 37100). 5 AMP tests and 73 existing
gait regression tests passed. Nominal physics, normal sensors, seed 20261006,
10 fixed commands x 4 environments x 12 s:

| Metric | A control | B retimed |
|---|---:|---:|
| Resets / maximum flight fraction | 0 / 0 | 0 / 0 |
| Mean per-command swing clearance P95, mm | 22.28 | 30.63 |
| Mean inward support fraction (<10 cm), % | 10.06 | 8.93 |
| Mean roll RMS, deg | 1.40 | 1.79 |
| Mean planar velocity bias norm, m/s | 0.0167 | 0.0112 |
| Peak raw PD request before limiting, Nm | 15.64 | 21.46 |

B improves measured foot clearance but increases roll and requested torque.
Raw PD request is NOT applied motor torque. Neither smoke model replaces A37000;
the long run is an experimental comparison, not an approved deployment model.
Evidence: `artifacts/rs01_amp/ab_smoke_20261006_145212/` (training, audits, switches).
