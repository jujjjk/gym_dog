# V21: phase coordination and moderate clearance

## Scope

Task `rs01_omni_v21_phase_coord` starts from V20 B18000. No hardware commands or long training launched. Keep B18000 as deployment baseline; smoke checkpoints are not approved deployment replacements.

- Replace V19 filtered mean sagittal foot bias with half-cycle-aligned contralateral foot X/Z trajectories. Front and rear pairs compared separately, not simultaneous action mirrors. Deadbands X=15 mm, Z=6 mm; Huber scales X=30 mm, Z=15 mm.
- Gate by raw intent: full on march/straight, half on reverse, fades with lateral/yaw commands. No new reward term; preserve width, roll and diagonal support objectives.
- Replace existing swing clearance target 18 mm with 25 mm. This is a phase-shaped target, NOT guaranteed physical clearance. Keep stance ratio 0.72 and gait frequency unchanged.
- History interpolates actual accumulated phase (32 policy steps), clears on episode reset/command changes, and contributes only after half a cycle. No privileged history added to actor observations. Z comparison assumes existing flat terrain.
- RS01 actuator, torque/speed/thermal constraints, V20 bounded target mapping, 61D observation and action interfaces are unchanged. Fixed learning rate 1e-4; fresh optimizer when resuming.

## Reproducible smoke audit (2026-09-21)

Two seeds 16/17, 512 environments, 100 added iterations, B18000 to 18100:

```
logs/rs01_omni_v21_phase_coord/Sep21_16-48-22_v21_smoke_seed16
logs/rs01_omni_v21_phase_coord/Sep21_16-48-23_v21_smoke_seed17
```

34 unit tests pass, including unchanged actuator/observation/reward-scale contracts, phase interpolation, startup and unwrapped phase. Logs `/tmp/rs01_v21_smoke16.log`, `/tmp/rs01_v21_smoke17.log`.

Evaluation logs: `../artifacts/rs01_v21/`. Same nominal physics, seed 20260925, four environments per command, 12 s fixed-command runs (first 2 s excluded in balance audit). Baseline is evaluated using V21 reward instrumentation but unchanged B18000 actor/physics; reward numbers across objectives are not a model ranking.

| Model | Front X mismatch mm, march/forward/reverse | Mid-swing clearance median mm, march/forward/reverse | Roll RMS degrees, march/forward/reverse |
|---|---|---|---|
| B18000 | 18.2 / 18.8 / 24.9 | 13.6 / 15.6 / 16.3 | 0.69 / 0.77 / 0.47 |
| V21 seed16, 18100 | 22.4 / 16.3 / 17.1 | 20.1 / 20.9 / 21.4 | 0.65 / 0.58 / 0.80 |
| V21 seed17, 18100 | 19.0 / 17.7 / 27.6 | 18.8 / 16.0 / 19.5 | 0.45 / 0.66 / 0.46 |

Mismatch is measured mean absolute front-foot X difference against the opposite foot half a cycle earlier, not mean foot offset. It must not be interpreted as observed hardware root cause.

All three fixed-command balance audits: zero resets, finite outputs. Seed16 forward actual vx 0.221 vs target 0.200 m/s (baseline 0.205); reverse -0.177 vs -0.200 (baseline -0.205). Seed17 march vx 0.022 vs zero (baseline 0.007). Short smoke improves some geometry/clearance but NOT every coordination or tracking metric. Do not deploy smoke weights; long training needs later selection across modes, disturbances and both simulators, followed by logged real-machine validation.

Wide audit, 23 march/moving commands (static stand excluded), 12 s each × 4 environments. All finite, zero resets and observed flight. Mean four-foot support B/seed16/seed17 = 21.6/22.0/22.6%; mean vertical velocity RMS = 0.084/0.081/0.081 m/s. No observed support/bounce regression in this short suite.

| Model | Mean roll RMS degrees | Pooled vx / vy / yaw-rate RMSE (m/s, m/s, rad/s) | Raw PD peak Nm | Maximum joint speed rad/s |
|---|---|---|---|---|
| B18000 | 1.67 | 0.033 / 0.022 / 0.086 | 15.68 | 17.24 |
| seed16 | 1.81 | 0.036 / 0.028 / 0.104 | 16.35 | 18.99 |
| seed17 | 1.43 | 0.037 / 0.029 / 0.110 | 15.51 | 17.61 |

Raw PD demand is pre-actuator demand, not a claim of delivered torque exceeding the operating limit. Tracking has regressed during adaptation; model selection must retain B18000 and require recovering this accuracy, not rank only symmetry or clearance.

Continuous sequence: both seeds completed 125 s × 4 environments, changing every 5 s (march interleaved with all 12 moving commands), **zero resets, finite outputs**. Seed16 JSON: `../artifacts/rs01_v16_selection/sequence_18100_20260925_1789980912275744854.json`; seed17 JSON: `../artifacts/rs01_v16_selection/sequence_18100_20260925_1789980899190122174.json`. This turn tested Isaac Gym only; MuJoCo and real-machine efficacy remain unverified for V21.

## User-started long training

```bash
cd /home/nszb/gym/unitree_rl_gym
bash legged_gym/scripts/train_rs01_v21_dual.sh --start
```

Two GPUs, same task with seeds 16/17, 4096 environments each, 3000 additional iterations from original B18000 (not smoke weights). Logs automatically shown; Ctrl+C stops viewing only, not training. Launcher refuses duplicate V21 training. Both runs can still learn similar biases: two seeds are a reproducibility check, not proof of reward efficacy.
