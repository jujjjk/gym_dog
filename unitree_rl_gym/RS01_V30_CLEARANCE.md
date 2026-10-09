# V30: small directional clearance increase

## Scope and baseline

Branch `fix/rs01-heading-odom-soft-inhibit`, HEAD `4fee2ce`; existing dirty files preserved.
Baseline is V29 GPU0 seed16 A34000 (Oct04_16-52-45), not model_9200.
Actual asset dog_rs01.urdf: mass 11.7317368323 kg, previously audited links
180/202.158 mm, spherical foot radius 16 mm. No physical-model edits.
PD 40/1, policy 50 Hz, physics 2.5 ms, 61 observations / 12 actions unchanged.
Real RS01 delay, target rate/acceleration, thermal and torque guards retained.
17 Nm is peak capability, 6 Nm continuous reference, not a replacement hard cap.

## Evidence, mechanism and change

| Problem | Evidence | Change | Check |
|---|---|---|---|
| User sees low foot lift | V29 audit records swing clearance | Existing clearance target only: straight/backward 30→35 mm; lateral/turn/march 25→28 mm | Actual swing P95 by direction |
| Risk of hopping or clipping | Higher target demands more motion within same swing time | Keep cadence, PD, actuator constraints and stability costs unchanged | Resets, flight, Z excursion, raw torque |
| Risk of inward support regression | A34000 nonstraight mean inside-10cm ratio 15.69% | Preserve existing landing/support objective | Same metric under same commands |

V30 reuses V28Robot/V25 clearance calculation: continuous directional blend,
no command-mode hard height switch. Abrupt command jumps can still jump the reward
target; there is no added temporal target filter or direct joint lift injection.
The command-dependent height is a reward reference, NOT guaranteed realized height.
Existing reward scales and all other config fields are identical (unit-tested).
The sole active clearance reward is phase_swing_clearance=-1; rear duplicate is
disabled. No active opposing foot-height cap was found. Body vertical-velocity,
torque and smoothing costs remain deliberately active; no new reward added.
FL+RR / FR+RL clock, 3 Hz lateral/turn target, slew, and duty factor unchanged.

## Files and validation

- New `rs01_omni_v30_config.py`: two height parameters, new experiment name.
- Task registration plus existing audit/play task allowlists extended.
- `tests/test_rs01_omni_v30.py`: config identity, mixed command targets,
  finite one-sided height cost. All 73 RS01 V-series tests pass.
- `train_rs01_v30_dual.sh`: user-only launcher, both seeds start at A34000.
- Short training: 256 env ×100 iterations each, seeds16/17, no long training.

Existing checkpoints are untouched.
Rollback: use `rs01_omni_v29_mince` and original A34000.

## Fixed-direction smoke results

Artifacts: `/home/nszb/gym/artifacts/rs01_v30/{baseline,fast_16,fast_17}.log`.
Same nominal physics, normal sensor profile, evaluation seed20260925,
12 directions ×4 environments ×12 seconds; steady metrics discard first2 seconds.
Baseline A34000 rerun in V30 (reward-reference-only change does not alter execution).
Short candidates: Oct04_19-22-58_v30_smoke_seed16/17, model_34100.pt.

| Metric | A34000 baseline | V30 seed16 | V30 seed17 |
|---|---:|---:|---:|
| Resets / flight | 0 / 0 | 0 / 0 | 0 / 0 |
| Forward swing height P95 mm | 24.3 | 27.5 | 26.6 |
| Backward swing height P95 mm | 22.8 | 28.9 | 24.4 |
| Left / right swing height P95 mm | 12.0 / 11.2 | 16.2 / 16.3 | 14.8 / 13.3 |
| Mean nonstraight support-inside-10cm % | 15.69 | 12.92 | 12.96 |
| Mean per-mode roll RMS deg | 1.57 | 1.48 | 1.38 |
| Mean per-mode Z peak-to-peak mm | 16.03 | 16.89 | 16.02 |
| Raw PD peak Nm, not applied torque | 13.96 | 16.37 | 16.63 |
| Mean per-mode stance slip m/s | .041 | .050 | .048 |

These are aggregated noncontact foot-height P95 values, not guaranteed minimum
clearance or every-stride peaks. V30 seed16 turning height did NOT increase
(left13.0→12.9, right14.5→13.8 mm); seed17 increased to14.1/15.1 mm.
Thus higher reference improved several modes but has not solved every mode.
Higher raw peak and slip are explicit tradeoffs, not a blanket improvement.
Keep A34000 as baseline; short candidates do not replace it for deployment.

Direct-switch smoke: seed20261004, normal sensors, 4env, 12 commands ×3s,
no interleaved march. Both candidates finite, zero resets and zero flight.
Maximum roll: seed16 6.98°, seed17 6.19°; baseline same protocol 5.62°.
This transient regression must be checked after long training, despite lower
fixed-command mean roll. Results: `switch_16.json`, `switch_17.json`.
Smoke only; no 60s fixed-direction or thermal/second-simulator approval this turn.

## Long training (not launched)

```bash
cd /home/nszb/gym/unitree_rl_gym
bash legged_gym/scripts/train_rs01_v30_dual.sh --start
```

Two GPUs, 4096 env each, 3000 additional iterations; do not interpret a smoke pass
as full gait acceptance. No speed expansion or hardware deployment. Phase parity,
long fixed-direction tests, perturbation/thermal tests and MuJoCo remain necessary
before any Sim2Real approval.
