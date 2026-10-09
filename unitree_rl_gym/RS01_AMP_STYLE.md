# RS01 command-conditioned AMP style experiment

## Scope / repository

Branch fix/rs01-heading-odom-soft-inhibit, HEAD4fee2ce; existing dirty edits preserved.
Independent task `rs01_amp_style`, warm-start V29 A34000. No existing task weights,
URDF, actuator implementation, actor input or deployment code replaced.
Actual URDF reread: 11.7317368323kg; thigh~180mm, calf~202.158mm, foot radius16mm.
Retains RS01 PD40/1, target rate/acceleration limits, measured delay/FOPDT, friction,
thermal/torque guards. Physics2.5ms, actor50Hz, sensor-derived61D actor and12actions.
17Nm peak capability and6Nm continuous reference remain distinct. Kp40 yields
2/4/8Nm for .05/.1/.2rad error before damping; ideal actuators are not substituted.

## Evidence/change matrix

| Problem | Evidence | Change | Acceptance |
|---|---|---|---|
| Higher reward target did not retain foot lift | V30 A35000 forward P95 21.5mm vs V29 A34000 24.3mm | Learned motion-transition style reward | Actual foot lift/contact/roll, not discriminator score alone |
| Ideal reference not executable open loop | 60mm design physically~27mm; forward request~1m actually-.19m | Treat reference as style prior, keep real plant | No promise to reproduce impossible timing |
| Old3Hz lateral vs reference2Hz conflict | Reference manifest2Hz,.65 duty | Single2Hz clock,.65 duty in AMP task only | Actor/contact/style timing consistent |
| Duplicate style objectives | Old clearance and half-cycle coordination | Disable clearance reward and coordination weight | Retain only task/contact/safety/minimum-footprint requirements |
| Reset/command discontinuities look like motion | Auto reset returns initial next observation | Capture truth before reset; exclude done, previous-reset and command changes | Unit tests, rollout finite checks |

## Reference and learning

Source `/home/nszb/gym/artifacts/rs01_reference/design_10s_20261005`: ten clips,
500frames each,50Hz. Explicit opt-in to unvalidated kinematic references. No dynamics
approval is implied. Training samples the same ten body-frame commands uniformly,
resamples every5s; no unsupported high-speed extrapolation claimed.

Discriminator: two46D state vectors +3D raw command =95D. Features are joint q/dq,
body-frame foot positions, true body linear/angular velocities, projected gravity,
height above environment origin. Privileged truth is discriminator-only; policy
continues seeing unchanged sensor-derived61D. Same FL/FR/RL/RR ordering.
Reference states never set the simulated robot's base or joints during training.

Least-squares discriminator real+1/fake-1, gradient penalty10, Adam1e-4,
4batches×512 per PPO update,131072-transition policy replay. Real/fake minibatches
match command distribution; normalisation uses reference mean/std with physical
minimum scales. Uniform finite reference transitions; no joining clip boundaries.
Style reward=max(0,1-.25*(D-1)^2), multiplied by4*dt at full weight, ramped over
200 **new AMP updates**, contact-legality and valid-transition gates. Maximum
style contribution .08/policy step versus tracking14*dt=.28 before gating.
The full rollout reward is logged consistently; AMP diagnostics are separate.

Full AMP checkpoints include discriminator, its optimizer, normalisation, update
count, reference SHA256; PPO checkpoint format retained. Reference changes cause
resume failure. Policy replay starts empty on resume. Old61D checkpoint initializes
actor/critic only; discriminator starts fresh. `load_optimizer=False` resets PPO
optimizer as inherited; AMP discriminator optimizer is restored for AMP resumes.

## Files / validation

- `rs01_amp_config.py`, `rs01_amp_env.py`: isolated task, sampler, clock and46D truth.
- `legged_gym/algorithms/rs01_amp.py`: conditional AMP and runner adapter.
- `task_registry.py`: AMP runner selected only by explicit config flag.
- Existing audit/play scripts accept task and use its10 supported commands.
- `tests/test_rs01_amp.py`: plant/actor unchanged, boundary rejection, discriminator
  update and checkpoint-state restoration. Three tests passed.
- Two-iteration integration smoke passed, strict61D checkpoint load succeeded.
- Two256env×100iteration training smokes completed;73 existing V-series regression
  tests plus3 new AMP tests passed. AMP updates100 saved in both checkpoints.

## Short-run evidence (not convergence)

Runs `logs/rs01_amp_style/Oct05_14-41-29_amp_smoke_seed16` and `...seed17`,
model_34100.pt, both initialized from original V29 A34000. Nominal physics,
normal sensor profile, seed20261005, ten reference commands ×4env ×12s.
Comparison baseline A34000 replayed in exactly the same new2Hz/.65 environment,
not compared to its old3Hz playback. Artifacts: `../artifacts/rs01_amp/`.

| Metric | Same-env A34000 | AMP seed16 | AMP seed17 |
|---|---:|---:|---:|
| Resets / flight | 0 / 0 | 0 / 0 | 0 / 0 |
| Mean per-mode inward-support<10cm % | 16.94 | 10.17 | 7.69 |
| Mean per-mode roll RMS deg | 1.60 | 1.63 | 1.50 |
| Raw PD peak Nm, NOT applied torque | 16.86 | 16.01 | 17.73 |
| Forward foot height P95 mm | 25.2 | 19.4 | 23.0 |
| March foot height P95 mm | 15.3 | 18.2 | 24.9 |
| Left foot height P95 mm | 18.2 | 24.4 | 26.3 |
| Right foot height P95 mm | 23.6 | 21.4 | 28.2 |

No claim of overall improvement: forward clearance fell; seed17 raw demand exceeds
17Nm transiently, so it is not a hardware-ready recommendation. Reference50/60mm
has NOT been reproduced. Discriminator expert score~+.90 and policy~- .91 means
the style difference remains easily distinguishable. Last rollout-step mean style
bonus .00486/.00386; valid-transition fractions .992/.996. Weight ramp at100updates
is only half complete. Lower inward-support is useful evidence, not proof of gait
imitation or that AMP alone caused it. No no-AMP training ablation was run.

Direct-switch validation:4env, seed20261006,10commands×3s=30s, normal sensors,
no interleaved stationary stand. Both candidates finite, zero resets and flight;
maximum roll seed16=5.19°, seed17=5.09°. These are short software/locomotion smoke
checks, not60s fixed-mode, thermal or Sim2Real acceptance. Long launcher was
syntax-checked only and was NOT executed.

## Commands (long training not launched)

```bash
cd /home/nszb/gym/unitree_rl_gym
bash legged_gym/scripts/train_rs01_amp_dual.sh --start
```

Two GPUs, seeds16/17,4096env each,3000 additional iterations, both warm-start A34000.
This is a learning experiment, not hardware certification. No deploy or ONNX export.
Retain V29 A34000 and V30 A35000. Rollback by selecting their original task/run.
Long physical/thermal, multi-seed and second-simulator gates remain required.
