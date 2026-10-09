# V23: full-direction balance, reward-only A/B

## State and physical baseline

Base commit4fee2ce, fix/rs01-heading-odom-soft-inhibit; prior dirty files preserved.
Original V22 B23500 and deployment retained. Actual dog_rs01.urdf mass checked:
11.7317368323 kg, joint rated torque17 Nm, rated speed32.9867 rad/s.
Geometry/PD/target limits/identified delay/FOPDT/thermal model are unchanged.
Operating torque cap14 Nm, continuous reference6 Nm, thermal full8 Nm/tau2 s.
Observations61D/actions12, policy50 Hz and V22 sensor snapshots unchanged.

## Evidence → change

The 2026-10-03 MuJoCo direct-sequence audit showed forward roll RMS1.22deg versus
left3.91deg; combined-reverse maximum10.12deg. Omni foot separation was much
narrower; host projection was more frequent. See
`../artifacts/rs01_v22_sim2sim/DIRECTION_BALANCE_REVIEW.md` for the protocol and caveats.

| Existing weakness | V23 replacement | A/B difference |
|---|---|---|
| Omni roll envelope tolerates5deg |3deg (straight stays2deg)|None|
| Omni minimum foot width14cm |A18cm/B20cm (straight stays22cm)|Width only|
| Placement penalty weight falls1→.5 in omni |1 for every direction|None|

No new reward names/scales; no duplicate roll, symmetry, contact or tracking penalty.
V18 still computes the sole envelope roll cost; V21 still computes width and conditional
half-cycle coordination; V19 subtracts the placement cost once. V23 overrides only the
existing placement weight. Basic diagonal support remains; no global action/force mirror.
Existing inward action map is unchanged (new hard clipping would change the execution contract).

Both tasks inherit the same sensor randomization, command distributions/ranges and PPO.
Warm-start B23500 with fresh optimizer, fixed LR1e-4, same seed16 for controlled A/B.
No command smoothing in this first experiment: previous audit found it insufficient alone,
and V21 history invalidation must be addressed before introducing continuously changing commands.

## Files

- `legged_gym/envs/rs01_omni_v2/rs01_omni_v23_config.py`: A/B parameters.
- `legged_gym/envs/rs01_omni_v2/rs01_omni_v23_env.py`: replacement placement weight.
- `legged_gym/envs/__init__.py`: two new task registrations, prior entries preserved.
- `tests/test_rs01_omni_v23.py`: isolation, reward topology, geometry and optimizer tests.
- `legged_gym/scripts/audit_rs01_v22.py`: accepts V23; optional12-direction fast suite;
  original six-command default unchanged.
- `legged_gym/scripts/train_rs01_v23_dual.sh`: explicit --start, duplicate launch protection,
  separate GPU/logs, original checkpoint, no automatic execution.

## Verification

15 V21/V22/V23 unit tests passed. Both A/B completed20 PPO iterations with256 envs,
seed16, resumed23500→23520. Training finite; no import/reward lookup/runtime error.
This is an integration smoke, not evidence of a learned stable improvement.

Isaac Gym evaluation: seed20260925, Normal sensors, 4 envs per command,12s per command,
statistics exclude first2s, reset counts include the complete interval. Actor observation,
sensor snapshot and reward effective-command assertions all passed; finite=True.

| Model | Common six commands: resets / flight | Fast12 commands: resets |
|---|---|---:|
| Original B23500 | Prior 60s test:0 /0 |5|
| A18 smoke23520 |0 /0 |3|
| B20 smoke23520 |0 /0 |6|

Common commands: march, vx±.2, vy+.2, wz+.3, combined(.2,.08,.25).
Fast commands: vx+.4/-.3, vy±.3, diagonals±.3/±.15, wz±.6,
combined±(.3,.15,.5). Original fast failures all occurred in reverse combined motion.
These fixed-command failures do not contradict the previous36s direct sequence passing:
the dwell duration, episode history and environment allocation differ.
Common-speed mean roll RMS A1.604deg/B1.722deg. Fast mean RMS is contaminated by resets;
do not rank failed models using average posture statistics. Do not deploy either smoke model.
No V23 MuJoCo/hardware acceptance claim; future selected ONNX needs fresh parity tests.

Logs: `../artifacts/rs01_v23/rs01_v23_*.log`.
Smoke runs: `logs/rs01_omni_v23_balance18/Oct03_14-44-02_v23_smoke18` and
`logs/rs01_omni_v23_balance20/Oct03_14-44-03_v23_smoke20`.

## User-started training

```bash
cd /home/nszb/gym/unitree_rl_gym
bash legged_gym/scripts/train_rs01_v23_dual.sh --start
```

GPU0=A18, GPU1=B20;4096 envs each; same seed16;3000 additional iterations,
expected checkpoint range23500→26500. Both resume the original
`logs/rs01_omni_v22_sensor/Sep22_09-57-44_v22_gpu1_seed16_20260922_095740/model_23500.pt`.
The launcher prints PIDs/log paths and follows both logs. Ctrl+C stops viewing, not training.
No long training was started during implementation.

## Acceptance and rollback

Evaluate intermediate models rather than assuming26500 is best. Require fixed commands
AND direct reversals across both simulators, matched seeds/thermal start and shuffled order.
Report worst-direction/seed roll/pitch, crossing/foot width, contact/flight, guard projection,
actual speed RMSE and thermal headroom; reject falls irrespective of total reward.
Keep existing RS01 constraints. If20cm sacrifices tracking or increases clipping, prefer18cm
or revisit geometry; do not raise torque limits. Command slew is a later isolated experiment.
Rollback: select original V22 task/checkpoint; no old config or artifact overwritten.
