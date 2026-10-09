# V22 omnidirectional inward guard

## Scope / repository

Based on `d2597ab`, branch `fix/rs01-heading-odom-soft-inhibit`.
Preserves existing tasks/checkpoints and the user's unrelated `11.md` changes.
New tasks: `rs01_v22_phase_guard5` and `rs01_v22_phase_guard3`.
No long training or deployment is authorized/performed by this change.

## Physical and control audit

URDF: `dog_urdf/urdf/dog_rs01.urdf`, mass 11.7317368323 kg,
thigh 0.180 m, calf-foot 0.202158 m. Inward rotation is negative on the
left and positive on the right (all hip axes +X). At the original default
thigh/calf angles, URDF forward kinematics gives:

| Hip inward rotation | Front/rear foot spacing | Foot Z relative to base |
|---|---:|---:|
| 0 degrees | .294500 m | -.300000 m |
| 3 degrees | .262860 m | -.304155 m |
| 5 degrees | .241542 m | -.306463 m |
| 8 degrees | .209298 m | -.309223 m |

These are static geometry, NOT evidence of dynamic stability or torque feasibility.
Other leg angles and body roll can still move a foot inward. Prior complete physical
audit: `../artifacts/rs01_v22_phase_scratch/REPORT.md`.
Unchanged real RS01 chain: clipped/scaled actor target -> **new projection** ->
target rate/acceleration limits -> identified delay/response/backlash -> PD ->
motor/thermal limits. Policy 50 Hz, physics 400 Hz, Kp/Kd 40/1,
configured motor peak 14 Nm; hardware 17 Nm capability and 6 Nm continuous
reference are NOT reinterpreted or changed. No qpos/velocity clamp, no URDF edits.

## Evidence and minimal change

| Problem | Evidence | Change | Check |
|---|---|---|---|
| Inward restriction disappears for lateral/yaw commands | V20 straight factor becomes zero | Replace V20 map with command-independent inward compression | All command modes have same signed bound |
| Phase does not constrain foot placement | Phase-lift actor can still request inward targets | Bound inward target to 5 or 3 degrees | Actual hip and stance-foot metrics, not target alone |
| High-speed instability has more than one possible cause | 29750 failed fast switching and long right strafe | Keep baseline and evaluate same sequence | Resets, roll/pitch, raw/motor torque, flight |

We deliberately do not simultaneously change rewards, cadence, PD, width thresholds,
sensor/effective-command semantics or action observations. This isolates the guard's
effect and avoids stacking constraints/rewards. Stance-region reward replacement
and command-transition shaping remain follow-ups only if measured necessary.
The existing placement reward already prefers wider support and does not demand
the inward targets being removed here. No exact left/right symmetry is enforced.

## Implementation / deployment contract

`rs01_v22_phase_guard.py` overrides projection without calling V20's projection:
for an inward hip target, multiply by `limit / (action_scale * clip_actions)`;
outward hips and thigh/calf targets are unchanged. Mapping is continuous at zero
and has no command/phase boundary jumps. Limits are constant across support/swing;
the existing target rate limiter smooths changes. Zero default hip angles are
validated. The target bound does NOT guarantee the actual joint stays in range
under inertia, compliance or contact load.

**Do not deploy/export a new guard actor as a bare V22 actor.** MuJoCo and the real
controller must reproduce this exact replacement before their target rate limiter,
not apply both V20 and this mapping. Deployment parity is not implemented here;
the new candidates are training-only until that parity is tested.

## Validation

Unit tests: `tests/test_rs01_v22_phase_guard.py` covers signed limits, all command
modes, no double mapping, outward/non-hip preservation, invalid defaults, and exact
equality of inherited plant/reward/sensor/PPO configs except guard/run name.
Existing phase and V22 sensor tests also run.
`select_play.py` accepts both new tasks and reports stance-inside fraction using
the existing per-command half-width threshold, actual inward hip peak, roll/pitch,
raw and motor torque peaks and flight. These are policy-rate samples; inside fraction
is a body-frame foot-position proxy, NOT a dynamic balance margin. Sequence metrics
include transitions; post-reset samples are not treated as uninterrupted success.

Comparison protocol: checkpoint29750, seed20261010, NORMAL sensors, 4 environments,
13 commands switched directly every3 seconds, 39 simulated seconds/environment,
vx/vy up to .2m/s and yaw .3rad/s. Same saved actor, no adaptation, across all three
tasks. This initial intervention test is not a trained candidate comparison.

### Same-actor intervention results

| Metric | Original | 5-degree guard | 3-degree guard |
|---|---:|---:|---:|
| Resets / flight | 0 / 0 | 0 / 0 | 0 / 0 |
| Mean of per-command roll RMS, degrees | 1.784 | 1.335 | 2.137 |
| Mean of per-command pitch RMS, degrees | .947 | 1.350 | 3.808 |
| Peak actual inward hip angle, degrees | 19.050 | 11.951 | 12.820 |
| Mean per-command stance-inside fraction | .0163 | 0 | 0 |
| Raw torque peak, Nm | 15.240 | 14.718 | 18.230 |
| Forward mean vx (command .2m/s) | .1824 | .1717 | .1637 |
| Left mean vy (command .2m/s) | .1799 | .1458 | .0953 |
| Right mean vy (command -.2m/s) | -.1542 | -.1407 | -.1194 |

Select **5 degrees as the adaptation candidate**, not a deployment-ready model.
Do not promote 3 degrees: it worsens posture and lateral tracking with this actor.
Actual angles exceeding target limits confirm that this is NOT a physical clamp.
All sampled motor peaks remained <=14Nm. The 3-degree case requested 18.23Nm raw,
which the real motor model could not deliver. The zero stance-inside proxy does
not mean zero adduction (the omni threshold remains only 7cm per side).
Logs: `../artifacts/rs01_v22_phase_guard/`.

### Adaptation smoke (not a best-model selection)

21 tests passed (14 phase/guard + 7 V22 sensor tests). Registry check: 61D/12
actions, all20 enabled reward methods available, inherited fixed learning rate
1e-4. Exact config equality tests confirm no actuator/reward/sensor changes.

Ran **30 PPO updates only**, 256 environments, seed16, GPU0, resume29750:
`logs/rs01_v22_phase_guard5/Oct09_15-06-48_guard5_smoke30_from29750/model_29780.pt`.
184320 environment steps, about28s reported training time; finite saved weights,
optimizer LR1e-4. Checkpoint numbers are inherited iteration counters, not29780 new
updates. Replayed29780 with the exact39s/4env/seed20261010 protocol above:

| Metric | Guard5 after30 updates |
|---|---:|
| Resets / flight / finite | 0 / 0 / true |
| Mean per-command roll / pitch RMS | 1.869 / 2.392 degrees |
| Peak actual inward hip | 13.413 degrees |
| Mean stance-inside fraction | 0 |
| Peak raw / motor torque | 15.999 / 14.000 Nm |
| Forward actual vx, command .2 | .1909 m/s |
| Left actual vy, command .2 | .1548 m/s |
| Right actual vy, command -.2 | -.1330 m/s |

It still moves with finite observations/rewards and no reset, but posture regressed
relative to the unadapted5-degree intervention. **Do not promote29780 as a better
model.** The smoke validates executable training, not convergence or a solved
balance problem. The optional long command intentionally starts from preserved29750,
not this short-run candidate. Save/evaluate intermediate models rather than selecting
the last model by default.

## Optional long training (USER starts only)

The 5-degree variant is the preferred next experiment; task names create separate
log roots and leave model29750 untouched. `max_iterations` means additional PPO
updates when resuming. Start from the preserved actor rather than overwrite it:

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python legged_gym/scripts/train.py \
  --task=rs01_v22_phase_guard5 --num_envs=4096 --max_iterations=3000 \
  --run_name=guard5_from29750_seed16 --seed=16 --resume \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_phase_lift/Oct08_19-28-46_v22_phase_lift_seed16 \
  --checkpoint=29750 --headless --sim_device=cuda:0 --rl_device=cuda:0
```

The 3-degree task is retained for controlled experiments only, not recommended for
long training from the initial results. A single seed cannot prove superiority.

## Remaining risks / rollback

Smaller target ranges can reduce lateral recovery authority and change a trained
actor's control distribution. Fine-tuning and independent-seed, longer, faster,
perturbation and second-simulator tests are required before sim2real. Wider support
can increase torque. No claim that all instability is caused by adduction.
Rollback is selecting unchanged `rs01_v22_phase_lift` with model29750; do not remove
models or revert unrelated user files.
