# RS01 AMP repair — 2026-10-07

## Scope and preserved contract

User approved stopping old long runs (checkpoints retained), rebuilding a small
style-first task, and later explicitly approved temporary reference-motion
pretraining if pure AMP fails to discover locomotion. No hardware deployment.

All tasks below retain the RS01 URDF, sensor-derived 61D actor input, 50 Hz policy,
2.5 ms physics, PD40/1, actuator delay/filter, target rate/acceleration, torque and
thermal guards. They inherit the Core policy target scales hip=.22, thigh=.40,
calf=.55 rad. These are NOT the old deployment action scales; do not deploy these
checkpoints with an older model's metadata.

Existing files/experiments from other editors were preserved. The shared AMP
checkpoint `iteration` argument matches the base runner and is retained. No
evidence supports blaming all unsuccessful learning on another editor's changes.

## Verified findings

- Original ten reference clips agree with RS01 forward kinematics within 0.0011 mm.
  This verifies geometry/order only, not dynamic feasibility.
- Existing from-scratch/RSI audits did not show locomotion. The later in-plant
  37250/37300 results resumed an old policy; they do not prove scratch learning.
- Core's style reward required every foot to have lifted recently and exact
  diagonal/four-foot support. Three-foot handoff zeroed style reward. This is an
  exploration barrier, not proof of the sole cause of failed learning.

## Isolated repairs

`rs01_amp_style_first` / `rs01_amp_style_first_slow`:

- No recent-four-foot-lift or exact-contact prerequisite for receiving style reward.
- Three-foot load transfer allowed; single-foot, flight and non-diagonal two-foot
  support retain a cost. Upright attenuation is continuous.
- Removed the prolonged-four-contact penalty; motion style belongs to AMP.
- Kept velocity task and hardware/safety terms. No old placement/mirror/clearance
  rewards inherited. Neither task resumes an old actor.

`rs01_amp_bootstrap` / `rs01_amp_bootstrap_slow` (user-authorized second stage):

- One temporary phase-indexed joint position/velocity reference reward, weight8.
- AMP reward weight0 during this supervised acquisition stage, avoiding competition.
- Fixed PPO learning rate3e-4; actual actuator constraints unchanged.
- These are reference-tracking pretraining, NOT pure AMP successes.
- To remove the teacher completely, use the corresponding `style_first` task;
  its reward scales contain no `reference_motion`. Transition is gated on physical
  rollout evidence, not training reward. No transition has been approved yet.

70% of training resets may use reference states; 30% use ordinary resets. Reference
states are never imposed during rollouts. Evaluation sets `env.test=True`, which
disables reference initialization entirely; tests verify this guard.

## Evidence and acceptance

Fixed-command audit: `legged_gym/scripts/evaluate_rs01_amp_core.py`, ten commands,
two robots each,12s, discard first2s,seed20261008. JSON/NPZ under
`../artifacts/rs01_amp/core_audit/`. Includes foot lift events, alternating diagonals,
actual velocity, resets, contact fractions, roll/pitch and policy-rate torque.
Torques are sampled at50Hz, NOT an exhaustive physics-substep peak certification.

The existing forward gate is deliberately unchanged for comparability: no reset
or flight, strict non-diagonal/non-four-foot fraction<10%, vx>=half request, all
feetP95>=10mm, >=4 diagonal alternations, >=3 lift events/foot, raw sampled torque<17Nm.
Three-foot fraction is separately reported. Passing is not Sim2Real approval.

Pure AMP checkpoint100 after gate repair: A vx=-.00011m/s, B=-.00163m/s;
both zero diagonal alternations. Those short pilots were stopped, models retained.

Bootstrap checkpoint100: A vx=-.0052m/s, B=-.0038m/s; both zero alternations.
B rear-footP95 reached14.1/25.9mm, but this is not sustained locomotion.

Current bounded bootstrap runs (400 iterations,2048env/GPU):

- `logs/rs01_amp_bootstrap/Oct07_22-09-46_reference_bootstrap_short400_seed16`
- `logs/rs01_amp_bootstrap_slow/Oct07_22-09-46_reference_bootstrap_short400_seed17`

Each run saves exact configs, reference SHA256 and33 source-file hashes in
`experiment_contract.json`. Updated shared AMP logging separates raw style score
from the applied gate and bonus. Long training remains blocked until physical
motion acceptance; no old model/checkpoint was removed.
