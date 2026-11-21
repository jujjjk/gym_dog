# Battery mass correction — 2026-11-16

Active asset: `urdf/urdf4_rs01.urdf`.

| Quantity | Before (kg) | After (kg) |
|---|---:|---:|
| trunk | 5.1784426438803 | 5.9784426438803 |
| Whole robot | 11.569864831733414 | 12.369864831733413 |

The user requested the missing 0.800 kg battery be included. Battery placement
and dimensions were not supplied. This is a provisional point-mass addition at
the original trunk centre of mass; **trunk CoM and rotational inertia are unchanged**.
The robot's aggregate CoM follows the new link masses normally. A measured
battery pose/shape is needed for a finite-volume inertia correction.

`urdf/urdf4_rs01_no_battery_legacy.urdf` is an exact snapshot of the previous asset
(SHA256 `528c088ff240916b04dfc98611ae9437b92a31cfafc9a297399632e8d0774b91`).
`source.json` records the active hash, mass delta and approximation.

Legacy AMP datasets, validation receipts, reference banks and checkpoints were
not relabeled. Their old mass/URDF validation does not certify this corrected
asset; new-motion validation and a matching reference bank are needed before
new AMP training. Existing validation/export hash guards intentionally reject
old data paired with the new asset. Historical replay must explicitly use the
legacy asset and its matching contracts.

MuJoCo scene generation verifies mass against the contracted URDF rather than a
hard-coded 11.57 kg. Cached policy bundles also check the URDF hash and scene
mass to prevent silent replay of a stale scene after a mass correction.

Import checks and exact numeric results:
`artifacts/urdf4_battery_mass_20261116/` at the repository root. These verify
asset loading and mass only; no new gait, policy-transfer or hardware validation
is claimed. Joint order, collision geometry, RS01 actuator and PD are unchanged.
