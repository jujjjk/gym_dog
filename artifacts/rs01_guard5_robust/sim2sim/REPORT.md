# Guard5 Robust 36100 — MuJoCo validation, 2026-10-10

## Scope and provenance

Repository: fix/rs01-heading-odom-soft-inhibit, 10f80e8, dirty worktree preserved.
Candidate: rs01_v22_guard5_robust/Oct09_22-16-53_robust_push_payload_seed16/model_36100.pt.
Baseline: rs01_v22_phase_guard5/Oct09_15-13-52_guard5_from29750_seed16/model_32750.pt.
No training, rewards, deployed policies, or actuator parameters changed.

The exporter and Guard5 viewer now accept the robust task explicitly, preserving task identity instead of relabeling the checkpoint. Export parity error: 7.153e-06 (61 observations, 12 actions); strict checkpoint loading passed.

RS01 URDF SHA256: 48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4. Existing audited mass 11.7317368 kg, thigh 0.180 m, calf-to-foot 0.202158 m, foot radius 0.016 m. Scene conversion re-read and checksum-validated this URDF.

The bridge uses actual torque-driven data.ctrl, not animated joint positions. Policy 50 Hz; physics and PD feedback 400 Hz; identified delay/response, target rate/acceleration limits, sensor-derived 61D observation and host guard retained. The 5-degree inward limit constrains requested targets, not physical joint state. No automatic resets or hidden recovery.

## Protocol

NORMAL sensor profile; phase 0; flat nominal ground; no external pushes or payload. Initial march, then forward/backward/left/right, left/right yaw, four diagonals, combined/reverse combined; no interleaved march. Normal commands: cardinal translation 0.2 m/s, yaw 0.3 rad/s, diagonals (±0.2, ±0.1), combined ±(0.2, 0.08, 0.25). Fast scales all components by 1.5.

## Results

| Model / test | Seed | Seconds | Fall / overspeed | Sampled flight | Roll RMS deg | Vertical velocity RMS m/s | Sampled raw peak Nm |
|---|---:|---:|---|---:|---:|---:|---:|
| 36100 normal, 3s/action | 20261014 | 39 | 0 / 0 | 0 | 1.448 | 0.0494 | 17.494 |
| 36100 fast, 3s/action | 20261014 | 39 | 0 / 0 | 0 | 1.561 | 0.0677 | 19.003 |
| 32750 fast, 3s/action | 20261014 | 39 | 0 / 0 | 0 | 1.355 | 0.0728 | 18.129 |
| 36100 fast, 5s/action | 20261015 | 65 | 0 / 0 | 0 | 2.783 | 0.0638 | 17.536 |

All completed and finite. First three tests: zero sampled illegal contact, sampled motor peak 14 Nm. Raw torque and motor torque are distinct; raw requests exceed the operating cap occasionally. Torque/contact recorder samples at 50 Hz and does not certify absence of substep events. Joint speed maxima are tracked across physics substeps.

Fast 39s candidate: pitch RMS 1.375 degrees versus baseline 1.609. Full-sequence vx/vy/wz RMSE including switches: candidate 0.1035/0.0844 m/s / 0.1909 rad/s, baseline 0.1012/0.0819 / 0.1805. These are not steady-state tracking errors.

Candidate fast cardinal stages, mean over final 2 seconds of each 3s stage:

| Stage | Command | Actual primary speed | Unwanted vx or vy |
|---|---:|---:|---:|
| Forward | vx +0.300 | +0.272 m/s | vy +0.020 m/s |
| Backward | vx -0.300 | -0.313 m/s | vy -0.009 m/s |
| Left | vy +0.300 | +0.255 m/s | vx -0.068 m/s |
| Right | vy -0.300 | -0.202 m/s | vx -0.008 m/s |
| Left yaw | wz +0.450 | +0.489 rad/s | vx -0.040 m/s |
| Right yaw | wz -0.450 | -0.507 rad/s | vx -0.031 m/s |

## Decision / limitations

Suitable for viewing and further simulation evaluation, not proof of Sim2Real readiness. Fast lateral tracking remains asymmetric and left translation has unintended backward drift. Longer independent-seed run has larger roll RMS; seed and dwell time changed together, so neither is isolated as the cause. Robust candidate is not universally better than 32750 in nominal MuJoCo. Push/payload transfer, sustained thermal behavior, actual inward stance margin and detailed diagonal contact timing are not certified by this test. Keep deployed 32750 unchanged.

## Reproduce

From /home/nszb/gym, activate unitree-rl, then:

```bash
python mujoko/rs01_go2/sim2sim_v22_guard.py \
  --task rs01_v22_guard5_robust \
  --scene artifacts/rs01_guard5_robust/sim2sim/scene.xml \
  --policy artifacts/rs01_guard5_robust/sim2sim/robust_36100.onnx \
  --interval 3 --speed_scale 1.5 --sensor_seed 20261014 \
  --output artifacts/rs01_guard5_robust/sim2sim/viewer_fast --viewer
```

Omit --viewer for headless reproduction; use a new output basename to preserve original measurements. For the 65s test use --interval 5 --sensor_seed 20261015. Normal test uses --speed_scale 1. Baseline uses the original guard5_32750.onnx and --task rs01_v22_phase_guard5, otherwise identical settings.

Rollback: existing Guard5 default task and original ONNX remain unchanged; use original model32750. No retraining is needed to undo this export/viewer-only change.
