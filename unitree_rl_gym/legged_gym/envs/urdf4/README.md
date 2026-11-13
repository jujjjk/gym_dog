# URDF4 omnidirectional RS01 task

`urdf4_omni_v1` trains the URDF4.0 quadruped with 12 direct actions, 240 actor observations (5 × 48), and 96 critic observations. Left/right thigh and knee signs follow the exported joint axes. The RS01 actuator uses 50 Hz target limiting, 40 ms delay, 5 ms response updates, a 31.9 ms first-order response, Kp=40/Kd=1, and a 14 N·m motor torque cap.

## Play the selected checkpoint

With the repository's Isaac Gym/PyTorch environment available, run from the repository root:

```bash
bash artifacts/urdf4_selection_20261113/play_selected.sh --sweep
```

This uses the native `play.py` viewer, leaves the camera under mouse control, and switches directions every 10 seconds. Without `--sweep`, use `--vx 0.6 --vy 0 --wz 0` for a fixed command. Nominal playback disables sensor noise, pushes and plant randomization.

The selected checkpoint is committed at `artifacts/urdf4_selection_20261113/model_5400.pt`; it does not require the ignored training logs directory. It resumed the gyroobs400 checkpoint for 5000 further iterations. All 51 saved checkpoints were screened; 5400, 3400, 3600 and 4900 received 60-second nominal and two-seed disturbed evaluations. Model 5400 ranked first in all three finalist evaluations. See the selection report and JSON results in the same artifact directory.

## Prepare, check and train

From `unitree_rl_gym` with the environment activated:

```bash
python legged_gym/scripts/prepare_urdf4_asset.py
python -m unittest tests.test_urdf4_omni
python legged_gym/scripts/check_urdf4_omni.py --headless
python legged_gym/scripts/check_urdf4_omni.py --headless --run_name randomized
bash legged_gym/scripts/train_urdf4_omni.sh --start
```

The launcher starts a fresh 5000-iteration run by default. The original SolidWorks package is in `URDF4.0.SLDASM/`; the training copy changes mesh paths, robot name and foot visuals, preserving physics. Both packages are included. The preparation script records SHA256 provenance.

## Evidence and limits

- `artifacts/urdf4_validation_20261112/`: URDF checks and early policy reviews.
- `artifacts/urdf4_selection_20261113/`: checkpoint selection, candidate summaries, preview and playback helper.
- `artifacts/urdf4_torque_20261113/`: 400 Hz per-action torque audit, JSON and per-joint CSV. Every action runs independently for 60 seconds; the steady window is 2–60 seconds. This does not measure the 10-second direction-switching sequence.

Self-collision remains disabled. Contact is sampled at 50 Hz; strict diagonal timing has not passed a dedicated acceptance test. There is no complete thermal or torque-speed model. Fast forward motion reaches 6.23 N·m RMS at the left rear knee in the nominal torque audit. A separate disturbed torque audit saw one reset during lateral motion. The selected model is a simulation candidate, not a real-robot deployment approval.

Intermediate checkpoints, full training logs, raw NPZ torque traces and raw preview frames remain local. Only the selected checkpoint is bundled. Audit scripts retain paths from the recorded workstation; use the portable playback helper above for another checkout.
