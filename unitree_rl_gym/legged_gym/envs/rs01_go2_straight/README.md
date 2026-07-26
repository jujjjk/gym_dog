# RS01 Go2-style straight task

Task name: `rs01_go2_straight`

This task is independent of the existing dog and fanfan tasks. Its learning
surface follows the repository's Go2 task:

- 48 observations: body velocity, gravity, command, joint state and last action;
- 12 direct joint-position residual actions;
- flat ground and forward-only velocity commands;
- the compact Go2 reward set, with no CPG, phase schedule, symmetry target,
  fixed stride, contact schedule or reference policy.

The robot-specific substitutions are:

- `dog_urdf/urdf/dog_rs01.urdf`;
- the supplied 0.316 m standing reset and joint angles;
- real controller gains and target rate/acceleration limits;
- per-motor response gain, time constant, observed closed-loop delay and
  Coulomb friction from `rs01shujv/rs01_actuator_data_20260720.json`;
- 6 N·m continuous rating retained as telemetry and a 17 N·m hard
  electromagnetic peak limit.

The measured delay is intentionally named `observed_closed_loop_delay_s`. The
identification report states that it cannot be interpreted as pure
communication delay.

Train from scratch:

```bash
python3 legged_gym/scripts/train.py \
  --task=rs01_go2_straight \
  --headless \
  --run_name=go2_minimal_from_scratch
```

Play a checkpoint:

```bash
python3 legged_gym/scripts/play_rs01_go2_straight.py \
  --task=rs01_go2_straight \
  --load_run=<run-directory> \
  --checkpoint=<iteration>
```
