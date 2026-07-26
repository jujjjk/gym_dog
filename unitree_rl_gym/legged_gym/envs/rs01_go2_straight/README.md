# RS01 Go2-style straight task

Task name: `rs01_go2_straight`

This task is independent of the existing dog and fanfan tasks. Its learning
surface follows the repository's Go2 task:

- 84 observations: the original 48-D body/command/joint input plus four
  executed-action frames in total;
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

Observation indices `36:48` contain the current executed action. Indices
`48:60`, `60:72` and `72:84` contain the three preceding executed actions.
At 50 Hz this gives the policy four action frames spanning the actuator's
roughly 40–55 ms observed closed-loop delay. It improves observability but
does not change the delay calculation inside the actuator model.

The earlier 48-D checkpoints are intentionally incompatible with this task.
Train the 84-D policy from scratch; logs are written under
`logs/rs01_go2_straight_84`.

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
