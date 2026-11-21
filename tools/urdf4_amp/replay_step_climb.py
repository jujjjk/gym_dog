"""Inspect recorded MuJoCo step-climb states in the native viewer."""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error('--speed must be positive')

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    state = mujoco.MjData(model)
    with np.load(args.rollout) as archive:
        times = archive['time_s']
        poses = archive['qpos']
        velocities = archive['qvel']
    if poses.shape != (len(times), model.nq) or velocities.shape != (len(times), model.nv):
        raise ValueError('Rollout and scene state dimensions differ')
    times = times - times[0]
    state.qpos[:] = poses[0]
    mujoco.mj_forward(model, state)
    with mujoco.viewer.launch_passive(model, state) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [-.04, 0., .22]
            viewer.cam.distance = 1.45
            viewer.cam.azimuth = 118
            viewer.cam.elevation = -22
        print(f'Native replay: {times[-1]:.2f} s recorded trajectory, {args.speed:g}x, loops after 3 s hold', flush=True)
        start = time.monotonic()
        cycle_wall_seconds = times[-1] / args.speed + 3.
        while viewer.is_running():
            elapsed = ((time.monotonic() - start) % cycle_wall_seconds) * args.speed
            index = int(np.clip(np.searchsorted(times, elapsed, side='right') - 1, 0, len(times) - 1))
            state.qpos[:] = poses[index]
            state.qvel[:] = velocities[index]
            state.time = times[index]
            mujoco.mj_forward(model, state)
            viewer.sync()
            time.sleep(.016)


if __name__ == '__main__':
    main()
