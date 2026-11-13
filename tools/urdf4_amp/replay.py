"""Native MuJoCo replay of recorded AMP poses; this is not a controller rollout."""
import argparse
import json
import tempfile
import time
from pathlib import Path
import numpy as np
import mujoco as mj
import mujoco.viewer
from design import make_model, setup

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/urdf4_amp_design_20261113/amp_reference_v1')
p.add_argument('--clip', default='forward')
p.add_argument('--sweep', action='store_true')
p.add_argument('--follow', action='store_true', help='Translate camera with the robot, preserving mouse rotation/zoom/pan')
p.add_argument('--start-clip', default=None)
p.add_argument('--seconds', type=float, default=8)
a = p.parse_args()
if a.seconds <= 0:
    p.error('--seconds must be positive')
manifest = json.loads((a.dataset / 'manifest.json').read_text())
all_names = [c['clip'] for c in manifest['clips']]
if not a.sweep and a.clip not in all_names:
    p.error('Unknown clip; choose: ' + ', '.join(all_names))
names = all_names if a.sweep else [a.clip]
if a.start_clip is not None:
    if a.start_clip not in names:
        p.error('--start-clip must be a selected motion')
    pivot = names.index(a.start_clip)
    names = names[pivot:] + names[:pivot]
requests = {'delta': 0, 'pause': False, 'restart': False}

def on_key(key):
    if key == 78:  # N
        requests['delta'] += 1
    elif key == 66:  # B
        requests['delta'] -= 1
    elif key == 32:
        requests['pause'] = not requests['pause']
    elif key == 82:  # R
        requests['restart'] = True

with tempfile.TemporaryDirectory() as tmp:
    m = make_model(Path(tmp))
    d = mj.MjData(m)
    qa, va, _, _ = setup(m)
    clips = {name: dict(np.load(a.dataset / (name + '.npz'), allow_pickle=False)) for name in names}
    index, frame = 0, 0
    with mj.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, .18]
            viewer.cam.distance = 1.
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -20
        announced = None
        previous_xy = None
        while viewer.is_running():
            start = time.monotonic()
            if requests['delta'] or requests['restart']:
                index = (index + requests['delta']) % len(names)
                frame = 0
                requests['delta'] = 0
                requests['restart'] = False
            name = names[index]
            z = clips[name]
            length = min(len(z['amp_features']), max(1, round(a.seconds / .02)))
            if name != announced:
                print(f'Recorded motion {index + 1}/{len(names)}: {name}', flush=True)
                announced = name
            with viewer.lock():
                d.qpos[:3] = z['root_pos_world_m'][frame]
                d.qpos[:2] -= z['root_pos_world_m'][0, :2]
                d.qpos[3:7] = z['root_quat_wxyz'][frame]
                d.qpos[qa] = z['joint_pos_rad'][frame]
                mj.mj_forward(m, d)
                if a.follow and previous_xy is not None:
                    viewer.cam.lookat[:2] += d.qpos[:2] - previous_xy
                previous_xy = d.qpos[:2].copy()
            viewer.set_texts((None, None,
                              f'{a.dataset.name} | {index + 1}/{len(names)}  {name}\nRecorded poses, not live control\nN: next | B: previous | Space: pause | R: restart',
                              f'{frame * .02:.2f} / {length * .02:.1f} s\n' + ('PAUSED' if requests['pause'] else 'PLAYING')))
            viewer.sync()
            if not requests['pause']:
                frame += 1
                if frame >= length:
                    frame = 0
                    index = (index + 1) % len(names)
            time.sleep(max(0, .02 - (time.monotonic() - start)))
