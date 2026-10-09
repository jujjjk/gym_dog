"""Independent V22 worlds, assembled into ONE render-only grid.

No robot state is cloned from another robot. Grid offsets affect rendering only;
worlds cannot collide with one another, like Isaac Gym independent environments.
Real RS01 dynamics, sensor snapshots and recurrent controller histories are kept.
"""
import argparse
import copy
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from sim2sim_v22 import V22Sim, FAST_MOVEMENTS
from sim2sim_v20 import MOVEMENTS
from sim2sim import roll_pitch_yaw
import mujoco
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]


class IndependentSim(V22Sim):
    sessions = {}

    def make_policy_session(self, policy):
        key = str(Path(policy).resolve())
        if key not in self.sessions:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            self.sessions[key] = ort.InferenceSession(key, sess_options=options,
                                                     providers=['CPUExecutionProvider'])
        # ONNX actor is feed-forward. Only immutable weights are shared.
        return self.sessions[key]

    def __init__(self, scene, policy, seed, profile):
        self.seed = seed
        self.sensor_profile = profile
        with torch.random.fork_rng(devices=[]):
            super().__init__(scene, policy, [0., 0., 0.])
            self.rng_state = torch.get_rng_state().clone()

    def control_step(self):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng_state)
            super().control_step()
            self.rng_state = torch.get_rng_state().clone()


def display_world(scene, sims, spacing):
    root = ET.parse(scene).getroot()
    world = root.find('worldbody')
    bodies = world.findall('body')
    if len(bodies) != 1 or root.find('asset') is not None:
        raise ValueError('Use the exported primitive RS01 scene with one root body')
    source = bodies[0]
    world.remove(source)
    for tag in ('actuator', 'sensor', 'contact', 'equality', 'tendon', 'keyframe'):
        for element in root.findall(tag):
            root.remove(element)
    columns = math.ceil(math.sqrt(len(sims)))
    rows = math.ceil(len(sims)/columns)
    offsets = np.array([[(i % columns-(columns-1)/2)*spacing,
                         (i // columns-(rows-1)/2)*spacing, 0.] for i in range(len(sims))])
    for i in range(len(sims)):
        body = copy.deepcopy(source)
        for element in body.iter():
            if 'name' in element.attrib:
                element.set('name', 'env%d_%s' % (i, element.get('name')))
        world.append(body)
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('contact', 'disable')  # Display only. Individual worlds keep contacts.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    if model.nq != sum(s.model.nq for s in sims):
        raise RuntimeError('Display joint layout mismatch')
    data = mujoco.MjData(model)
    return model, data, offsets


def update_display(model, data, offsets, sims):
    qi = vi = 0
    for sim, offset in zip(sims, offsets):
        nq, nv = sim.model.nq, sim.model.nv
        # Keep the last finite display pose if this independent world failed.
        if np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all():
            data.qpos[qi:qi+nq] = sim.data.qpos
            data.qpos[qi:qi+3] += offset
            data.qvel[vi:vi+nv] = sim.data.qvel
        qi += nq
        vi += nv
    mujoco.mj_forward(model, data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, default=ROOT/'artifacts/rs01_v22_sim2sim/B23500.onnx')
    p.add_argument('--scene', type=Path, default=ROOT/'artifacts/rs01_v20_sim2sim/scene.xml')
    p.add_argument('--num_envs', type=int, default=48)
    p.add_argument('--interval', type=float, default=3.)
    p.add_argument('--spacing', type=float, default=2.5)
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--sensor_profile', choices=('clean', 'normal', 'robust'), default='normal')
    p.add_argument('--fast', action='store_true')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/rs01_v22_sim2sim/multi48.json')
    args = p.parse_args()
    if not 1 <= args.num_envs <= 128 or not math.isfinite(args.interval) or args.interval < .02:
        p.error('Require 1..128 environments and finite interval >= .02 seconds')
    if not math.isfinite(args.spacing) or args.spacing <= 0:
        p.error('Require positive finite spacing')
    torch.set_num_threads(1)
    sims = [IndependentSim(args.scene, args.policy, args.seed+i, args.sensor_profile)
            for i in range(args.num_envs)]
    model, data, offsets = display_world(args.scene, sims, args.spacing)
    update_display(model, data, offsets, sims)
    failures = [None]*len(sims)
    steps = np.zeros(len(sims), dtype=int)
    sequence = [('march', (0., 0., 0.))]
    for movement in FAST_MOVEMENTS if args.fast else MOVEMENTS:
        sequence.extend([movement, ('march', (0., 0., 0.))])
    dt = sims[0].policy_dt
    count = max(1, round(args.interval/dt))
    viewer = None
    completed = 0
    closed = False
    start = time.monotonic()
    last_render = 0.
    try:
        if not args.headless:
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(model, data)
            with viewer.lock():
                viewer.cam.lookat[:] = [0., 0., .3]
                viewer.cam.distance = max(2., math.ceil(math.sqrt(len(sims)))*args.spacing*1.4)
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -30
            viewer.sync()
        with torch.no_grad():
            for name, command in sequence:
                print('%s %s: %.2fs, active=%d/%d' %
                      (name, command, count*dt, failures.count(None), len(sims)), flush=True)
                for sim in sims:
                    sim.set_command(command, 1.)
                for _ in range(count):
                    begin = time.monotonic()
                    for i, sim in enumerate(sims):
                        if failures[i] is not None:
                            continue
                        sim.control_step()
                        steps[i] += 1
                        finite = np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
                        roll, pitch, _ = roll_pitch_yaw(sim.data.qpos[3:7])
                        reason = ('nonfinite' if not finite else 'overspeed' if sim.step_overspeed
                                  else 'fall' if sim.data.qpos[2] < .18 or max(abs(roll), abs(pitch)) > .8
                                  else None)
                        if reason:
                            failures[i] = dict(reason=reason, time_s=int(steps[i])*dt, stage=name)
                            print('env%d stopped: %s' % (i, failures[i]), flush=True)
                    completed += 1
                    if viewer:
                        if not viewer.is_running():
                            closed = True
                            break
                        if time.monotonic()-last_render >= 1/20:
                            with viewer.lock():
                                update_display(model, data, offsets, sims)
                            viewer.sync()
                            last_render = time.monotonic()
                        time.sleep(max(0., dt-(time.monotonic()-begin)))
                    if all(failures):
                        break
                if closed or all(failures):
                    break
    finally:
        if viewer:
            viewer.close()
    update_display(model, data, offsets, sims)
    report = dict(policy=str(args.policy), num_envs=len(sims), independent_physics=True,
                  independent_sensor_rng=True, cross_environment_collisions=False,
                  interval_s=count*dt, requested_s=count*dt*len(sequence),
                  simulated_s=completed*dt, wall_s=time.monotonic()-start, viewer_closed=closed,
                  sensor_profile=args.sensor_profile, fast=args.fast,
                  environments=[dict(id=i, seed=args.seed+i, steps=int(steps[i]), failure=failures[i],
                                     final_qpos=s.data.qpos.tolist()) for i, s in enumerate(sims)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Finished: %.2fs simulated, %.2fs wall, stopped=%d/%d; %s' %
          (report['simulated_s'], report['wall_s'], sum(x is not None for x in failures),
           len(sims), args.output), flush=True)


if __name__ == '__main__':
    main()
