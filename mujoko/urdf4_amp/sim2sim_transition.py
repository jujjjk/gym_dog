"""Verify the exported URDF4 transition student, using the Gym command controller."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import isaacgym  # Gym's shared controller classes must be imported before torch.
import numpy as np
import torch

import legged_gym.envs
from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat
from legged_gym.envs.urdf4.amp_rough_env import rough_height_samples
from legged_gym.envs.urdf4.amp_transition_env import TransitionPathController, Urdf4AMPTransitionRobot
from legged_gym.envs.urdf4.amp_v3_env import Urdf4AMPV3Robot
from legged_gym.envs.urdf4.gait_clock import CycleMean
from legged_gym.envs.urdf4.path_tracking import PathGains, rotate_xy, wrap
from legged_gym.envs.urdf4.urdf4_omni_env import Urdf4OmniRobot

from sim2sim import ROOT, StudentSim, prepare_scene


OMNI = [
    (.18, 0, 0), (.50, 0, 0), (.25, 0, 0), (-.15, 0, 0), (-.30, 0, 0),
    (0, .10, 0), (0, .20, 0), (0, -.10, 0), (0, -.20, 0),
    (0, 0, .40), (0, 0, .60), (0, 0, -.40), (0, 0, -.60),
    (.15, .08, 0), (.15, -.08, 0), (-.15, .08, 0), (-.15, -.08, 0),
    (.20, 0, .30), (.20, 0, -.30), (0, 0, 0),
]


class NominalTransitionController:
    _init_transition_control = Urdf4AMPTransitionRobot._init_transition_control
    _reference_touchdown_phases = Urdf4AMPTransitionRobot._reference_touchdown_phases
    _contact_cadence = Urdf4AMPTransitionRobot._contact_cadence
    _pure_fast = staticmethod(Urdf4AMPTransitionRobot._pure_fast)
    _command_ramp = Urdf4AMPTransitionRobot._command_ramp
    _transition_torque_cap = Urdf4AMPTransitionRobot._transition_torque_cap
    _update_transition_command = Urdf4AMPTransitionRobot._update_transition_command
    _transition_amplitude = Urdf4AMPTransitionRobot._transition_amplitude
    _select_reference = Urdf4AMPV3Robot._select_reference
    _frame = Urdf4OmniRobot._frame

    def __init__(self, contract, sim):
        self.device, self.num_envs, self.dt = 'cpu', 1, contract['dt']
        self.cfg = SimpleNamespace(
            commands=SimpleNamespace(fast_ramp_s=contract['fast_ramp_s']),
            amp_cts=SimpleNamespace(reference=contract['amp_reference']),
            transition_control=contract['transition_control'],
        )
        self.foot_legs = contract['foot_order']
        self.obs_scales = SimpleNamespace(**contract['normalization']['obs_scales'])
        self.commands_scale = torch.tensor([2., 2., .25])
        self.default_dof_pos = torch.from_numpy(sim.default).reshape(1, 12)
        self.commands = torch.zeros(1, 3)
        self.target_commands = torch.zeros_like(self.commands)
        self.previous_target = torch.zeros_like(self.commands)
        self.command_start = torch.zeros_like(self.commands)
        self.external_commands = torch.zeros_like(self.commands)
        self.root_states = torch.zeros(1, 13)
        self.obs_history = torch.zeros(1, 20, 51)
        self.stale = True
        self.yaw_mean = CycleMean(1, 1, 25, self.device)
        self.command_elapsed = torch.zeros(1)
        self.episode_steps = torch.zeros(1)
        self.motion_phase = torch.zeros(1)
        self.motion_frequency = torch.full((1,), 2.)
        self.phase_was_active = torch.zeros(1, dtype=torch.bool)
        self.cold_start = torch.ones(1, dtype=torch.bool)
        self.gait_enabled = torch.zeros_like(self.cold_start)
        self.heading_target = torch.zeros(1)
        self.heading_error = torch.zeros(1)
        self.public_heading_target = torch.zeros(1)
        self.path_public_offset = torch.zeros(1)
        self.nominal_mask = torch.ones_like(self.cold_start)
        self.yaw_drift = torch.zeros(1)
        self.recovery_left = torch.zeros(1)
        self.imu = SimpleNamespace(gyro_bias=torch.zeros(1, 3))
        self.reference_id = torch.zeros(1, dtype=torch.long)
        clock = contract['phase_clock']
        self.reference_commands = torch.tensor(clock['commands'])
        self.reference_phase_offsets = torch.tensor(clock['offsets'])
        self.reference_frequencies = torch.tensor(clock['frequencies_hz'])
        self.path = TransitionPathController(
            1, self.dt, self.device, gains=PathGains(**contract['direction_gains']),
            speed_enabled=contract['speed_feedback_enabled'], speed_settings=contract['speed_feedback'],
            rough_mask=torch.tensor([contract.get('terrain_kind') == 'rough']),
            rough_cfg=SimpleNamespace(**contract['rough_ground']),
        )
        self._init_transition_control()
        self._read_state(sim)
        xy, yaw, _ = self._localized_pose()
        self.path.reset_ids(torch.arange(1), xy, yaw)

    def _read_state(self, sim):
        gyro, velocity, gravity = sim.truth()
        self.base_ang_vel = torch.from_numpy(gyro).reshape(1, 3)
        self.base_lin_vel = torch.from_numpy(velocity).reshape(1, 3)
        self.projected_gravity = torch.from_numpy(gravity).reshape(1, 3)
        self.root_states[0, :3] = torch.tensor(sim.d.qpos[:3])
        w, x, y, z = sim.d.qpos[3:7]
        self.root_states[0, 3:7] = torch.tensor([x, y, z, w])
        self.root_states[0, 7:10] = torch.tensor(sim.d.xmat[sim.trunk].reshape(3, 3) @ velocity)
        self.contacts = torch.from_numpy(sim.contacts.copy()).reshape(1, 4)
        self.actions = torch.from_numpy(sim.action).reshape(1, 12)

    def _localized_pose(self):
        yaw = yaw_from_quat(self.root_states[:, 3:7])
        xy, velocity = self.root_states[:, :2], self.root_states[:, 7:9]
        self.measured_planar_velocity = rotate_xy(-yaw, velocity)
        return xy, yaw, velocity

    def get_foot_contact_mask(self):
        return self.contacts

    def _clock_phase(self):
        return self.motion_phase

    def observe(self, sim):
        self._read_state(sim)
        self._update_transition_command()
        self.episode_steps += 1
        q = torch.tensor(sim.d.qpos[sim.qa], dtype=torch.float32).reshape(1, 12)
        dq = torch.tensor(sim.d.qvel[sim.va], dtype=torch.float32).reshape(1, 12)
        self.yaw_mean.push(self.commands[:, 2:3] - self.base_ang_vel[:, 2:3])
        frame = self._frame(self.base_ang_vel, self.projected_gravity, q, dq, self.yaw_mean.mean())
        heading = wrap(self.public_heading_target-yaw_from_quat(self.root_states[:, 3:7]))
        frame = torch.cat((frame, heading.sin()[:, None], heading.cos()[:, None]-1, self.gait_enabled.float()[:, None]), 1)
        self.obs_history = torch.roll(self.obs_history, 1, 1)
        self.obs_history[:, 0] = frame
        if self.stale:
            self.obs_history[:] = frame[:, None]
            self.stale = False
        public = torch.cat((self.obs_history.flatten(1), self.transition_goal, self.external_commands,
                            self.measured_planar_velocity, self._transition_amplitude()[:, None]), 1)
        limits = sim.c['normalization']['clip_observations']
        return public.clamp(-limits, limits)


class TransitionSim(StudentSim):
    def __init__(self, bundle, policy):
        super().__init__(bundle, policy=policy)

    def reset(self, seed=1, noise=0., offset=(0., 0.)):
        self.controller = None
        super().reset(seed=seed, noise=noise)
        self.d.qpos[:2] += offset
        self.d.qpos[2] += self.ground_height(self.d.qpos[:2])
        self.mj.mj_forward(self.m, self.d)
        self.refresh_contacts()
        self.controller = NominalTransitionController(self.c, self)
        self.public = None

    def ground_height(self, xy):
        if self.m.nhfield == 0:
            return 0.
        half_width, half_length, scale, _ = self.m.hfield_size[0]
        geom = self.m.geom_pos[self.ground]
        samples = self.m.hfield_data.reshape(int(self.m.hfield_nrow[0]), int(self.m.hfield_ncol[0]))
        x, y = ((xy-geom[:2])/[2*half_width, 2*half_length]+.5)*[samples.shape[1]-1, samples.shape[0]-1]
        ix, iy = int(np.clip(x, 0, samples.shape[1]-2)), int(np.clip(y, 0, samples.shape[0]-2))
        u, v = np.clip([x-ix, y-iy], 0., 1.)
        return geom[2]+scale*((1-v)*((1-u)*samples[iy, ix]+u*samples[iy, ix+1])
                              +v*((1-u)*samples[iy+1, ix]+u*samples[iy+1, ix+1]))

    def observe(self):
        if self.controller is not None:
            self.refresh_contacts()
            self.public = self.controller.observe(self)

    def start(self, command, enabled=True):
        self.controller.target_commands[:] = torch.tensor(command)

    def step(self, action=None):
        if action is None:
            if self.public is None:
                action = np.zeros(12, np.float32)
            else:
                with torch.inference_mode():
                    action = self.policy(self.public)[0].numpy()
        limit = None
        if self.controller.torque_burst is not None:
            limit = float(self.controller._transition_torque_cap(self.c['actuator']['peak_torque_limit_nm'])[0])
        return super().step(action, torque_limit=limit)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_rough_scene(scene, rough):
    import mujoco as mj

    heights = rough_height_samples(SimpleNamespace(**rough))
    low, high = float(heights.min()), float(heights.max())
    elevations = (heights.T[::-1]-low)/(high-low)
    tree = ET.parse(scene)
    root = tree.getroot()
    ET.SubElement(root.find('asset'), 'hfield', name='rough_floor', nrow=str(heights.shape[1]),
                  ncol=str(heights.shape[0]), size=f'40 20 {high-low:.9g} .1',
                  elevation=' '.join(f'{value:.8g}' for value in elevations.flat))
    ground = root.find("worldbody/geom[@name='ground']")
    ground.attrib.pop('size')
    ground.attrib.update(type='hfield', hfield='rough_floor', pos=f'20 0 {low:.9g}')
    tree.write(scene)
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    for i, j in ((214, 209), (188, 221), (213, 186)):
        origin = np.array([-20+i*rough['grid_m'], -20+j*rough['grid_m'], .5])
        hit = np.array([-1], dtype=np.int32)
        distance = mj.mj_ray(model, data, origin, np.array([0., 0., -1.]), None, True, -1, hit)
        if abs((.5-distance)-heights[i, j]) > 1.e-6:
            raise ValueError('MuJoCo heightfield disagrees with the shared sample grid')
    return model


def prepare_bundle(bundle, model, rough):
    checkpoint = torch.load(model, map_location='cpu')
    iteration = checkpoint['iter']
    policy_path = model.parent/'exported'/f'student_{iteration}.pt'
    meta = json.loads(policy_path.with_suffix('.pt.json').read_text())
    stairs = checkpoint['contract'].get('stairs')
    if (checkpoint['contract']['transition_control'] != meta['transition_control']
            or meta['public_size'] != (1041 if stairs else 1029)
            or meta.get('stairs') != stairs):
        raise ValueError('Student export and transition checkpoint do not agree')
    original = checkpoint['model_state_dict']
    exported = torch.jit.load(str(policy_path), map_location='cpu').state_dict()
    for key, value in exported.items():
        if stairs and key.startswith('base.'):
            key = key[len('base.'):]
        elif stairs and key.startswith('residual.'):
            key = 'stair_residual.'+key[len('residual.'):]
        source = 'student_encoder.'+key[len('encoder.'):] if key.startswith('encoder.') else key
        if source not in original or not torch.equal(value, original[source]):
            raise ValueError(f'Student export differs from checkpoint at {key}')
    env = checkpoint['experiment']['env']
    urdf = ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf'
    root = env['init_state']
    x, y, z, w = root['rot']
    meta.update(urdf=str(urdf), initial_root=root['pos']+[w, x, y, z],
                checkpoint=str(model.resolve()), checkpoint_sha256=sha(model),
                torchscript=str(policy_path.resolve()), torchscript_sha256=sha(policy_path),
                physics_dt_s=env['sim']['dt'], fast_ramp_s=env['commands']['fast_ramp_s'],
                min_base_height_m=env['rewards']['min_base_height_m'],
                amp_reference=env['amp_cts']['reference'], terrain_kind='rough' if rough else 'flat')
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle/'contract.json').write_text(json.dumps(meta, indent=2)+'\n')
    m = prepare_scene(bundle)
    if rough:
        m = prepare_rough_scene(bundle/'scene.xml', meta['rough_ground'])
    print('Scene', m.nq, m.nv, 'mass_kg', m.body_mass.sum(), flush=True)


def assess_segment(matrix, command, dt, full):
    terminal = matrix[-round(1/dt):]
    contact = matrix[:, 6:10] > 0
    count = contact.sum(1)
    diagonal = (contact[:, 0] & contact[:, 3]) | (contact[:, 1] & contact[:, 2])
    illegal = (count < 2) | ((count == 2) & ~diagonal)
    final = contact[-round(2/dt):]
    clearance = matrix[-round(2/dt):, 21:25]
    all_feet_lift = bool((~final & (clearance > .012)).any(0).all())
    two = final.sum(1)==2
    both_diagonals = bool((two & final[:, 0] & final[:, 3]).any() and (two & final[:, 1] & final[:, 2]).any())
    return dict(command=command, completed=full, seconds=len(matrix)*dt,
                actual_twist_last_second=terminal[:, :3].mean(0).tolist(),
                peak_roll_pitch_deg=abs(matrix[:, 3:5]).max(0).tolist(),
                cross_peak_m=float(abs(matrix[:, 5]).max()), raw_peak_Nm=float(matrix[:, 10].max()),
                motor_peak_Nm=float(matrix[:, 11].max()), illegal_support_s=float(illegal.sum()*dt),
                flight_s=float((count == 0).sum()*dt), fell=bool(matrix[-1, 13]),
                all_four_feet_lift=all_feet_lift, both_diagonals=both_diagonals)


def evaluate(sim, commands, offset, segment_s):
    sim.reset(offset=offset)
    dt = float(sim.policy_dt)
    steps = round(segment_s/dt)
    cases, records = [], []
    for index, command in enumerate(commands):
        sim.start(command)
        if index == 0:
            sim.step()
        rows = []
        for _ in range(steps):
            torque = sim.step()
            c = sim.controller
            gx, gy, gz = c.projected_gravity[0].tolist()
            tilt = np.degrees([np.arctan2(-gy, -gz), np.arcsin(np.clip(gx, -1, 1))])
            z = float(sim.d.qpos[2])
            fall = sim.fallen() or z-sim.ground_height(sim.d.qpos[:2]) < sim.c['min_base_height_m']
            foot_clearance = [foot[2]-sim.ground_height(foot[:2])-.02 for foot in sim.d.xpos[sim.feet]]
            rows.append(np.r_[c.base_lin_vel[0, :2].numpy(), float(c.base_ang_vel[0, 2]), tilt,
                             float(c.path.cross_error[0]), sim.contacts,
                             abs(torque[:, 0]).max(), abs(torque[:, 1]).max(), z, fall,
                             c.external_commands[0].numpy(), c.commands[0].numpy(), float(c.motion_phase[0]),
                             foot_clearance])
            if fall:
                break
        matrix = np.stack(rows)
        records.append(matrix)
        row = assess_segment(matrix, command, dt, len(rows)==steps and not fall)
        print(index, json.dumps(row), flush=True)
        cases.append(row)
        if fall:
            break
    return cases, np.concatenate(records)


def view_sequence(sim, commands, offset, segment_s):
    import mujoco.viewer

    request = {'paused': False, 'restart': True}

    def key_callback(key):
        if key == 32:
            request['paused'] = not request['paused']
        elif key == 82:
            request.update(paused=False, restart=True)

    for geom in range(sim.m.ngeom):
        if sim.m.geom_bodyid[geom] and (sim.m.geom_contype[geom] or sim.m.geom_conaffinity[geom]):
            sim.m.geom_group[geom] = 3
    dt = float(sim.policy_dt)
    segment_steps = round(segment_s/dt)
    with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = 0
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        print('MuJoCo viewer ready. Mouse: camera. Space: pause. R: restart.', flush=True)
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                if request['restart']:
                    sim.reset(offset=offset)
                    sim.start(commands[0])
                    sim.step()
                    tick = 0
                    viewer.cam.lookat[:] = sim.d.qpos[:3]
                    request['restart'] = False
                if not request['paused']:
                    index, within = divmod(tick, segment_steps)
                    if not within:
                        command = commands[index]
                        sim.start(command)
                        print(f'Segment {index+1}/{len(commands)} vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f}', flush=True)
                    previous_xy = sim.d.qpos[:2].copy()
                    sim.step()
                    viewer.cam.lookat[:2] += sim.d.qpos[:2] - previous_xy
                    tick += 1
                    if sim.fallen() or sim.d.qpos[2]-sim.ground_height(sim.d.qpos[:2]) < sim.c['min_base_height_m']:
                        request['paused'] = True
                        print('Fall detected. Press R to restart.', flush=True)
                    elif tick == segment_steps*len(commands):
                        request['restart'] = True
            viewer.sync()
            time.sleep(max(0., dt-(time.monotonic()-started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prepare-model', type=Path)
    parser.add_argument('--rough', action='store_true', help='Prepare the shared rough heightfield')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--viewer', action='store_true', help='Play the continuous sequence in the native MuJoCo viewer')
    parser.add_argument('--suite', choices=('omni', 'forward', 'endpoints'), default='omni')
    parser.add_argument('--offset', type=float, nargs=2, default=(0., 0.))
    parser.add_argument('--segment-s', type=float, default=6.)
    parser.add_argument('--peak-speed', type=float, default=.50)
    args = parser.parse_args()
    if not .35 < args.peak_speed <= .50 or args.segment_s < 2.:
        parser.error('Use a peak in (0.35, 0.50] m/s and segments of at least two seconds')
    torch.set_num_threads(1)
    if args.prepare_model:
        prepare_bundle(args.bundle, args.prepare_model, args.rough)
    if args.viewer and args.suite == 'endpoints':
        parser.error('Use the omni or forward sequence in the viewer')
    if args.output is None and not args.viewer:
        return
    contract = json.loads((args.bundle/'contract.json').read_text())
    policy_path = Path(contract['torchscript'])
    if sha(policy_path) != contract['torchscript_sha256']:
        raise ValueError('Exported policy hash changed')
    sim = TransitionSim(args.bundle, torch.jit.load(str(policy_path), map_location='cpu').eval())
    commands = list(OMNI)
    commands[1] = (args.peak_speed, 0, 0)
    if args.suite == 'forward':
        commands = [*commands[:4], commands[-1]]
    if args.viewer:
        view_sequence(sim, commands, args.offset, args.segment_s)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if args.suite == 'endpoints':
        traces = []
        cases = []
        for command in (commands[i] for i in (0, 1, 2, 4, 6, 8, 10, 12, 13, 17, 19)):
            rows, trace = evaluate(sim, [command], args.offset, args.segment_s)
            cases.extend(rows)
            traces.append(trace)
        np.savez_compressed(args.output/'traces.npz', **{f'case_{i}': t for i,t in enumerate(traces)})
    else:
        cases, trace = evaluate(sim, commands, args.offset, args.segment_s)
        np.savez_compressed(args.output/'traces.npz', samples=trace)
    report = dict(suite=args.suite, terrain=contract.get('terrain_kind', 'flat'), offset_m=args.offset,
                  model=contract['checkpoint'], sha256=contract['checkpoint_sha256'], mass_kg=float(sim.m.body_mass.sum()),
                  dt_s=float(sim.policy_dt), physics_dt_s=sim.dt, requested_segment_s=args.segment_s, cases=cases,
                  completed_all=len(cases)==(11 if args.suite=='endpoints' else len(commands)) and all(c['completed'] for c in cases))
    if args.suite == 'forward':
        for case in cases[1:3]:
            checks = {
                'fall': case['completed'] and not case['fell'],
                'speed': abs(case['actual_twist_last_second'][0]-case['command'][0]) <= .05,
                'direction': case['cross_peak_m'] <= .05,
                'body_tilt': all(x <= limit for x, limit in zip(case['peak_roll_pitch_deg'], (5., 4.))),
                'torque': case['raw_peak_Nm'] <= 17.,
                'support': case['illegal_support_s'] <= .04 and case['flight_s'] == 0.,
                'swing': case['all_four_feet_lift'],
                'diagonal': case['both_diagonals'],
            }
            case['failed_gates'] = [name for name, passed in checks.items() if not passed]
        report['fast_windows_passed'] = len(cases) >= 3 and all(not case['failed_gates'] for case in cases[1:3])
        report['peak_speed_m_s'] = args.peak_speed
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
