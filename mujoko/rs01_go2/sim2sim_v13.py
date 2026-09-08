"""V13 61-D actor bridge. Reuses unchanged RS01 motor and URDF machinery."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import mujoco
from sim2sim import Rs01Go2Sim, quaternion_rotation_matrix, roll_pitch_yaw, wrapped_angle


class V13Sim(Rs01Go2Sim):
    def reset(self):
        super().reset()
        if 'v13' not in self.cfg or self.cfg['dimensions']['observations'] != 61:
            raise ValueError('Use an export_v13.py policy, not a legacy export')
        self.phase_value = 0.0
        self.gait_enable = 1.0
        self.turning = abs(self.command[2]) >= self.cfg['v13']['commands']['direction_turn_enter_rad_s']
        self.heading_target = roll_pitch_yaw(self.data.qpos[3:7])[2]
        self.reference_xy = self.data.qpos[:2].copy()
        self.confidence = 0.0
        self.last_observation = None

    @property
    def phase(self):
        return self.phase_value

    def set_command(self, command, gait=1.0):
        c = self.cfg['v13']['commands']
        command = np.asarray(command, dtype=np.float64)
        turning = (abs(command[2]) > c['direction_turn_exit_rad_s'] if self.turning
                   else abs(command[2]) >= c['direction_turn_enter_rad_s'])
        if turning != self.turning:
            self.heading_target = roll_pitch_yaw(self.data.qpos[3:7])[2]
            self.reference_xy = self.data.qpos[:2].copy()
        self.turning = turning
        self.command = command
        self.gait_enable = gait

    def frequency(self):
        c = self.cfg['v13']['commands']
        caps = [c['forward_velocity_range_m_s'][1] if self.command[0] >= 0
                else c['backward_speed_range_m_s'][1],
                c['lateral_speed_range_m_s'][1], c['yaw_speed_range_rad_s'][1]]
        intensity = np.max(np.abs(self.command) / caps)
        dead = self.cfg['v13']['speed_deadband']
        blend = np.clip((intensity - dead) / (1 - dead), 0, 1)
        low = 1 / self.gait_cfg['period_s']
        return low + (self.cfg['v13']['max_frequency_hz'] - low) * blend

    def effective_command(self):
        c = self.cfg['v13']['commands']
        error = wrapped_angle(self.heading_target - roll_pitch_yaw(self.data.qpos[3:7])[2])
        blend = np.clip(1 - abs(self.command[2]) / c['direction_blend_yaw_rad_s'], 0, 1)
        blend *= not self.turning
        angle = np.clip(error, -c['direction_rotation_limit_rad'], c['direction_rotation_limit_rad'])
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        target = self.command.copy()
        correction = (rot @ target[:2] - target[:2]) * blend
        correction *= min(1, c['direction_planar_correction_limit_m_s'] / max(np.linalg.norm(correction), 1e-6))
        target[:2] += correction
        limit = c['direction_yaw_correction_limit_rad_s']
        target[2] += blend * limit * math.tanh(c['direction_heading_gain'] * error / limit)
        for i, key in enumerate(('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')):
            target[i] = np.clip(target[i], *self.cfg['commands']['ranges'][key])
        return target

    def observation(self):
        _, angular = self.base_velocity_body()
        linear = self.last_estimated_linear_velocity
        gravity = quaternion_rotation_matrix(self.data.qpos[3:7]).T @ np.array([0., 0., -1.])
        target = self.effective_command()
        error = wrapped_angle(self.heading_target - roll_pitch_yaw(self.data.qpos[3:7])[2])
        o = self.obs_cfg
        obs = np.concatenate((
            linear * o['lin_vel_scale'], angular * o['ang_vel_scale'], gravity,
            target * o['command_scale'],
            (self.data.qpos[self.qpos_indices] - self.default) * o['dof_pos_scale'],
            self.data.qvel[self.qvel_indices] * o['dof_vel_scale'], self.action,
            [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)],
            [math.sin(error), math.cos(error), 0., np.clip(2 * (linear[1] - target[1]), -10, 10),
             self.gait_enable, 0., np.clip(2 * (linear[0] - target[0]), -10, 10)],
            self.command * o['command_scale'], [np.clip(self.confidence, 0, 1)],
        )).astype(np.float32)
        assert obs.shape == (61,)
        self.last_observation = np.clip(obs, -o['clip'], o['clip'])
        return self.last_observation

    def control_step(self):
        super().control_step()
        self.phase_value = (self.phase_value + self.policy_dt * self.frequency()) % 1
        h = self.heading_target
        rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        self.reference_xy += self.policy_dt * (rot @ self.command[:2])
        self.heading_target = wrapped_angle(h + self.policy_dt * self.command[2])
        _, angular = self.base_velocity_body()
        result = self.leg_odometry.estimate(self.data.qpos[self.qpos_indices],
                                           self.data.qvel[self.qvel_indices], angular)
        self.last_estimated_linear_velocity = np.asarray(result['base_linear_velocity'])
        self.confidence = float(result['confidence'])


def run(args, sim_class=V13Sim):
    sim = sim_class(args.scene, args.policy, args.command)
    sim.gait_enable = float(args.march or np.linalg.norm(args.command) > 0)
    if args.settle_seconds:
        update = sim.update_policy
        sim.update_policy = lambda: None
        for _ in range(round(args.settle_seconds / sim.policy_dt)):
            sim.control_step()
        sim.update_policy = update
        sim.heading_target = roll_pitch_yaw(sim.data.qpos[3:7])[2]
        sim.reference_xy = sim.data.qpos[:2].copy()
    sim.phase_value = args.phase
    if args.zero_action:
        sim.update_policy = lambda: None  # Diagnostic: fixed default joint target.
    rows = []
    start_xy = sim.data.qpos[:2].copy()
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        viewer.cam.distance = 2.0
    reason = None
    try:
        for step in range(round(args.duration / sim.policy_dt)):
            begin = time.monotonic()
            sim.control_step()
            linear, angular = sim.base_velocity_body()
            roll, pitch, yaw = roll_pitch_yaw(sim.data.qpos[3:7])
            force, illegal, _ = sim.contact_diagnostics()
            contact = force >= sim.gait_cfg['contact_threshold_n']
            desired = sim.desired_contact()
            error = sim.data.qpos[:2] - sim.reference_xy
            h = sim.heading_target
            lateral = np.dot(error, [-math.sin(h), math.cos(h)])
            row = dict(time_s=(step+1)*sim.policy_dt, vx=linear[0], vy=linear[1], wz=angular[2],
                       x=sim.data.qpos[0], y=sim.data.qpos[1], z=sim.data.qpos[2], roll=roll,
                       pitch=pitch, yaw=yaw, heading_error=wrapped_angle(h-yaw), lateral_error=lateral,
                       phase=sim.phase, confidence=sim.confidence, estimated_vx=sim.last_estimated_linear_velocity[0],
                       estimated_vy=sim.last_estimated_linear_velocity[1], illegal_contacts=illegal,
                       exact_diagonal=int(contact.sum()==2 and np.array_equal(contact, desired)),
                       phase_match=int(np.array_equal(contact, desired)), flight=int(not contact.any()),
                       four_feet=int(contact.all()))
            if hasattr(sim, 'step_overspeed'):
                row['speed_domain_violation'] = int(sim.step_overspeed)
                row['max_joint_speed_rad_s'] = sim.step_max_speed
            for name, values in [('raw',sim.last_raw_torque),('motor',sim.last_motor_torque),
                                 ('applied',sim.last_applied_torque),('action',sim.action),
                                 ('q',sim.data.qpos[sim.qpos_indices]),('dq',sim.data.qvel[sim.qvel_indices])]:
                row.update({f'{name}_{joint}': float(v) for joint,v in zip(sim.names,values)})
            for i,leg in enumerate(('FR','FL','RR','RL')):
                row[f'contact_{leg}']=int(contact[i]); row[f'force_{leg}']=float(force[i])
                row[f'foot_height_{leg}']=float(sim.foot_heights()[i])
            rows.append(row)
            if not np.isfinite(list(row.values())).all(): reason='nonfinite'
            elif getattr(sim, 'step_overspeed', False): reason='rs01_speed_domain_exceeded'
            elif row['z'] < .18: reason='base_height_below_0.18m'
            elif abs(roll) > .8: reason='roll_above_0.8rad'
            elif abs(pitch) > .8: reason='pitch_above_0.8rad'
            if reason: break
            if viewer:
                if not viewer.is_running(): reason='viewer_closed'; break
                viewer.cam.lookat[:]=sim.data.qpos[:3]; viewer.sync()
                time.sleep(max(0.,sim.policy_dt-(time.monotonic()-begin)))
    finally:
        if viewer: viewer.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(args.output) + '.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    measured=[r for r in rows if r['time_s'] > 2.]
    summary=dict(command=args.command, march=args.march, initial_phase=args.phase, zero_action=args.zero_action,
                 settle_seconds=args.settle_seconds,
                 requested_duration_s=args.duration, completed_duration_s=rows[-1]['time_s'],
                 stop_reason=reason, policy=str(args.policy.resolve()), scene=str(args.scene.resolve()),
                 warmup_skip_s=2., finite=bool(np.isfinite([list(r.values()) for r in rows]).all()),
                 displacement_xy_m=(sim.data.qpos[:2]-start_xy).tolist(),
                 note='No automatic reset. Metrics stop at first fall; initial phases are not random seeds.')
    if measured:
        avg=lambda k:float(np.mean([r[k] for r in measured]))
        rms=lambda k:float(np.sqrt(np.mean([r[k]**2 for r in measured])))
        summary.update(mean_velocity=[avg(k) for k in ('vx','vy','wz')],
                       rmse_velocity=[float(np.sqrt(np.mean([(r[k]-c)**2 for r in measured])))
                                      for k,c in zip(('vx','vy','wz'),args.command)],
                       lateral_rms_m=rms('lateral_error'), heading_rms_rad=rms('heading_error'),
                       roll_rms_rad=rms('roll'), pitch_rms_rad=rms('pitch'),
                       exact_diagonal_ratio=avg('exact_diagonal'),phase_match_ratio=avg('phase_match'),
                       flight_ratio=avg('flight'),four_feet_ratio=avg('four_feet'),
                       illegal_contact_ratio=float(np.mean([r['illegal_contacts']>0 for r in measured])))
        raw=np.array([[r[f'raw_{j}'] for j in sim.names] for r in measured])
        motor=np.array([[r[f'motor_{j}'] for j in sim.names] for r in measured])
        summary.update(raw_p95_nm=float(np.percentile(np.abs(raw),95)),raw_peak_nm=float(np.abs(raw).max()),
                       motor_peak_nm=float(np.abs(motor).max()),motor_over6_ratio=float(np.mean(np.abs(motor)>6)),
                       saturation_ratio=float(np.mean(np.abs(motor)>=sim.peak_limit-1e-4)))
    if hasattr(sim, 'step_overspeed'):
        summary['speed_domain_violations'] = sum(r['speed_domain_violation'] for r in rows)
        summary['max_joint_speed_rad_s'] = max(r['max_joint_speed_rad_s'] for r in rows)
    Path(str(args.output) + '.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def get_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',type=Path,required=True); p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--command',nargs=3,type=float,default=[.4,0,0])
    p.add_argument('--march',action='store_true'); p.add_argument('--phase',type=float,default=0.)
    p.add_argument('--zero-action',action='store_true',help='Diagnostic only: hold default joint targets without actor')
    p.add_argument('--settle-seconds',type=float,default=0.,help='Diagnostic: settle with zero action before starting actor')
    p.add_argument('--duration',type=float,default=30.); p.add_argument('--viewer',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    return p


if __name__ == '__main__':
    p=get_parser();args=p.parse_args()
    if args.duration <= 0 or args.settle_seconds < 0 or not 0 <= args.phase < 1:
        p.error('Require duration > 0, settle >= 0 and 0 <= phase < 1')
    run(args)
