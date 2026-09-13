"""V20 B bridge: mapping -> limiter -> V16 guard -> identified RS01 plant.

No physical state projection, ideal motors, automatic resets or hidden recovery.
"""
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import mujoco
from sim2sim import limited_target_step, roll_pitch_yaw
from sim2sim_v14 import V14Sim, get_parser, run
from sim2sim_sensor_sync import SensorSyncSim


def map_targets(target, command, ids, sides, ranges, cfg):
    g = np.clip(1-abs(command[1])/cfg['vy_gate']-abs(command[2])/cfg['wz_gate'], 0, 1)
    hip = target[ids]
    out = target.copy()
    out[ids] = np.where(hip*sides < 0, hip*(1-g*(1-cfg['inward_target_rad']/ranges)), hip)
    return out


def project_guard(target, q, dq, kp, kd, limit, lower, upper):
    raw = kp*(target-q)-kd*dq
    safe = np.clip(raw, -limit, limit)
    return np.clip(q+(safe+kd*dq)/kp, lower, upper), raw, safe


def update_guard(rms_sq, torque, dt, cfg):
    alpha = math.exp(-dt/cfg['tau'])
    rms_sq = alpha*rms_sq+(1-alpha)*torque**2
    fraction = np.clip((np.sqrt(rms_sq)-cfg['continuous'])/(cfg['full']-cfg['continuous']), 0, 1)
    return rms_sq, cfg['peak']-fraction*(cfg['peak']-cfg['continuous'])


class V20Sim(V14Sim):
    task_name = 'rs01_omni_v20_bounded_hip'
    base_velocity_world = SensorSyncSim.base_velocity_world

    def __init__(self, *args):
        super().__init__(*args)
        c = self.cfg.get('v20', {})
        if c.get('mapping') != 'conditional_inward_scale_v1' or c.get('odometry') != 'legacy':
            raise ValueError('Use export_v20.py; old ONNX contracts are incompatible')
        self.hip_ids = [self.names.index(x+'_hip_joint') for x in ('FL','FR','RL','RR')]
        self.hip_sides = np.array([1.,-1.,1.,-1.])
        if not np.allclose(self.default[self.hip_ids], 0.):
            raise ValueError('V20 requires zero default hip angles')
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in self.names]
        self.lower, self.upper = self.model.jnt_range[joint_ids].T.copy()

    def reset(self):
        super().reset()
        # Preserve guard heating if a caller explicitly resets an existing sim.
        if not hasattr(self, 'guard_rms_sq'):
            self.guard_rms_sq = np.zeros(len(self.names))
            self.guard_limit = np.full(len(self.names), self.peak_limit)
        self.guard_target = self.default.copy()

    def frequency(self):
        return super().frequency() if self.gait_enable > .5 else 0.

    def update_policy(self):
        action = self.session.run(['actions'], {'observations': self.observation()[None,:]})[0][0]
        self.action = np.clip(action.astype(np.float64), -self.action_clip, self.action_clip)
        desired = map_targets(self.default+self.action_scale*self.action, self.command,
                              self.hip_ids, self.hip_sides,
                              self.action_scale[self.hip_ids]*self.action_clip, self.cfg['v20'])
        self.desired_target = desired.copy()
        self.limited_target, self.target_rate = limited_target_step(
            desired, self.limited_target, self.target_rate, self.rate_limit,
            self.acceleration_limit, self.policy_dt)
        self.guard_target, self.guard_raw_pd, self.guard_safe_pd = project_guard(
            self.limited_target, self.data.qpos[self.qpos_indices], self.data.qvel[self.qvel_indices],
            self.kp, self.kd, self.guard_limit, self.lower, self.upper)
        thermal_input = np.maximum(abs(self.guard_safe_pd), abs(self.last_motor_torque))
        self.guard_rms_sq, self.guard_limit = update_guard(
            self.guard_rms_sq, thermal_input, self.policy_dt, self.cfg['v20']['guard'])

    def physics_step(self):
        original = self.limited_target
        self.limited_target = self.guard_target
        try:
            super().physics_step()
        finally:
            self.limited_target = original


MOVEMENTS = [('forward',(.2,0,0)),('backward',(-.2,0,0)),
    ('left',(0,.2,0)),('right',(0,-.2,0)),('forward_left',(.2,.1,0)),
    ('forward_right',(.2,-.1,0)),('backward_left',(-.2,.1,0)),
    ('backward_right',(-.2,-.1,0)),('turn_left',(0,0,.3)),('turn_right',(0,0,-.3)),
    ('combined',(.2,.08,.25)),('combined_reverse',(-.2,-.08,-.25))]


def sequence_run(args):
    sim = V20Sim(args.scene, args.policy, [0.,0.,0.])
    sim.phase_value = args.phase
    sequence = [('march',(0.,0.,0.))]
    for move in MOVEMENTS:
        sequence.extend([move, ('march',(0.,0.,0.))])
    rows=[]; reason=None; viewer=None
    if args.viewer:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(sim.model,sim.data)
        viewer.cam.distance=2.
    try:
        for name,command in sequence:
            print(name, command, '5s', flush=True)
            sim.set_command(command, 1.)
            for _ in range(round(5/sim.policy_dt)):
                begin=time.monotonic(); sim.control_step()
                linear,angular=sim.base_velocity_body()
                roll,pitch,yaw=roll_pitch_yaw(sim.data.qpos[3:7])
                force,illegal,_=sim.contact_diagnostics()
                contact=force>=sim.gait_cfg['contact_threshold_n']
                rows.append(dict(stage=name,time_s=(len(rows)+1)*sim.policy_dt,
                    command_vx=command[0],command_vy=command[1],command_wz=command[2],
                    vx=float(linear[0]),vy=float(linear[1]),vz=float(linear[2]),wz=float(angular[2]),
                    roll=roll,pitch=pitch,height=float(sim.data.qpos[2]),
                    flight=int(not contact.any()),four_feet=int(contact.all()),
                    illegal_contacts=illegal,raw_peak_nm=float(abs(sim.last_raw_torque).max()),
                    motor_peak_nm=float(abs(sim.last_motor_torque).max()),
                    guard_min_nm=float(sim.guard_limit.min()),max_speed_rad_s=sim.step_max_speed))
                if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all(): reason='nonfinite'
                elif sim.step_overspeed: reason='overspeed'
                elif sim.data.qpos[2]<.18 or max(abs(roll),abs(pitch))>.8: reason='fall'
                if reason: break
                if viewer:
                    if not viewer.is_running(): reason='viewer_closed'; break
                    viewer.cam.lookat[:]=sim.data.qpos[:3]; viewer.sync()
                    time.sleep(max(0.,sim.policy_dt-(time.monotonic()-begin)))
            if reason: break
    finally:
        if viewer: viewer.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(args.output)+'.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(policy=str(args.policy),scene=str(args.scene),initial_phase=args.phase,
        requested_duration_s=125.,completed_duration_s=rows[-1]['time_s'],stop_reason=reason,
        finite=all(np.isfinite([v for k,v in r.items() if k!='stage']).all() for r in rows),
        flight_ratio=float(np.mean([r['flight'] for r in rows])),
        roll_rms_deg=float(np.sqrt(np.mean([r['roll']**2 for r in rows]))*180/np.pi),
        vz_rms_m_s=float(np.sqrt(np.mean([r['vz']**2 for r in rows]))),
        raw_peak_nm=max(r['raw_peak_nm'] for r in rows),
        max_joint_speed_rad_s=max(r['max_speed_rad_s'] for r in rows))
    Path(str(args.output)+'.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=get_parser();p.add_argument('--sequence',action='store_true')
    args=p.parse_args()
    if args.duration<=0 or not 0<=args.phase<1: p.error('Invalid duration or phase')
    if args.sequence: sequence_run(args)
    else: run(args,V20Sim)
