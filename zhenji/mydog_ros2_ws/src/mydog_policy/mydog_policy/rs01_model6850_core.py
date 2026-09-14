"""61-D A6850 inference core. No ROS, motor I/O, simulated delay or FOPDT.

Use once per 20ms synchronized sensor sample. Outputs are UNARMED candidate
targets; hardware limits, timestamp validation and startup remain external.
"""
import math
from dataclasses import replace
import numpy as np
from .rs01_model930_core import Model930Contract, Rs01Model930Mapper, Rs01NewMachineLegOdometry, wrap_pi

EXPECTED_ONNX_SHA256 = 'fa5988d2c29bb91bfd5e064532261a04f7ea7eefdffa9aeedb9ab14e197ab556'


class Model6850Contract(Model930Contract):
    expected_task = 'rs01_omni_v15_stand_phase'
    expected_observations = 61
    model_label = 'A6850'

    @classmethod
    def from_onnx_session(cls, session, onnx_path, expected_sha256=EXPECTED_ONNX_SHA256):
        if expected_sha256 != EXPECTED_ONNX_SHA256:
            raise RuntimeError('A6850 artifact hash override is not supported')
        c=super().from_onnx_session(session,onnx_path,expected_sha256)
        if c.raw.get('v15') != dict(stand_phase='freeze_and_resume',odometry='legacy'):
            raise RuntimeError('Requires V15 stand-only contract, not experimental estimator')
        if c.raw['observations']['base_linear_velocity_source'] != 'rs01_leg_odometry':
            raise RuntimeError('Actor must use sensor-only velocity')
        # Preserve exported precision at the limiter's target-crossing branch;
        # float32 rounding can change whether its stored rate resets to zero.
        arrays=dict(default=np.asarray(c.raw['default_joint_angles_rad'],dtype=np.float64))
        for field,key in [('action_scale','action_scale_rad'),('rate_limit','target_rate_limit_rad_s'),
                          ('accel_limit','target_acceleration_limit_rad_s2')]:
            arrays[field]=np.asarray(c.raw['control'][key],dtype=np.float64)
        return replace(c,**arrays)


class Rs01Model6850Core:
    def project_policy_target(self, target, command):
        return target

    def __init__(self,session,contract):
        self.session=session; self.contract=contract; self.mapper=Rs01Model930Mapper(contract)
        o=contract.raw['observations']['rs01_leg_odometry']
        self.odometry=Rs01NewMachineLegOdometry(
            nominal_base_height=o['nominal_base_height_m'],foot_radius=o['foot_radius_m'],
            height_margin=o['height_margin_m'],vertical_speed_threshold=o['vertical_speed_threshold_m_s'],
            velocity_residual_threshold=o['velocity_residual_threshold_m_s'],
            filter_alpha=o['filter_alpha'],no_contact_decay=o['no_contact_decay'],
            previous_stance_score_bonus=o['previous_stance_score_bonus'],strict_diagonal_pairs=True)
        self.reset(0.)

    def reset(self,yaw,q_policy=None,phase=0.):
        if not np.isfinite(yaw) or not 0<=phase<1: raise ValueError('Invalid reset')
        self.phase=float(phase);self.heading=float(yaw);self.turning=False;self.steps=0
        self.command=np.zeros(3);self.gait=0.;self.action=np.zeros(12)
        self.target=self.contract.default.astype(np.float64).copy() if q_policy is None else np.asarray(q_policy,dtype=np.float64).reshape(12).copy()
        if not np.isfinite(self.target).all():raise ValueError('Invalid reset target')
        self.rate=np.zeros(12);self.odometry.reset()

    def frequency(self,command):
        v=self.contract.raw['v13'];c=v['commands']
        caps=[c['forward_velocity_range_m_s'][1] if command[0]>=0 else c['backward_speed_range_m_s'][1],
              c['lateral_speed_range_m_s'][1],c['yaw_speed_range_rad_s'][1]]
        dead=v['speed_deadband'];blend=np.clip((np.max(np.abs(command)/caps)-dead)/(1-dead),0,1)
        low=1/self.contract.gait_period
        return low+(v['max_frequency_hz']-low)*blend

    def tick(self,q,dq,gyro,gravity,yaw,command,gait=1.,kinematics=None):
        q=np.asarray(q,dtype=np.float64).reshape(12);dq=np.asarray(dq,dtype=np.float64).reshape(12)
        gyro=np.asarray(gyro,dtype=np.float64).reshape(3);gravity=np.asarray(gravity,dtype=np.float64).reshape(3)
        command=np.asarray(command,dtype=np.float64).reshape(3)
        c=self.contract; dt=c.policy_dt; conf=c.raw['v13']['commands']
        if not np.isfinite(np.r_[q,dq,gyro,gravity,yaw,command,gait]).all():raise ValueError('NaN/Inf input')
        if gait not in (0,1):raise ValueError('gait must be 0 or 1')
        if np.max(np.abs(dq))>=c.raw['v14']['speed_validity_limit_rad_s']:raise ValueError('RS01 overspeed')
        if np.any(q<c.lower) or np.any(q>c.upper):raise ValueError('Joint outside URDF domain')
        if not .9<=np.linalg.norm(gravity)<=1.1:raise ValueError('Invalid projected gravity')
        if not gait and np.any(command):raise ValueError('Stand cannot have a moving command')
        for value,key in zip(command,('lin_vel_x','lin_vel_y','ang_vel_yaw')):
            lo,hi=c.raw['commands']['ranges'][key]
            if not lo<=value<=hi:raise ValueError('Command outside training range')
        # Advance the previously executed action interval before observing the
        # new state; mode switches never reset the gait clock.
        if self.steps:
            self.phase=(self.phase+dt*self.frequency(self.command)*(self.gait>.5))%1
            self.heading=wrap_pi(self.heading+dt*self.command[2])
            odom=self.odometry.estimate(q,dq,gyro,kinematics=kinematics)
            velocity=odom['base_linear_velocity'];confidence=odom['confidence']
        else:velocity=np.zeros(3);confidence=0.
        turning=(abs(command[2])>conf['direction_turn_exit_rad_s'] if self.turning
                 else abs(command[2])>=conf['direction_turn_enter_rad_s'])
        if turning!=self.turning:self.heading=float(yaw)
        self.turning=turning;self.command=command.copy();self.gait=gait
        error=wrap_pi(self.heading-yaw)
        blend=np.clip(1-abs(command[2])/conf['direction_blend_yaw_rad_s'],0,1)*(not turning)
        angle=np.clip(error,-conf['direction_rotation_limit_rad'],conf['direction_rotation_limit_rad'])
        rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        correction=(rot@command[:2]-command[:2])*blend
        correction*=min(1,conf['direction_planar_correction_limit_m_s']/max(np.linalg.norm(correction),1e-6))
        target=command.copy();target[:2]+=correction
        limit=conf['direction_yaw_correction_limit_rad_s']
        target[2]+=blend*limit*math.tanh(conf['direction_heading_gain']*error/limit)
        for i,key in enumerate(('lin_vel_x','lin_vel_y','ang_vel_yaw')):
            target[i]=np.clip(target[i],*c.raw['commands']['ranges'][key])
        obs=np.r_[velocity*c.lin_vel_scale,gyro*c.ang_vel_scale,gravity,target*c.command_scale,
                  (q-c.default)*c.dof_pos_scale,dq*c.dof_vel_scale,self.action,
                  math.sin(2*math.pi*self.phase),math.cos(2*math.pi*self.phase),
                  math.sin(error),math.cos(error),0,np.clip(2*(velocity[1]-target[1]),-10,10),
                  gait,0,np.clip(2*(velocity[0]-target[0]),-10,10),command*c.command_scale,confidence]
        obs=np.clip(obs,-c.obs_clip,c.obs_clip).astype(np.float32)
        if obs.shape!=(61,):raise RuntimeError('Observation layout mismatch')
        raw=self.session.run(['actions'],{'observations':obs[None]})[0][0]
        if np.shape(raw)!=(12,) or not np.isfinite(raw).all():raise RuntimeError('Invalid ONNX action')
        action=np.clip(raw,-c.action_clip,c.action_clip).astype(np.float64)
        desired=c.default+c.action_scale*action
        desired=self.project_policy_target(desired,command)
        # Exact current training/MuJoCo limiter, NOT the old braking-aware
        # model930 deployment limiter. Crossing resets rate as in simulation.
        desired_rate=np.clip((desired-self.target)/dt,-c.rate_limit,c.rate_limit)
        rate=np.clip(self.rate+np.clip(desired_rate-self.rate,-c.accel_limit*dt,c.accel_limit*dt),-c.rate_limit,c.rate_limit)
        nxt=self.target+rate*dt;cross=(desired-self.target)*(desired-nxt)<=0
        self.target=np.where(cross,desired,nxt);self.rate=np.where(cross,0,rate)
        if np.any(self.target<c.lower) or np.any(self.target>c.upper):raise RuntimeError('Target outside URDF')
        self.action=action;self.steps+=1
        return dict(observation=obs,action=action.copy(),target_policy=self.target.copy(),
                    target_real=self.mapper.policy_target_to_real(self.target),confidence=float(confidence),
                    phase=self.phase,estimated_velocity=np.asarray(velocity).copy())


def validate_sample_times(now,motor_times,imu_time,last_motor_times=None,last_imu_time=None,
                          max_age=.04,max_skew=.01):
    """Require same-clock acquisition times; reception times are insufficient.

    Limits are initial guarded-test bounds, NOT proven hardware tolerances.
    """
    motor_times=np.asarray(motor_times,dtype=np.float64).reshape(12)
    times=np.r_[motor_times,imu_time]
    if not np.isfinite(np.r_[times,now]).all():raise ValueError('Invalid timestamps')
    if np.any(now-times<0) or np.any(now-times>max_age):raise ValueError('Stale/future feedback')
    if np.ptp(times)>max_skew:raise ValueError('Unsynchronized joint/IMU samples')
    if last_motor_times is not None and np.any(motor_times<=last_motor_times):raise ValueError('Repeated/out-of-order motor feedback')
    if last_imu_time is not None and imu_time<=last_imu_time:raise ValueError('Repeated/out-of-order IMU feedback')
