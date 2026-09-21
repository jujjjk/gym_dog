"""B18000 deployment contract. Physical delay/FOPDT is NOT simulated on hardware."""
import math
from dataclasses import replace
import numpy as np
from .rs01_model930_core import Model930Contract
from .rs01_model6850_core import Rs01Model6850Core
from .rs01_model6850_guard import Guarded6850PolicyCore

EXPECTED_ONNX_SHA256 = '1dd9e9e337fcba95a6f4fa1126894c36675af08933b5bdaf0d552835a6de4361'


class Model18000Contract(Model930Contract):
    expected_task = 'rs01_omni_v20_bounded_hip'
    expected_observations = 61
    model_label = 'B18000'

    @classmethod
    def from_onnx_session(cls, session, onnx_path, expected_sha256=EXPECTED_ONNX_SHA256):
        if expected_sha256 != EXPECTED_ONNX_SHA256:
            raise RuntimeError('B18000 hash override is not permitted')
        c=super().from_onnx_session(session,onnx_path,expected_sha256)
        expected=dict(mapping='conditional_inward_scale_v1',inward_target_rad=np.deg2rad(8.),
                      vy_gate=.08,wz_gate=.20,stand_phase='freeze_and_resume',odometry='legacy',
                      guard=dict(continuous=6.,peak=14.,full=8.,tau=2.))
        if c.raw.get('v20') != expected or c.policy_dt != .02:
            raise RuntimeError('B18000 mapping/guard/20ms contract mismatch')
        if c.raw['observations']['base_linear_velocity_source'] != 'rs01_leg_odometry':
            raise RuntimeError('B18000 requires sensor-only odometry')
        arrays=dict(default=np.asarray(c.raw['default_joint_angles_rad'],dtype=np.float64))
        for field,key in [('action_scale','action_scale_rad'),('rate_limit','target_rate_limit_rad_s'),
                          ('accel_limit','target_acceleration_limit_rad_s2')]:
            arrays[field]=np.asarray(c.raw['control'][key],dtype=np.float64)
        return replace(c,**arrays)


class Rs01Model18000Core(Rs01Model6850Core):
    # Straight translation retains symmetric yaw feedback, with no added bias.
    straight_heading_deadband_rad = math.radians(2.)
    straight_yaw_gain_scale = .5
    straight_lateral_cmd_mps = .02
    def project_policy_target(self, target, command):
        cfg=self.contract.raw['v20']
        ids=[self.contract.joint_names.index(x+'_hip_joint') for x in ('FL','FR','RL','RR')]
        sides=np.array([1.,-1.,1.,-1.])
        gate=np.clip(1-abs(command[1])/cfg['vy_gate']-abs(command[2])/cfg['wz_gate'],0,1)
        ranges=self.contract.action_scale[ids]*self.contract.action_clip
        hip=target[ids];out=target.copy()
        out[ids]=np.where(hip*sides<0,hip*(1-gate*(1-cfg['inward_target_rad']/ranges)),hip)
        return out

    def _mix_direction_command(self, command, error, turning, conf):
        command=np.asarray(command,dtype=np.float64).reshape(3)
        operator_straight = (
            abs(command[1]) < self.straight_lateral_cmd_mps
            and abs(command[2]) < conf['direction_turn_enter_rad_s'])
        if not operator_straight:
            return super()._mix_direction_command(command, error, turning, conf)
        target=command.copy()
        blend=float(not turning)
        deadband=self.straight_heading_deadband_rad
        adjusted=0. if abs(error) <= deadband else math.copysign(abs(error)-deadband, error)
        limit=conf['direction_yaw_correction_limit_rad_s']
        gain=conf['direction_heading_gain']*self.straight_yaw_gain_scale
        target[2]+=blend*limit*math.tanh(gain*adjusted/limit)
        ranges=self.contract.raw['commands']['ranges']
        for i,key in enumerate(('lin_vel_x','lin_vel_y','ang_vel_yaw')):
            target[i]=np.clip(target[i],*ranges[key])
        return target


class Guarded18000PolicyCore(Guarded6850PolicyCore):
    actor_type = Rs01Model18000Core
