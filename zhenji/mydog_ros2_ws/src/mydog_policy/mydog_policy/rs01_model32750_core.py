"""Guard5 B32750: unconditional 5-degree inward hip compression on the V22 stack.

Replaces the B23500/B18000 conditional inward map for every command, applied
after action clipping/scaling and before the rate/acceleration limiter, the
same order as the rs01_v22_phase_guard5 training task and the MuJoCo bridge.
Sensor snapshot, direction controller and guarded limiter are unchanged.
"""
from dataclasses import replace
import numpy as np
from .rs01_model930_core import Model930Contract
from .rs01_model6850_guard import Guarded6850PolicyCore
from .rs01_model23500_core import Rs01Model23500Core

EXPECTED_ONNX_SHA256 = 'c7e9ba1561fe8fe2f3a90381e0c9fecbec1413d568cbc9c09e512225a386d2a7'


class Model32750Contract(Model930Contract):
    expected_task = 'rs01_v22_phase_guard5'
    expected_observations = 61
    model_label = 'B32750'

    @classmethod
    def from_onnx_session(cls, session, onnx_path, expected_sha256=EXPECTED_ONNX_SHA256):
        if expected_sha256 != EXPECTED_ONNX_SHA256:
            raise RuntimeError('B32750 hash override is not permitted')
        c = super().from_onnx_session(session, onnx_path, expected_sha256)
        expected = dict(mapping='omnidirectional_inward_scale_v1',
                        inward_target_rad=np.deg2rad(5.),
                        vy_gate=.08, wz_gate=.20, stand_phase='freeze_and_resume',
                        odometry='legacy', guard=dict(continuous=6., peak=14., full=8., tau=2.),
                        mapping_note='Replace V20 map; all commands; before rate/acceleration limiter; no physical-state clamp')
        v22 = c.raw.get('v22', {})
        if c.raw.get('v20') != expected or c.policy_dt != .02:
            raise RuntimeError('B32750 mapping/guard/20ms contract mismatch')
        if (v22.get('host_guard') != 'policy_sensor_snapshot'
                or v22.get('observation') != 'sensor_snapshot_61'
                or c.raw['observations']['base_linear_velocity_source'] != 'rs01_leg_odometry'
                or v22.get('odometry') != c.raw['observations']['rs01_leg_odometry']):
            raise RuntimeError('B32750 sensor/odometry contract mismatch')
        arrays = dict(default=np.asarray(c.raw['default_joint_angles_rad'], dtype=np.float64))
        for field, key in [('action_scale', 'action_scale_rad'), ('rate_limit', 'target_rate_limit_rad_s'),
                           ('accel_limit', 'target_acceleration_limit_rad_s2')]:
            arrays[field] = np.asarray(c.raw['control'][key], dtype=np.float64)
        return replace(c, **arrays)


class Rs01Model32750Core(Rs01Model23500Core):
    # Guard5 keeps the trained V13 direction controller and V22 numerics;
    # only the inward hip target mapping differs from B23500.
    def project_policy_target(self, target, command):
        cfg = self.contract.raw['v20']
        ids = [self.contract.joint_names.index(x+'_hip_joint') for x in ('FL', 'FR', 'RL', 'RR')]
        sides = np.array([1., -1., 1., -1.])
        if not np.allclose(self.contract.default[ids], 0.):
            raise RuntimeError('Inward guard requires zero default hip angles')
        ranges = self.contract.action_scale[ids]*self.contract.action_clip
        limit = cfg['inward_target_rad']
        if not np.isfinite(limit) or limit <= 0 or np.any(ranges < limit):
            raise RuntimeError('Invalid omnidirectional inward limit')
        hip = target[ids]
        out = target.copy()
        out[ids] = np.where(hip*sides < 0, hip*limit/ranges, hip)
        return out


class Guarded32750PolicyCore(Guarded6850PolicyCore):
    actor_type = Rs01Model32750Core
