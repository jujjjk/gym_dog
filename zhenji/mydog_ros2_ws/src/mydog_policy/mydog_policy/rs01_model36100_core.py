"""Guard5 robust B36100: push/payload-trained actor on the unchanged V22 stack.

Push/payload robustness is training-only. The deployment contract, 5-degree
unconditional inward hip mapping, direction controller, sensor snapshot and
guarded limiter are identical to B32750; only the task name and locked
artifact hash differ.
"""
from dataclasses import replace
import numpy as np
from .rs01_model930_core import Model930Contract
from .rs01_model6850_guard import Guarded6850PolicyCore
from .rs01_model32750_core import Rs01Model32750Core

EXPECTED_ONNX_SHA256 = '6a879329945cd403f23d5566dda5b345f495eb3fd50a1c8a9476bcd7671e62d9'


class Model36100Contract(Model930Contract):
    expected_task = 'rs01_v22_guard5_robust'
    expected_observations = 61
    model_label = 'B36100'

    @classmethod
    def from_onnx_session(cls, session, onnx_path, expected_sha256=EXPECTED_ONNX_SHA256):
        if expected_sha256 != EXPECTED_ONNX_SHA256:
            raise RuntimeError('B36100 hash override is not permitted')
        c = super().from_onnx_session(session, onnx_path, expected_sha256)
        expected = dict(mapping='omnidirectional_inward_scale_v1',
                        inward_target_rad=np.deg2rad(5.),
                        vy_gate=.08, wz_gate=.20, stand_phase='freeze_and_resume',
                        odometry='legacy', guard=dict(continuous=6., peak=14., full=8., tau=2.),
                        mapping_note='Replace V20 map; all commands; before rate/acceleration limiter; no physical-state clamp')
        v22 = c.raw.get('v22', {})
        if c.raw.get('v20') != expected or c.policy_dt != .02:
            raise RuntimeError('B36100 mapping/guard/20ms contract mismatch')
        if (v22.get('host_guard') != 'policy_sensor_snapshot'
                or v22.get('observation') != 'sensor_snapshot_61'
                or c.raw['observations']['base_linear_velocity_source'] != 'rs01_leg_odometry'
                or v22.get('odometry') != c.raw['observations']['rs01_leg_odometry']):
            raise RuntimeError('B36100 sensor/odometry contract mismatch')
        arrays = dict(default=np.asarray(c.raw['default_joint_angles_rad'], dtype=np.float64))
        for field, key in [('action_scale', 'action_scale_rad'), ('rate_limit', 'target_rate_limit_rad_s'),
                           ('accel_limit', 'target_acceleration_limit_rad_s2')]:
            arrays[field] = np.asarray(c.raw['control'][key], dtype=np.float64)
        return replace(c, **arrays)


class Rs01Model36100Core(Rs01Model32750Core):
    """Unconditional 5-degree inward mapping, identical to B32750."""


class Guarded36100PolicyCore(Guarded6850PolicyCore):
    actor_type = Rs01Model36100Core
