"""Gravity-referenced support guard; the actor's trained odometry is unchanged."""
import numpy as np
from .rs01_model930_core import Rs01NewMachineLegOdometry


class SupportPlaneOdometry(Rs01NewMachineLegOdometry):
    def estimate_aligned(self, q, dq, gyro, gravity):
        gravity=np.asarray(gravity,dtype=float).reshape(3)
        norm=float(np.linalg.norm(gravity))
        if not np.isfinite(gravity).all() or not .9 <= norm <= 1.1:
            raise ValueError('Invalid gravity for support geometry')
        up=-gravity/norm
        original=self.compute_kinematics(q,dq,gyro)
        positions=original['foot_position'].copy()
        velocities=original['foot_velocity'].copy()
        # Height and vertical motion use the gravity axis, not the tilted
        # body z axis. Include omega x r before testing vertical foot speed.
        positions[:,2]=original['foot_position']@up
        vertical=-original['velocity_by_foot']@up
        velocities[:,2]=vertical
        geometry=dict(original,foot_position=positions,foot_velocity=velocities)
        result=super().estimate(q,dq,gyro,kinematics=geometry)
        result['foot_position']=original['foot_position']
        result['foot_velocity']=original['foot_velocity']
        result['support_vertical_speed']=vertical.copy()
        return result

    @classmethod
    def from_estimator(cls, source):
        names=('nominal_base_height','foot_radius','height_margin',
               'vertical_speed_threshold','velocity_residual_threshold',
               'filter_alpha','no_contact_decay','previous_stance_score_bonus')
        return cls(**{name:getattr(source,name) for name in names},strict_diagonal_pairs=True)
