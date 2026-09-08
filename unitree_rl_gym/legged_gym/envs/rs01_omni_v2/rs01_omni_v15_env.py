"""Stand clock and gravity-aware sensor odometry; V14 motors/rewards unchanged."""
from .rs01_omni_v14_env import Rs01OmniV14Robot
from ..rs01_go2_straight.rs01_support_odometry import GravitySupportOdometry


class Rs01OmniV15StandRobot(Rs01OmniV14Robot):
    def _advance_wide_gait_phase(self):
        self.wide_gait_phase.add_(
            self.dt * self._command_gait_frequency() * (self.gait_enable > .5)
        ).remainder_(1.0)


class Rs01OmniV15Robot(Rs01OmniV15StandRobot):
    """Experimental estimator: NOT accepted for long training/deployment."""
    def _initialize_rs01_observation_estimator(self):
        super()._initialize_rs01_observation_estimator()
        self.rs01_leg_odometry = GravitySupportOdometry(self.rs01_leg_odometry)

    def _update_rs01_observation_estimator(self):
        self.rs01_leg_odometry.gravity = self.projected_gravity
        super()._update_rs01_observation_estimator()
