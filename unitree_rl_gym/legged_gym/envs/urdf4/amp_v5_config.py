"""Command-conditioned fast feedback with an independently qualified Gym clip."""
from .amp_v4_config import Urdf4AMPV4Cfg,Urdf4AMPV4PPO,ROOT


class Urdf4AMPV5Cfg(Urdf4AMPV4Cfg):
    class amp_cts(Urdf4AMPV4Cfg.amp_cts):
        reward_revision='v12_measured_state_feedback_without_action_copy'
        balance_variant=2
        extra_reference=str(ROOT/'artifacts/urdf4_speed_fix_20261117/feedback_extra040_v3.pt')
        adaptive_feedback_phase=False
        protected_policy=str(ROOT/'artifacts/urdf4_speed_fix_20261117/selected/exported/student_50.pt')
        use_public_prior=False  # Enabled only for explicitly prepared v13 seeds.
        public_prior_archive=str(ROOT/'artifacts/urdf4_speed_fix_20261117/public_prior1/prior.pt')
        public_feedback_history=False
    class direction(Urdf4AMPV4Cfg.direction):
        speed_feedback_enabled=True
        mid_cross_yaw_gain=2.
        mid_cross_yaw_damping=.2
        mid_cross_yaw_limit_rad_s=.08
        mid_cross_integral_gain=0.  # Opt-in measured direction-loop experiment.
    class expand(Urdf4AMPV4Cfg.expand):
        initial_stage=1
        max_stage=1  # 0.60 remains an unqualified diagnostic, not a ready range.


class Urdf4AMPV5PPO(Urdf4AMPV4PPO):
    class runner(Urdf4AMPV4PPO.runner):
        cts_v5_enabled=True
        cts_v4_enabled=False
        experiment_name='urdf4_amp_cts_v5_feedback'
        training_objective='command_conditioned_fast_gait_and_direction'
        v3_preflight=str(ROOT/'artifacts/urdf4_speed_fix_20261117/v5_preflight.json')
