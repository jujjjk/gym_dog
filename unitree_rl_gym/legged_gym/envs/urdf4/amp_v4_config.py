"""Sensor-complete, supervised-feedback V3 continuation; old V3 remains intact."""
from .amp_v3_config import Urdf4AMPV3Cfg,Urdf4AMPV3PPO,ROOT


class Urdf4AMPV4Cfg(Urdf4AMPV3Cfg):
    class env(Urdf4AMPV3Cfg.env):
        public_obs_size=1029
        num_observations=1130
        num_privileged_obs=1130
    class amp_cts(Urdf4AMPV3Cfg.amp_cts):
        feedback_v4=True
        reward_revision='v7_native_phase_public_velocity_anchored_feedback'
        native_prior=str(ROOT/'artifacts/urdf4_speed_fix_20261117/native_probe/native_prior.pt')
        feedback_samples=str(ROOT/'artifacts/urdf4_speed_fix_20261117/feedback_data/samples.pt')
        balance_archive=str(ROOT/'artifacts/urdf4_speed_fix_20261117/balance_temporal2.pt')
        balance_variant=1
    class expand(Urdf4AMPV3Cfg.expand):
        cold_settle_s=2.
        curriculum_min_updates=500
    class commands(Urdf4AMPV3Cfg.commands):
        reference_probability=.70
        stand_probability=.10
        fast_ramp_s=4.
    class direction(Urdf4AMPV3Cfg.direction):
        speed_feedback_enabled=False
        speed_feedback=dict(proportional=.5,integral=.5,filter_tau_s=.30,correction_limit=.20,correction_acceleration=.30,output_x_limit=.90)


class Urdf4AMPV4PPO(Urdf4AMPV3PPO):
    class algorithm(Urdf4AMPV3PPO.algorithm):
        learning_rate=2.e-5
        entropy_coef=.0002
    class runner(Urdf4AMPV3PPO.runner):
        cts_v4_enabled=True
        cts_v3_enabled=False
        experiment_name='urdf4_amp_cts_v4_feedback'
        training_objective='v4_feedback_anchored_omni'
        anchor_coefficient=20.
        v3_preflight=str(ROOT/'artifacts/urdf4_speed_fix_20261117/preflight.json')
