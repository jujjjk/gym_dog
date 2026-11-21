"""First rough-ground continuation of the selected capped V4 policy."""
from .amp_v4_config import ROOT, Urdf4AMPV4Cfg, Urdf4AMPV4PPO


class Urdf4AMPRoughCfg(Urdf4AMPV4Cfg):
    class rough:
        amplitude_m = .005
        grid_m = .10
        wavelengths_m = [.44, .67, 1.05, 1.62]
        terrain_seed = 18051
        flat_stride = 4
        spawn_jitter_m = 1.0
        handoff_tolerance_s = .02
        rough_amp_weight = .5
        exact_fast_fraction = .10
        fast_cross_yaw_gain = 1.5
        fast_cross_yaw_damping = .6
        fast_cross_yaw_limit = .07

    class commands(Urdf4AMPV4Cfg.commands):
        class ranges(Urdf4AMPV4Cfg.commands.ranges):
            lin_vel_x = [-.30, .50]
            lin_vel_y = [-.20, .20]
            ang_vel_yaw = [-.60, .60]

    class expand(Urdf4AMPV4Cfg.expand):
        stages = [[-.20, .30, .10, .40], [-.30, .50, .20, .60]]
        initial_stage = 1
        max_stage = 1

    class direction(Urdf4AMPV4Cfg.direction):
        speed_feedback_enabled = True

    class amp_cts(Urdf4AMPV4Cfg.amp_cts):
        balance_variant = 2
        reward_revision = 'v4_rough_ground_handoff_v1'


class Urdf4AMPRoughPPO(Urdf4AMPV4PPO):
    class runner(Urdf4AMPV4PPO.runner):
        cts_rough_enabled = True
        experiment_name = 'urdf4_amp_cts_rough'
        training_objective = 'rough_ground_diagonal_omni'
        v3_preflight = str(ROOT / 'artifacts/urdf4_rough_20261118/training_receipt.json')
