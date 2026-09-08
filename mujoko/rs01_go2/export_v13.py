"""Export V13 without changing the legacy straight-policy exporter."""
import export_policy as legacy
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_config import (
    Rs01OmniV13DirectionCfg, Rs01OmniV13DirectionCfgPPO,
)

original_build_contract = legacy.build_contract


def build_contract(task, cfg, checkpoint, output):
    # Legacy scalar-heading/default-speed metadata are not consumed by V13.
    # Supply them on an isolated proxy, never mutate the training config.
    class Proxy(cfg):
        class commands(cfg.commands):
            straight_heading_observation_scale = 1.0
            playback_speed_mps = 0.4
    contract = original_build_contract(task, Proxy, checkpoint, output)
    contract['v13'] = {
        'commands': legacy.class_to_dict(cfg.commands),
        'max_frequency_hz': cfg.rewards.wide_gait_max_frequency_hz,
        'speed_deadband': cfg.rewards.wide_gait_speed_deadband,
        'checkpoint_sha256': legacy.sha256_file(checkpoint),
    }
    contract['observations']['layout'] = [
        ['estimated_body_velocity_scaled', 3], ['body_angular_velocity_scaled', 3],
        ['projected_gravity', 3], ['effective_command_scaled', 3],
        ['joint_position_error_scaled', 12], ['joint_velocity_scaled', 12],
        ['previous_clipped_action', 12], ['integrated_phase_sin_cos', 2],
        ['heading_error_sin_cos', 2], ['zero', 1],
        ['estimated_vy_minus_effective_target_times2', 1], ['gait_enable', 1],
        ['zero', 1], ['estimated_vx_minus_effective_target_times2', 1],
        ['raw_command_scaled', 3], ['odometry_confidence', 1],
    ]
    assert sum(size for _, size in contract['observations']['layout']) == 61
    return contract


if __name__ == '__main__':
    legacy.TASKS = {'rs01_omni_v13_direction': (
        Rs01OmniV13DirectionCfg, Rs01OmniV13DirectionCfgPPO)}
    legacy.build_contract = build_contract
    legacy.main()
