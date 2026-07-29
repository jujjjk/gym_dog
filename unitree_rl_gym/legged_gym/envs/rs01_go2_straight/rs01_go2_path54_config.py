"""54-D lateral-path continuation from the selected model_1425."""

from .rs01_go2_drift_repair_config import (
    Rs01Go2Model930DriftRepairCfg,
    Rs01Go2Model930DriftRepairCfgPPO,
)


class Rs01Go2Model1425Path54Cfg(Rs01Go2Model930DriftRepairCfg):
    """Add only the path state that the 52-D drift task could not observe."""

    class env(Rs01Go2Model930DriftRepairCfg.env):
        # Existing 52-D proprioception and heading sin/cos, followed by:
        # [path lateral displacement, path-frame lateral velocity].
        num_observations = 54

    class commands(Rs01Go2Model930DriftRepairCfg.commands):
        observe_straight_path_state = True
        # +/-0.5 m maps to the clipped +/-1 displacement observation.
        straight_path_lateral_position_scale = 2.0
        # Use the existing linear-velocity normalization for path-frame vy.
        straight_path_lateral_velocity_scale = 2.0

    class init_state(Rs01Go2Model930DriftRepairCfg.init_state):
        # Start some environments beside the reference line without changing
        # the physical spawn pose. This creates a two-sided recovery signal.
        reset_path_lateral_error_noise_m = 0.15
        reset_heading_noise_rad = 0.12
        reset_yaw_rate_noise_rad_s = 0.08

    class rewards(Rs01Go2Model930DriftRepairCfg.rewards):
        lateral_path_recovery_gain_per_s = 0.60
        lateral_path_recovery_max_velocity_mps = 0.10
        lateral_path_recovery_velocity_scale_mps = 0.08

        class scales(Rs01Go2Model930DriftRepairCfg.rewards.scales):
            # The old unconditional vy cost would oppose useful motion back to
            # the line. Replace it with tracking of a restoring vy target.
            lateral_velocity = 0.0
            lateral_path_recovery = -0.50


class Rs01Go2Model1425Path54CfgPPO(
    Rs01Go2Model930DriftRepairCfgPPO
):
    class policy(Rs01Go2Model930DriftRepairCfgPPO.policy):
        init_noise_std = 0.035

    class algorithm(Rs01Go2Model930DriftRepairCfgPPO.algorithm):
        learning_rate = 1.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Model930DriftRepairCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.035
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        observation_column_migration = {
            "source_width": 52,
            "destination_width": 54,
            "copy_prefix": 52,
            "column_mappings": [],
        }
        # Keep model_1425 as a light executed-action gait anchor. The two new
        # columns start at zero weight and are free to learn path correction.
        reference_policy_coef = 0.03
        reference_action_deadband = 0.10
        reference_action_hinge_coef = 0.40
        reference_action_transform = "clip"
        reference_action_clip = 1.0
