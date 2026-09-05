from types import SimpleNamespace

import isaacgym  # noqa: F401 - Isaac Gym must be loaded before torch
import torch

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from legged_gym.envs.rs01_omni_v2 import (
    Rs01OmniV2Cfg,
    Rs01OmniV2CfgPPO,
    Rs01OmniV2Robot,
    Rs01OmniV3Contact1Cfg,
    Rs01OmniV3Contact1CfgPPO,
    Rs01OmniV3Contact15Cfg,
    Rs01OmniV3Contact15CfgPPO,
    Rs01OmniV3Robot,
    Rs01OmniV4Contact1Cfg,
    Rs01OmniV4Contact1CfgPPO,
    Rs01OmniV4Contact15Cfg,
    Rs01OmniV4Contact15CfgPPO,
    Rs01OmniV4Robot,
    Rs01OmniV5Odd05Cfg,
    Rs01OmniV5Odd05CfgPPO,
    Rs01OmniV5Odd10Cfg,
    Rs01OmniV5Odd10CfgPPO,
    Rs01OmniV5Robot,
    Rs01OmniV6Seed08Cfg,
    Rs01OmniV6Seed08CfgPPO,
    Rs01OmniV6Seed11Cfg,
    Rs01OmniV6Seed11CfgPPO,
    Rs01OmniV6Robot,
    Rs01OmniV7Seed14Cfg,
    Rs01OmniV7Seed14CfgPPO,
    Rs01OmniV7Seed18Cfg,
    Rs01OmniV7Seed18CfgPPO,
    Rs01OmniV8Clearance2Cfg,
    Rs01OmniV8Clearance2CfgPPO,
    Rs01OmniV8Clearance4Cfg,
    Rs01OmniV8Clearance4CfgPPO,
    Rs01OmniV9Speed10Cfg,
    Rs01OmniV9Speed10CfgPPO,
    Rs01OmniV9Speed14Cfg,
    Rs01OmniV9Speed14CfgPPO,
)


def _sampling_robot(count=20000):
    robot = object.__new__(Rs01OmniV2Robot)
    robot.device = "cpu"
    robot.cfg = SimpleNamespace(commands=Rs01OmniV2Cfg.commands)
    robot.num_envs = count
    robot.commands = torch.zeros(count, 4)
    robot.command_mode = torch.zeros(count, dtype=torch.long)
    robot.gait_enable = torch.zeros(count)
    robot.moving_mode_probabilities = torch.tensor(
        Rs01OmniV2Cfg.commands.moving_mode_probabilities
    )
    robot._v2_command_ready = True
    return robot


def test_v2_isolated_config_and_fixed_ppo_contract():
    assert Rs01OmniV2Cfg.__bases__ == (LeggedRobotCfg,)
    assert Rs01OmniV2CfgPPO.__bases__ == (LeggedRobotCfgPPO,)
    assert Rs01OmniV2Cfg.env.num_observations == 55
    assert Rs01OmniV2Cfg.env.num_actions == 12
    assert Rs01OmniV2Cfg.rs01_actuator.peak_torque_limit_nm == 17.0
    assert Rs01OmniV2Cfg.control.stiffness == {
        "hip": 40.0,
        "thigh": 40.0,
        "calf": 40.0,
    }
    assert Rs01OmniV2Cfg.control.action_scale_by_joint == {
        "hip": 0.18,
        "thigh": 0.18,
        "calf": 0.14,
    }
    assert Rs01OmniV2CfgPPO.algorithm.learning_rate == 1.0e-4
    assert Rs01OmniV2CfgPPO.algorithm.schedule == "fixed"
    assert Rs01OmniV2CfgPPO.algorithm.entropy_coef == 0.0
    assert Rs01OmniV2CfgPPO.runner.action_std_value == 0.20
    assert Rs01OmniV2CfgPPO.runner.freeze_action_std is True
    assert Rs01OmniV2CfgPPO.runner.reference_policy_coef == 0.0


def test_zero_command_distinguishes_stand_from_in_place_march():
    robot = _sampling_robot(2)
    env_ids = torch.arange(2)
    modes = torch.tensor([robot.COMMAND_STAND, robot.COMMAND_MARCH])
    robot._set_command_modes(env_ids, modes)
    assert torch.count_nonzero(robot.commands[:, :3]) == 0
    assert torch.equal(robot.gait_enable, torch.tensor([0.0, 1.0]))
    assert torch.equal(robot._walking_command_gate(), robot.gait_enable)


def test_transition_sampler_teaches_march_and_motion_switches():
    robot = _sampling_robot()
    robot.command_mode[:10000] = robot.COMMAND_STAND
    robot.command_mode[10000:] = robot.COMMAND_MARCH

    torch.manual_seed(20260904)
    robot._resample_commands(torch.arange(robot.num_envs))

    assert torch.all(robot.command_mode[:10000] == robot.COMMAND_MARCH)
    assert torch.all(
        robot.command_mode[10000:] >= robot.COMMAND_FORWARD
    )
    empirical = (
        torch.bincount(
            robot.command_mode[10000:] - robot.COMMAND_FORWARD,
            minlength=5,
        ).float()
        / 10000
    )
    assert torch.all(
        torch.abs(empirical - robot.moving_mode_probabilities) < 0.02
    )


def test_moving_commands_can_switch_to_stand_march_or_new_motion():
    robot = _sampling_robot()
    robot.command_mode[:] = robot.COMMAND_FORWARD
    torch.manual_seed(20260906)
    robot._resample_commands(torch.arange(robot.num_envs))

    stand_ratio = float(
        (robot.command_mode == robot.COMMAND_STAND).float().mean()
    )
    march_ratio = float(
        (robot.command_mode == robot.COMMAND_MARCH).float().mean()
    )
    moving_ratio = float(
        (robot.command_mode >= robot.COMMAND_FORWARD).float().mean()
    )
    assert abs(stand_ratio - 0.10) < 0.015
    assert abs(march_ratio - 0.35) < 0.015
    assert abs(moving_ratio - 0.55) < 0.02


def test_sampled_commands_have_no_near_zero_direction_ambiguity():
    robot = _sampling_robot()
    robot.command_mode[:] = robot.COMMAND_MARCH
    torch.manual_seed(20260905)
    robot._resample_commands(torch.arange(robot.num_envs))

    forward = robot.command_mode == robot.COMMAND_FORWARD
    backward = robot.command_mode == robot.COMMAND_BACKWARD
    lateral = robot.command_mode == robot.COMMAND_LATERAL
    yaw = robot.command_mode == robot.COMMAND_YAW
    combined = robot.command_mode == robot.COMMAND_COMBINED
    assert torch.all(robot.commands[forward, 0] >= 0.08)
    assert torch.all(robot.commands[backward, 0] <= -0.06)
    assert torch.all(torch.abs(robot.commands[lateral, 1]) >= 0.04)
    assert torch.all(torch.abs(robot.commands[yaw, 2]) >= 0.12)
    assert torch.all(torch.abs(robot.commands[combined, 0]) >= 0.06)
    assert torch.all(torch.abs(robot.commands[combined, 1]) >= 0.03)
    assert torch.all(torch.abs(robot.commands[combined, 2]) >= 0.10)


def test_lateral_and_heading_path_costs_are_bounded():
    robot = object.__new__(Rs01OmniV2Robot)
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            trajectory_lateral_scale_m=0.10,
            heading_error_scale_rad=0.20,
        )
    )
    robot._straight_path_state = lambda: (
        torch.tensor([0.0, 0.10, 100.0]),
        torch.zeros(3),
    )
    robot._straight_heading_error = lambda: torch.tensor(
        [0.0, 0.20, 3.14]
    )
    robot._walking_command_gate = lambda: torch.ones(3)
    lateral = robot._reward_trajectory_lateral_error()
    heading = robot._reward_omni_heading_error()
    assert torch.all((lateral >= 0.0) & (lateral <= 1.0))
    assert torch.all((heading >= 0.0) & (heading <= 1.0))
    assert lateral[0] == 0.0
    assert heading[0] == 0.0


def test_v3_ab_changes_only_the_prolonged_four_foot_weight():
    a = Rs01OmniV3Contact1Cfg
    b = Rs01OmniV3Contact15Cfg
    assert a.rewards.scales.prolonged_all_feet_contact == -1.0
    assert b.rewards.scales.prolonged_all_feet_contact == -1.5
    assert a.rewards.scales.tracking_planar_velocity == 0.0
    assert a.rewards.scales.tracking_yaw_velocity == 0.0
    assert a.rewards.scales.tracking_command_velocity == 3.0
    assert a.rewards.scales.phase_support_tracking == 0.0
    assert a.rewards.scales.phase_contact_error == -1.5
    assert (
        Rs01OmniV3Contact1CfgPPO.algorithm.learning_rate
        == Rs01OmniV3Contact15CfgPPO.algorithm.learning_rate
        == 1.0e-4
    )
    assert (
        Rs01OmniV3Contact1CfgPPO.runner.freeze_action_std
        is Rs01OmniV3Contact15CfgPPO.runner.freeze_action_std
        is True
    )


def _v3_reward_robot(commands, linear_velocity, yaw_velocity):
    robot = object.__new__(Rs01OmniV3Robot)
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            command_planar_tracking_sigma=0.010,
            command_yaw_tracking_sigma=0.040,
        )
    )
    robot.commands = torch.tensor(commands, dtype=torch.float)
    robot.base_lin_vel = torch.tensor(linear_velocity, dtype=torch.float)
    robot.base_ang_vel = torch.zeros(len(commands), 3)
    robot.base_ang_vel[:, 2] = torch.tensor(yaw_velocity, dtype=torch.float)
    return robot


def test_v3_command_reward_is_zero_for_stationary_nonzero_commands():
    robot = _v3_reward_robot(
        [[0.10, 0.00, 0.00, 0.0], [0.00, 0.00, 0.30, 0.0]],
        [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]],
        [0.00, 0.00],
    )
    assert torch.equal(
        robot._reward_tracking_command_velocity(), torch.zeros(2)
    )


def test_v3_command_reward_is_one_at_the_requested_velocity():
    robot = _v3_reward_robot(
        [[0.10, 0.00, 0.00, 0.0], [0.00, 0.00, -0.30, 0.0]],
        [[0.10, 0.00, 0.00], [0.00, 0.00, 0.00]],
        [0.00, -0.30],
    )
    assert torch.allclose(
        robot._reward_tracking_command_velocity(), torch.ones(2)
    )


def test_v3_reset_mode_probabilities_are_stratified_20_20_60():
    robot = object.__new__(Rs01OmniV3Robot)
    robot.command_mode_probabilities = torch.tensor(
        Rs01OmniV3Contact1Cfg.commands.mode_probabilities
    )
    torch.manual_seed(20260904)
    modes = robot._sample_initial_modes(50000)
    stand = float((modes == robot.COMMAND_STAND).float().mean())
    march = float((modes == robot.COMMAND_MARCH).float().mean())
    moving = float((modes >= robot.COMMAND_FORWARD).float().mean())
    assert abs(stand - 0.20) < 0.01
    assert abs(march - 0.20) < 0.01
    assert abs(moving - 0.60) < 0.01


def test_v4_ab_has_one_controlled_difference_and_fixed_optimizer():
    a = Rs01OmniV4Contact1Cfg
    b = Rs01OmniV4Contact15Cfg
    assert a.rewards.scales.prolonged_all_feet_contact == -1.0
    assert b.rewards.scales.prolonged_all_feet_contact == -1.5
    for cfg in (a, b):
        assert cfg.rewards.scales.tracking_command_velocity == 4.0
        assert cfg.rewards.scales.phase_contact_error == -0.75
        assert cfg.rewards.scales.phase_two_contact_quality == 1.5
        assert cfg.rewards.scales.alive == 1.0
        assert cfg.rewards.scales.termination == -50.0
    for ppo in (Rs01OmniV4Contact1CfgPPO, Rs01OmniV4Contact15CfgPPO):
        assert ppo.algorithm.learning_rate == 1.0e-4
        assert ppo.algorithm.schedule == "fixed"
        assert ppo.algorithm.entropy_coef == 0.0
        assert ppo.policy.init_noise_std == 0.35
        assert ppo.runner.action_std_value == 0.35
        assert ppo.runner.freeze_action_std is True


def test_v4_phase_quality_only_rewards_exact_two_foot_diagonal_support():
    robot = object.__new__(Rs01OmniV4Robot)
    robot.num_envs = 4
    robot.device = "cpu"
    desired = torch.tensor(
        [
            [True, False, False, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )
    actual = torch.tensor(
        [
            [True, False, False, True],  # exact diagonal
            [True, True, True, True],    # static four-foot support
            [True, False, True, True],   # three-foot escape
            [True, True, True, True],    # phase handoff: not a two-foot target
        ]
    )
    robot._desired_contact_mask = lambda: desired
    robot.get_foot_contact_mask = lambda: actual
    robot._walking_command_gate = lambda: torch.ones(4)
    reward = robot._reward_phase_two_contact_quality()
    assert torch.equal(reward, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_v4_phase_quality_is_disabled_for_stand_mode():
    robot = object.__new__(Rs01OmniV4Robot)
    robot.num_envs = 2
    robot.device = "cpu"
    diagonal = torch.tensor(
        [[True, False, False, True], [False, True, True, False]]
    )
    robot._desired_contact_mask = lambda: diagonal
    robot.get_foot_contact_mask = lambda: diagonal
    robot._walking_command_gate = lambda: torch.zeros(2)
    assert torch.equal(
        robot._reward_phase_two_contact_quality(), torch.zeros(2)
    )


def test_v5_ab_changes_only_odd_support_penalty():
    a = Rs01OmniV5Odd05Cfg
    b = Rs01OmniV5Odd10Cfg
    assert a.rewards.scales.odd_feet_contact == -0.5
    assert b.rewards.scales.odd_feet_contact == -1.0
    for cfg in (a, b):
        assert cfg.rewards.phase_support_sigma == 0.25
        assert cfg.rewards.scales.tracking_command_velocity == 6.0
        assert cfg.rewards.scales.phase_support_tracking == 2.0
        assert cfg.rewards.scales.phase_two_contact_quality == 1.0
        assert cfg.rewards.scales.prolonged_all_feet_contact == -1.0
    for ppo in (Rs01OmniV5Odd05CfgPPO, Rs01OmniV5Odd10CfgPPO):
        assert ppo.algorithm.learning_rate == 1.0e-4
        assert ppo.runner.action_std_value == 0.35
        assert ppo.runner.freeze_action_std is True


def test_v5_odd_support_penalty_rejects_one_and_three_contacts_only():
    robot = object.__new__(Rs01OmniV5Robot)
    robot.get_foot_contact_mask = lambda: torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, False, False, True],
            [True, True, False, True],
            [True, True, True, True],
        ]
    )
    robot._walking_command_gate = lambda: torch.ones(5)
    assert torch.equal(
        robot._reward_odd_feet_contact(),
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    )


def test_v5_dense_support_reward_excludes_four_contact_handoff():
    robot = object.__new__(Rs01OmniV5Robot)
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(phase_support_sigma=0.25)
    )
    robot._desired_contact_mask = lambda: torch.tensor(
        [
            [True, False, False, True],
            [True, True, True, True],
            [False, True, True, False],
        ]
    )
    robot._phase_support_error = lambda: torch.tensor([0.0, 0.0, 0.25])
    robot._walking_command_gate = lambda: torch.ones(3)
    reward = robot._reward_phase_support_tracking()
    assert torch.allclose(
        reward, torch.tensor([1.0, 0.0, torch.exp(torch.tensor(-1.0))])
    )


def _v6_seed_robot(test=False, step=0, amplitude=0.8):
    robot = object.__new__(Rs01OmniV6Robot)
    robot.cfg = SimpleNamespace(
        env=SimpleNamespace(test=test),
        rewards=SimpleNamespace(gait_stance_ratio=0.65),
        control=SimpleNamespace(
            structured_exploration_amplitude=amplitude,
            structured_exploration_decay_steps=4800,
            structured_exploration_calf_action=-2.0,
            structured_exploration_swing_thigh_action=-1.0,
            structured_exploration_stride_thigh_action=0.55,
            structured_exploration_full_stride_speed_m_s=0.20,
            structured_exploration_profile="sine",
            structured_exploration_lift_fraction=0.18,
            structured_exploration_lower_start_fraction=0.70,
            structured_exploration_phase_lead=0.0,
        ),
    )
    robot.common_step_counter = step
    robot.actions = torch.zeros(1, 12)
    robot.commands = torch.tensor([[0.10, 0.0, 0.0, 0.0]])
    robot.dof_slot_by_leg = {
        "FL": {"hip": 0, "thigh": 1, "calf": 2},
        "FR": {"hip": 3, "thigh": 4, "calf": 5},
        "RL": {"hip": 6, "thigh": 7, "calf": 8},
        "RR": {"hip": 9, "thigh": 10, "calf": 11},
    }
    robot._gait_phase = lambda: torch.tensor([0.825])
    robot._walking_command_gate = lambda: torch.ones(1)
    return robot


def test_v6_ab_changes_only_structured_seed_amplitude():
    assert Rs01OmniV6Seed08Cfg.control.structured_exploration_amplitude == 0.8
    assert Rs01OmniV6Seed11Cfg.control.structured_exploration_amplitude == 1.1
    for cfg in (Rs01OmniV6Seed08Cfg, Rs01OmniV6Seed11Cfg):
        assert cfg.control.structured_exploration_decay_steps == 4800
        assert cfg.rewards.scales.odd_feet_contact == -1.0
    for ppo in (Rs01OmniV6Seed08CfgPPO, Rs01OmniV6Seed11CfgPPO):
        assert ppo.algorithm.learning_rate == 1.0e-4
        assert ppo.runner.action_std_value == 0.35


def test_v6_seed_moves_only_one_physical_diagonal_in_mid_swing():
    robot = _v6_seed_robot()
    seed = robot._structured_exploration_action()[0]
    assert seed[2] < 0.0 and seed[11] < 0.0
    assert seed[5] == 0.0 and seed[8] == 0.0
    assert torch.isclose(seed[2], seed[11])
    assert torch.isclose(seed[1], seed[10])


def test_v6_seed_is_absent_in_test_mode_and_after_decay():
    assert torch.count_nonzero(
        _v6_seed_robot(test=True)._structured_exploration_action()
    ) == 0
    assert torch.count_nonzero(
        _v6_seed_robot(step=4800)._structured_exploration_action()
    ) == 0


def test_v7_ab_uses_delay_aware_plateau_and_only_changes_amplitude():
    assert Rs01OmniV7Seed14Cfg.control.structured_exploration_amplitude == 1.4
    assert Rs01OmniV7Seed18Cfg.control.structured_exploration_amplitude == 1.8
    for cfg in (Rs01OmniV7Seed14Cfg, Rs01OmniV7Seed18Cfg):
        assert cfg.control.structured_exploration_profile == "plateau"
        assert cfg.control.structured_exploration_phase_lead == 0.12
        assert cfg.control.structured_exploration_decay_steps == 4800
        assert cfg.rewards.scales.odd_feet_contact == -1.0
    for ppo in (Rs01OmniV7Seed14CfgPPO, Rs01OmniV7Seed18CfgPPO):
        assert ppo.algorithm.learning_rate == 1.0e-4
        assert ppo.runner.action_std_value == 0.35


def test_v8_is_pure_scratch_with_rs01_physical_limits_intact():
    a = Rs01OmniV8Clearance2Cfg
    b = Rs01OmniV8Clearance4Cfg
    assert a.rewards.scales.phase_swing_clearance == -2.0
    assert b.rewards.scales.phase_swing_clearance == -4.0
    for cfg in (a, b):
        assert cfg.control.action_scale_by_joint == {
            "hip": 0.22,
            "thigh": 0.22,
            "calf": 0.22,
        }
        assert cfg.rs01_actuator.peak_torque_limit_nm == 17.0
        assert cfg.rs01_actuator.target_rate_limit_rad_s == {
            "hip": 2.0,
            "thigh": 2.6,
            "calf": 3.2,
        }
        assert cfg.rs01_actuator.target_acceleration_limit_rad_s2 == {
            "hip": 60.0,
            "thigh": 78.0,
            "calf": 96.0,
        }
    for ppo in (Rs01OmniV8Clearance2CfgPPO, Rs01OmniV8Clearance4CfgPPO):
        assert ppo.policy.init_noise_std == 0.45
        assert ppo.runner.action_std_value == 0.45
        assert ppo.runner.freeze_action_std is True
        assert ppo.algorithm.learning_rate == 1.0e-4


def test_v9_ab_changes_only_command_tracking_strength():
    a = Rs01OmniV9Speed10Cfg
    b = Rs01OmniV9Speed14Cfg
    assert a.rewards.scales.tracking_command_velocity == 10.0
    assert b.rewards.scales.tracking_command_velocity == 14.0
    for cfg in (a, b):
        assert cfg.rewards.scales.phase_support_tracking == 1.0
        assert cfg.rewards.scales.phase_two_contact_quality == 0.75
        assert cfg.rewards.scales.phase_swing_clearance == -1.0
        assert cfg.control.action_scale_by_joint["calf"] == 0.22
        assert cfg.rs01_actuator.target_rate_limit_rad_s["calf"] == 3.2
    for ppo in (Rs01OmniV9Speed10CfgPPO, Rs01OmniV9Speed14CfgPPO):
        assert ppo.policy.init_noise_std == 0.30
        assert ppo.runner.action_std_value == 0.30
        assert ppo.runner.freeze_action_std is True
        assert ppo.algorithm.learning_rate == 1.0e-4
