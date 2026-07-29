import isaacgym  # noqa: F401 - Isaac Gym must be loaded before torch
import torch
from types import SimpleNamespace

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from legged_gym.envs.rs01_go2_straight.rs01_actuator import (
    compute_rs01_joint_torques,
    limit_position_target,
    step_identified_position_response,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_straight_config import (
    Rs01Go2StraightCfg,
    Rs01Go2StraightCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_straight_env import (
    Rs01Go2StraightRobot,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_rear_coord_config import (
    Rs01Go2RearCoordCfg,
    Rs01Go2RearCoordCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_path_polish_config import (
    Rs01Go2PathPolishCfg,
    Rs01Go2PathPolishCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_kp40_config import (
    Rs01Go2Kp40Cfg,
    Rs01Go2Kp40CfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_kp40_polish_config import (
    Rs01Go2Kp40PolishCfg,
    Rs01Go2Kp40PolishCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_sim2sim_config import (
    Rs01Go2Sim2SimAdaptCfg,
    Rs01Go2Sim2SimAdaptCfgPPO,
    Rs01Go2Sim2SimCalfRepairCfg,
    Rs01Go2Sim2SimCalfRepairCfgPPO,
    Rs01Go2Sim2SimKd050Cfg,
    Rs01Go2Sim2SimKd050CfgPPO,
    Rs01Go2Sim2SimRobustCfg,
    Rs01Go2MatchedTransferCfg,
    Rs01Go2MatchedTransferCfgPPO,
    Rs01Go2Heading52Cfg,
    Rs01Go2Heading52CfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_drift_repair_config import (
    Rs01Go2Model930DriftRepairCfg,
    Rs01Go2Model930DriftRepairCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_path54_config import (
    Rs01Go2Model1425Path54Cfg,
    Rs01Go2Model1425Path54CfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_path54_sim2sim_config import (
    Rs01Go2Path54Sim2SimTransferCfg,
    Rs01Go2Path54Sim2SimTransferCfgPPO,
)
from legged_gym.utils.checkpoint_adapter import (
    adapt_observation_input_state,
)


def test_task_contract_is_minimal_go2_shape_and_real_stand():
    assert Rs01Go2StraightCfg.env.num_observations == 50
    assert Rs01Go2StraightCfg.env.num_actions == 12
    assert Rs01Go2StraightCfg.init_state.pos == [0.0, 0.0, 0.316]
    assert set(Rs01Go2StraightCfg.init_state.default_joint_angles.values()) == {
        0.0,
        -0.32987297,
        1.31853104,
    }
    assert Rs01Go2StraightCfg.rs01_actuator.peak_torque_limit_nm == 17.0
    assert Rs01Go2StraightCfg.rs01_actuator.continuous_torque_nm == 6.0
    assert Rs01Go2StraightCfg.rs01_actuator.runtime_dof_order == [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]


def test_real_rs01_motor_ids_match_verified_leg_wiring():
    mapping = Rs01Go2StraightCfg.rs01_actuator.joint_to_motor_id
    assert mapping == {
        "FR_hip_joint": "0x11",
        "FR_thigh_joint": "0x12",
        "FR_calf_joint": "0x13",
        "FL_hip_joint": "0x21",
        "FL_thigh_joint": "0x22",
        "FL_calf_joint": "0x23",
        "RL_hip_joint": "0x31",
        "RL_thigh_joint": "0x32",
        "RL_calf_joint": "0x33",
        "RR_hip_joint": "0x41",
        "RR_thigh_joint": "0x42",
        "RR_calf_joint": "0x43",
    }


def test_contact_solver_and_reset_noise_are_scaled_to_rs01_feet():
    assert Rs01Go2StraightCfg.init_state.reset_dof_position_noise_rad == 0.003
    assert Rs01Go2StraightCfg.sim.substeps == 2
    assert Rs01Go2StraightCfg.sim.physx.num_position_iterations == 8
    assert Rs01Go2StraightCfg.sim.physx.num_velocity_iterations == 2
    assert Rs01Go2StraightCfg.sim.physx.contact_offset == 0.003
    assert Rs01Go2StraightCfg.sim.physx.rest_offset == 0.0
    assert Rs01Go2StraightCfg.sim.physx.max_depenetration_velocity == 0.25


def test_initial_exploration_cannot_grow_back_to_go2_noise():
    assert Rs01Go2StraightCfgPPO.policy.init_noise_std == 0.25
    assert Rs01Go2StraightCfgPPO.algorithm.entropy_coef == 0.0
    assert Rs01Go2StraightCfgPPO.runner.action_std_value == 0.25
    assert Rs01Go2StraightCfgPPO.runner.freeze_action_std is True


def test_standing_reward_shortcut_is_closed_but_handoff_has_grace():
    scales = Rs01Go2StraightCfg.rewards.scales
    assert Rs01Go2StraightCfg.rewards.tracking_sigma == 0.05
    assert Rs01Go2StraightCfg.rewards.only_positive_rewards is False
    assert Rs01Go2StraightCfg.rewards.all_feet_contact_grace_s == 0.12
    assert Rs01Go2StraightCfg.rewards.phase_support_sigma == 0.10
    assert Rs01Go2StraightCfg.rewards.foot_collision_radius_m == 0.016
    assert Rs01Go2StraightCfg.rewards.swing_clearance_m == 0.014
    assert scales.tracking_ang_vel == 0.0
    assert scales.yaw_rate < 0.0
    assert scales.prolonged_all_feet_contact < 0.0
    assert scales.phase_support_tracking > 0.0
    assert scales.phase_swing_clearance < 0.0
    assert scales.same_axle_flight < 0.0
    assert scales.flight < 0.0
    assert scales.raw_torque_over_peak < 0.0
    assert scales.motor_saturation < 0.0

    robot = object.__new__(Rs01Go2StraightRobot)
    robot.dt = 0.02
    robot.cfg = type(
        "Cfg",
        (),
        {
            "rewards": type(
                "Rewards",
                (),
                {"all_feet_contact_grace_s": 0.12},
            )()
        },
    )()
    robot.commands = torch.tensor(
        [[0.4, 0.0, 0.0, 0.0]] * 4
    )
    robot.all_feet_contact_time_s = torch.tensor(
        [0.10, 0.12, 0.18, 0.24]
    )
    penalty = robot._reward_prolonged_all_feet_contact()
    assert torch.allclose(
        penalty,
        torch.tensor([0.0, 0.0, 0.5, 1.0]),
        atol=1.0e-6,
    )


def test_phase_schedule_has_two_diagonals_and_four_foot_handoff():
    diagonal_a = torch.tensor([True, False, False, True])
    diagonal_b = torch.tensor([False, True, True, False])
    desired = Rs01Go2StraightRobot._desired_contact_mask_from_phase(
        torch.tensor([0.25, 0.55, 0.75]),
        diagonal_a,
        diagonal_b,
        stance_ratio=0.65,
    )
    assert torch.equal(
        desired,
        torch.tensor(
            [
                [True, False, False, True],
                [True, True, True, True],
                [False, True, True, False],
            ]
        ),
    )


def test_phase_load_error_rejects_front_rear_pair_and_wrong_diagonal():
    desired = torch.tensor(
        [
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )
    force = torch.tensor(
        [
            [50.0, 0.0, 0.0, 50.0],
            [50.0, 50.0, 0.0, 0.0],
            [25.0, 25.0, 25.0, 25.0],
        ]
    )
    error = Rs01Go2StraightRobot._foot_load_distribution_error(
        force, desired
    )
    assert torch.allclose(
        error,
        torch.tensor([0.0, 0.5, 0.0]),
        atol=1.0e-6,
    )


def test_phase_support_error_requires_swing_feet_to_leave_contact():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.4, 0.0, 0.0, 0.0]] * 2)
    robot.feet_indices = torch.arange(4)
    desired = torch.tensor(
        [[True, False, False, True]] * 2
    )
    robot._desired_contact_mask = lambda: desired
    robot.contact_forces = torch.zeros(2, 4, 3)
    robot.contact_forces[:, :, 2] = torch.tensor(
        [
            [50.0, 0.0, 0.0, 50.0],
            [50.0, 1.0, 1.0, 50.0],
        ]
    )
    robot.foot_contact_mask = torch.tensor(
        [
            [True, False, False, True],
            [True, True, True, True],
        ]
    )
    error = robot._phase_support_error()
    assert error[0].item() == 0.0
    assert error[1].item() > 0.5


def test_swing_height_target_is_low_and_diagonal():
    diagonal_a = torch.tensor([True, False, False, True])
    diagonal_b = torch.tensor([False, True, True, False])
    target, swing = Rs01Go2StraightRobot._swing_height_target_from_phase(
        torch.tensor([0.825, 0.325, 0.55]),
        diagonal_a,
        diagonal_b,
        stance_ratio=0.65,
        foot_radius_m=0.016,
        swing_clearance_m=0.014,
    )
    assert torch.equal(swing[0], diagonal_a)
    assert torch.equal(swing[1], diagonal_b)
    assert not torch.any(swing[2])
    assert torch.allclose(
        target[0],
        torch.tensor([0.030, 0.016, 0.016, 0.030]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        target[1],
        torch.tensor([0.016, 0.030, 0.030, 0.016]),
        atol=1.0e-6,
    )


def test_same_axle_flight_rejects_bound_but_not_diagonal_swing():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.4, 0.0, 0.0, 0.0]] * 3)
    robot.foot_slot_by_leg = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    robot.foot_contact_mask = torch.tensor(
        [
            [True, False, False, True],
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    penalty = robot._reward_same_axle_flight()
    assert torch.equal(penalty, torch.tensor([0.0, 1.0, 2.0]))


def test_rear_coord_polish_preserves_control_contract_and_uses_small_updates():
    assert Rs01Go2RearCoordCfg.env.num_observations == 50
    assert Rs01Go2RearCoordCfg.env.num_actions == 12
    assert Rs01Go2RearCoordCfg.control.stiffness == (
        Rs01Go2StraightCfg.control.stiffness
    )
    assert Rs01Go2RearCoordCfg.control.damping == (
        Rs01Go2StraightCfg.control.damping
    )
    assert Rs01Go2RearCoordCfg.control.action_scale == 0.14
    assert Rs01Go2RearCoordCfg.commands.ranges.lin_vel_x == [0.30, 0.45]
    assert Rs01Go2RearCoordCfg.rewards.scales.diagonal_contact_sync < 0.0
    assert Rs01Go2RearCoordCfg.rewards.scales.rear_swing_clearance < 0.0
    assert Rs01Go2RearCoordCfgPPO.policy.init_noise_std == 0.15
    assert Rs01Go2RearCoordCfgPPO.algorithm.learning_rate == 2.0e-4
    assert Rs01Go2RearCoordCfgPPO.algorithm.schedule == "fixed"
    assert Rs01Go2RearCoordCfgPPO.runner.load_optimizer is False
    assert Rs01Go2RearCoordCfgPPO.runner.reference_policy_coef > 0.0


def test_diagonal_contact_sync_penalizes_only_pair_disagreement():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.4, 0.0, 0.0, 0.0]] * 3)
    robot.foot_slot_by_leg = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    robot.foot_contact_mask = torch.tensor(
        [
            [True, False, False, True],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    penalty = robot._reward_diagonal_contact_sync()
    assert torch.allclose(penalty, torch.tensor([0.0, 0.5, 1.0]))


def test_path_polish_widens_only_heading_observation_and_adapts_checkpoint():
    assert Rs01Go2PathPolishCfg.env.num_observations == 51
    assert Rs01Go2PathPolishCfg.env.num_actions == 12
    assert Rs01Go2PathPolishCfg.commands.observe_straight_heading_error
    assert Rs01Go2PathPolishCfg.rewards.scales.yaw_rate == 0.0
    assert Rs01Go2PathPolishCfg.rewards.scales.heading_recovery < 0.0
    assert Rs01Go2PathPolishCfg.control.stiffness == (
        Rs01Go2RearCoordCfg.control.stiffness
    )
    assert Rs01Go2PathPolishCfg.control.action_scale == 0.14
    assert Rs01Go2PathPolishCfgPPO.runner.adapt_observation_input is True
    assert Rs01Go2PathPolishCfgPPO.runner.load_optimizer is False
    assert Rs01Go2PathPolishCfgPPO.algorithm.learning_rate == 5.0e-5
    assert Rs01Go2PathPolishCfgPPO.runner.save_interval == 5


def test_kp40_task_is_isolated_and_keeps_rs01_motor_contract():
    assert Rs01Go2Kp40Cfg.env.num_observations == 51
    assert Rs01Go2Kp40Cfg.env.num_actions == 12
    assert Rs01Go2Kp40Cfg.control.stiffness == {
        "hip": 40.0,
        "thigh": 40.0,
        "calf": 40.0,
    }
    assert Rs01Go2Kp40Cfg.control.damping == {
        "hip": 1.0,
        "thigh": 1.0,
        "calf": 1.0,
    }
    assert Rs01Go2Kp40Cfg.control.action_scale == 0.18
    assert Rs01Go2Kp40Cfg.commands.ranges.lin_vel_x == [0.18, 0.28]
    assert Rs01Go2Kp40Cfg.commands.playback_speed_mps == 0.23
    assert Rs01Go2Kp40Cfg.rewards.tracking_sigma == 0.015
    assert Rs01Go2Kp40Cfg.rs01_actuator.continuous_torque_nm == 6.0
    assert Rs01Go2Kp40Cfg.rs01_actuator.peak_torque_limit_nm == 17.0
    assert Rs01Go2Kp40CfgPPO.algorithm.learning_rate == 1.0e-4
    assert Rs01Go2Kp40CfgPPO.runner.reference_policy_coef == 0.05


def test_kp40_polish_changes_only_small_update_and_quality_terms():
    assert Rs01Go2Kp40PolishCfg.control.stiffness == (
        Rs01Go2Kp40Cfg.control.stiffness
    )
    assert Rs01Go2Kp40PolishCfg.control.damping == (
        Rs01Go2Kp40Cfg.control.damping
    )
    assert Rs01Go2Kp40PolishCfg.control.action_scale == 0.18
    assert Rs01Go2Kp40PolishCfg.commands.ranges.lin_vel_x == [0.18, 0.28]
    assert Rs01Go2Kp40PolishCfg.rewards.gait_period_s == (
        Rs01Go2Kp40Cfg.rewards.gait_period_s
    )
    assert Rs01Go2Kp40PolishCfg.rewards.gait_stance_ratio == (
        Rs01Go2Kp40Cfg.rewards.gait_stance_ratio
    )
    scales = Rs01Go2Kp40PolishCfg.rewards.scales
    assert scales.raw_torque_over_peak == -1.0
    assert scales.motor_saturation == -1.0
    assert scales.diagonal_contact_sync == -1.0
    assert Rs01Go2Kp40PolishCfgPPO.algorithm.learning_rate == 5.0e-5
    assert Rs01Go2Kp40PolishCfgPPO.policy.init_noise_std == 0.07
    assert Rs01Go2Kp40PolishCfgPPO.runner.save_interval == 5
    assert Rs01Go2Kp40PolishCfgPPO.runner.reference_policy_coef == 0.15


def test_sim2sim_adapt_reduces_only_calf_target_authority():
    assert Rs01Go2Sim2SimAdaptCfg.env.num_observations == 51
    assert Rs01Go2Sim2SimAdaptCfg.env.num_actions == 12
    assert Rs01Go2Sim2SimAdaptCfg.control.stiffness == {
        "hip": 40.0,
        "thigh": 40.0,
        "calf": 40.0,
    }
    assert Rs01Go2Sim2SimAdaptCfg.control.action_scale_by_joint == {
        "hip": 0.18,
        "thigh": 0.18,
        "calf": 0.14,
    }
    assert (
        Rs01Go2Sim2SimAdaptCfg.rs01_actuator
        .target_rate_limit_rad_s["calf"]
    ) == 2.6
    assert (
        Rs01Go2Sim2SimAdaptCfg.rs01_actuator
        .target_acceleration_limit_rad_s2["calf"]
    ) == 72.0
    assert (
        Rs01Go2Sim2SimAdaptCfgPPO.runner
        .reference_action_transform
    ) == "clip"


def test_sim2sim_robust_randomization_is_narrow_and_progressive():
    domain = Rs01Go2Sim2SimRobustCfg.domain_rand
    assert domain.friction_range == [0.85, 1.15]
    assert domain.added_mass_range == [-0.30, 0.30]
    assert domain.rs01_response_gain_scale_range == [0.95, 1.05]
    assert domain.rs01_time_constant_scale_range == [0.90, 1.10]
    assert domain.rs01_friction_scale_range == [0.90, 1.10]
    assert domain.rs01_delay_step_offset_range == [-1, 1]


def test_matched_transfer_preserves_contract_and_randomizes_each_motor():
    assert Rs01Go2MatchedTransferCfg.env.num_observations == 51
    assert Rs01Go2MatchedTransferCfg.env.num_actions == 12
    assert Rs01Go2MatchedTransferCfg.control.stiffness == {
        "hip": 40.0,
        "thigh": 40.0,
        "calf": 40.0,
    }
    assert Rs01Go2MatchedTransferCfg.control.damping["calf"] == 0.50
    assert Rs01Go2MatchedTransferCfg.rs01_actuator.peak_torque_limit_nm == 17.0
    domain = Rs01Go2MatchedTransferCfg.domain_rand
    assert domain.rs01_independent_motor_randomization is True
    assert domain.rs01_independent_delay_randomization is True
    assert domain.rs01_response_gain_scale_range == [0.97, 1.03]
    assert domain.rs01_time_constant_scale_range == [0.95, 1.05]
    assert domain.rs01_friction_scale_range == [0.95, 1.05]
    assert domain.rs01_delay_step_offset_range == [-1, 1]
    assert Rs01Go2MatchedTransferCfg.init_state.reset_heading_noise_rad == 0.12
    assert Rs01Go2MatchedTransferCfg.rewards.scales.heading_recovery == -0.40
    assert Rs01Go2MatchedTransferCfgPPO.algorithm.learning_rate == 2.0e-5
    assert Rs01Go2MatchedTransferCfgPPO.runner.save_interval == 5


def test_heading52_replaces_saturated_scalar_without_changing_motor_contract():
    assert Rs01Go2Heading52Cfg.env.num_observations == 52
    assert Rs01Go2Heading52Cfg.env.num_actions == 12
    assert Rs01Go2Heading52Cfg.commands.observe_straight_heading_error is False
    assert (
        Rs01Go2Heading52Cfg.commands.observe_straight_heading_sin_cos
        is True
    )
    assert Rs01Go2Heading52Cfg.init_state.reset_heading_noise_rad == 0.30
    assert (
        Rs01Go2Heading52Cfg.init_state.reset_yaw_rate_noise_rad_s
        == 0.20
    )
    assert (
        Rs01Go2Heading52Cfg.control.action_scale_by_joint
        == Rs01Go2MatchedTransferCfg.control.action_scale_by_joint
    )
    assert (
        Rs01Go2Heading52Cfg.rs01_actuator.runtime_dof_order
        == Rs01Go2MatchedTransferCfg.rs01_actuator.runtime_dof_order
    )
    migration = (
        Rs01Go2Heading52CfgPPO.runner.observation_column_migration
    )
    assert migration["source_width"] == 51
    assert migration["destination_width"] == 52
    assert migration["copy_prefix"] == 50


def test_model930_drift_repair_is_nominal_52d_fixed_speed_continuation():
    cfg = Rs01Go2Model930DriftRepairCfg
    ppo = Rs01Go2Model930DriftRepairCfgPPO
    assert cfg.env.num_observations == 52
    assert cfg.env.num_actions == 12
    assert cfg.commands.ranges.lin_vel_x == [0.23, 0.23]
    assert cfg.commands.ranges.lin_vel_y == [0.0, 0.0]
    assert cfg.commands.ranges.ang_vel_yaw == [0.0, 0.0]
    assert cfg.commands.observe_straight_heading_sin_cos is True
    assert cfg.noise.add_noise is False
    assert cfg.domain_rand.randomize_friction is False
    assert cfg.domain_rand.randomize_base_mass is False
    assert cfg.domain_rand.randomize_rs01_actuator is False
    assert cfg.domain_rand.push_robots is False
    assert cfg.rs01_actuator.joint_to_motor_id["RL_hip_joint"] == "0x31"
    assert cfg.rs01_actuator.joint_to_motor_id["RR_hip_joint"] == "0x41"
    scales = cfg.rewards.scales
    assert scales.tracking_lin_vel == 0.0
    assert scales.tracking_forward_velocity > 0.0
    assert scales.lateral_velocity < 0.0
    assert scales.heading_recovery < 0.0
    assert scales.near_heading_yaw_rate < 0.0
    assert scales.roll_posture < 0.0
    assert scales.left_right_foot_force_balance < 0.0
    assert scales.rear_motor_torque_balance < 0.0
    assert scales.lateral_foot_slip < 0.0
    assert scales.action_saturation < 0.0
    assert scales.diagonal_contact_sync < 0.0
    assert ppo.algorithm.learning_rate == 1.0e-5
    assert ppo.runner.save_interval == 5
    assert ppo.runner.load_optimizer is False
    assert ppo.runner.adapt_observation_input is False
    assert ppo.runner.reference_policy_coef == 0.03


def test_drift_repair_rewards_separate_forward_and_lateral_errors():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor(
        [[0.23, 0.0, 0.0, 0.0], [0.23, 0.0, 0.0, 0.0]]
    )
    robot.base_lin_vel = torch.tensor(
        [[0.23, 0.0, 0.0], [0.23, 0.08, 0.0]]
    )
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            tracking_sigma=0.015,
            lateral_velocity_scale_mps=0.08,
        )
    )
    forward = robot._reward_tracking_forward_velocity()
    lateral = robot._reward_lateral_velocity()
    assert torch.allclose(forward, torch.ones(2))
    assert torch.allclose(lateral, torch.tensor([0.0, 1.0]))


def test_model1425_path54_preserves_gait_and_adds_only_path_feedback():
    cfg = Rs01Go2Model1425Path54Cfg
    ppo = Rs01Go2Model1425Path54CfgPPO
    assert cfg.env.num_observations == 54
    assert cfg.env.num_actions == 12
    assert cfg.commands.observe_straight_heading_sin_cos is True
    assert cfg.commands.observe_straight_path_state is True
    assert cfg.commands.ranges.lin_vel_x == [0.23, 0.23]
    assert (
        cfg.control.action_scale_by_joint
        == Rs01Go2Model930DriftRepairCfg.control.action_scale_by_joint
    )
    assert (
        cfg.rewards.gait_period_s
        == Rs01Go2Model930DriftRepairCfg.rewards.gait_period_s
    )
    assert (
        cfg.rewards.gait_stance_ratio
        == Rs01Go2Model930DriftRepairCfg.rewards.gait_stance_ratio
    )
    assert cfg.rewards.scales.lateral_velocity == 0.0
    assert cfg.rewards.scales.lateral_path_recovery < 0.0
    assert cfg.rs01_actuator.peak_torque_limit_nm == 17.0
    assert (
        cfg.rs01_actuator.runtime_dof_order
        == Rs01Go2Model930DriftRepairCfg.rs01_actuator.runtime_dof_order
    )
    assert ppo.algorithm.learning_rate == 1.0e-5
    assert ppo.runner.load_optimizer is False
    assert ppo.runner.adapt_observation_input is True
    migration = ppo.runner.observation_column_migration
    assert migration == {
        "source_width": 52,
        "destination_width": 54,
        "copy_prefix": 52,
        "column_mappings": [],
    }


def test_path_state_and_recovery_target_use_the_initial_heading_frame():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.straight_heading_target_rad = torch.tensor(
        [0.0, torch.pi / 2.0]
    )
    robot.straight_path_origin_xy = torch.zeros(2, 2)
    robot.root_states = torch.zeros(2, 13)
    # At heading 0 the lateral axis is +y. At heading pi/2 it is -x.
    robot.root_states[0, 1] = 0.10
    robot.root_states[0, 8] = -0.06
    robot.root_states[1, 0] = -0.20
    robot.root_states[1, 7] = 0.10
    robot.commands = torch.tensor([[0.23, 0.0, 0.0, 0.0]] * 2)
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            lateral_path_recovery_gain_per_s=0.60,
            lateral_path_recovery_max_velocity_mps=0.10,
            lateral_path_recovery_velocity_scale_mps=0.08,
        )
    )

    lateral_error, lateral_velocity = robot._straight_path_state()
    target_velocity = robot._lateral_path_target_velocity()
    penalty = robot._reward_lateral_path_recovery()

    assert torch.allclose(
        lateral_error, torch.tensor([0.10, 0.20]), atol=1.0e-6
    )
    assert torch.allclose(
        lateral_velocity, torch.tensor([-0.06, -0.10]), atol=1.0e-6
    )
    assert torch.allclose(
        target_velocity, torch.tensor([-0.06, -0.10]), atol=1.0e-6
    )
    assert torch.allclose(penalty, torch.zeros(2), atol=1.0e-6)


def test_model1425_checkpoint_columns_widen_from_52_to_54_without_aliasing():
    old_actor = torch.arange(
        512 * 52, dtype=torch.float
    ).reshape(512, 52)
    old_critic = old_actor + 1.0
    loaded = {
        "actor.0.weight": old_actor,
        "critic.0.weight": old_critic,
    }
    current = {
        "actor.0.weight": torch.full((512, 54), -1.0),
        "critic.0.weight": torch.full((512, 54), -1.0),
    }
    adapted, changes = adapt_observation_input_state(
        loaded,
        current,
        column_migration=(
            Rs01Go2Model1425Path54CfgPPO.runner
            .observation_column_migration
        ),
    )

    assert len(changes) == 2
    for key, source in (
        ("actor.0.weight", old_actor),
        ("critic.0.weight", old_critic),
    ):
        assert torch.equal(adapted[key][:, :52], source)
        assert torch.count_nonzero(adapted[key][:, 52:]) == 0
        assert adapted[key].data_ptr() != current[key].data_ptr()
        assert adapted[key].data_ptr() != source.data_ptr()


def test_path54_sim2sim_transfer_restores_only_narrow_dynamics_spread():
    cfg = Rs01Go2Path54Sim2SimTransferCfg
    ppo = Rs01Go2Path54Sim2SimTransferCfgPPO
    assert cfg.env.num_observations == 54
    assert cfg.env.num_actions == 12
    assert cfg.commands.observe_straight_path_state is True
    assert cfg.commands.ranges.lin_vel_x == [0.23, 0.23]
    assert cfg.noise.add_noise is False
    domain = cfg.domain_rand
    assert domain.randomize_friction is True
    assert domain.friction_range == [0.90, 1.10]
    assert domain.randomize_base_mass is True
    assert domain.added_mass_range == [-0.20, 0.20]
    assert domain.randomize_rs01_actuator is True
    assert domain.rs01_response_gain_scale_range == [0.97, 1.03]
    assert domain.rs01_time_constant_scale_range == [0.95, 1.05]
    assert domain.rs01_friction_scale_range == [0.95, 1.05]
    assert domain.rs01_delay_step_offset_range == [-1, 1]
    assert domain.rs01_independent_motor_randomization is True
    assert domain.rs01_independent_delay_randomization is True
    assert domain.push_robots is False
    assert (
        cfg.control.action_scale_by_joint
        == Rs01Go2Model1425Path54Cfg.control.action_scale_by_joint
    )
    assert (
        cfg.rewards.gait_period_s
        == Rs01Go2Model1425Path54Cfg.rewards.gait_period_s
    )
    assert ppo.algorithm.learning_rate == 5.0e-6
    assert ppo.runner.action_std_value == 0.025
    assert ppo.runner.load_optimizer is False
    assert ppo.runner.adapt_observation_input is False
    assert ppo.runner.reference_policy_coef == 0.05


def test_near_heading_yaw_cost_does_not_fight_large_heading_recovery():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.23, 0.0, 0.0, 0.0]] * 3)
    robot.straight_heading_target_rad = torch.zeros(3)
    robot.rpy = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.10],
            [0.0, 0.0, 0.25],
        ]
    )
    robot.base_ang_vel = torch.tensor(
        [[0.0, 0.0, 0.60]] * 3
    )
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            near_heading_window_rad=0.20,
            yaw_rate_scale_rad_s=0.60,
        )
    )
    penalty = robot._reward_near_heading_yaw_rate()
    assert torch.allclose(
        penalty,
        torch.tensor([1.0, 0.5, 0.0]),
        atol=1.0e-6,
    )


def test_left_right_load_and_cycle_rear_torque_balance_are_normalized():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.23, 0.0, 0.0, 0.0]] * 2)
    robot.feet_indices = torch.arange(4)
    robot.foot_slot_by_leg = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    robot.contact_forces = torch.zeros(2, 4, 3)
    robot.contact_forces[:, :, 2] = torch.tensor(
        [
            [30.0, 30.0, 30.0, 30.0],
            [45.0, 15.0, 45.0, 15.0],
        ]
    )
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            foot_contact_force_threshold=2.0,
            gait_period_s=0.60,
        )
    )
    load_penalty = robot._reward_left_right_foot_force_balance()
    assert torch.allclose(load_penalty, torch.tensor([0.0, 0.25]))

    robot.rear_motor_torque_abs_ema_nm = torch.tensor(
        [
            [[3.0, 6.0, 9.0], [3.0, 6.0, 9.0]],
            [[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]],
        ]
    )
    robot.episode_length_buf = torch.tensor([30, 30])
    robot.dt = 0.02
    torque_penalty = robot._reward_rear_motor_torque_balance()
    assert torch.allclose(torque_penalty, torch.tensor([0.0, 0.25]))


def test_lateral_slip_penalizes_only_contacting_feet():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.23, 0.0, 0.0, 0.0]])
    robot.feet_state = torch.zeros(1, 4, 13)
    robot.feet_state[0, :, 8] = torch.tensor([0.20, 1.0, -0.20, -1.0])
    robot.foot_contact_mask = torch.tensor(
        [[True, False, True, False]]
    )
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(lateral_foot_slip_scale_mps=0.20)
    )
    penalty = robot._reward_lateral_foot_slip()
    assert torch.allclose(penalty, torch.tensor([1.0]))


def test_calf_repair_uses_measured_kd_sweep_and_direct_feasibility_terms():
    assert Rs01Go2Sim2SimCalfRepairCfg.env.num_observations == 51
    assert Rs01Go2Sim2SimCalfRepairCfg.env.num_actions == 12
    assert Rs01Go2Sim2SimCalfRepairCfg.control.damping == {
        "hip": 1.0,
        "thigh": 1.0,
        "calf": 0.55,
    }
    assert (
        Rs01Go2Sim2SimCalfRepairCfg.rewards
        .calf_velocity_soft_limit_rad_s
    ) == 8.0
    assert (
        Rs01Go2Sim2SimCalfRepairCfg.rewards
        .action_saturation_soft_limit
    ) == 0.90
    scales = Rs01Go2Sim2SimCalfRepairCfg.rewards.scales
    assert scales.calf_velocity_excess < 0.0
    assert scales.action_saturation < 0.0
    assert scales.raw_torque_over_peak == -1.0
    assert scales.motor_saturation == -1.0
    assert Rs01Go2Sim2SimCalfRepairCfgPPO.runner.action_std_value == 0.08
    assert Rs01Go2Sim2SimCalfRepairCfgPPO.runner.freeze_action_std is True


def test_kd050_stage_is_a_small_checkpoint_815_continuation():
    assert Rs01Go2Sim2SimKd050Cfg.env.num_observations == 51
    assert Rs01Go2Sim2SimKd050Cfg.env.num_actions == 12
    assert Rs01Go2Sim2SimKd050Cfg.control.stiffness == {
        "hip": 40.0,
        "thigh": 40.0,
        "calf": 40.0,
    }
    assert Rs01Go2Sim2SimKd050Cfg.control.damping == {
        "hip": 1.0,
        "thigh": 1.0,
        "calf": 0.50,
    }
    assert (
        Rs01Go2Sim2SimKd050Cfg.control.action_scale_by_joint
        == Rs01Go2Sim2SimCalfRepairCfg.control.action_scale_by_joint
    )
    assert (
        Rs01Go2Sim2SimKd050Cfg.rs01_actuator
        .target_rate_limit_rad_s
        == Rs01Go2Sim2SimCalfRepairCfg.rs01_actuator
        .target_rate_limit_rad_s
    )
    assert Rs01Go2Sim2SimKd050CfgPPO.algorithm.learning_rate == 2.0e-5
    assert Rs01Go2Sim2SimKd050CfgPPO.runner.action_std_value == 0.06
    assert Rs01Go2Sim2SimKd050CfgPPO.runner.freeze_action_std is True
    assert Rs01Go2Sim2SimKd050CfgPPO.runner.save_interval == 5
    assert Rs01Go2Sim2SimRobustCfg.control.damping["calf"] == 0.50


def test_calf_velocity_reward_is_zero_below_soft_limit_and_quadratic_above():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.calf_dof_indices = torch.tensor([2, 5, 8, 11])
    robot.dof_vel = torch.zeros(2, 12)
    robot.dof_vel[0, robot.calf_dof_indices] = torch.tensor(
        [4.0, 8.0, -6.0, -7.0]
    )
    robot.dof_vel[1, robot.calf_dof_indices] = torch.tensor(
        [16.0, -8.0, 8.0, 8.0]
    )
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(calf_velocity_soft_limit_rad_s=8.0)
    )
    penalty = robot._reward_calf_velocity_excess()
    assert penalty[0].item() == 0.0
    assert torch.allclose(penalty[1], torch.tensor(0.25))


def test_action_saturation_reward_observes_preclip_policy_output():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.policy_actions_unclipped = torch.tensor(
        [[0.5, 0.9, 1.1, -1.3]]
    )
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(action_saturation_soft_limit=0.9),
        normalization=SimpleNamespace(clip_actions=1.0),
    )
    penalty = robot._reward_action_saturation()
    assert torch.allclose(
        penalty,
        torch.tensor([(0.2**2 + 0.4**2) / 4.0]),
        atol=1.0e-7,
    )


def test_checkpoint_state_loading_reapplies_destination_action_std():
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.std = torch.nn.Parameter(torch.full((3,), 0.15))

    class ResettableEnv:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            return None, None

    runner = object.__new__(OnPolicyRunner)
    runner.cfg = {
        "action_std_value": 0.08,
        "freeze_action_std": True,
    }
    runner.env = ResettableEnv()
    actor = Actor()
    runner.alg = SimpleNamespace(actor_critic=actor)
    runner.load_actor_critic_state_dict(
        {"std": torch.full((3,), 0.15)}
    )
    assert torch.allclose(actor.std, torch.full((3,), 0.08))
    assert actor.std.requires_grad is False
    assert runner.env.reset_count == 1


def test_heading_recovery_tracks_restoring_rate_instead_of_zero_rate():
    robot = object.__new__(Rs01Go2StraightRobot)
    robot.commands = torch.tensor([[0.4, 0.0, 0.0, 0.0]] * 2)
    robot.straight_heading_target_rad = torch.zeros(2)
    robot.rpy = torch.tensor(
        [[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]]
    )
    robot.base_ang_vel = torch.tensor(
        [[0.0, 0.0, -0.3], [0.0, 0.0, 0.0]]
    )
    robot.cfg = type(
        "Cfg",
        (),
        {
            "rewards": type(
                "Rewards",
                (),
                {
                    "heading_recovery_gain_rad_s_per_rad": 1.5,
                    "heading_recovery_max_rate_rad_s": 0.6,
                },
            )()
        },
    )()
    penalty = robot._reward_heading_recovery()
    assert penalty[0].item() < 1.0e-6
    assert torch.allclose(penalty[1], torch.tensor(0.25))


def test_position_target_obeys_rate_and_acceleration_limits():
    desired = torch.tensor([[1.0]])
    previous = torch.zeros_like(desired)
    previous_rate = torch.zeros_like(desired)
    rate_limit = torch.tensor([2.0])
    acceleration_limit = torch.tensor([60.0])
    target, rate = limit_position_target(
        desired,
        previous,
        previous_rate,
        rate_limit,
        acceleration_limit,
        0.02,
    )
    assert torch.allclose(rate, torch.tensor([[1.2]]))
    assert torch.allclose(target, torch.tensor([[0.024]]))


def test_identified_response_uses_gain_and_time_constant():
    response = torch.zeros(1, 1)
    delayed = torch.ones(1, 1)
    default = torch.zeros(1, 1)
    gain = torch.tensor([0.9])
    tau = torch.tensor([0.03])
    updated = step_identified_position_response(
        response, delayed, default, gain, tau, 0.005
    )
    assert 0.0 < updated.item() < 0.9


def test_motor_torque_is_peak_limited_non_aliasing_and_friction_opposes_motion():
    raw, motor, applied = compute_rs01_joint_torques(
        response_target_rad=torch.tensor([[1.0, -1.0]]),
        joint_position_rad=torch.zeros(1, 2),
        joint_velocity_rad_s=torch.tensor([[1.0, -1.0]]),
        kp_nm_per_rad=torch.tensor([70.0, 70.0]),
        kd_nm_per_rad_s=torch.zeros(2),
        peak_torque_limit_nm=torch.full((1, 2), 17.0),
        coulomb_friction_nm=torch.full((2,), 0.156),
        friction_smoothing_rad_s=0.05,
    )
    assert torch.max(torch.abs(motor)).item() <= 17.0
    assert raw.data_ptr() != motor.data_ptr()
    assert motor.data_ptr() != applied.data_ptr()
    assert applied[0, 0] < motor[0, 0]
    assert applied[0, 1] > motor[0, 1]
