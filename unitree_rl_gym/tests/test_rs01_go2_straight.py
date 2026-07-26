import isaacgym  # noqa: F401 - Isaac Gym must be loaded before torch
import torch

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
