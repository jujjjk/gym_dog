import isaacgym  # noqa: F401 - Isaac Gym must be loaded before torch
import torch

from legged_gym.envs.rs01_go2_straight.rs01_actuator import (
    compute_rs01_joint_torques,
    limit_position_target,
    step_identified_position_response,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_straight_config import (
    Rs01Go2StraightCfg,
)


def test_task_contract_is_minimal_go2_shape_and_real_stand():
    assert Rs01Go2StraightCfg.env.num_observations == 48
    assert Rs01Go2StraightCfg.env.num_actions == 12
    assert Rs01Go2StraightCfg.init_state.pos == [0.0, 0.0, 0.316]
    assert set(Rs01Go2StraightCfg.init_state.default_joint_angles.values()) == {
        0.0,
        -0.32987297,
        1.31853104,
    }
    assert Rs01Go2StraightCfg.rs01_actuator.peak_torque_limit_nm == 17.0
    assert Rs01Go2StraightCfg.rs01_actuator.continuous_torque_nm == 6.0


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
