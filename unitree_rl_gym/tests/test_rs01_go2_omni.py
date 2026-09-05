from types import SimpleNamespace

import isaacgym  # noqa: F401 - Isaac Gym must be loaded before torch
import torch

from legged_gym.envs.rs01_go2_straight.rs01_go2_omni_config import (
    Rs01Go2OmniDiagonalCfg,
    Rs01Go2OmniDiagonalCfgPPO,
)
from legged_gym.envs.rs01_go2_straight.rs01_go2_omni_env import (
    Rs01Go2OmniDiagonalRobot,
)
from legged_gym.utils.helpers import get_load_path, update_cfg_from_args


def test_omni_task_keeps_checkpoint_and_measured_actuator_contract():
    cfg = Rs01Go2OmniDiagonalCfg
    ppo = Rs01Go2OmniDiagonalCfgPPO
    assert cfg.env.num_observations == 54
    assert cfg.env.num_actions == 12
    assert cfg.commands.use_rs01_estimated_observations is True
    assert cfg.rs01_actuator.peak_torque_limit_nm == 17.0
    assert cfg.control.action_scale_by_joint == {
        "hip": 0.18,
        "thigh": 0.18,
        "calf": 0.14,
    }
    assert ppo.runner.reference_policy_coef == 0.0
    assert ppo.runner.adapt_observation_input is False
    assert ppo.runner.load_optimizer is False
    assert ppo.runner.freeze_action_std is False
    assert ppo.algorithm.schedule == "adaptive"


def test_omni_rewards_do_not_penalize_requested_lateral_or_yaw_motion():
    scales = Rs01Go2OmniDiagonalCfg.rewards.scales
    assert scales.tracking_planar_velocity > 0.0
    assert scales.tracking_yaw_velocity > 0.0
    assert scales.tracking_forward_velocity == 0.0
    assert scales.lateral_velocity == 0.0
    assert scales.yaw_rate == 0.0
    assert scales.lateral_path_recovery == 0.0
    assert scales.left_right_foot_force_balance == 0.0
    assert scales.rear_motor_torque_balance == 0.0
    assert scales.phase_support_tracking > 0.0
    assert scales.same_axle_flight < 0.0
    assert scales.flight < 0.0


def test_stratified_sampler_covers_every_mode_and_respects_ranges():
    robot = object.__new__(Rs01Go2OmniDiagonalRobot)
    robot.device = "cpu"
    robot.cfg = SimpleNamespace(commands=Rs01Go2OmniDiagonalCfg.commands)
    robot.command_mode_probabilities = torch.tensor(
        Rs01Go2OmniDiagonalCfg.commands.mode_probabilities
    )
    robot.commands = torch.zeros(20000, 4)
    robot.command_mode = torch.zeros(20000, dtype=torch.long)
    env_ids = torch.arange(20000)

    torch.manual_seed(20260904)
    robot._resample_commands(env_ids)

    empirical = torch.bincount(robot.command_mode, minlength=6).float() / 20000
    expected = robot.command_mode_probabilities
    assert torch.all(torch.abs(empirical - expected) < 0.015)

    stand = robot.command_mode == robot.COMMAND_STAND
    forward = robot.command_mode == robot.COMMAND_FORWARD
    backward = robot.command_mode == robot.COMMAND_BACKWARD
    lateral = robot.command_mode == robot.COMMAND_LATERAL
    yaw = robot.command_mode == robot.COMMAND_YAW
    assert torch.count_nonzero(robot.commands[stand, :3]) == 0
    assert torch.all((robot.commands[forward, 0] >= 0.10) & (robot.commands[forward, 0] <= 0.23))
    assert torch.all((robot.commands[backward, 0] >= -0.15) & (robot.commands[backward, 0] <= -0.08))
    assert torch.count_nonzero(robot.commands[lateral][:, [0, 2]]) == 0
    assert torch.count_nonzero(robot.commands[yaw, :2]) == 0


def test_walking_gate_accepts_forward_backward_lateral_and_yaw():
    robot = object.__new__(Rs01Go2OmniDiagonalRobot)
    robot.cfg = SimpleNamespace(commands=Rs01Go2OmniDiagonalCfg.commands)
    robot.commands = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0],
            [-0.10, 0.0, 0.0, 0.0],
            [0.0, 0.10, 0.0, 0.0],
            [0.0, 0.0, 0.20, 0.0],
        ]
    )
    assert torch.equal(
        robot._walking_command_gate(),
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]),
    )


def test_vector_tracking_is_maximal_only_at_requested_velocity():
    robot = object.__new__(Rs01Go2OmniDiagonalRobot)
    robot.cfg = SimpleNamespace(
        rewards=SimpleNamespace(planar_tracking_sigma=0.010)
    )
    robot.commands = torch.tensor(
        [[0.20, 0.0, 0.0, 0.0], [0.0, 0.10, 0.0, 0.0]]
    )
    robot.base_lin_vel = torch.tensor(
        [[0.20, 0.0, 0.0], [0.05, 0.10, 0.0]]
    )
    reward = robot._reward_tracking_planar_velocity()
    assert reward[0].item() == 1.0
    assert torch.allclose(reward[1], torch.exp(torch.tensor(-0.25)))


def test_phase_support_error_depends_on_contact_pattern_not_equal_load():
    robot = object.__new__(Rs01Go2OmniDiagonalRobot)
    robot._desired_contact_mask = lambda: torch.tensor(
        [[True, False, False, True], [True, False, False, True]]
    )
    robot.get_foot_contact_mask = lambda: torch.tensor(
        [[True, False, False, True], [True, True, False, True]]
    )
    assert torch.allclose(
        robot._phase_support_error(), torch.tensor([0.0, 0.25])
    )


def test_command_line_seed_updates_environment_seed():
    env_cfg = SimpleNamespace(env=SimpleNamespace(num_envs=8), seed=1)
    args = SimpleNamespace(seed=20260904, num_envs=None)
    updated, _ = update_cfg_from_args(env_cfg, None, args)
    assert updated.seed == 20260904


def test_absolute_cross_task_checkpoint_does_not_need_destination_log_root():
    checkpoint = get_load_path(
        None,
        "/tmp/rs01_source_run",
        checkpoint=1950,
    )
    assert checkpoint == "/tmp/rs01_source_run/model_1950.pt"
