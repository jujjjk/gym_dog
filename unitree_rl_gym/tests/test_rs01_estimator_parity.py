from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ROS_PACKAGE = ROOT / "zhenji/mydog_ros2_ws/src/mydog_policy"
sys.path.insert(0, str(ROS_PACKAGE))

ODOMETRY_PATH = (
    ROOT
    / "unitree_rl_gym/legged_gym/envs/rs01_go2_straight"
    / "rs01_odometry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rs01_training_odometry", ODOMETRY_PATH
)
ODOMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ODOMETRY)
Rs01TorchLegOdometry = ODOMETRY.Rs01TorchLegOdometry
Rs01TorchStraightPathEstimator = (
    ODOMETRY.Rs01TorchStraightPathEstimator
)
from mydog_policy.rs01_model1850_core import (  # noqa: E402
    Rs01StraightPathEstimator,
)
from mydog_policy.rs01_model930_core import (  # noqa: E402
    Rs01NewMachineLegOdometry,
)


DEFAULT = np.tile(
    np.asarray([0.0, -0.32987297, 1.31853104], dtype=np.float32),
    4,
)


def test_torch_fk_matches_deployment_fk_for_actual_rs01_geometry():
    torch_estimator = Rs01TorchLegOdometry(1, "cpu")
    foot, jacobian = torch_estimator.foot_position_and_jacobian(
        torch.tensor(DEFAULT[None, :])
    )
    for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
        expected_foot, expected_jacobian = (
            Rs01NewMachineLegOdometry.foot_position_and_jacobian(
                leg,
                DEFAULT[leg_index * 3:leg_index * 3 + 3],
            )
        )
        np.testing.assert_allclose(
            foot[0, leg_index].numpy(),
            expected_foot,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            jacobian[0, leg_index].numpy(),
            expected_jacobian,
            atol=2.0e-6,
        )


def test_torch_and_deployment_odometry_match_over_stateful_sequence():
    rng = np.random.default_rng(1850)
    torch_estimator = Rs01TorchLegOdometry(2, "cpu")
    numpy_estimators = [
        Rs01NewMachineLegOdometry(),
        Rs01NewMachineLegOdometry(),
    ]
    for _ in range(40):
        q = (
            DEFAULT[None, :]
            + rng.normal(0.0, 0.015, size=(2, 12))
        ).astype(np.float32)
        dq = rng.normal(0.0, 0.08, size=(2, 12)).astype(np.float32)
        omega = rng.normal(0.0, 0.03, size=(2, 3)).astype(np.float32)
        torch_result = torch_estimator.estimate(
            torch.tensor(q),
            torch.tensor(dq),
            torch.tensor(omega),
        )
        for env_index, numpy_estimator in enumerate(numpy_estimators):
            numpy_result = numpy_estimator.estimate(
                q[env_index], dq[env_index], omega[env_index]
            )
            np.testing.assert_allclose(
                torch_result["base_linear_velocity"][
                    env_index
                ].numpy(),
                numpy_result["base_linear_velocity"],
                atol=3.0e-6,
            )
            np.testing.assert_array_equal(
                torch_result["stance_mask"][env_index].numpy(),
                numpy_result["stance_mask"],
            )
            np.testing.assert_allclose(
                torch_result["foot_position"][env_index].numpy(),
                numpy_result["foot_position"],
                atol=3.0e-6,
            )
            np.testing.assert_allclose(
                torch_result["velocity_by_foot"][
                    env_index
                ].numpy(),
                numpy_result["velocity_by_foot"],
                atol=3.0e-6,
            )


def test_strict_diagonal_torch_and_deployment_estimators_match():
    rng = np.random.default_rng(1951)
    torch_estimator = Rs01TorchLegOdometry(
        2, "cpu", strict_diagonal_pairs=True
    )
    numpy_estimators = [
        Rs01NewMachineLegOdometry(strict_diagonal_pairs=True),
        Rs01NewMachineLegOdometry(strict_diagonal_pairs=True),
    ]
    for _ in range(30):
        q = (
            DEFAULT[None, :]
            + rng.normal(0.0, 0.015, size=(2, 12))
        ).astype(np.float32)
        dq = rng.normal(0.0, 0.08, size=(2, 12)).astype(np.float32)
        omega = rng.normal(0.0, 0.03, size=(2, 3)).astype(np.float32)
        torch_result = torch_estimator.estimate(
            torch.tensor(q), torch.tensor(dq), torch.tensor(omega)
        )
        for env_index, numpy_estimator in enumerate(numpy_estimators):
            numpy_result = numpy_estimator.estimate(
                q[env_index], dq[env_index], omega[env_index]
            )
            np.testing.assert_allclose(
                torch_result["base_linear_velocity"][
                    env_index
                ].numpy(),
                numpy_result["base_linear_velocity"],
                atol=3.0e-6,
            )
            np.testing.assert_allclose(
                torch_result["confidence"][env_index].item(),
                numpy_result["confidence"],
                atol=3.0e-6,
            )
            np.testing.assert_array_equal(
                torch_result["stance_mask"][env_index].numpy(),
                numpy_result["stance_mask"],
            )


def test_odometry_and_path_reset_remove_previous_session_memory():
    torch_odometry = Rs01TorchLegOdometry(2, "cpu")
    q = torch.tensor(np.tile(DEFAULT, (2, 1)))
    dq = torch.full((2, 12), 0.1)
    omega = torch.zeros(2, 3)
    torch_odometry.estimate(q, dq, omega)
    torch_odometry.reset(torch.tensor([1]))
    assert torch.count_nonzero(torch_odometry.filtered[1]) == 0
    assert not torch.any(torch_odometry.last_stance[1])

    numpy_odometry = Rs01NewMachineLegOdometry()
    numpy_odometry.estimate(DEFAULT, np.full(12, 0.1), np.zeros(3))
    numpy_odometry.reset()
    np.testing.assert_array_equal(numpy_odometry.filtered, 0.0)
    np.testing.assert_array_equal(numpy_odometry.last_stance, False)

    torch_path = Rs01TorchStraightPathEstimator(1, "cpu", 0.02)
    numpy_path = Rs01StraightPathEstimator()
    torch_path.reset(torch.tensor([0]), torch.tensor([0.3]))
    numpy_path.reset(0.0, 0.3)
    for step in range(1, 21):
        yaw = 0.3 + 0.01 * step
        velocity = [0.23, 0.04, 0.0]
        torch_state = torch_path.update(
            torch.tensor([yaw]), torch.tensor([velocity])
        )
        numpy_state = numpy_path.update(
            0.02 * step, yaw, velocity
        )
        np.testing.assert_allclose(
            [
                torch_state[0][0].item(),
                torch_state[1][0].item(),
            ],
            numpy_state,
            atol=2.0e-7,
        )
    torch_path.reset(torch.tensor([0]), torch.tensor([-0.4]))
    assert torch_path.lateral_displacement[0].item() == 0.0
    assert torch_path.lateral_velocity[0].item() == 0.0
