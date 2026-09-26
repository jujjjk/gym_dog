"""Scalar four-leg FK must reproduce the numpy reference it replaced."""
import numpy as np
from mydog_policy.rs01_model930_core import Rs01NewMachineLegOdometry, _axis_angle


def reference_fk(leg, q_leg):
    cls = Rs01NewMachineLegOdometry
    q_leg = np.asarray(q_leg, dtype=np.float64).reshape(3)
    origins = cls.ORIGINS[leg]
    rotation = np.eye(3)
    position = np.zeros(3)
    joint_positions, joint_axes = [], []
    for index in range(3):
        position += rotation @ np.asarray(origins[index])
        joint_positions.append(position.copy())
        joint_axes.append(rotation @ cls.AXES[index])
        rotation = rotation @ _axis_angle(cls.AXES[index], float(q_leg[index]))
    foot = position + rotation @ np.asarray(origins[3])
    jacobian = np.zeros((3, 3))
    for index in range(3):
        jacobian[:, index] = np.cross(joint_axes[index], foot - joint_positions[index])
    return foot.astype(np.float32), jacobian.astype(np.float32)


def test_scalar_fk_matches_numpy_reference_over_joint_range():
    rng = np.random.default_rng(23500)
    for _ in range(400):
        q = rng.uniform([-.8, -1.6, .6], [.8, 1.6, 2.7])
        for leg in Rs01NewMachineLegOdometry.LEG_ORDER:
            foot, jac = Rs01NewMachineLegOdometry.foot_position_and_jacobian(leg, q)
            ref_foot, ref_jac = reference_fk(leg, q)
            assert foot.dtype == np.float32 and jac.dtype == np.float32
            np.testing.assert_allclose(foot, ref_foot, rtol=0, atol=2e-6)
            np.testing.assert_allclose(jac, ref_jac, rtol=0, atol=2e-6)


def test_kinematics_spin_term_matches_numpy_cross():
    odometry = Rs01NewMachineLegOdometry()
    rng = np.random.default_rng(1)
    q = rng.uniform(-.5, .5, 12) + np.tile([0., -.33, 1.32], 4)
    dq = rng.uniform(-3., 3., 12)
    omega = np.array([.3, -.2, .5], dtype=np.float32)
    result = odometry.compute_kinematics(q, dq, omega)
    for leg_index, leg in enumerate(Rs01NewMachineLegOdometry.LEG_ORDER):
        foot, jac = reference_fk(leg, q[leg_index*3:leg_index*3+3])
        expected = -(jac @ dq[leg_index*3:leg_index*3+3].astype(np.float32) + np.cross(omega, foot))
        np.testing.assert_allclose(result['velocity_by_foot'][leg_index], expected, atol=5e-6)
