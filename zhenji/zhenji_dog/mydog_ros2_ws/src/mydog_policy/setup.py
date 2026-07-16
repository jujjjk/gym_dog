from glob import glob

from setuptools import find_packages, setup

package_name = "mydog_policy"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            glob("launch/*.launch.py"),
        ),
        (
            "share/" + package_name + "/models",
            glob("resource/fanfan_clearance_robust_5730.*")
            + glob("resource/fanfan_hardware_balance_5530_best.*")
            + glob("resource/fanfan_realdata_best.*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jetson",
    maintainer_email="jetson@todo.todo",
    description="Fanfan quadruped robot policy deployment package",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "mydog_policy_node = mydog_policy.mydog_policy_node:main",
            "mydog_check_5100_start_pose = mydog_policy.check_5100_start_pose:main",
            "mydog_check_force_coord_start_pose = mydog_policy.check_force_coord_start_pose:main",
            "mydog_validate_force_coord_model = mydog_policy.validate_force_coord_model:main",
            "mydog_state_estimator_node = mydog_policy.state_estimator_fixed_node:main",
            "mydog_openloop_gait_node = mydog_policy.openloop_gait_node:main",
            "fanfan_ik_gait_node = mydog_policy.fanfan_ik_gait_node:main",
            "fanfan_step_in_place_node = mydog_policy.fanfan_step_in_place_node:main",
            "fanfan_cpg_vmc_v4_node = mydog_policy.fanfan_cpg_vmc_v4_node:main",
            "fanfan_cpg_vmc_v4_migration_node = mydog_policy.fanfan_cpg_vmc_v4_migration_node:main",
            "mydog_soft_body_gait_node = mydog_policy.openloop_gait_node_soft_body_urdf:main",
            "mydog_joint_semantic_test_node = mydog_policy.joint_semantic_test_node:main",
            "mydog_validate_omni_fast = mydog_policy.validate_omni_fast:main",
            "mydog_validate_omni_yaw_clean = mydog_policy.validate_omni_fast:main",
            "mydog_omni_fast_command = mydog_policy.omni_fast_command:main",
            "mydog_omni_yaw_clean_command = mydog_policy.omni_fast_command:main",
            "mydog_policy_parity_node = mydog_policy.sim2real_parity_fixed_node:main",
            "mydog_force_coord_node = mydog_policy.sim2real_force_coord_node:main",
            "mydog_symmetric_transition_node = mydog_policy.sim2real_symmetric_transition_node:main",
            "mydog_validate_symmetric_transition_model = mydog_policy.validate_symmetric_transition_model:main",
            "mydog_symmetric_transition_command = mydog_policy.symmetric_transition_command:main",
            # Clearance-robust checkpoint 5730 deployment.
            "mydog_clearance_robust_node = mydog_policy.sim2real_clearance_robust_node:main",
            (
                "mydog_validate_clearance_robust_model = "
                "mydog_policy.validate_clearance_robust_model:main"
            ),
            "mydog_clearance_robust_command = mydog_policy.clearance_robust_command:main",
            # Selected final hardware-balance deployment.
            (
                "mydog_hardware_balance_node = "
                "mydog_policy.sim2real_hardware_balance_node:main"
            ),
            (
                "mydog_validate_hardware_balance_model = "
                "mydog_policy.validate_hardware_balance_model:main"
            ),
            (
                "mydog_hardware_balance_command = "
                "mydog_policy.hardware_balance_command:main"
            ),
            (
                "mydog_realdata_node = "
                "mydog_policy.sim2real_realdata_node:main"
            ),
        ],
    },
)
