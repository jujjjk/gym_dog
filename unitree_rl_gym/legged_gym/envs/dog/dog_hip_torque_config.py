"""Compact-hip, lower-torque continuation from selected model_6100."""

from .dog_low_twist_config import (
    DogRs01LowTwistCfg,
    DogRs01LowTwistCfgPPO,
)


class DogRs01HipTorqueCfg(DogRs01LowTwistCfg):
    """Preserve the diagonal gait while reducing hip sweep and motor demand."""

    class control(DogRs01LowTwistCfg.control):
        # model_6100 averages 0.328 rad hip peak-to-peak with 0.22 rad target
        # authority.  A moderate 18% reduction keeps balance authority while
        # asking PPO to replace visible lateral sweep with quieter support.
        # The actor still directly outputs all 12 motor targets.
        hip_action_scale = 0.18

        # A narrower hip target safety envelope is still far outside the
        # learned motion.  It prevents a future continuation from exploiting
        # the old +/-0.60 rad envelope without changing the RS01 torque caps.
        target_position_limits_by_joint = {
            "hip": [-0.35, 0.35],
            "thigh": [-1.20, 0.45],
            "calf": [0.45, 1.75],
        }

    class rewards(DogRs01LowTwistCfg.rewards):
        # Keep a 4 degree balance band, then progressively penalize excursion.
        hip_excursion_soft_limit_rad = 0.070
        hip_excursion_penalty_width_rad = 0.090
        hip_target_soft_limit_rad = 0.080
        hip_target_penalty_width_rad = 0.080

        class scales(DogRs01LowTwistCfg.rewards.scales):
            # The positive term prevents the clipped-reward objective from
            # finding "low torque by standing still".
            compact_hip_low_torque_forward = 12.0

            # Directly compress achieved and requested hip sweep.  Target and
            # action-rate terms act before the motion reaches the trunk.
            hip_joint_excursion = -7.0
            hip_target_excursion = -3.0
            hip_policy_action_rate = -0.20
            hip_velocity = -0.004

            # Dense torque shaping acts below saturation.  Retain the hard
            # 14/16/17 Nm caps and the manual's 6 Nm sustained-rating costs.
            motor_torque_usage = -4.0
            sagittal_motor_saturation = -4.0
            torque_clip = -1.8
            torque_near_limit = -0.70
            peak_torque = -0.85
            sustained_torque = -1.20
            sustained_torque_max = -1.30
            torques = -2.0e-5
            mechanical_power = -0.0035

            # Do not trade compact hips for renewed trunk twisting or slower
            # shuffling; preserve the selected model's key objectives.
            smooth_low_torque_forward = 12.0
            smooth_diagonal_handoff = 10.0
            handoff_body_twist = -7.0
            body_angular_velocity = -4.0
            tracking_lin_vel = 8.0
            command_velocity_progress = 20.0
            normalized_command_tracking = 13.0
            exact_diagonal_swing = 7.0


class DogRs01HipTorqueCfgPPO(DogRs01LowTwistCfgPPO):
    """Conservative continuation from the selected low-twist model_6100."""

    class algorithm(DogRs01LowTwistCfgPPO.algorithm):
        learning_rate = 1.5e-5
        entropy_coef = 5.0e-5
        schedule = "fixed"

    class runner(DogRs01LowTwistCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_direct12_hip_compact_torque_v1"
        resume = True
        load_run = "Jul23_17-34-12_rs01_direct12_low_twist_desat_v1"
        checkpoint = 6100
        load_optimizer = False
        symmetry_coef = 0.75
        max_iterations = 1000
        save_interval = 25
