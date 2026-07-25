"""Stage-2 RS01 actuator-feasibility A/B tasks derived from model_9200."""

from .dog_torque_straight_v5_config import (
    DogRs01TorqueStraightV5Cfg,
    DogRs01TorqueStraightV5CfgPPO,
)


class DogRs01Stage2ActuatorACfg(DogRs01TorqueStraightV5Cfg):
    """A: preserve support PD and make sagittal targets executable."""

    class control(DogRs01TorqueStraightV5Cfg.control):
        # Hip torque is already healthy (P95 about 6 Nm). Reduce only the
        # thigh/calf authority responsible for 35--62 Nm raw requests.
        hip_action_scale = 0.16
        thigh_action_scale = 0.21
        calf_action_scale = 0.20
        rear_action_scale = 0.22

        # Preserve the 60/70/70 support stiffness. Limit target reversals
        # before the identified RS01 delay/tau and PD stages amplify error.
        final_target_rate_limits_initial = {
            "hip": 2.0, "thigh": 2.5, "calf": 3.05,
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_initial = {
            "hip": 60.0, "thigh": 72.0, "calf": 88.0,
        }
        final_target_accel_limits_final = final_target_accel_limits_initial

        # Motor heat does not disappear when an episode is reset.
        preserve_thermal_state_on_reset = False
        preserve_thermal_state_in_test = True
        thermal_reset_ratio_range = [0.75, 0.95]
        motor_temperature_reset_range_c = [30.0, 45.0]

    class commands(DogRs01TorqueStraightV5Cfg.commands):
        stand_probability = 0.05

        class ranges(DogRs01TorqueStraightV5Cfg.commands.ranges):
            lin_vel_x = [0.08, 0.10]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(DogRs01TorqueStraightV5Cfg.rewards):
        # Direct-12 has no hidden swing trajectory. This is a scuffing floor,
        # not an instruction to raise the feet higher.
        swing_clearance_minimum = 0.016
        torque_near_limit_ratio = 0.85
        peak_torque_soft_ratio = 1.0

        # One progress objective and orthogonal gait, actuator, and body terms.
        # This replaces V5's 51-term reward farm for the new Stage-2 task only.
        class scales:
            termination = -12.0
            stand_height = 2.0
            stand_posture = 0.5

            normalized_command_tracking = 4.0
            commanded_smooth_straight_progress = 6.0

            exact_diagonal_swing = 10.0
            diagonal_load_transfer = 5.0
            phase_contact_mismatch = -6.0
            non_diagonal_swing = -20.0
            flight = -30.0
            all_feet_contact = -0.5
            swing_clearance_shortfall = -2.0

            straight_path_recovery_velocity = -2.0
            straight_heading_recovery_rate = -2.0
            body_angular_velocity = -1.5
            orientation = -2.0

            final_target_acceleration = -0.10
            torque_clip = -4.0
            raw_torque_rate = -0.60
            motor_thermal_overload = -2.0
            mechanical_power = -0.0015


class DogRs01Stage2ActuatorACfgPPO(DogRs01TorqueStraightV5CfgPPO):
    class algorithm(DogRs01TorqueStraightV5CfgPPO.algorithm):
        learning_rate = 7.5e-6
        entropy_coef = 2.0e-5
        clip_param = 0.08
        max_grad_norm = 0.35
        schedule = "fixed"

    class runner(DogRs01TorqueStraightV5CfgPPO.runner):
        run_name = "rs01_stage2_actuator_a_target_limits"
        load_run = "Jul24_15-09-38_rs01_direct12_structural_rebuild_v1"
        checkpoint = 9200
        load_optimizer = False
        adapt_observation_input = False
        reference_policy_coef = 0.25
        reference_action_deadband = 0.04
        reference_action_hinge_coef = 2.0
        max_iterations = 1200
        save_interval = 25


class DogRs01Stage2ActuatorBCfg(DogRs01Stage2ActuatorACfg):
    """B: A's target limits plus a deliberately modest sagittal PD change."""

    class control(DogRs01Stage2ActuatorACfg.control):
        stiffness = {"hip": 60.0, "thigh": 68.0, "calf": 68.0}
        damping = {"hip": 1.2, "thigh": 1.5, "calf": 1.5}


class DogRs01Stage2ActuatorBCfgPPO(DogRs01Stage2ActuatorACfgPPO):
    class runner(DogRs01Stage2ActuatorACfgPPO.runner):
        run_name = "rs01_stage2_actuator_b_modest_pd"
