"""Progressive Sim2Sim adaptation tasks derived from the selected model_730."""

from .rs01_go2_kp40_config import (
    Rs01Go2Kp40Cfg,
    Rs01Go2Kp40CfgPPO,
)


class Rs01Go2Sim2SimAdaptCfg(Rs01Go2Kp40Cfg):
    """First adapt model_730 to executable calf targets at nominal dynamics."""

    class control(Rs01Go2Kp40Cfg.control):
        # Keep hip/thigh authority and reduce only the joint type responsible
        # for the 46--49 Nm MuJoCo P95. The actor remains 12 direct outputs.
        action_scale_by_joint = {
            "hip": 0.18,
            "thigh": 0.18,
            "calf": 0.14,
        }

    class rs01_actuator(Rs01Go2Kp40Cfg.rs01_actuator):
        # The original 0.18-rad calf channel can reverse too quickly relative
        # to the measured 40--56 ms response. These limits remain implementable
        # by the real 50 Hz controller and are mirrored by Sim2Sim deployment.
        target_rate_limit_rad_s = {
            "hip": 2.0,
            "thigh": 2.6,
            "calf": 2.6,
        }
        target_acceleration_limit_rad_s2 = {
            "hip": 60.0,
            "thigh": 78.0,
            "calf": 72.0,
        }

    class commands(Rs01Go2Kp40Cfg.commands):
        playback_speed_mps = 0.23

        class ranges(Rs01Go2Kp40Cfg.commands.ranges):
            lin_vel_x = [0.21, 0.25]

    class domain_rand(Rs01Go2Kp40Cfg.domain_rand):
        # Stage A establishes a nominal gait under the new hard target
        # contract. Robustness is introduced only by the following task.
        randomize_friction = False
        randomize_base_mass = False
        randomize_rs01_actuator = False
        push_robots = False


class Rs01Go2Sim2SimAdaptCfgPPO(Rs01Go2Kp40CfgPPO):
    class policy(Rs01Go2Kp40CfgPPO.policy):
        init_noise_std = 0.08

    class algorithm(Rs01Go2Kp40CfgPPO.algorithm):
        learning_rate = 5.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Kp40CfgPPO.runner):
        run_name = ""
        # Keep the established experiment root so resume can resolve the
        # selected model_730 without copying or modifying the checkpoint.
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.08
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        reference_policy_coef = 0.10
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 1.0
        # This environment executes clamp(action, -1, 1), not tanh(action).
        reference_action_transform = "clip"
        reference_action_clip = 1.0


class Rs01Go2Sim2SimCalfRepairCfg(Rs01Go2Sim2SimAdaptCfg):
    """Repair the measured calf swing-speed saturation from model_785."""

    class control(Rs01Go2Sim2SimAdaptCfg.control):
        # A fixed-checkpoint sweep at 0.23 m/s showed that Kd=0.55 preserves
        # more of the selected gait's 30-s path/contact symmetry than Kd=0.50
        # while reducing whole-body raw P95 from 32.90 to 18.00 N.m. Kd=0.35
        # destroyed contact timing; Kd=0.7 left raw P95 above 23 N.m.
        damping = {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 0.55,
        }

    class rewards(Rs01Go2Sim2SimAdaptCfg.rewards):
        # A normal 0.14-rad swing over this task's 0.21-s swing window needs
        # only a few rad/s. Eight rad/s leaves ample gait headroom while
        # exposing the measured 32.99-rad/s swing snap directly to PPO.
        calf_velocity_soft_limit_rad_s = 8.0
        # Penalize the actor mean before the environment's hard +/-1 clamp so
        # PPO can move channels back into an executable region.
        action_saturation_soft_limit = 0.90

        class scales(Rs01Go2Sim2SimAdaptCfg.rewards.scales):
            calf_velocity_excess = -0.15
            action_saturation = -0.50
            raw_torque_over_peak = -1.0
            motor_saturation = -1.0


class Rs01Go2Sim2SimCalfRepairCfgPPO(Rs01Go2Sim2SimAdaptCfgPPO):
    class algorithm(Rs01Go2Sim2SimAdaptCfgPPO.algorithm):
        learning_rate = 3.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Sim2SimAdaptCfgPPO.runner):
        save_interval = 5
        action_std_value = 0.08
        freeze_action_std = True
        reference_policy_coef = 0.10
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 1.0


class Rs01Go2Sim2SimKd050Cfg(Rs01Go2Sim2SimCalfRepairCfg):
    """Second nominal adaptation after selecting the straight model_815."""

    class control(Rs01Go2Sim2SimCalfRepairCfg.control):
        # Fixed model_815 A/B at 0.23 m/s:
        # Kd 0.55 -> 15.06% calf velocity-ceiling use, 18.22 N.m raw P95.
        # Kd 0.50 ->  7.86% calf velocity-ceiling use, 16.19 N.m raw P95,
        # while retaining 0.217 m/s, 65.18% exact contact and zero flight.
        # Kd 0.45 lost the accepted gait, so it is deliberately not used.
        damping = {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 0.50,
        }


class Rs01Go2Sim2SimKd050CfgPPO(Rs01Go2Sim2SimCalfRepairCfgPPO):
    class policy(Rs01Go2Sim2SimCalfRepairCfgPPO.policy):
        init_noise_std = 0.06

    class algorithm(Rs01Go2Sim2SimCalfRepairCfgPPO.algorithm):
        learning_rate = 2.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Sim2SimCalfRepairCfgPPO.runner):
        save_interval = 5
        action_std_value = 0.06
        freeze_action_std = True
        # Keep model_815's path solution while allowing the smaller damping
        # change to recover the remaining 0.013 m/s command shortfall.
        reference_policy_coef = 0.12
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 1.0


class Rs01Go2Sim2SimRobustCfg(Rs01Go2Sim2SimKd050Cfg):
    """Narrow, measured RS01 and contact randomization after Stage A passes."""

    class domain_rand(Rs01Go2Sim2SimKd050Cfg.domain_rand):
        randomize_friction = True
        friction_range = [0.85, 1.15]
        randomize_base_mass = True
        added_mass_range = [-0.30, 0.30]
        randomize_rs01_actuator = True
        rs01_response_gain_scale_range = [0.95, 1.05]
        rs01_time_constant_scale_range = [0.90, 1.10]
        rs01_friction_scale_range = [0.90, 1.10]
        # A shared per-environment offset preserves the measured differences
        # between individual motors while covering one 5 ms physics step.
        rs01_delay_step_offset_range = [-1, 1]
        push_robots = False


class Rs01Go2Sim2SimRobustCfgPPO(Rs01Go2Sim2SimKd050CfgPPO):
    class policy(Rs01Go2Sim2SimKd050CfgPPO.policy):
        init_noise_std = 0.07

    class algorithm(Rs01Go2Sim2SimKd050CfgPPO.algorithm):
        learning_rate = 3.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Sim2SimKd050CfgPPO.runner):
        save_interval = 5
        action_std_value = 0.07
        reference_policy_coef = 0.05
        reference_action_deadband = 0.10
        reference_action_hinge_coef = 0.5


class Rs01Go2MatchedTransferCfg(Rs01Go2Sim2SimRobustCfg):
    """Short transfer whose hard acceptance target is the matched MuJoCo scene."""

    class init_state(Rs01Go2Sim2SimRobustCfg.init_state):
        # The matched model_870 accumulates yaw rather than falling.  A wider
        # two-sided reset exposes heading recovery without changing the
        # standing pose or injecting an initial angular velocity.
        reset_heading_noise_rad = 0.12

    class domain_rand(Rs01Go2Sim2SimRobustCfg.domain_rand):
        # Keep the measured values as the centre. Independent joint samples
        # prevent PPO from assuming that both sides have identical response.
        # Ranges stay deliberately narrow for this short migration.
        rs01_response_gain_scale_range = [0.97, 1.03]
        rs01_time_constant_scale_range = [0.95, 1.05]
        rs01_friction_scale_range = [0.95, 1.05]
        rs01_delay_step_offset_range = [-1, 1]
        rs01_independent_motor_randomization = True
        rs01_independent_delay_randomization = True

    class rewards(Rs01Go2Sim2SimRobustCfg.rewards):
        class scales(Rs01Go2Sim2SimRobustCfg.rewards.scales):
            # This remains a restoring-rate objective, not a blanket
            # suppression of yaw. The increase is intentionally modest so
            # contact timing and forward motion keep their existing solution.
            heading_recovery = -0.40


class Rs01Go2MatchedTransferCfgPPO(Rs01Go2Sim2SimRobustCfgPPO):
    class policy(Rs01Go2Sim2SimRobustCfgPPO.policy):
        init_noise_std = 0.05

    class algorithm(Rs01Go2Sim2SimRobustCfgPPO.algorithm):
        learning_rate = 2.0e-5
        schedule = "fixed"

    class runner(Rs01Go2Sim2SimRobustCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.05
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        # Preserve model_870's usable PhysX gait while allowing small
        # asymmetric corrections. Dense checkpoints are selected by the
        # matched MuJoCo acceptance environment, never by final iteration.
        reference_policy_coef = 0.12
        reference_action_deadband = 0.06
        reference_action_hinge_coef = 1.0


class Rs01Go2Heading52Cfg(Rs01Go2MatchedTransferCfg):
    """Global heading recovery after closing the Sim2Sim runtime contract."""

    class env(Rs01Go2MatchedTransferCfg.env):
        # Replace the saturated scalar heading error with sin/cos.
        num_observations = 52

    class commands(Rs01Go2MatchedTransferCfg.commands):
        observe_straight_heading_error = False
        observe_straight_heading_sin_cos = True

    class init_state(Rs01Go2MatchedTransferCfg.init_state):
        # First recovery stage: wider than the old local +/-0.12 rad range,
        # but still conservative enough to preserve the accepted gait.
        reset_heading_noise_rad = 0.30
        reset_yaw_rate_noise_rad_s = 0.20


class Rs01Go2Heading52CfgPPO(Rs01Go2MatchedTransferCfgPPO):
    class policy(Rs01Go2MatchedTransferCfgPPO.policy):
        init_noise_std = 0.05

    class algorithm(Rs01Go2MatchedTransferCfgPPO.algorithm):
        learning_rate = 2.0e-5
        schedule = "fixed"

    class runner(Rs01Go2MatchedTransferCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight_phase_load"
        save_interval = 5
        action_std_value = 0.05
        freeze_action_std = True
        load_optimizer = False
        adapt_observation_input = True
        # The first 50 columns are unchanged. The old final scalar represented
        # approximately 2*heading_error near zero; map its first-layer weight
        # to 2*sin(error), and initialize cos(error) to zero.
        observation_column_migration = {
            "source_width": 51,
            "destination_width": 52,
            "copy_prefix": 50,
            "column_mappings": [
                {
                    "source": 50,
                    "destination": 50,
                    "scale": 2.0,
                },
            ],
        }
        # Keep the nominal gait prior, but allow more recovery authority than
        # the local-heading transfer used before runtime parity was fixed.
        reference_policy_coef = 0.08
        reference_action_deadband = 0.08
        reference_action_hinge_coef = 0.75
