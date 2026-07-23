"""Training configuration for the new dog_urdf and measured RS01 motors."""

from legged_gym.envs.fanfan.fanfan_config import (
    FanfanRoughCfg,
    FanfanRoughCfgPPO,
)


# Real motor IDs by URDF joint.  0x11 and 0x41 were removed for repair during
# the 2026-07-20 identification, so those two deliberately use the aggregate
# 5%-95% range instead of invented per-motor values.
MOTOR_ID_BY_JOINT = {
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


def _band(value, fraction=0.04, floor=0.0):
    """Narrow identification uncertainty band around a tested motor value."""
    return [max(floor, value * (1.0 - fraction)), value * (1.0 + fraction)]


AGGREGATE_GAIN_RANGE = [0.8610138814, 1.0472019796]
AGGREGATE_TAU_RANGE_S = [0.0211223794, 0.0389344554]
AGGREGATE_FRICTION_RANGE_NM = [0.1327624374, 0.1928178369]
AGGREGATE_REVERSAL_RANGE_RAD = [0.0044620321, 0.0061166694]

GAIN_BY_JOINT = {
    "FR_hip_joint": AGGREGATE_GAIN_RANGE,
    "FR_thigh_joint": _band(1.0486707534),
    "FR_calf_joint": _band(0.8901892350),
    "FL_hip_joint": _band(0.9820176769),
    "FL_thigh_joint": _band(0.8766154442),
    "FL_calf_joint": _band(1.0140210102),
    "RL_hip_joint": _band(0.9033451910),
    "RL_thigh_joint": _band(0.9197254975),
    "RL_calf_joint": _band(0.8493146574),
    "RR_hip_joint": AGGREGATE_GAIN_RANGE,
    "RR_thigh_joint": _band(0.9472898466),
    "RR_calf_joint": _band(0.9921300052),
}

TAU_BY_JOINT_S = {
    "FR_hip_joint": AGGREGATE_TAU_RANGE_S,
    "FR_thigh_joint": _band(0.0332403879, 0.10),
    "FR_calf_joint": _band(0.0349193071, 0.10),
    "FL_hip_joint": _band(0.0315726532, 0.10),
    "FL_thigh_joint": _band(0.0252413929, 0.10),
    "FL_calf_joint": _band(0.0380176152, 0.10),
    "RL_hip_joint": _band(0.0377307586, 0.10),
    "RL_thigh_joint": _band(0.0277814454, 0.10),
    "RL_calf_joint": _band(0.0307921073, 0.10),
    "RR_hip_joint": AGGREGATE_TAU_RANGE_S,
    "RR_thigh_joint": _band(0.0192696944, 0.10),
    "RR_calf_joint": _band(0.0345461150, 0.10),
}

FRICTION_BY_JOINT_NM = {
    "FR_hip_joint": AGGREGATE_FRICTION_RANGE_NM,
    "FR_thigh_joint": _band(0.1456268675, 0.12),
    "FR_calf_joint": _band(0.1574945718, 0.12),
    "FL_hip_joint": _band(0.1227865306, 0.12),
    "FL_thigh_joint": _band(0.1470226303, 0.12),
    "FL_calf_joint": _band(0.1701760180, 0.12),
    "RL_hip_joint": _band(0.1765830058, 0.12),
    "RL_thigh_joint": _band(0.1545462338, 0.12),
    "RL_calf_joint": _band(0.2061008805, 0.12),
    "RR_hip_joint": AGGREGATE_FRICTION_RANGE_NM,
    "RR_thigh_joint": _band(0.1619854995, 0.12),
    "RR_calf_joint": _band(0.1449552123, 0.12),
}

REVERSAL_BY_JOINT_RAD = {
    "FR_hip_joint": AGGREGATE_REVERSAL_RANGE_RAD,
    "FR_thigh_joint": _band(0.0049512871, 0.10),
    "FR_calf_joint": _band(0.0052637389, 0.10),
    "FL_hip_joint": _band(0.0040617325, 0.10),
    "FL_thigh_joint": _band(0.0055072621, 0.10),
    "FL_calf_joint": _band(0.0056321717, 0.10),
    "RL_hip_joint": _band(0.0054141406, 0.10),
    "RL_thigh_joint": _band(0.0053630479, 0.10),
    "RL_calf_joint": _band(0.0065130766, 0.10),
    "RR_hip_joint": AGGREGATE_REVERSAL_RANGE_RAD,
    "RR_thigh_joint": _band(0.0055556673, 0.10),
    "RR_calf_joint": _band(0.0054570812, 0.10),
}


class DogRs01TrotCfg(FanfanRoughCfg):
    """Forward diagonal-pair walk with a measured, non-ideal actuator."""

    class env(FanfanRoughCfg.env):
        num_envs = 4096
        num_observations = 52
        num_actions = 12
        episode_length_s = 20

    class init_state(FanfanRoughCfg.init_state):
        pos = [0.0, 0.0, 0.316]
        rot = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.32987297,
            "FR_calf_joint": 1.31853104,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": -0.32987297,
            "FL_calf_joint": 1.31853104,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": -0.32987297,
            "RR_calf_joint": 1.31853104,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.32987297,
            "RL_calf_joint": 1.31853104,
        }

    class asset(FanfanRoughCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/../dog_urdf/urdf/"
            "dog_rs01.urdf"
        )
        name = "dog_rs01"
        foot_name = "foot"
        penalize_contacts_on = ["thigh_joint", "calf_joint"]
        terminate_after_contacts_on = ["Trunk"]
        collapse_fixed_joints = False
        flip_visual_attachments = False
        self_collisions = 1
        # The bench data cannot identify reflected rotor inertia. Do not
        # invent an armature value; the URDF already contains link/motor mass.
        armature = 0.0

    class control(FanfanRoughCfg.control):
        # Actual Isaac Gym/URDF DOF order. Export and real deployment must use
        # this exact contract; motor order is handled by the semantic mapper.
        policy_joint_order = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        control_type = "P"
        decimation = 4  # 0.005 * 4 = the measured 50 Hz controller
        # dog_rs01 is 11.73 kg versus Fanfan's 7.24 kg.  Its 17 Nm RS01 peak
        # limit does not allow a blind 1.62x gain scaling, but gains below the
        # lighter Fanfan make this robot visibly soft.  This set is at/above
        # Fanfan's 60/70/70 while the extra damping and target-rate guard keep
        # the measured 39--55 ms delayed actuator out of the pogo regime.
        stiffness = {"hip": 60.0, "thigh": 75.0, "calf": 80.0}
        damping = {"hip": 1.0, "thigh": 1.3, "calf": 1.4}
        # At or below the RS01 manual's 17 Nm peak. The separate sustained-
        # torque rewards still use the manual's 6 Nm continuous rating.
        torque_limits_by_joint = {"hip": 14.0, "thigh": 16.0, "calf": 17.0}

        # Go2 uses 0.25 rad. This heavier RS01 dog receives 0.22 rad so PPO
        # can create useful stance sweep while the measured target-rate,
        # acceleration, position and torque guards remain downstream.
        action_scale = 0.22
        rear_action_scale = 0.22
        hip_action_scale = 0.07
        filter_policy_actions = True
        policy_action_filter_alpha = 0.24
        policy_action_filter_alpha_range = [0.20, 0.28]
        policy_action_rate_limits = {"hip": 3.5, "thigh": 4.0, "calf": 4.5}
        policy_action_accel_limits = {
            "hip": 50.0, "thigh": 60.0, "calf": 70.0
        }

        # Conservative working envelope inside the mechanical URDF limits.
        target_position_limits_by_joint = {
            "hip": [-0.60, 0.60],
            "thigh": [-1.20, 0.45],
            "calf": [0.45, 1.75],
        }
        # The previous 2 rad/s calf target limit needed at least 75 ms just to
        # issue a 0.15 rad lift, before the measured 39--55 ms RS01 delay. It
        # forced tentative stepping. These remain far below the manual's
        # 315 rpm (about 33 rad/s) no-load speed and all torque caps stay active.
        final_target_rate_limits_initial = {
            "hip": 1.2, "thigh": 2.6, "calf": 3.2
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_initial = {
            "hip": 30.0, "thigh": 50.0, "calf": 65.0
        }
        final_target_accel_limits_final = final_target_accel_limits_initial
        final_target_limit_open_start_iteration = 0
        final_target_limit_open_end_iteration = 1
        rear_calf_target_rate_scale = 1.0
        project_straight_diagonal_actions = True
        # Treat each physical diagonal as one gait unit, but retain a bounded
        # 7% per-leg residual. The deterministic reference compensates the
        # measured position gains; this small residual lets PPO correct the
        # remaining front/rear load and tau differences. An A/B test with the
        # same checkpoint raised exact paired flight from 8.79% to 9.45%
        # without changing torque saturation (16.4%).
        straight_diagonal_projection_blend = 0.93

        # Deterministic CPG base controller. Two coupled limit cycles drive
        # FL+RR and FR+RL at exactly pi phase separation. The trajectory is
        # built in foot space and converted with the dog_urdf analytic IK;
        # PPO learns only a bounded residual on top.
        use_rs01_diagonal_cpg = True
        cpg_radial_rate = 30.0
        cpg_coupling_gain = 18.0
        # Gym load measurements showed a rear-biased support line. Moving the
        # complete footprint 15 mm rearward centers the trunk mass over each
        # diagonal without changing the supplied zero-command stand pose.
        cpg_nominal_foot_x_m = -0.015
        cpg_nominal_foot_z_m = -0.300
        # The 15 mm footprint shift restores useful propulsion at the smaller
        # 1.4 gain. At 0.15 m/s command the support-gated zero-residual CPG
        # realizes about 0.10--0.12 m/s without the unequal, high-impact step
        # of gain 2.0; PPO learns only the remaining speed residual.
        cpg_stride_gain = 1.4
        cpg_max_stride_m = 0.085
        cpg_swing_clearance_m = 0.030
        # Do not hide an incorrect phase schedule with four independent lift
        # amplitudes.  A physical diagonal receives one identical trajectory;
        # measured motor differences remain in the actuator model downstream.
        cpg_swing_clearance_scale_by_leg = {
            "FL": 1.0, "FR": 1.0, "RL": 1.0, "RR": 1.0,
        }
        cpg_full_clearance_speed_m_s = 0.12
        cpg_lift_fraction = 0.18
        cpg_lower_start_fraction = 0.62
        # Disabled by default: force-impulse feedback did not improve paired
        # contact in Gym and made the nominal CPG unnecessarily asymmetric.
        cpg_force_balance_gain_m_per_weight = 0.0
        cpg_force_balance_max_m = 0.010
        cpg_force_balance_time_constant_s = 0.12
        cpg_front_rear_load_bias_m = 0.0
        # One millimetre equalizes the measured FL+RR / FR+RL load totals and
        # their exact paired-air fractions; larger values simply reverse the
        # imbalance and increase flight events.
        cpg_diagonal_load_bias_m = 0.001
        # Contact-following term selected in Gym with the full measured RS01
        # actuator chain. Negative means extend a stance leg while the trunk
        # rises; the 6 mm cap prevents it from becoming a height controller.
        cpg_vertical_velocity_damping_s = -0.03
        cpg_vertical_velocity_damping_max_m = 0.006

        use_continuous_gait_scaling = True
        gait_equivalent_speed_weights = [1.0, 1.5, 0.18]
        gait_speed_knots = [0.0, 0.02, 0.05, 0.12, 0.22, 0.35]
        # Fanfan-timed lift envelope. At the 0.15 m/s viewer command this is
        # 0.191 rad, while 0.23 rad is reached only at the highest command.
        # A measured sweep showed that 0.17/0.19 rad remained tentative;
        # 0.23 rad produced decisive paired lift without extra saturation.
        gait_calf_amplitude_knots = [0.0, 0.0, 0.115, 0.183, 0.210, 0.230]
        # 0.25 * 0.38 = 95 ms. This covers the measured 38.6--60 ms pure delay
        # plus most of the 19--38 ms motor time constant. The contact handoff
        # gate is shifted by the same amount, so the old support pair cannot
        # receive an early swing command while the new pair is still landing.
        gait_target_phase_lead = 0.25
        gait_transition_ramp_s = 0.20
        compensate_identified_position_gain_in_gait = True
        use_fast_swing_profile = True
        # Fast smoothstep rise/fall has zero endpoint velocity.  A sine arch
        # changes target velocity abruptly at lift-off/touch-down and becomes
        # an impact source once the heavy robot uses realistic high Kp.
        fast_swing_profile_shape = "plateau"
        swing_lift_fraction = 0.12
        # Hold full clearance until 65% of swing, then descend quickly. The
        # following 76 ms double-support window lets the landed diagonal take
        # load before the previous support diagonal is released.
        swing_lower_start_fraction = 0.65
        enforce_swing_calf_reference = True
        swing_calf_reference_scale = 1.0
        # A stance leg may still make small load-bearing corrections, but it
        # may not be commanded into the flexion direction used for swing.
        # Together with the strong diagonal projection this prevents a policy
        # from inventing a left/right pace or a front/rear bound.
        enforce_stance_leg_extension = True
        stance_guard_preserve_gait_reference = True
        # Calf extension keeps the support pair out of swing. Paired thigh
        # sweep must stay free because it creates forward ground reaction.
        enforce_stance_thigh_reference = False
        preserve_forward_gait = False
        stance_calf_extension = 0.0
        stance_thigh_extension = 0.0
        # A scheduled pair may leave the ground only after the opposite
        # physical diagonal is actually supporting the body. This contact
        # interlock prevents rear-pair/front-pair bounds after landing impact.
        # With the final 0.70 duty factor this smooth secondary interlock no
        # longer deadlocks the gait: continuous tests cut physical flight by
        # more than half while retaining useful forward motion.
        gate_swing_on_opposite_diagonal_support = True
        # Use a smooth release instead of the old binary 0.85 threshold, which
        # deadlocked when a landed pair was touching but not yet fully loaded.
        opposite_diagonal_support_floor_score = 0.30
        opposite_diagonal_support_full_score = 0.75
        hold_blocked_swing_thigh = True
        use_contact_aware_phase_transfer = True
        phase_transfer_min_pair_weight_fraction = 0.42
        phase_transfer_min_load_fraction = 0.58
        # The target release is held until both toes of the arriving diagonal
        # carry load. Four 20 ms samples cover the 38.6--60 ms pure delay while
        # retaining a measured 2.3 Hz dynamic cycle instead of a slow walk.
        phase_transfer_max_wait_steps = 4
        # The contact-aware CPG now performs the handoff itself. The former
        # calf-extension clamp distorted the analytic foot-space trajectory.
        use_active_diagonal_load_transfer = False
        active_transfer_target_weight_fraction = 0.52
        active_transfer_max_calf_extension_rad = 0.025

        use_real_actuator_model = True
        actuator_time_constant_ranges = {
            "hip": AGGREGATE_TAU_RANGE_S,
            "thigh": AGGREGATE_TAU_RANGE_S,
            "calf": AGGREGATE_TAU_RANGE_S,
        }
        actuator_time_constant_ranges_by_joint = TAU_BY_JOINT_S
        actuator_position_gain_ranges_by_joint = GAIN_BY_JOINT
        # This is the identified closed-loop pure delay, not a claim that the
        # transport layer alone takes this long.
        command_delay_range_s = [0.0386, 0.0550]
        command_delay_slow_probability = 0.03
        command_delay_slow_range_s = [0.0550, 0.0600]
        command_delay_max_s = 0.0600
        coulomb_friction_velocity_smoothing_rad_s = 0.05
        training_torque_limit_ranges = {
            "hip": [12.0, 14.0],
            "thigh": [15.0, 16.0],
            "calf": [16.0, 17.0],
        }

    class commands(FanfanRoughCfg.commands):
        heading_command = False
        observe_heading_error = True
        resampling_time = 4.0
        stand_probability = 0.10
        pure_sagittal_probability = 0.90
        pure_yaw_probability = 0.0
        pure_lateral_probability = 0.0
        hard_transition_probability = 0.0

        class ranges(FanfanRoughCfg.commands.ranges):
            # Zero is sampled explicitly by ``stand_probability``.  Moving
            # commands start above the dead zone so walking gets a clear
            # learning signal instead of mostly near-zero stand samples.
            lin_vel_x = [0.15, 0.30]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class terrain(FanfanRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        static_friction = 0.9
        dynamic_friction = 0.85

    class domain_rand(FanfanRoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.65, 1.15]
        # URDF total is 11.73 kg.  Cover battery/harness/payload variation on
        # the trunk while keeping the nominal CAD inertia as the center case.
        randomize_base_mass = True
        added_mass_range = [-0.30, 0.80]
        randomize_base_com = True
        base_com_x_range = [-0.010, 0.010]
        base_com_y_range = [-0.008, 0.008]
        base_com_z_range = [-0.005, 0.005]
        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]
        rear_calf_strength_range = [0.95, 1.05]
        pair_diagonal_motor_strength = False
        kp_multiplier_range = [0.95, 1.05]
        kd_multiplier_range = [0.90, 1.10]
        joint_zero_offset_ranges = {
            "hip": [-0.008, 0.008],
            "thigh": [-0.010, 0.010],
            "calf": [-0.012, 0.012],
        }
        # Kept for compatibility; the exact per-motor reversal ranges below
        # take precedence in the actuator implementation.
        joint_backlash_ranges = {
            "hip": AGGREGATE_REVERSAL_RANGE_RAD,
            "thigh": AGGREGATE_REVERSAL_RANGE_RAD,
            "calf": AGGREGATE_REVERSAL_RANGE_RAD,
        }
        effective_reversal_gap_ranges_by_joint = REVERSAL_BY_JOINT_RAD
        coulomb_friction_ranges_by_joint = FRICTION_BY_JOINT_NM
        randomize_joint_friction_damping = False

        gait_calf_amplitude_max_range = [0.230, 0.230]
        # The 11.73 kg trunk and measured RS01 delay require a real load
        # handoff window. At 0.70 duty the new diagonal has 76 ms to accept
        # load before the old pair leaves, while swing remains a decisive
        # 114 ms rather than a slow static step.
        gait_stance_ratio_range = [0.70, 0.70]
        gait_low_speed_period_range = [0.38, 0.38]
        gait_high_speed_period_range = [0.38, 0.38]
        gait_backward_scale_range = [1.0, 1.0]
        randomize_gait_phase_on_reset = False
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.12

    class noise(FanfanRoughCfg.noise):
        add_noise = True
        noise_level = 0.6
        use_real_observation_model = False

    class rewards(FanfanRoughCfg.rewards):
        gym_settled_base_height_m = 0.309475
        mujoco_settled_base_height_m = 0.307120
        base_height_target = 0.309475
        min_base_height = 0.245
        min_base_height_soft = 0.275
        soft_dof_pos_limit = 0.92
        # 2.63 Hz dynamic diagonal gait: pairs exchange every 0.19 s. A 0.70
        # duty factor gives the real actuator 76 ms of double support and a
        # 114 ms swing; no ballistic interval exists in the CPG schedule.
        gait_period = 0.38
        gait_stance_ratio = 0.70
        # A small Fanfan-shaped seed starts diagonal unloading without trying
        # to make an untrained (model_0) actor walk open-loop. PPO learns the
        # actual toe lift and stride through the pair-minimum lift/load reward.
        gait_thigh_amplitude = 0.055
        gait_swing_thigh_lift_amplitude = 0.070
        gait_calf_amplitude = -0.230
        gait_lateral_hip_amplitude = 0.0
        swing_height_target = 0.030
        diagonal_pair_lift_start_height = 0.015
        diagonal_pair_lift_target_height = 0.030
        swing_height_sigma = 0.00045
        swing_clearance_minimum = 0.025
        phase_foot_velocity_sigma = 0.04
        foot_contact_force_threshold = 1.0
        transition_nominal_weight_n = 115.1
        transition_new_pair_weight_fraction = 0.45
        transition_total_weight_fraction = 0.70
        diagonal_support_min_foot_force_n = 5.0
        diagonal_support_min_pair_weight_fraction = 0.30
        nominal_foot_x_by_leg_m = {
            "FL": 0.216, "FR": 0.216, "RL": -0.216, "RR": -0.216
        }
        diagonal_stride_position_sigma_m = 0.018
        diagonal_stride_height_sigma_m = 0.012
        diagonal_stride_velocity_sigma_m_s = 0.30
        diagonal_vertical_velocity_sigma_m_s = 0.25
        diagonal_load_transfer_sigma = 0.12
        # Permit the designed 76 ms load-transfer overlap, then penalize a
        # robot that remains planted instead of beginning the next swing.
        max_all_feet_contact_time_s = 0.08
        all_feet_contact_penalty_saturation_s = 0.08
        max_foot_contact_time_s = 0.42
        foot_contact_time_penalty_saturation_s = 0.20
        gate_phase_rewards_with_command = True
        gate_stand_posture_with_command = True
        phase_command_gate_sigma = 0.0004
        # Match standard Go2 training: shaping penalties subtract from useful
        # locomotion reward, but early exploration is clipped at zero. The
        # termination penalty is added afterward and remains a hard negative.
        only_positive_rewards = True
        tracking_sigma = 0.04
        lateral_tracking_sigma = 0.001
        longitudinal_tracking_sigma = 0.003
        max_contact_force = 65.0

        # Hard clipping at or below the manual's 17 Nm peak is always active. The
        # *reward* guard ramps after locomotion is acquired so its large early
        # gradient cannot make standing the only profitable policy.
        torque_curriculum = True
        torque_near_limit_ratio = 0.80
        peak_torque_soft_ratio = 0.90
        sustained_torque_ratio = 0.60
        continuous_torque_limits_by_joint = {
            "hip": 6.0, "thigh": 6.0, "calf": 6.0
        }
        torque_ema_alpha = 0.99
        torque_curriculum_stage2_iteration = 100
        torque_curriculum_stage3_iteration = 500
        torque_curriculum_stage4_iteration = 1000
        torque_curriculum_blend_iterations = 100
        torque_curriculum_stage2 = {
            "torque_clip": -0.60,
            "torque_near_limit": -0.20,
            "peak_torque": -0.25,
            "sustained_torque": -0.35,
        }
        torque_curriculum_stage3 = {
            "torque_clip": -1.20,
            "torque_near_limit": -0.40,
            "peak_torque": -0.50,
            "sustained_torque": -0.70,
        }
        torque_curriculum_stage4 = {
            "torque_clip": -2.00,
            "torque_near_limit": -0.80,
            "peak_torque": -1.00,
            "sustained_torque": -1.40,
        }
        pd_pos_err_soft_limit = {
            "hip": 0.08, "thigh": 0.10, "calf": 0.12
        }
        # Training torque is still hard-clipped to 14/16/17 Nm and all
        # continuous/peak torque penalties remain active.  Do not terminate
        # from the larger *unclipped* PD request: that previously reset most
        # exploratory episodes after roughly one second and prevented PPO
        # from learning forward motion.
        enable_actuator_safety_termination = False
        # The delivered torque remains hard-clipped to 14/16/17 Nm. These
        # thresholds only decide whether a simulated exploratory episode is
        # reset; allowing a short clipped transient prevents PPO from being
        # reset every ~1 s before it observes a complete gait cycle.
        actuator_safety_grace_steps = 50
        raw_torque_termination_ratio = 2.0
        raw_torque_termination_steps = 5
        torque_saturation_window_steps = 25
        torque_saturation_window_ratio = 0.25
        calf_error_termination_rad = 0.24
        calf_error_termination_steps = 3
        calf_angle_limits = [0.35, 1.82]
        terminate_on_calf_angle = True
        terminate_rear_sit_pitch = -0.45
        # During locomotion, two or more airborne feet are legal only when
        # they are exactly FL+RR or FR+RL. A rear pair, front pair, same-side
        # pair, triple or flight pattern is rejected on its first sample.
        enable_non_diagonal_swing_termination = True
        non_diagonal_swing_grace_steps = 10
        non_diagonal_swing_termination_steps = 1
        # Play/test is always one-frame strict. Training observes full cycles
        # first, then progressively adopts that deployment contract.
        non_diagonal_termination_curriculum = [
            {"until_iteration": 300, "steps": 4},
            {"until_iteration": 700, "steps": 3},
            {"until_iteration": 1100, "steps": 2},
            {"until_iteration": 1.0e12, "steps": 1},
        ]
        # Complete flight is never part of this walking gait. Reject it on
        # the first 50 Hz sample instead of sharing the 60 ms chatter margin
        # used for other invalid multi-foot patterns.
        enable_flight_termination = True
        flight_termination_grace_steps = 10

        class scales(FanfanRoughCfg.rewards.scales):
            termination = -80.0
            stand_height = 1.0
            stand_posture = 0.5
            # Go2 treats commanded body velocity as the task and foot timing
            # as shaping. Signed progress must dominate every phase reward.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # Dense velocity terms make standing under a non-zero command
            # strictly worse than making forward progress.  The narrow
            # exponential tracker alone gave too little PPO gradient while
            # the robot was initially stationary.
            # The previous 30:-8 balance made 0.35 m/s more profitable than a
            # requested 0.15 m/s, encouraging impacts and apparent motion that
            # was mostly reset-to-origin hopping. Prefer commanded velocity.
            command_velocity_progress = 8.0
            normalized_command_tracking = 8.0
            absolute_longitudinal_tracking_error = -20.0
            heading_tracking = 0.0
            lateral_velocity = -2.0
            backward_velocity = -12.0
            diagonal_gait = 3.0
            exact_diagonal_swing = 14.0
            scheduled_diagonal_pair_lift = 16.0
            touchdown_pair_support = 8.0
            touchdown_pair_support_shortfall = -12.0
            # Require a complete handoff, not merely touchdown while both
            # diagonals continue sharing load indefinitely.
            diagonal_load_transfer = 14.0
            diagonal_load_transfer_error = -12.0
            # Positive credit exists only while FL+RR or FR+RL really carries
            # load. The shortfall signal begins before binary contact is lost.
            diagonal_support = 10.0
            diagonal_support_shortfall = -20.0
            non_diagonal_swing = -30.0
            single_foot_swing = -12.0
            # Linear contact-pattern error closes the loophole where both
            # diagonal pairs jump together and still look "synchronized".
            phase_contact_mismatch = -4.0
            # A continuous load-transfer signal acts before a toe actually
            # leaves the floor.  This lets PPO learn diagonal unloading from
            # the small reference seed instead of requiring a lucky jump.
            phase_foot_force_tracking = -4.0
            # This exponential score was still positive for stationary feet.
            # Signed body progress above is the unambiguous forward objective.
            phase_foot_velocity_tracking = 0.0
            diagonal_contact_sync_all = -16.0
            diagonal_foot_height_sync_all = -10.0
            diagonal_stride_sync_all = 12.0
            diagonal_stride_sync_shortfall = -8.0
            swing_height = 0.4
            swing_clearance_shortfall = -4.0
            swing_contact = -5.0
            # Standing/shuffling remains costly, but complete flight must be
            # strictly worse than escaping both contact-duration terms.  This
            # ordering removes the reward loophole that produced pogo hops.
            all_feet_contact = -10.0
            excessive_foot_contact_time = -0.75
            flight = -40.0
            lin_vel_z = -12.0
            ang_vel_xy = -0.8
            yaw_rate = -0.5
            hip_velocity = -0.002
            hip_symmetry = -0.3
            diagonal_joint_sync = -1.0
            action_magnitude = -0.003
            orientation = -4.0
            base_height = -30.0
            low_base_height = -15.0
            rear_sit = 0.0
            front_feet_contact = 0.0
            rear_calf_fold = 0.0
            rear_load_bias = 0.0
            rear_leg_posture = 0.0
            torques = -3.0e-6
            torque_clip = -0.25
            torque_near_limit = -0.08
            peak_torque = -0.12
            sustained_torque = -0.15
            sustained_torque_max = -0.20
            mechanical_power = -0.001
            pd_position_error_over_limit = -0.8
            motor_target_tracking_error = -0.08
            final_target_velocity = -0.08
            final_target_acceleration = -0.03
            dof_vel = -1.0e-4
            dof_acc = -4.0e-7
            action_rate = -0.02
            policy_action_rate = -0.02
            collision = -3.0
            dof_pos_limits = -4.0
            calf_angle_limits = -6.0
            feet_contact_forces = -0.004
            feet_air_time = 0.0

    class sim(FanfanRoughCfg.sim):
        dt = 0.005
        substeps = 2

        class physx(FanfanRoughCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 4


class DogRs01TrotCfgPPO(FanfanRoughCfgPPO):
    class policy(FanfanRoughCfgPPO.policy):
        # Exact diagonal projection averages each pair's exploration sample.
        # 0.05 therefore produced only milliradian residuals and never crossed
        # the physical toe-liftoff threshold. This remains far below the old
        # large-action initialization that caused hopping.
        init_noise_std = 0.18

    class algorithm(FanfanRoughCfgPPO.algorithm):
        entropy_coef = 0.005
        # The adaptive schedule reached its 1e-2 hard maximum by iteration
        # 43 because early KL was small, destabilizing the policy before a
        # complete trot was learned.  A fixed rate is predictable here.
        schedule = "fixed"
        learning_rate = 1.0e-4
        desired_kl = 0.01

    class runner(FanfanRoughCfgPPO.runner):
        experiment_name = "rough_dog_rs01_trot"
        run_name = "rs01_cpg_residual_balance"
        # If an explicit resume is requested, do not inherit stale Adam
        # moments. The intended CPG-residual run starts without --resume so
        # its near-zero actor initially leaves the validated CPG unchanged.
        load_optimizer = False
        # This is a residual policy on top of a calibrated phase reference.
        # Begin with an almost-zero mean instead of random large joint targets.
        actor_output_init_scale = 1.0e-3
        max_iterations = 3000
        save_interval = 25
        resume = False
