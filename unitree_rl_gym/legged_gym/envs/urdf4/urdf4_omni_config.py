"""URDF4.0 quadruped, RS01 actuators: flat-ground omnidirectional task, version 1.

Independent of the dog_rs01 task lineage.  It reuses only the identified RS01
actuator primitives (rate/acceleration-limited position target, closed-loop
delay, first-order response, PD, torque clip, Coulomb friction).

Joint sign conventions differ from the old machine: in this URDF the right-leg
thigh/calf axes are -Y and the left-leg axes +Y, so the default pose is
mirrored between sides (right calf negative, left calf positive).
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# Foot directly below the thigh joint, 0.27 m below it (trunk origin 0.28 m
# above flat ground with the 20 mm foot spheres).  Solved from the URDF
# kinematics; knee interior angle ~95 deg, 0.64 rad short of the extension stop.
_THIGH_RAD = 0.41428
_CALF_RAD = 1.28031


class Urdf4OmniCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        # One actor frame (48): gyro 3, gravity 3, command 3, joint position
        # error 12, joint velocity 12, previous action 12, gait clock sin/cos 2,
        # stride-averaged yaw-rate error 1 (command wz minus gyro z, mean of the
        # last gait period).  No base linear velocity: the robot does not measure it.
        frame_size = 48
        # Newest frame first.  The RS01 loop delays a target by ~40 ms before a
        # ~32 ms lag, so the actor needs its own recent actions and responses.
        history_length = 5
        num_observations = 48 * 5
        # Critic: noise-free frame 48 + base velocity 3 + height 1 + foot
        # contact 4 + scheduled stance 4 + foot height 4 + push force 3 +
        # friction 1 + payload 1 + CoM offset 3 + applied torque 12 + motor
        # strength 12.
        num_privileged_obs = 96
        episode_length_s = 20
        env_spacing = 2.5

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        # PhysX averages the two contact materials.  With the plane at 0.8 the
        # per-robot shape friction range below gives roughly 0.45-1.15.
        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # The base x-only curriculum is replaced below.
        num_commands = 4
        resampling_time = 5.0
        heading_command = False
        # Fraction of resamples that command standing still.
        stand_probability = 0.10
        # Fraction of moving resamples restricted to one axis (pure forward/
        # backward, pure lateral or pure turn), split evenly over the three
        # axes.  These are the usual joystick commands and where drift shows.
        single_axis_probability = 0.50
        # Commands smaller than both thresholds are treated as standing.
        min_planar_speed_m_s = 0.05
        min_yaw_speed_rad_s = 0.10
        # Playback/evaluation: [vx, vy, wz] applied to every environment.
        fixed_command = None

        class ranges:
            # Full envelope, reached at command level 1.0.  Same envelope as the
            # deployed dog_rs01 policies; the 2.6 rad/s thigh target-rate limit
            # caps useful forward speed near 0.6 m/s at this cadence.
            lin_vel_x = [-0.4, 0.6]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]

    class command_curriculum:
        enabled = True
        initial_level = 0.4
        level_step = 0.1
        # Episode-mean tracking rewards (0-1) and timeout share required to grow.
        min_linear_tracking = 0.70
        min_yaw_tracking = 0.60
        min_timeout_fraction = 0.85
        ema_rate = 0.02

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]  # 20 mm above touchdown: tolerates joint noise.
        default_joint_angles = {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": _THIGH_RAD,
            "FR_calf_joint": -_CALF_RAD,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": -_THIGH_RAD,
            "FL_calf_joint": _CALF_RAD,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": _THIGH_RAD,
            "RR_calf_joint": -_CALF_RAD,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -_THIGH_RAD,
            "RL_calf_joint": _CALF_RAD,
        }
        reset_dof_position_noise_rad = 0.08
        reset_linear_velocity_m_s = 0.3
        reset_angular_velocity_rad_s = 0.3
        randomize_yaw = True

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # Gains of the latest deployed RS01 controller (B36100).
        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}
        # target = default + scale * clip(action, +-clip_actions).  With the
        # clip of 2.0: hip +-0.30 rad, thigh/calf +-0.50 rad about the stance.
        action_scale_by_joint = {"hip": 0.15, "thigh": 0.25, "calf": 0.25}
        action_scale = 0.25  # Unused; kept for tools that read the base field.
        # 8 x 2.5 ms physics steps = 20 ms: the real 50 Hz policy/control loop.
        decimation = 8

    class rs01_actuator:
        # Source: rs01shujv/RS01_ACTUATOR_IDENTIFICATION_20260720.md.  The
        # motors of this machine are not individually identified, so every
        # joint uses the ten-motor median; domain_rand below spans the measured
        # 5-95 % band independently per motor.
        control_dt_s = 0.02
        response_update_dt_s = 0.005
        response_gain = 0.928
        time_constant_s = 0.0319
        closed_loop_delay_s = 0.040  # Measured 42.3 ms on the 5 ms clock.
        coulomb_friction_nm = 0.156
        friction_smoothing_rad_s = 0.05
        # Software limits of the real 50 Hz controller, not motor physics.
        target_rate_limit_rad_s = {"hip": 2.0, "thigh": 2.6, "calf": 3.2}
        target_acceleration_limit_rad_s2 = {"hip": 60.0, "thigh": 78.0, "calf": 96.0}
        # RS01: 17 Nm peak, ~6 Nm continuous.  The controller's operating cap
        # is 14 Nm; training against the cap keeps 3 Nm of real headroom.
        peak_torque_limit_nm = 14.0
        continuous_torque_nm = 6.0
        # Isaac Gym's DOF order for this asset.  Actor outputs, observations
        # and the exported contract all use it; a mismatch aborts start-up.
        runtime_dof_order = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]

    class asset(LeggedRobotCfg.asset):
        # Generated by legged_gym/scripts/prepare_urdf4_asset.py.
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/urdf4/urdf/urdf4_rs01.urdf"
        name = "urdf4_rs01"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 1  # Bitwise filter: 1 disables self-collision.
        # Merges imu_Link into the trunk; feet carry dont_collapse and stay.
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        armature = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Contact.
        randomize_friction = True
        friction_range = [0.1, 1.5]
        randomize_restitution = True
        restitution_range = [0.0, 0.2]
        # Inertial.
        randomize_base_mass = True
        added_mass_range = [-0.5, 2.0]
        randomize_base_com = True
        base_com_offset_m = [0.03, 0.02, 0.02]  # +- per axis x, y, z.
        randomize_link_mass = True
        link_mass_scale_range = [0.9, 1.1]
        # Reflected rotor inertia is unidentified; cover zero to a geared-rotor estimate.
        randomize_armature = True
        armature_range_kg_m2 = [0.0, 0.006]
        # Controller and actuator, independent per motor, resampled each episode.
        randomize_gains = True
        stiffness_scale_range = [0.9, 1.1]
        damping_scale_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.05]
        randomize_rs01_actuator = True
        rs01_response_gain_scale_range = [0.92, 1.13]
        rs01_time_constant_scale_range = [0.65, 1.25]
        rs01_friction_scale_range = [0.7, 1.5]
        rs01_delay_step_offset_range = [-1, 3]  # x 5 ms: 35-55 ms.
        # The base velocity-overwrite push stays off; see class push.
        push_robots = False

    class push:
        """Smooth horizontal force pulses on the trunk (no velocity edits)."""

        enabled = True
        peak_force_n = [10.0, 40.0]
        duration_s = [0.10, 0.30]
        interval_s = [5.0, 9.0]
        first_delay_s = [3.0, 6.0]
        application_height_m = 0.05  # Above the trunk CoM; adds a tipping moment.
        # Strength ramps from initial to full over this much simulated time.
        initial_strength = 0.3
        ramp_s = 600.0

    class imu:
        """Body IMU error model; see disturbances.ImuDisturbance."""

        enabled = True
        clean_env_fraction = 0.10
        latency_range_s = [0.0, 0.03]
        mount_roll_pitch_deg = 2.0
        mount_yaw_deg = 2.0
        # Gyro errors follow the real-robot logs (V21 observation-robustness
        # notes, IMU calibrated at power-on): noise sigma 0.004 rad/s, bias
        # +-0.005 rad/s.  The bias must stay well below the yaw drift worth
        # correcting (~0.05 rad/s) or the actor learns to ignore the gyro's
        # steady reading and cannot hold a straight line.
        gyro_noise_std_rad_s = 0.01
        gyro_bias_rad_s = 0.005
        gyro_bias_walk_rad_s_sqrt_s = 0.0005
        gyro_bias_limit_rad_s = 0.01
        gyro_scale_error = 0.03
        gravity_noise_std = 0.01
        attitude_drift_std_deg = 0.8
        attitude_drift_limit_deg = 2.5
        attitude_drift_tau_s = 4.0
        spike_probability = 0.01
        gyro_spike_rad_s = 0.4
        gravity_spike = 0.05
        stale_frame_probability = 0.01

    class joint_sensor:
        """Motor encoder feedback errors."""

        enabled = True
        latency_range_s = [0.0, 0.010]
        position_noise_rad = 0.01  # Uniform +-.
        velocity_noise_rad_s = 0.5  # Uniform +-.
        zero_offset_rad = 0.02  # Per-joint calibration error, fixed per episode.

    class gait:
        period_s = 0.50
        stance_ratio = 0.60  # 50 ms of four-foot overlap at each handoff, no flight.
        swing_clearance_m = 0.03
        foot_radius_m = 0.02
        randomize_phase_on_reset = True

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        # Commands are small, so the base sigma of 0.25 would pay a standing
        # robot ~70 % at 0.3 m/s.  0.06 pays 22 % there and 85 % at 0.1 m/s error.
        tracking_sigma = 0.06
        yaw_tracking_sigma = 0.10
        # Stride-averaged velocity error, exp(-|e|/sigma); see cycle_* scales.
        cycle_lin_vel_sigma_m_s = 0.03
        cycle_yaw_rate_sigma_rad_s = 0.05
        base_height_target = 0.27
        min_base_height_m = 0.14  # Below this the episode ends.
        foot_contact_force_threshold = 3.0
        foot_contact_release_force_threshold = 1.5
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.6  # Of the 32.99 rad/s no-load speed.
        max_contact_force = 100.0  # Static diagonal support is ~57 N per foot.

        class scales(LeggedRobotCfg.rewards.scales):
            # Task.
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            # Straight paths: a steady 0.08 rad/s yaw offset costs only 6 % of
            # tracking_ang_vel (flat Gaussian peak) yet turns the robot 46 deg
            # in 10 s.  These terms charge that steady offset directly.
            cycle_lin_vel = 0.5
            cycle_ang_vel = 0.5
            # Diagonal trot: contact must follow the clock, swing feet must lift.
            gait_contact = 1.0
            swing_clearance = -0.6
            # Body stability.
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -2.0
            base_height = -15.0
            # Actuator feasibility and smoothness.
            torques = -2.0e-4
            raw_torque_over_peak = -0.5
            dof_acc = -2.5e-7
            dof_vel_limits = -1.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            # Contact quality.
            feet_slip = -0.25
            feet_contact_forces = -0.01
            collision = -1.0
            # Posture.
            hip_deviation = -0.3
            stand_still = -0.5
            termination = -100.0
            # Base terms not used here.
            feet_air_time = 0.0
            dof_vel = 0.0
            feet_stumble = 0.0

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 2.0

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            # 0.05 rad/s of steady yaw error (3 deg/s of path bending) reads 0.2.
            yaw_rate_error_mean = 4.0

    class noise(LeggedRobotCfg.noise):
        # Master switch for the imu and joint_sensor models above.
        add_noise = True

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [1.6, -1.8, 0.9]
        lookat = [0.0, 0.0, 0.25]

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            # Solver settings validated on the previous RS01 machine.
            num_position_iterations = 8
            num_velocity_iterations = 2
            contact_offset = 0.003
            rest_offset = 0.0
            max_depenetration_velocity = 0.25


class Urdf4OmniCfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 1.0e-3
        schedule = "adaptive"
        desired_kl = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "urdf4_omni_v1"
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 100
