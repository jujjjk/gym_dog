from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class FanfanRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 50
        num_actions = 12

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 1.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.295]
        default_joint_angles = {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.563,
            "FR_calf_joint": -0.95,

            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.563,
            "FL_calf_joint": -0.95,

            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.563,
            "RR_calf_joint": -0.95,

            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.563,
            "RL_calf_joint": -0.95,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip": 60.0,
            "thigh": 70.0,
            "calf": 70.0,
        }
        damping = {
            "hip": 0.6,
            "thigh": 0.8,
            "calf": 0.8,
        }
        action_scale = 0.18
        rear_action_scale = 0.20
        hip_action_scale = 0.08
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/fanfan/urdf/fanfan.urdf"
        name = "fanfan"
        foot_name = "foot"
        # 截图里的失效模式是用小腿/膝部接地前蹭；只匹配 calf，
        # 避免 thigh 同时选中 thigh_shoulder 的内部接触力。
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["Trunk"]

        # Isaac Gym uses 1 to filter self-collisions. The hip and shoulder
        # collision shapes overlap by design and otherwise create huge forces.
        self_collisions = 1

        # 关键：不要合并 fixed joint，避免 thigh_shoulder / imu_link 导致形态异常
        collapse_fixed_joints = False

        # 关键：不要 flip visual
        flip_visual_attachments = False
        armature = 0.01

    class commands(LeggedRobotCfg.commands):
        heading_command = False
        resampling_time = 10.0

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.15, 0.30]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        # URDF total mass/inertia and contact modelling differ slightly between
        # PhysX and MuJoCo. Keep the ranges narrow enough to preserve the gait,
        # while exposing the policy to the asymmetries seen in Sim2Sim.
        randomize_friction = True
        friction_range = [0.75, 1.20]
        randomize_base_mass = True
        added_mass_range = [-0.20, 0.20]
        randomize_motor_strength = True
        motor_strength_range = [0.90, 1.10]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.15

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.275
        min_base_height = 0.225
        min_base_height_soft = 0.255
        calf_angle_limits = [-2.65, -0.90]
        terminate_on_calf_angle = False
        terminate_rear_sit_pitch = -0.25
        max_rear_sit_pitch = 0.03
        front_feet_contact_height = 0.25
        rear_calf_fold_limit = -2.20
        rear_load_bias_force = 20.0
        stand_height_sigma = 0.0008
        stand_posture_sigma = 0.12
        rear_leg_posture_height = 0.25
        gait_period = 0.54
        gait_stance_ratio = 0.62
        gait_thigh_amplitude = 0.0
        gait_calf_amplitude = -0.30
        swing_height_target = 0.045
        swing_height_sigma = 0.0004
        max_contact_force = 60.0
        only_positive_rewards = True
        tracking_sigma = 0.02
        torque_near_limit_ratio = 0.90
        peak_torque_soft_ratio = 0.95
        sustained_torque_ratio = 0.75
        pd_pos_err_soft_limit = 0.22
        torque_curriculum = True
        torque_curriculum_steps_per_iteration = 24
        torque_curriculum_stage2_iteration = 300
        torque_curriculum_stage3_iteration = 1000
        torque_curriculum_stage4_iteration = 2000
        torque_curriculum_blend_iterations = 100
        torque_curriculum_stage2 = {
            "torque_clip": -0.4,
            "torque_near_limit": -0.04,
            "peak_torque": -0.04,
            "sustained_torque": -0.08,
        }
        torque_curriculum_stage3 = {
            "torque_clip": -0.6,
            "torque_near_limit": -0.06,
            "peak_torque": -0.06,
            "sustained_torque": -0.10,
        }
        torque_curriculum_stage4 = {
            "torque_clip": -0.8,
            "torque_near_limit": -0.08,
            "peak_torque": -0.08,
            "sustained_torque": -0.12,
        }

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -5.0
            stand_height = 2.0
            stand_posture = 0.2
            tracking_lin_vel = 6.0
            tracking_ang_vel = 2.0
            backward_velocity = -10.0
            diagonal_gait = 6.0
            swing_height = 0.2
            flight = -2.0

            lin_vel_z = -0.75
            ang_vel_xy = -0.7
            yaw_rate = -2.0
            hip_velocity = -0.003
            hip_symmetry = -1.0
            diagonal_joint_sync = -0.5
            action_magnitude = -0.012
            orientation = -3.0
            base_height = -10.0
            low_base_height = -10.0
            rear_sit = -10.0
            front_feet_contact = -0.5
            rear_calf_fold = -2.0
            rear_load_bias = -1.5
            rear_leg_posture = -1.0

            torques = -2.0e-6
            torque_clip = -0.2
            torque_near_limit = -0.02
            peak_torque = -0.02
            sustained_torque = -0.05
            mechanical_power = -1.0e-6
            pd_position_error_over_limit = -0.3
            dof_vel = -0.0
            dof_acc = -8.0e-8
            action_rate = -0.025

            # 明确压掉膝盖/小腿接地的投机解。
            collision = -2.0

            # 保守动作下适度提高限位惩罚，避免靠打限位撑地。
            dof_pos_limits = -2.0
            calf_angle_limits = -4.0

            feet_air_time = 0.0

    class sim(LeggedRobotCfg.sim):
        substeps = 2

        class physx(LeggedRobotCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 4


class FanfanRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.002

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_fanfan"
