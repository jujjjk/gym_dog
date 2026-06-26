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
            "hip": 100.0,
            "thigh": 100.0,
            "calf": 100.0,
        }
        damping = {
            "hip": 4.0,
            "thigh": 4.0,
            "calf": 4.0,
        }
        action_scale = 0.18
        rear_action_scale = 0.22
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
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

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

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -5.0
            stand_height = 2.0
            stand_posture = 0.2
            tracking_lin_vel = 5.0
            tracking_ang_vel = 2.0
            backward_velocity = -5.0
            diagonal_gait = 4.0
            swing_height = 0.2
            flight = -2.0

            lin_vel_z = -0.75
            ang_vel_xy = -0.7
            yaw_rate = -2.0
            hip_velocity = -0.003
            hip_symmetry = -1.0
            diagonal_joint_sync = -0.5
            action_magnitude = -0.01
            orientation = -3.0
            base_height = -10.0
            low_base_height = -10.0
            rear_sit = -10.0
            front_feet_contact = -0.5
            rear_calf_fold = -2.0
            rear_load_bias = -1.5
            rear_leg_posture = -1.0

            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -5.0e-8
            action_rate = -0.015

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
