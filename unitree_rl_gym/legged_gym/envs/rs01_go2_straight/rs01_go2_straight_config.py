"""Minimal Go2-style straight-walking task for the user's RS01 quadruped."""

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class Rs01Go2StraightCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = None
        num_actions = 12
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges:
            # Positive-only commands keep this a straight-forward locomotion task.
            lin_vel_x = [0.2, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        # The supplied settled standing state is the reset state; zero pose is
        # never part of this task.
        pos = [0.0, 0.0, 0.316]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
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
        reset_dof_position_noise_rad = 0.015

    class control(LeggedRobotCfg.control):
        control_type = "P"
        # Current gains read from the real 50 Hz controller.
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}
        action_scale = 0.20
        decimation = 4

    class rs01_actuator:
        # Source: rs01shujv/rs01_actuator_data_20260720.json.
        control_dt_s = 0.02
        peak_torque_limit_nm = 17.0
        continuous_torque_nm = 6.0
        friction_smoothing_rad_s = 0.05

        target_rate_limit_rad_s = {
            "hip": 2.0,
            "thigh": 2.6,
            "calf": 3.2,
        }
        target_acceleration_limit_rad_s2 = {
            "hip": 60.0,
            "thigh": 78.0,
            "calf": 96.0,
        }
        joint_to_motor_id = {
            "FR_hip_joint": "0x11",
            "FR_thigh_joint": "0x12",
            "FR_calf_joint": "0x13",
            "FL_hip_joint": "0x21",
            "FL_thigh_joint": "0x22",
            "FL_calf_joint": "0x23",
            "RR_hip_joint": "0x31",
            "RR_thigh_joint": "0x32",
            "RR_calf_joint": "0x33",
            "RL_hip_joint": "0x41",
            "RL_thigh_joint": "0x42",
            "RL_calf_joint": "0x43",
        }

        # 0x11 and 0x41 were under repair during identification, so those two
        # joints deliberately use the measured ten-motor median.
        response_gain = {
            "0x11": 0.927855661680751,
            "0x12": 1.0486707533878392,
            "0x13": 0.8901892350113182,
            "0x21": 0.9820176769446501,
            "0x22": 0.8766154442027798,
            "0x23": 1.0140210101567217,
            "0x31": 0.9033451910468997,
            "0x32": 0.9197254975107978,
            "0x33": 0.84931465736209,
            "0x41": 0.927855661680751,
            "0x42": 0.947289846573121,
            "0x43": 0.992130005156902,
        }
        time_constant_s = {
            "0x11": 0.03185165908596266,
            "0x12": 0.03324038786959833,
            "0x13": 0.0349193070801588,
            "0x21": 0.03157265316505583,
            "0x22": 0.02524139288123714,
            "0x23": 0.038017615155043115,
            "0x31": 0.03773075858427813,
            "0x32": 0.027781445421318836,
            "0x33": 0.030792107301988197,
            "0x41": 0.03185165908596266,
            "0x42": 0.019269694387369536,
            "0x43": 0.03454611495612375,
        }
        observed_closed_loop_delay_s = {
            "0x11": 0.04231169551050045,
            "0x12": 0.04366661031515827,
            "0x13": 0.03949328236918313,
            "0x21": 0.04210807646841948,
            "0x22": 0.0485772301146227,
            "0x23": 0.040431084377084446,
            "0x31": 0.040563384905597216,
            "0x32": 0.048059893495805577,
            "0x33": 0.04352540237277155,
            "0x41": 0.04231169551050045,
            "0x42": 0.05617128701672144,
            "0x43": 0.04017254586680312,
        }
        coulomb_friction_nm = {
            "0x11": 0.15602040281564622,
            "0x12": 0.14562686751879023,
            "0x13": 0.1574945718465066,
            "0x21": 0.12278653063604661,
            "0x22": 0.1470226303215849,
            "0x23": 0.170176017952754,
            "0x31": 0.17658300576584962,
            "0x32": 0.15454623378478588,
            "0x33": 0.20610088046732755,
            "0x41": 0.15602040281564622,
            "0x42": 0.16198549951689997,
            "0x43": 0.14495521234045125,
        }

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/../dog_urdf/urdf/dog_rs01.urdf"
        name = "rs01_go2_straight"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["Trunk"]
        self_collisions = 1
        collapse_fixed_joints = False
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        armature = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        foot_contact_force_threshold = 2.0
        foot_contact_release_force_threshold = 2.0
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        max_contact_force = 200.0

        class scales(LeggedRobotCfg.rewards.scales):
            # This is intentionally the compact Go2 reward set: no CPG, phase,
            # symmetry, contact schedule, fixed step length or body-yaw shaping.
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
            dof_pos_limits = -10.0

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 1.0

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [1.8, -2.2, 1.1]
        lookat = [0.4, 0.0, 0.3]


class Rs01Go2StraightCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rs01_go2_straight"
        max_iterations = 3000
        save_interval = 100
