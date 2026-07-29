#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class JointSemanticMapper:
    """
    真机电机语义 <-> IsaacLab 训练策略语义映射器。

    你当前真机电机 ID：
        0x11 右前髋关节   FR_hip_joint
        0x12 右前大臂     FR_thigh_joint
        0x13 右前小腿     FR_calf_joint

        0x21 左前髋关节   FL_hip_joint
        0x22 左前大臂     FL_thigh_joint
        0x23 左前小腿     FL_calf_joint

        0x31 左后髋关节   RL_hip_joint
        0x32 左后大臂     RL_thigh_joint
        0x33 左后小腿     RL_calf_joint

        0x41 右后髋关节   RR_hip_joint
        0x42 右后大臂     RR_thigh_joint
        0x43 右后小腿     RR_calf_joint

    真机返回顺序：
        FR, FL, RL, RR

    训练策略顺序：
        FR, FL, RR, RL

    所以进入神经网络前，需要把后腿顺序从：
        RL, RR
    交换成：
        RR, RL

    同时，如果某些关节正负方向和训练不一致，需要修改 joint_sign。
    """

    def __init__(self, policy_joint_names=None, default_joint_angles=None):
        # ============================================================
        # 1. 真机电机返回顺序
        #    必须和 motor_state_interface.py 里 q_real/dq_real 的顺序一致
        # ============================================================
        self.real_motor_ids = [
            0x11, 0x12, 0x13,   # FR: 右前 hip/thigh/calf
            0x21, 0x22, 0x23,   # FL: 左前 hip/thigh/calf
            0x31, 0x32, 0x33,   # RL: 左后 hip/thigh/calf
            0x41, 0x42, 0x43,   # RR: 右后 hip/thigh/calf
        ]

        self.real_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",

            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",

            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",

            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]

        # ============================================================
        # 2. 训练策略 joint_names_expr 顺序
        #    来自你刚刚给出的 IsaacLab 配置
        # ============================================================
        self.policy_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",

            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",

            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",

            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

        # ============================================================
        # 3. policy 顺序 -> real 数组索引
        #
        # real 顺序:
        #   FR, FL, RL, RR
        #
        # policy 顺序:
        #   FR, FL, RR, RL
        #
        # 所以：
        #   policy FR <- real FR: index 0,1,2
        #   policy FL <- real FL: index 3,4,5
        #   policy RR <- real RR: index 9,10,11
        #   policy RL <- real RL: index 6,7,8
        # ============================================================
        self.policy_to_real_index = np.array([
            0, 1, 2,       # FR
            3, 4, 5,       # FL
            9, 10, 11,     # RR
            6, 7, 8,       # RL
        ], dtype=np.int64)

        # ============================================================
        # 4. 关节正负方向映射
        #
        # 这里是你后面自己要改的地方。
        #
        # 含义：
        #   q_policy_abs = joint_sign * q_real_abs
        #
        # 如果某个关节真机正方向和训练正方向一致：+1
        # 如果某个关节真机正方向和训练正方向相反：-1
        #
        # 顺序必须按 policy_joint_names：
        #   0  FR_hip
        #   1  FR_thigh
        #   2  FR_calf
        #   3  FL_hip
        #   4  FL_thigh
        #   5  FL_calf
        #   6  RR_hip
        #   7  RR_thigh
        #   8  RR_calf
        #   9  RL_hip
        #   10 RL_thigh
        #   11 RL_calf
        # ============================================================
        self.joint_sign = np.array([
            -1, +1, +1,    # FR
            -1, -1, -1,    # FL
            +1, +1, +1,    # RR
            +1, -1, -1,    # RL
        ], dtype=np.float32)

        # ============================================================
        # 5. 训练默认站姿角
        #
        # 来自你给出的 init_state.joint_pos。
        #
        # 注意：
        #   这是训练策略语义下的默认角度。
        #   obs 里的 joint_pos 不是直接用绝对角，而是：
        #
        #       q_obs = q_abs_policy - default_joint_angle
        #
        # 顺序同 policy_joint_names：
        #   FR, FL, RR, RL
        # ============================================================
        # Keep this aligned with FanfanRlCpg training:
        # FANFAN_TEXT_STAND_JOINT_POS in fanfan_robot_cfg.py.
        # Policy order is FR, FL, RR, RL.
        self.default_joint_angle = np.array([
            -0.1571, 0.3491, -0.7854,    # FR
            0.1571, 0.3491, -0.7854,     # FL
            -0.1571, 0.2618, -0.5236,    # RR  thigh: real motor 0x42 = +15 deg; calf: 0x43 = -30 deg
            0.1571, 0.2618, -0.5236,     # RL  thigh: real motor 0x32 = -15 deg because joint_sign=-1; calf: 0x33 = +30 deg
        ], dtype=np.float32)

        # ============================================================
        # 6. 真机零点偏移
        #
        # 你前面说 0 点位置一致，所以这里先全 0。
        #
        # 如果后面发现某个真机电机零点和 URDF/训练零点存在固定偏差，
        # 可以在这里补偿。
        #
        # 顺序也是 policy_joint_names。
        # ============================================================
        self.real_zero_offset_policy_order = np.array([
            0.0, 0.0, 0.0,     # FR
            0.0, 0.0, 0.0,     # FL
            0.0, 0.0, 0.0,     # RR
            0.0, 0.0, 0.0,     # RL
        ], dtype=np.float32)

        # ============================================================
        # 7. 安全限幅
        #
        # 这里先给一个保守大范围。
        # 真正上机前建议按 URDF limit 或实机安全范围重新填写。
        #
        # 注意：这个限幅只用于输出动作时保护，不参与 obs 拼接。
        # ============================================================
        # Updated Fanfan training URDF limits in policy joint coordinates.
        # Keep these aligned with fanfan_urdf/fanfan_urdf/urdf/fanfan.urdf.
        self.policy_lower_limit = np.array([
            -0.8, -0.5, -2.7,    # FR
            -0.8, -0.5, -2.7,    # FL
            -0.8, -0.5, -2.7,    # RR
            -0.8, -0.5, -2.7,    # RL
        ], dtype=np.float32)

        self.policy_upper_limit = np.array([
            0.8, 4.0, -0.85,    # FR
            0.8, 4.0, -0.85,    # FL
            0.8, 4.0, -0.85,    # RR
            0.8, 4.0, -0.85,    # RL
        ], dtype=np.float32)

        if policy_joint_names is not None:
            self.configure_policy_contract(policy_joint_names, default_joint_angles)

    def configure_policy_contract(self, policy_joint_names, default_joint_angles=None):
        """Reorder real/policy semantics to match an exported ONNX contract.

        Motor signs, zero offsets and joint limits are properties of the real
        robot, so they are looked up by joint name before changing the policy
        order.  The training default pose may then be replaced by the values
        embedded in the model.
        """
        names = list(policy_joint_names)
        if len(names) != 12 or len(set(names)) != 12:
            raise ValueError("policy_joint_names must contain 12 unique names")
        unknown = sorted(set(names) - set(self.real_joint_names))
        if unknown:
            raise ValueError(f"ONNX contract contains unknown joints: {unknown}")

        old_names = list(self.policy_joint_names)
        by_name = {
            name: (
                float(self.joint_sign[i]),
                float(self.real_zero_offset_policy_order[i]),
                float(self.policy_lower_limit[i]),
                float(self.policy_upper_limit[i]),
                float(self.default_joint_angle[i]),
            )
            for i, name in enumerate(old_names)
        }

        self.policy_joint_names = names
        self.policy_to_real_index = np.asarray(
            [self.real_joint_names.index(name) for name in names], dtype=np.int64
        )
        self.joint_sign = np.asarray([by_name[name][0] for name in names], dtype=np.float32)
        self.real_zero_offset_policy_order = np.asarray(
            [by_name[name][1] for name in names], dtype=np.float32
        )
        self.policy_lower_limit = np.asarray(
            [by_name[name][2] for name in names], dtype=np.float32
        )
        self.policy_upper_limit = np.asarray(
            [by_name[name][3] for name in names], dtype=np.float32
        )
        if default_joint_angles is None:
            self.default_joint_angle = np.asarray(
                [by_name[name][4] for name in names], dtype=np.float32
            )
        else:
            default = np.asarray(default_joint_angles, dtype=np.float32).reshape(-1)
            if default.shape[0] != 12 or not np.all(np.isfinite(default)):
                raise ValueError("default_joint_angles must contain 12 finite values")
            self.default_joint_angle = default.copy()

    # ============================================================
    # 输入方向：真机电机 angle/speed -> 策略 obs
    # ============================================================
    def real_to_policy_q_dq(self, q_real, dq_real):
        """
        将真机返回的 12 个电机 angle/speed 转成策略输入需要的 joint_pos/joint_vel。

        输入：
            q_real:
                真机原始角度，顺序为 real_motor_ids。
            dq_real:
                真机原始速度，顺序为 real_motor_ids。

        输出：
            q_obs_policy:
                策略 obs 里的 joint_pos[12]。
                注意它不是绝对角，而是相对默认站姿角的偏差。

            dq_policy:
                策略 obs 里的 joint_vel[12]。
        """
        q_real = np.asarray(q_real, dtype=np.float32).reshape(12)
        dq_real = np.asarray(dq_real, dtype=np.float32).reshape(12)

        # 1. 真机顺序 -> 策略顺序
        q_real_ordered = q_real[self.policy_to_real_index]
        dq_real_ordered = dq_real[self.policy_to_real_index]

        # 2. 真机角度语义 -> 训练角度语义
        #
        # 如果 joint_sign = -1，则说明该关节真机正方向和训练正方向相反。
        # real_zero_offset_policy_order 默认是 0，因为你说零点一致。
        q_abs_policy = self.joint_sign * (
            q_real_ordered - self.real_zero_offset_policy_order
        )

        dq_policy = self.joint_sign * dq_real_ordered

        # 3. 训练绝对角 -> 策略观测角
        #
        # IsaacLab 常见 obs 是 joint_pos - default_joint_pos。
        q_obs_policy = q_abs_policy - self.default_joint_angle

        return q_obs_policy.astype(np.float32), dq_policy.astype(np.float32)

    def real_to_policy_abs_q_dq(self, q_real, dq_real):
        """
        Convert real motor angle/speed to absolute policy/URDF joint angle space.

        This is for kinematics and leg odometry. Do not feed q_abs_policy directly
        into the policy obs; policy obs still uses q_abs_policy - default_joint_angle.
        """
        q_real = np.asarray(q_real, dtype=np.float32).reshape(12)
        dq_real = np.asarray(dq_real, dtype=np.float32).reshape(12)

        q_real_ordered = q_real[self.policy_to_real_index]
        dq_real_ordered = dq_real[self.policy_to_real_index]

        q_abs_policy = self.joint_sign * (
            q_real_ordered - self.real_zero_offset_policy_order
        )
        dq_policy = self.joint_sign * dq_real_ordered

        return q_abs_policy.astype(np.float32), dq_policy.astype(np.float32)

    # ============================================================
    # 输出方向：策略 action -> 真机目标角
    # ============================================================
    def policy_action_to_real_target(self, action_policy, action_scale=0.25, clamp=True):
        """
        将神经网络 action 转成真机电机目标角度。

        IsaacLab 常见 joint position action 逻辑：
            target_policy_abs = default_joint_angle + action_scale * action_policy

        然后训练语义 -> 真机语义：
            target_real = joint_sign * target_policy_abs + real_zero_offset

        最后再从 policy 顺序转回 real_motor_ids 顺序。
        """
        action_policy = np.asarray(action_policy, dtype=np.float32).reshape(12)
        action_scale = np.asarray(action_scale, dtype=np.float32)
        if action_scale.shape == ():
            action_scale = np.full(12, float(action_scale), dtype=np.float32)
        else:
            action_scale = action_scale.reshape(12)

        # 1. action -> 训练语义下的目标绝对关节角
        target_policy_abs = self.default_joint_angle + action_scale * action_policy

        # 2. 训练语义下限幅
        if clamp:
            target_policy_abs = np.clip(
                target_policy_abs,
                self.policy_lower_limit,
                self.policy_upper_limit,
            )

        # 3. 训练语义 -> 真机语义，仍然是 policy 顺序
        #
        # 因为：
        #   q_policy_abs = sign * (q_real - offset)
        #
        # 所以：
        #   q_real = sign * q_policy_abs + offset
        target_real_policy_order = (
            self.joint_sign * target_policy_abs
            + self.real_zero_offset_policy_order
        )

        # 4. policy 顺序 -> real_motor_ids 顺序
        target_real = np.zeros(12, dtype=np.float32)
        target_real[self.policy_to_real_index] = target_real_policy_order

        return target_real.astype(np.float32)

    def policy_target_to_real_target(self, target_policy_abs, clamp=True):
        """
        如果后面你已经得到了训练语义下的 12 个目标绝对角，
        用这个函数直接转成真机电机目标角。

        输入顺序：
            policy_joint_names

        输出顺序：
            real_motor_ids
        """
        target_policy_abs = np.asarray(target_policy_abs, dtype=np.float32).reshape(12)

        if clamp:
            target_policy_abs = np.clip(
                target_policy_abs,
                self.policy_lower_limit,
                self.policy_upper_limit,
            )

        target_real_policy_order = (
            self.joint_sign * target_policy_abs
            + self.real_zero_offset_policy_order
        )

        target_real = np.zeros(12, dtype=np.float32)
        target_real[self.policy_to_real_index] = target_real_policy_order

        return target_real.astype(np.float32)

    def policy_values_to_real_order(self, values):
        """Reorder per-joint values without applying angle signs or offsets.

        Gains, torque limits and other magnitudes are indexed by joint name but
        are not joint coordinates.  They only need policy-order -> motor-order
        permutation; applying ``joint_sign`` to them would be incorrect.
        """
        policy_values = np.asarray(values, dtype=np.float32).reshape(-1)
        if policy_values.shape[0] != 12 or not np.all(np.isfinite(policy_values)):
            raise ValueError("policy values must contain 12 finite values")
        real_values = np.zeros(12, dtype=np.float32)
        real_values[self.policy_to_real_index] = policy_values
        return real_values

    # ============================================================
    # 调试辅助
    # ============================================================
    def real_default_pose_for_motor_order(self):
        """
        返回训练默认站姿对应的真机电机角度，顺序为 real_motor_ids。

        这个很有用：
        如果机器人真机摆成训练默认站姿，理论上 q_real 应该接近这个数组。
        """
        default_real_policy_order = (
            self.joint_sign * self.default_joint_angle
            + self.real_zero_offset_policy_order
        )

        default_real = np.zeros(12, dtype=np.float32)
        default_real[self.policy_to_real_index] = default_real_policy_order

        return default_real

    def print_mapping(self):
        print("========== Joint Semantic Mapping ==========")
        print("Real motor order:")
        for i, (mid, name) in enumerate(zip(self.real_motor_ids, self.real_joint_names)):
            print(f"  real[{i:02d}]  motor_id=0x{mid:02X}  {name}")

        print("\nPolicy joint order:")
        for i, name in enumerate(self.policy_joint_names):
            real_idx = int(self.policy_to_real_index[i])
            mid = self.real_motor_ids[real_idx]
            real_name = self.real_joint_names[real_idx]
            sign = self.joint_sign[i]
            default = self.default_joint_angle[i]
            print(
                f"  policy[{i:02d}] {name:16s} "
                f"<- real[{real_idx:02d}] motor_id=0x{mid:02X} {real_name:16s} "
                f"sign={sign:+.0f} default={default:+.4f}"
            )

        print("\nTraining default pose converted to real motor order:")
        default_real = self.real_default_pose_for_motor_order()
        for i, (mid, name) in enumerate(zip(self.real_motor_ids, self.real_joint_names)):
            print(f"  motor_id=0x{mid:02X} {name:16s} default_real={default_real[i]:+.4f}")

        print("============================================")

    def get_policy_joint_names(self):
        return list(self.policy_joint_names)

    def get_real_motor_ids(self):
        return list(self.real_motor_ids)


if __name__ == "__main__":
    mapper = JointSemanticMapper()
    mapper.print_mapping()

    # 测试 1：
    # 构造一个真机默认站姿角，如果映射正确，
    # 进入策略 obs 后 q_obs_policy 应该接近全 0。
    q_real_default = mapper.real_default_pose_for_motor_order()
    dq_real_zero = np.zeros(12, dtype=np.float32)

    q_obs_policy, dq_policy = mapper.real_to_policy_q_dq(
        q_real_default,
        dq_real_zero,
    )

    print("\nTest with default real pose:")
    print("q_real_default:")
    print(q_real_default)

    print("q_obs_policy should be near all zeros:")
    print(q_obs_policy)

    print("dq_policy:")
    print(dq_policy)

    # 测试 2：
    # 构造 action 全 0，如果 action_scale 任意，
    # 输出目标应该等于训练默认站姿对应的真机角度。
    action_zero = np.zeros(12, dtype=np.float32)
    target_real = mapper.policy_action_to_real_target(action_zero, action_scale=0.25)

    print("\nTest with zero action:")
    print("target_real should equal default real pose:")
    print(target_real)
