#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim_real_semantic_bridge.py

sim_semantic  ↔  real_policy  ↔  real_motor 三个 joint 空间的语义转换桥。

三个空间的精确定义
==================
A. sim_semantic
   IsaacLab / golden CSV 使用的空间 (q_ref_semantic_0~11 / q_ref_0~11 / q_cmd_final_0~11)。
   顺序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
         RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
   这是 V4 migration core 的内部空间。
   (IsaacLab joint_semantics.py: SIM_JOINT_SIGN_POLICY_ORDER = all +1, sim_offset = 0,
    policy_to_sim 是恒等。)

B. real_policy (mapper_semantic)
   ROS2 JointSemanticMapper.real_to_policy_abs_q_dq() 返回的 12 维空间。
   也是 policy 顺序 (FR,FL,RR,RL)，但符号/零点由真机 mapper 定义。
   不一定等于 sim_semantic，所以不能直接拿去和 golden CSV 比。

C. real_motor
   真正发给电机 0x11~0x43 的空间，由 mapper.policy_target_to_real_target() 得到。
   本桥不处理 real_motor，real_motor 仍由 JointSemanticMapper 负责。

桥的推导 (不是猜的)
==================
IsaacLab 部署契约 (joint_semantics.py FanfanJointSemanticAdapter):
    q_real_motor = real_sign_isaac ⊙ q_sim + real_zero_isaac        (real_zero_isaac = 0)
ROS2 mapper (semantic_mapper.JointSemanticMapper.real_to_policy_abs_q_dq):
    q_real_policy = mapper.joint_sign ⊙ (q_real_ordered - mapper.real_zero)   (mapper.real_zero = 0)
其中 q_real_ordered 就是 real_motor 重排到 policy 顺序。代入:
    q_real_policy = mapper.joint_sign ⊙ (real_sign_isaac ⊙ q_sim + real_zero_isaac - mapper.real_zero)
                  = (mapper.joint_sign · real_sign_isaac) ⊙ q_sim
                    + mapper.joint_sign ⊙ (real_zero_isaac - mapper.real_zero)
所以 (在同为 policy 顺序的前提下，逐关节 affine):
    sign[i]   = mapper.joint_sign[i] * real_sign_isaac[i]
    offset[i] = mapper.joint_sign[i] * (real_zero_isaac[i] - mapper.real_zero[i])
    q_real_policy = sign ⊙ q_sim + offset
    q_sim         = sign ⊙ (q_real_policy - offset)        (sign ∈ {±1})

当前仓库里 mapper.joint_sign == real_sign_isaac 且零点都为 0，所以推导结果是 **恒等** (sign=+1, offset=0)。
桥仍然保留这套显式 sign/offset/index，以便日后 mapper 改符号/零点时自动跟随，并在启动时打印 + 自检 roundtrip。
"""

from __future__ import annotations

import numpy as np

# sim_semantic policy 顺序 (与 fanfan_v4_migration_core / IsaacLab joint_semantics 一致)
SIM_JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

# IsaacLab joint_semantics.py: REAL_JOINT_SIGN_POLICY_ORDER (policy 顺序)
# q_real_motor = real_sign_isaac ⊙ q_sim
ISAAC_REAL_SIGN_POLICY_ORDER = np.array(
    [
        -1.0, 1.0, 1.0,    # FR
        -1.0, -1.0, -1.0,  # FL
        1.0, 1.0, 1.0,     # RR
        1.0, -1.0, -1.0,   # RL
    ],
    dtype=np.float64,
)
# IsaacLab real_zero_offset (policy 顺序) = 0
ISAAC_REAL_ZERO_POLICY_ORDER = np.zeros(12, dtype=np.float64)


def _strip_joint(name: str) -> str:
    return name[:-6] if name.endswith("_joint") else name


class SimRealSemanticBridge:
    """sim_semantic ↔ real_policy 的逐关节 affine 桥 (policy 顺序内, sign ∈ {±1})。"""

    def __init__(self, mapper=None, *, verbose: bool = True):
        """
        mapper: 可选的 JointSemanticMapper 实例。提供则从中读取
                joint_sign / real_zero_offset_policy_order / policy_joint_names；
                不提供则用 IsaacLab 默认 (与当前 mapper 一致)。
        """
        if mapper is not None:
            mapper_sign = np.asarray(mapper.joint_sign, dtype=np.float64).reshape(12)
            mapper_zero = np.asarray(mapper.real_zero_offset_policy_order, dtype=np.float64).reshape(12)
            mapper_names = [_strip_joint(n) for n in mapper.get_policy_joint_names()]
        else:
            mapper_sign = ISAAC_REAL_SIGN_POLICY_ORDER.copy()
            mapper_zero = np.zeros(12, dtype=np.float64)
            mapper_names = list(SIM_JOINT_NAMES)

        # sim policy index -> real_policy index (按关节名匹配; 当前两者同序 -> 恒等)
        index = []
        for sim_name in SIM_JOINT_NAMES:
            if sim_name in mapper_names:
                index.append(mapper_names.index(sim_name))
            else:
                index.append(len(index))  # 兜底: 同序
        self.sim_to_real_index = np.asarray(index, dtype=np.int64)

        # mapper 量重排到 sim 顺序，便于逐关节运算
        mapper_sign_simorder = mapper_sign[self.sim_to_real_index]
        mapper_zero_simorder = mapper_zero[self.sim_to_real_index]
        isaac_sign = ISAAC_REAL_SIGN_POLICY_ORDER
        isaac_zero = ISAAC_REAL_ZERO_POLICY_ORDER

        # 推导 (见文件头):
        self.sim_to_real_sign = mapper_sign_simorder * isaac_sign
        self.sim_to_real_offset = mapper_sign_simorder * (isaac_zero - mapper_zero_simorder)

        if np.any(self.sim_to_real_sign == 0.0):
            raise ValueError("SimRealSemanticBridge: sign 含 0，桥不可逆。")

        # 自检 roundtrip
        try:
            from .fanfan_v4_migration_core import SIM_V4_DEFAULT_JOINT_POS_SIM
        except ImportError:  # 允许脱离 ROS 包直接运行
            from fanfan_v4_migration_core import SIM_V4_DEFAULT_JOINT_POS_SIM
        q_test = SIM_V4_DEFAULT_JOINT_POS_SIM.copy()
        q_real = self.sim_to_real_policy(q_test)
        q_back = self.real_policy_to_sim(q_real)
        self.roundtrip_error_max = float(np.max(np.abs(q_back - q_test)))

        self.enabled = True
        self.is_identity = bool(
            np.allclose(self.sim_to_real_sign, 1.0, atol=1.0e-9)
            and np.allclose(self.sim_to_real_offset, 0.0, atol=1.0e-9)
            and np.array_equal(self.sim_to_real_index, np.arange(12))
        )
        if verbose:
            self.print_bridge()
        if self.roundtrip_error_max >= 1.0e-6:
            raise RuntimeError(
                f"SimRealSemanticBridge roundtrip_error_max={self.roundtrip_error_max:.3e} >= 1e-6"
            )

    # ------------------------------------------------------------------
    # 角度 (有 offset)
    # ------------------------------------------------------------------
    def sim_to_real_policy(self, q_sim: np.ndarray) -> np.ndarray:
        q_sim = np.asarray(q_sim, dtype=np.float64).reshape(12)
        return (self.sim_to_real_sign * q_sim + self.sim_to_real_offset).astype(np.float64)

    def real_policy_to_sim(self, q_real_policy: np.ndarray) -> np.ndarray:
        q = np.asarray(q_real_policy, dtype=np.float64).reshape(12)
        return (self.sim_to_real_sign * (q - self.sim_to_real_offset)).astype(np.float64)

    # ------------------------------------------------------------------
    # 速度 / 力矩 (只有 sign, 无 offset)
    # ------------------------------------------------------------------
    def sim_dq_to_real_policy(self, dq_sim: np.ndarray) -> np.ndarray:
        dq = np.asarray(dq_sim, dtype=np.float64).reshape(12)
        return (self.sim_to_real_sign * dq).astype(np.float64)

    def real_policy_dq_to_sim(self, dq_real_policy: np.ndarray) -> np.ndarray:
        dq = np.asarray(dq_real_policy, dtype=np.float64).reshape(12)
        return (self.sim_to_real_sign * dq).astype(np.float64)

    def sim_tau_to_real_policy(self, tau_sim: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_sim, dtype=np.float64).reshape(12)
        return (self.sim_to_real_sign * tau).astype(np.float64)

    def real_policy_tau_to_sim(self, tau_real_policy: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_real_policy, dtype=np.float64).reshape(12)
        return (self.sim_to_real_sign * tau).astype(np.float64)

    # ------------------------------------------------------------------
    def print_bridge(self, logger=None):
        lines = [
            f"[BRIDGE] enabled=True is_identity={self.is_identity} roundtrip_error_max={self.roundtrip_error_max:.3e}",
            f"[BRIDGE] sim joint order        = {list(SIM_JOINT_NAMES)}",
            f"[BRIDGE] real_policy joint order = (policy order, same as sim)",
            f"[BRIDGE] sim_to_real_index  = {self.sim_to_real_index.tolist()}",
            f"[BRIDGE] sim_to_real_sign   = {self.sim_to_real_sign.tolist()}",
            f"[BRIDGE] sim_to_real_offset = {self.sim_to_real_offset.tolist()}",
        ]
        for ln in lines:
            if logger is not None:
                logger.info(ln)
            else:
                print(ln)


if __name__ == "__main__":
    bridge = SimRealSemanticBridge(verbose=True)
    print(f"[BRIDGE] is_identity = {bridge.is_identity}")
    print(f"[BRIDGE] roundtrip_error_max = {bridge.roundtrip_error_max:.3e}")
