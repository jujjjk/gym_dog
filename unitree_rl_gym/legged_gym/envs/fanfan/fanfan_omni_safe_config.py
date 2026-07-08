"""Single-policy omnidirectional continuation from the sim2real forward gait."""
from legged_gym.envs.fanfan.fanfan_config import FanfanRoughCfg, FanfanRoughCfgPPO


class FanfanOmniSafeCfg(FanfanRoughCfg):
    class commands(FanfanRoughCfg.commands):
        heading_command = False
        observe_heading_error = True
        resampling_time = 8.0
        pure_yaw_probability = 0.18
        stand_probability = 0.08
        pure_lateral_probability = 0.20
        pure_sagittal_probability = 0.28
        omni_curriculum = True

        # Add one capability at a time while continuously replaying forward gait.
        omni_curriculum_stages = [
            {
                "until_iteration": 200,
                "lin_vel_x": [0.05, 0.28],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-0.35, 0.35],
            },
            {
                "until_iteration": 500,
                "lin_vel_x": [-0.05, 0.28],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-0.50, 0.50],
            },
            {
                "until_iteration": 900,
                "lin_vel_x": [-0.08, 0.28],
                "lin_vel_y": [-0.04, 0.04],
                "ang_vel_yaw": [-0.50, 0.50],
            },
            {
                "until_iteration": 1300,
                "lin_vel_x": [-0.10, 0.30],
                "lin_vel_y": [-0.06, 0.06],
                "ang_vel_yaw": [-0.60, 0.60],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.30],
                "lin_vel_y": [-0.08, 0.08],
                "ang_vel_yaw": [-0.70, 0.70],
            },
        ]

        class ranges(FanfanRoughCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.30]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.70, 0.70]

    class rewards(FanfanRoughCfg.rewards):
        lateral_tracking_sigma = 0.0004
        longitudinal_tracking_sigma = 0.002

        class scales(FanfanRoughCfg.rewards.scales):
            tracking_lin_vel = 6.0
            tracking_lateral_vel = 4.0
            tracking_longitudinal_vel = 4.0
            tracking_ang_vel = 4.0
            heading_tracking = 3.0

            backward_velocity = 0.0
            lateral_velocity = 0.0

            # yaw 基础约束。
            # 注意：这里不要过大，否则 omni 策略会变得不敢转向。
            yaw_rate = -0.20

            lateral_hip_common_mode = -1.0
            lateral_yaw_error = -8.0
            translation_yaw_error = -6.0


class FanfanOmniSafeCfgPPO(FanfanRoughCfgPPO):
    class algorithm(FanfanRoughCfgPPO.algorithm):
        entropy_coef = 0.003

    class runner(FanfanRoughCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_safe"
        run_name = "omni_safe"


class FanfanOmniSafe900Cfg(FanfanOmniSafeCfg):
    """Deployment snapshot: command envelope actually learned by model_900."""

    class commands(FanfanOmniSafeCfg.commands):
        omni_curriculum = False

        class ranges(FanfanOmniSafeCfg.commands.ranges):
            lin_vel_x = [-0.08, 0.28]
            lin_vel_y = [-0.04, 0.04]
            ang_vel_yaw = [-0.50, 0.50]


class FanfanOmniFastCfg(FanfanOmniSafeCfg):
    """Higher-speed continuation with a gradually enlarged action envelope."""

    class control(FanfanOmniSafeCfg.control):
        # Increase stride authority gradually through retraining, while keeping

        action_scale = 0.20
        rear_action_scale = 0.22
        hip_action_scale = 0.10

    class commands(FanfanOmniSafeCfg.commands):
        resampling_time = 7.0
        omni_curriculum = True

        omni_curriculum_stages = [
            {
                "until_iteration": 250,
                "lin_vel_x": [-0.08, 0.32],
                "lin_vel_y": [-0.05, 0.05],
                "ang_vel_yaw": [-0.55, 0.55],
            },
            {
                "until_iteration": 600,
                "lin_vel_x": [-0.10, 0.36],
                "lin_vel_y": [-0.06, 0.06],
                "ang_vel_yaw": [-0.65, 0.65],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.40],
                "lin_vel_y": [-0.08, 0.08],
                "ang_vel_yaw": [-0.80, 0.80],
            },
        ]

        class ranges(FanfanOmniSafeCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.40]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.80, 0.80]

    class rewards(FanfanOmniSafeCfg.rewards):
        class scales(FanfanOmniSafeCfg.rewards.scales):
            tracking_lin_vel = 7.0
            tracking_lateral_vel = 5.0
            tracking_longitudinal_vel = 5.0
            tracking_ang_vel = 5.0


class FanfanOmniFastCfgPPO(FanfanOmniSafeCfgPPO):
    class runner(FanfanOmniSafeCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_fast"
        run_name = "omni_fast"


class FanfanOmniSmoothRealCfg(FanfanOmniFastCfg):
    """Real-data correction: smooth outputs and match the deployed actuator envelope."""

    class control(FanfanOmniFastCfg.control):
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}
        torque_limit_override = 10.0

        action_scale = 0.19
        rear_action_scale = 0.21
        hip_action_scale = 0.09

        gate_gait_with_command = True
        gait_command_gate_sigma = 0.0004

    class commands(FanfanOmniFastCfg.commands):
        stand_probability = 0.25
        pure_yaw_probability = 0.15
        pure_lateral_probability = 0.15
        pure_sagittal_probability = 0.25

        omni_curriculum_stages = [
            {
                "until_iteration": 250,
                "lin_vel_x": [-0.10, 0.32],
                "lin_vel_y": [-0.05, 0.05],
                "ang_vel_yaw": [-0.55, 0.55],
            },
            {
                "until_iteration": 600,
                "lin_vel_x": [-0.12, 0.36],
                "lin_vel_y": [-0.06, 0.06],
                "ang_vel_yaw": [-0.65, 0.65],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.40],
                "lin_vel_y": [-0.08, 0.08],
                "ang_vel_yaw": [-0.80, 0.80],
            },
        ]

    class rewards(FanfanOmniFastCfg.rewards):
        # 原来是 0.75。
        # 降到 0.70 后，策略动作还没贴边时就会开始被惩罚。
        action_saturation_threshold = 0.70
        stand_command_sigma = 0.0004

        class scales(FanfanOmniFastCfg.rewards.scales):
            # 原来：
            # action_magnitude = -0.04
            # action_rate = -0.18
            # action_saturation = -0.30
            #
            # 这里小幅加重，重点压动作幅值和动作变化。
            action_magnitude = -0.055
            action_rate = -0.25
            action_saturation = -0.45

            # 站立时压原地抖动。
            stand_action = -0.50
            stand_dof_velocity = -0.004

            # 关节加速度惩罚略加，不要太大，否则腿会发软。
            dof_acc = -5.0e-7

            # 扭矩项先不猛加，避免学得太保守。
            torques = -6.0e-6


class FanfanOmniSmoothRealCfgPPO(FanfanOmniFastCfgPPO):
    class algorithm(FanfanOmniFastCfgPPO.algorithm):
        entropy_coef = 0.001

    class runner(FanfanOmniFastCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_smooth_real"
        run_name = "omni_smooth_real"

class FanfanOmniFilteredCfg(FanfanOmniSmoothRealCfg):
    """Velocity-preserving smooth policy for real deployment.

    v2 思路：
    1. 不再过度压动作幅值，否则速度跟不上；
    2. 保留动作滤波，但不要滤得太死；
    3. thigh/calf 的速度限制放开一些，保证步幅和摆腿速度；
    4. yaw 仍然压，但不让 yaw 惩罚影响正常转向；
    5. 提高速度跟踪奖励，避免策略为了省力而走慢。
    """

    class control(FanfanOmniSmoothRealCfg.control):
        filter_policy_actions = True

        # v1 是 0.22，太柔导致响应慢。
        # 这里改回略高，既保留滤波，又让速度跟得上。
        policy_action_filter_alpha = 0.28

        # 不要把 thigh/calf 限得太死。
        # 前进速度主要靠 thigh/calf 产生步幅和摆腿速度。
        # hip 继续稍微保守一点，避免左右摆和 yaw 抖。
        policy_action_rate_limits = {
            "hip": 0.75 / 0.09,
            "thigh": 1.25 / 0.19,
            "calf": 2.10 / 0.19,
        }

        # 加速度限制也放宽一点。
        # 上一版 20/32/55 会让腿明显发软。
        policy_action_accel_limits = {
            "hip": 24.0 / 0.09,
            "thigh": 42.0 / 0.19,
            "calf": 72.0 / 0.19,
        }

    class commands(FanfanOmniSmoothRealCfg.commands):
        # 站立概率保留 0.35 没问题，能让原地更稳。
        # 如果后面发现整体还是偏保守，可以降到 0.30。
        stand_probability = 0.35
        pure_yaw_probability = 0.12
        pure_lateral_probability = 0.13
        pure_sagittal_probability = 0.22

    class rewards(FanfanOmniSmoothRealCfg.rewards):
        only_positive_rewards = False

        # v1 降到 0.70 后，策略太早害怕动作贴边。
        # 改回 0.75，让它允许必要的大步幅。
        action_saturation_threshold = 0.75

        class scales(FanfanOmniSmoothRealCfg.rewards.scales):
            # =========================
            # 1. 速度跟踪：提高优先级
            # =========================
            # 原 FanfanOmniFastCfg 是 7/5/5/5。
            # 这里把前进/横向速度跟踪提高，避免策略为了平滑而走慢。
            tracking_lin_vel = 8.0
            tracking_longitudinal_vel = 6.0
            tracking_lateral_vel = 5.5

            # 转向跟踪不要加太猛，否则 yaw 会更积极。
            tracking_ang_vel = 5.0

            # =========================
            # 2. yaw 稳定：保留，但不压死
            # =========================
            # v1 是 -0.35，这里略降到 -0.30。
            # 过大容易让转向变慢，也可能影响横移。
            yaw_rate = -0.30

            # 重点还是压“没有转向命令时乱转头”。
            # 比原始版稍强，比 v1 稍弱。
            lateral_yaw_error = -9.0
            translation_yaw_error = -7.0

            # 保持朝向，不要过大。
            heading_tracking = 3.3

            # =========================
            # 3. 动作幅值：不要压太狠
            # =========================
            # 这个是速度变慢的关键。
            # action_magnitude / policy_action_magnitude 太大，会直接惩罚迈大步。
            action_magnitude = -0.040
            policy_action_magnitude = -0.110

            # =========================
            # 4. 动作平滑：主要压“变化率”，不压“幅值”
            # =========================
            # 原 SmoothReal：action_rate = -0.18
            # v1：-0.25 太重。
            # 这里折中。
            action_rate = -0.21

            # 原 Filtered：policy_action_rate = -0.30
            # v1：-0.45 太重。
            # 这里只小幅加。
            policy_action_rate = -0.36

            # 饱和惩罚也回退，不要把速度能力打掉。
            action_saturation = -0.35
            policy_action_saturation = -0.75

            # filter gap 不能太大，否则网络会强行贴滤波后的慢动作。
            policy_filter_gap = -0.42

            # =========================
            # 5. 站立和能耗
            # =========================
            # 站立压抖保留，但不要太大。
            stand_action = -0.65
            stand_dof_velocity = -0.0035

            # dof_acc 和 torques 回到温和一点。
            # 太大也会让腿软、速度慢。
            dof_acc = -3.5e-7
            torques = -5.5e-6


class FanfanOmniFilteredCfgPPO(FanfanOmniSmoothRealCfgPPO):
    class algorithm(FanfanOmniSmoothRealCfgPPO.algorithm):
        entropy_coef = 0.0005

    class runner(FanfanOmniSmoothRealCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_filtered"
        run_name = "omni_filtered_vel_smooth_v2"


class FanfanOmniVelTrackV3Cfg(FanfanOmniFilteredCfg):
    """Continuation focused on direction accuracy and higher commanded speed.

    This is intentionally a new task instead of overwriting
    ``fanfan_omni_filtered``.  The previous model_2000 is a good smooth/safe
    seed, but it under-tracks velocity and still has lateral/yaw drift.  V3
    makes velocity and heading errors more expensive while only mildly opening
    the stride envelope.
    """

    class control(FanfanOmniFilteredCfg.control):
        # Keep the real RS01 PD/10Nm envelope from SmoothReal/Filtered.
        # Slightly larger thigh/calf stride authority is needed because the
        # selected model tracks 0.35 m/s as ~0.23 m/s in Gym.
        action_scale = 0.205
        rear_action_scale = 0.225
        hip_action_scale = 0.09

        # A little more leg speed, hip still conservative to avoid side/yaw
        # whipping.  Values are normalized-action limits per second.
        policy_action_rate_limits = {
            "hip": 0.75 / 0.09,
            "thigh": 1.45 / 0.205,
            "calf": 2.35 / 0.205,
        }
        policy_action_accel_limits = {
            "hip": 26.0 / 0.09,
            "thigh": 48.0 / 0.205,
            "calf": 82.0 / 0.205,
        }

    class commands(FanfanOmniFilteredCfg.commands):
        resampling_time = 7.0
        stand_probability = 0.25
        pure_yaw_probability = 0.14
        pure_lateral_probability = 0.18
        pure_sagittal_probability = 0.28

        # Resume from a learned omni policy and expand gradually.  The first
        # stage is close to the selected checkpoint's envelope; later stages
        # ask for faster forward/lateral/yaw tracking.
        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 250,
                "lin_vel_x": [-0.10, 0.38],
                "lin_vel_y": [-0.08, 0.08],
                "ang_vel_yaw": [-0.75, 0.75],
            },
            {
                "until_iteration": 700,
                "lin_vel_x": [-0.12, 0.42],
                "lin_vel_y": [-0.09, 0.09],
                "ang_vel_yaw": [-0.82, 0.82],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.10, 0.10],
                "ang_vel_yaw": [-0.90, 0.90],
            },
        ]

        class ranges(FanfanOmniFilteredCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.46]
            lin_vel_y = [-0.10, 0.10]
            ang_vel_yaw = [-0.90, 0.90]

    class rewards(FanfanOmniFilteredCfg.rewards):
        # Tighter sigmas make small velocity errors matter.  This is the main
        # correction for forward/side drift instead of merely asking for larger
        # commands.
        tracking_sigma = 0.012
        longitudinal_tracking_sigma = 0.0012
        lateral_tracking_sigma = 0.00025

        class scales(FanfanOmniFilteredCfg.rewards.scales):
            # Stronger velocity tracking: selected model_2000 was stable but
            # slow, e.g. 0.35 command produced ~0.23 m/s in Gym.
            tracking_lin_vel = 12.0
            tracking_longitudinal_vel = 10.0
            tracking_lateral_vel = 9.0
            tracking_ang_vel = 7.0

            # Direction lock: punish yaw drift when the command is translation
            # only.  Keep plain yaw_rate moderate so real turn commands are not
            # suppressed.
            heading_tracking = 4.8
            lateral_yaw_error = -15.0
            translation_yaw_error = -13.0
            yaw_rate = -0.22

            # Allow larger useful steps.  We still keep rate/filter penalties so
            # this does not become the old aggressive raw-output policy again.
            action_magnitude = -0.025
            policy_action_magnitude = -0.070
            action_saturation = -0.24
            policy_action_saturation = -0.48
            policy_filter_gap = -0.32

            # Smoothness remains meaningful, just not strong enough to make the
            # robot choose under-speed shuffling.
            action_rate = -0.18
            policy_action_rate = -0.28

            # Keep real safety terms.
            dof_acc = -3.0e-7
            torques = -5.0e-6


class FanfanOmniVelTrackV3CfgPPO(FanfanOmniFilteredCfgPPO):
    class algorithm(FanfanOmniFilteredCfgPPO.algorithm):
        # Slightly more exploration helps escape the slow local optimum, but
        # keep it low because this is continuation from a real-capable seed.
        entropy_coef = 0.0008

    class runner(FanfanOmniFilteredCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_veltrack_v3"
        run_name = "omni_veltrack_v3"


class FanfanOmniLateralFixCfg(FanfanOmniVelTrackV3Cfg):
    """Fix lateral and diagonal drift from the best veltrack_v3 policy.

    The selected veltrack_v3 model is good for forward/back/yaw, but lateral
    commands still under-track vy and diagonal commands still create yaw drift.
    This continuation therefore over-samples pure lateral and diagonal-style
    translation while explicitly penalizing:
      * vx drift during pure lateral motion;
      * yaw drift during diagonal translation with yaw_cmd = 0.
    """

    class control(FanfanOmniVelTrackV3Cfg.control):
        # Do not further enlarge stride here.  The problem is command
        # decomposition, not raw authority.  Keeping the envelope avoids
        # reintroducing aggressive sim2real outputs.
        policy_action_filter_alpha = 0.28

    class commands(FanfanOmniVelTrackV3Cfg.commands):
        resampling_time = 6.0

        # More samples where vy matters.  Keep enough sagittal/yaw replay so the
        # already-good forward/back/yaw behavior does not regress.
        stand_probability = 0.16
        pure_yaw_probability = 0.10
        pure_lateral_probability = 0.34
        pure_sagittal_probability = 0.18

        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 350,
                "lin_vel_x": [-0.06, 0.28],
                "lin_vel_y": [-0.10, 0.10],
                "ang_vel_yaw": [-0.55, 0.55],
            },
            {
                "until_iteration": 900,
                "lin_vel_x": [-0.10, 0.38],
                "lin_vel_y": [-0.11, 0.11],
                "ang_vel_yaw": [-0.75, 0.75],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.90, 0.90],
            },
        ]

        class ranges(FanfanOmniVelTrackV3Cfg.commands.ranges):
            lin_vel_x = [-0.12, 0.46]
            lin_vel_y = [-0.12, 0.12]
            ang_vel_yaw = [-0.90, 0.90]

    class rewards(FanfanOmniVelTrackV3Cfg.rewards):
        # Make vy errors more visible.  Keep x sigma close to v3 to avoid
        # harming forward/back.
        lateral_tracking_sigma = 0.00016
        longitudinal_tracking_sigma = 0.0012
        tracking_sigma = 0.010

        class scales(FanfanOmniVelTrackV3Cfg.rewards.scales):
            tracking_lin_vel = 13.0
            tracking_longitudinal_vel = 10.5
            tracking_lateral_vel = 13.0
            tracking_ang_vel = 7.0

            # Existing yaw locks plus two new targeted drift terms.
            lateral_yaw_error = -17.0
            translation_yaw_error = -15.0
            lateral_forward_drift = -16.0
            diagonal_yaw_error = -12.0

            # Do not punish hip motion too much during lateral learning.
            lateral_hip_common_mode = -0.55

            # Slightly relax action magnitude so the policy can discover true
            # lateral stepping; keep rate/filter terms to preserve smoothness.
            action_magnitude = -0.020
            policy_action_magnitude = -0.060
            action_rate = -0.18
            policy_action_rate = -0.28
            action_saturation = -0.22
            policy_action_saturation = -0.45
            policy_filter_gap = -0.30

            dof_acc = -3.0e-7
            torques = -5.0e-6


class FanfanOmniLateralFixCfgPPO(FanfanOmniVelTrackV3CfgPPO):
    class algorithm(FanfanOmniVelTrackV3CfgPPO.algorithm):
        entropy_coef = 0.0009

    class runner(FanfanOmniVelTrackV3CfgPPO.runner):
        experiment_name = "rough_fanfan_omni_lateral_fix"
        run_name = "omni_lateral_fix"


class FanfanOmniLateralSpeedCleanCfg(FanfanOmniLateralFixCfg):
    """Continue from lateral_fix 4850: more vy, cleaner straight yaw, less saturation.

    Do not use this as a broad redesign.  It is a conservative polishing stage:
    preserve the improved diagonal/yaw behavior from lateral_fix while making
    pure lateral movement a little faster and preventing the actor from solving
    it by driving raw outputs into saturation.
    """

    class control(FanfanOmniLateralFixCfg.control):
        # Keep the same physical action envelope.  If we enlarge action scale
        # here, vy may rise but raw saturation/real risk will rise too.
        policy_action_filter_alpha = 0.26

    class commands(FanfanOmniLateralFixCfg.commands):
        resampling_time = 6.0

        # Slightly more pure lateral than lateral_fix, but not so much that
        # forward/back/yaw replay is forgotten.
        stand_probability = 0.14
        pure_yaw_probability = 0.10
        pure_lateral_probability = 0.38
        pure_sagittal_probability = 0.20

        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 300,
                "lin_vel_x": [-0.04, 0.32],
                "lin_vel_y": [-0.11, 0.11],
                "ang_vel_yaw": [-0.55, 0.55],
            },
            {
                "until_iteration": 800,
                "lin_vel_x": [-0.08, 0.40],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.75, 0.75],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.90, 0.90],
            },
        ]

    class rewards(FanfanOmniLateralFixCfg.rewards):
        lateral_tracking_sigma = 0.00012
        longitudinal_tracking_sigma = 0.0012
        tracking_sigma = 0.010

        # Start saturation penalty earlier than lateral_fix.  This directly
        # attacks the ~48% raw_sat75 issue while still allowing moderate steps.
        action_saturation_threshold = 0.70

        class scales(FanfanOmniLateralFixCfg.rewards.scales):
            # More vy pressure, but not huge: last run showed direction improves
            # faster than speed, and too much lateral pressure increases raw sat.
            tracking_lin_vel = 13.5
            tracking_longitudinal_vel = 10.5
            tracking_lateral_vel = 16.0
            tracking_ang_vel = 7.0

            # Preserve lateral/diagonal fixes and add a straight-only yaw lock.
            lateral_forward_drift = -15.0
            diagonal_yaw_error = -12.0
            lateral_yaw_error = -16.0
            translation_yaw_error = -14.0
            straight_yaw_error = -12.0

            # Reduce raw saturation and keep real output smoother.  This is the
            # main difference from lateral_fix.
            action_magnitude = -0.030
            policy_action_magnitude = -0.095
            action_saturation = -0.42
            policy_action_saturation = -0.95
            policy_filter_gap = -0.40
            action_rate = -0.20
            policy_action_rate = -0.34

            lateral_hip_common_mode = -0.55
            dof_acc = -3.5e-7
            torques = -5.5e-6


class FanfanOmniLateralSpeedCleanCfgPPO(FanfanOmniLateralFixCfgPPO):
    class algorithm(FanfanOmniLateralFixCfgPPO.algorithm):
        # Lower entropy: this is polish from a good seed, not exploration.
        entropy_coef = 0.00045

    class runner(FanfanOmniLateralFixCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_lateral_speed_clean"
        run_name = "omni_lateral_speed_clean"


class FanfanOmniDesatTorqueCfg(FanfanOmniLateralSpeedCleanCfg):
    """Polish from speed_clean 4850: reduce raw saturation while guarding torque.

    Raw outputs stayed near 48% saturation.  A small action-scale increase lets
    the same joint target be represented with less normalized action.  The
    scale increase is intentionally tiny and is paired with stronger torque
    boundary penalties so the policy does not trade raw saturation for real
    actuator risk.
    """

    class control(FanfanOmniLateralSpeedCleanCfg.control):
        # Previous effective scales from veltrack/lateral stages were:
        # action=0.205, rear=0.225, hip=0.09.
        # Open only ~4-5%; hip barely changes to preserve straight/yaw quality.
        action_scale = 0.215
        rear_action_scale = 0.235
        hip_action_scale = 0.092

        # Slightly softer command-to-target filter than lateral_fix, but not as
        # slow as speed_clean's safest setting.  This keeps response without
        # requiring the raw actor to bang against tanh.
        policy_action_filter_alpha = 0.26

    class commands(FanfanOmniLateralSpeedCleanCfg.commands):
        # Preserve the current command distribution.  This is not another
        # lateral capability stage; it is a desaturation/torque polish stage.
        stand_probability = 0.15
        pure_yaw_probability = 0.10
        pure_lateral_probability = 0.36
        pure_sagittal_probability = 0.21

        omni_curriculum_stages = [
            {
                "until_iteration": 300,
                "lin_vel_x": [-0.06, 0.36],
                "lin_vel_y": [-0.11, 0.11],
                "ang_vel_yaw": [-0.65, 0.65],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.90, 0.90],
            },
        ]

    class rewards(FanfanOmniLateralSpeedCleanCfg.rewards):
        # Earlier saturation threshold remains useful with larger scale.
        action_saturation_threshold = 0.70

        # Start torque penalties before the hard 10Nm guard.
        torque_near_limit_ratio = 0.82
        peak_torque_soft_ratio = 0.88
        sustained_torque_ratio = 0.68

        class scales(FanfanOmniLateralSpeedCleanCfg.rewards.scales):
            # Preserve speed/yaw benefits from speed_clean.
            tracking_lin_vel = 13.5
            tracking_longitudinal_vel = 10.5
            tracking_lateral_vel = 16.0
            tracking_ang_vel = 7.0

            lateral_forward_drift = -15.0
            diagonal_yaw_error = -12.0
            lateral_yaw_error = -16.0
            translation_yaw_error = -14.0
            straight_yaw_error = -13.0

            # Desaturation: stronger than speed_clean, but paired with slightly
            # larger action scale so it should reduce raw usage instead of just
            # making the gait weak.
            action_magnitude = -0.035
            policy_action_magnitude = -0.120
            action_saturation = -0.55
            policy_action_saturation = -1.35
            policy_filter_gap = -0.42
            action_rate = -0.21
            policy_action_rate = -0.36

            # Torque guard: these scales are intentionally much stronger than
            # the base defaults because speed_clean 4850 touched 10Nm.
            torque_near_limit = -0.16
            peak_torque = -0.16
            sustained_torque = -0.22
            torque_clip = -0.35
            torques = -7.0e-6
            dof_acc = -3.8e-7


class FanfanOmniDesatTorqueCfgPPO(FanfanOmniLateralSpeedCleanCfgPPO):
    class algorithm(FanfanOmniLateralSpeedCleanCfgPPO.algorithm):
        # Very low exploration; this is a local polish from a selected model.
        entropy_coef = 0.0003

    class runner(FanfanOmniLateralSpeedCleanCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "omni_desat_torque"
