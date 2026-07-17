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




class FanfanOmniYawDriftCleanCfg(FanfanOmniDesatTorqueCfg):
    """Polish from desat_torque 5750: fix straight lateral drift and left lateral yaw.

    Main target from play CSV:
    - straight cmd vx=0.35 has unwanted vy drift around -0.031
    - left lateral cmd vy=0.07 has yaw_rate around 0.042
    Keep desaturation and torque safety from 5750.
    """

    class control(FanfanOmniDesatTorqueCfg.control):
        # Keep the desat action envelope. Do not enlarge again.
        action_scale = 0.215
        rear_action_scale = 0.235
        hip_action_scale = 0.092
        policy_action_filter_alpha = 0.26

    class commands(FanfanOmniDesatTorqueCfg.commands):
        # Slightly more straight and lateral replay.
        # Do not over-sample yaw, because current issue is not turn capability.
        stand_probability = 0.14
        pure_yaw_probability = 0.08
        pure_lateral_probability = 0.38
        pure_sagittal_probability = 0.24
        resampling_time = 6.0

        omni_curriculum_stages = [
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

        class ranges(FanfanOmniDesatTorqueCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.46]
            lin_vel_y = [-0.12, 0.12]
            ang_vel_yaw = [-0.85, 0.85]

    class rewards(FanfanOmniDesatTorqueCfg.rewards):
        # Keep desat threshold.
        action_saturation_threshold = 0.70

        # Keep torque protection from desat.
        torque_near_limit_ratio = 0.82
        peak_torque_soft_ratio = 0.88
        sustained_torque_ratio = 0.68

        class scales(FanfanOmniDesatTorqueCfg.rewards.scales):
            # Keep velocity tracking. Do not keep pushing speed harder.
            tracking_lin_vel = 13.5
            tracking_longitudinal_vel = 10.5
            tracking_lateral_vel = 16.0
            tracking_ang_vel = 7.0

            # Existing yaw/drift terms: small targeted increase only.
            straight_yaw_error = -15.0        # desat was -13.0
            translation_yaw_error = -14.5     # desat was -14.0
            lateral_yaw_error = -16.5         # desat was -16.0
            diagonal_yaw_error = -12.0
            lateral_forward_drift = -15.0

            # New targeted fixes.
            straight_lateral_drift = -18.0    # fix forward vy drift
            left_lateral_yaw_error = -10.0    # fix left lateral yaw

            # Do not over-penalize general yaw, or turning gets worse.
            yaw_rate = -0.22

            # Keep desaturation, but do not make it more conservative.
            action_magnitude = -0.035
            policy_action_magnitude = -0.120
            action_saturation = -0.55
            policy_action_saturation = -1.35
            policy_filter_gap = -0.42
            action_rate = -0.21
            policy_action_rate = -0.36

            # Keep torque guard.
            torque_near_limit = -0.16
            peak_torque = -0.16
            sustained_torque = -0.22
            torque_clip = -0.35
            torques = -7.0e-6
            dof_acc = -3.8e-7


class FanfanOmniYawDriftCleanCfgPPO(FanfanOmniDesatTorqueCfgPPO):
    class algorithm(FanfanOmniDesatTorqueCfgPPO.algorithm):
        # Keep exploration very low; this is local polishing from 5750.
        entropy_coef = 0.00025

    class runner(FanfanOmniDesatTorqueCfgPPO.runner):
        # Keep same experiment folder so resume from desat_torque run is easy.
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "yaw_drift_clean_from_5750"


class FanfanOmniYawSymmetryCfg(FanfanOmniYawDriftCleanCfg):
    """Continue from yaw-clean 5100 and improve straight-motion symmetry."""

    class control(FanfanOmniYawDriftCleanCfg.control):
        torque_limit_override = None
        torque_limits_by_joint = {
            "hip": 10.0,
            "thigh": 10.0,
            "calf": 13.0,
        }

    class commands(FanfanOmniYawDriftCleanCfg.commands):
        stand_probability = 0.12
        pure_yaw_probability = 0.06
        pure_lateral_probability = 0.18
        pure_sagittal_probability = 0.50
        resampling_time = 6.0

        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 400,
                "lin_vel_x": [0.06, 0.35],
                "lin_vel_y": [-0.03, 0.03],
                "ang_vel_yaw": [-0.20, 0.20],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniYawDriftCleanCfg.rewards):
        class scales(FanfanOmniYawDriftCleanCfg.rewards.scales):
            straight_policy_side_balance = -0.08
            straight_torque_side_balance = -0.18
            straight_diagonal_target_sync = -0.35


class FanfanOmniYawSymmetryCfgPPO(FanfanOmniYawDriftCleanCfgPPO):
    class algorithm(FanfanOmniYawDriftCleanCfgPPO.algorithm):
        entropy_coef = 0.00015

    class runner(FanfanOmniYawDriftCleanCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "yaw_symmetry_from_5100"


class FanfanOmniYawPathFixCfg(FanfanOmniYawSymmetryCfg):
    """Correct visible straight-path drift found in symmetry model 5400."""

    class commands(FanfanOmniYawSymmetryCfg.commands):
        # Spend the first 150 continuation iterations correcting the measured
        # straight-path bias, then replay the complete omni envelope.
        stand_probability = 0.12
        pure_yaw_probability = 0.06
        pure_lateral_probability = 0.14
        pure_sagittal_probability = 0.58
        resampling_time = 6.0

        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 150,
                "lin_vel_x": [0.12, 0.38],
                "lin_vel_y": [-0.01, 0.01],
                "ang_vel_yaw": [-0.08, 0.08],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniYawSymmetryCfg.rewards):
        class scales(FanfanOmniYawSymmetryCfg.rewards.scales):
            # Directly optimize the world-visible path drift while retaining
            # the existing body-frame vy and heading constraints.
            straight_path_lateral_velocity = -28.0
            straight_lateral_drift = -28.0
            straight_yaw_error = -13.0

            # The first symmetry run increased raw-action saturation from
            # about 52.6% to 59.7%. Keep the objectives, but stop rewarding a
            # high-energy solution merely because both sides use equal energy.
            straight_policy_side_balance = -0.03
            straight_torque_side_balance = -0.08
            straight_diagonal_target_sync = -0.18
            policy_action_magnitude = -0.13
            policy_action_saturation = -1.60


class FanfanOmniYawPathFixCfgPPO(FanfanOmniYawSymmetryCfgPPO):
    class algorithm(FanfanOmniYawSymmetryCfgPPO.algorithm):
        entropy_coef = 0.00010

    class runner(FanfanOmniYawSymmetryCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "yaw_path_fix_from_symmetry_5400"


class FanfanOmniDiagonalCoordCfg(FanfanOmniYawDriftCleanCfg):
    """Joint-level diagonal coordination continuation from yaw-clean 5100."""

    class control(FanfanOmniYawDriftCleanCfg.control):
        torque_limit_override = None
        torque_limits_by_joint = {
            "hip": 10.0,
            "thigh": 10.0,
            "calf": 13.0,
        }

    class commands(FanfanOmniYawDriftCleanCfg.commands):
        stand_probability = 0.10
        pure_yaw_probability = 0.06
        pure_lateral_probability = 0.14
        pure_sagittal_probability = 0.62
        resampling_time = 6.0

        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 300,
                "lin_vel_x": [0.08, 0.38],
                "lin_vel_y": [-0.015, 0.015],
                "ang_vel_yaw": [-0.10, 0.10],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniYawDriftCleanCfg.rewards):
        class scales(FanfanOmniYawDriftCleanCfg.rewards.scales):
            # The old front/rear hip pairing compares opposite trot phases.
            # Replace it and the ungated sagittal-only sync with physical
            # same-phase diagonal mirror objectives.
            hip_symmetry = 0.0
            diagonal_joint_sync = 0.0
            straight_diagonal_target_mirror = -1.20
            straight_diagonal_joint_mirror = -0.80
            straight_diagonal_torque_mirror = -0.12

            # Require both a square body and a straight world path. The path
            # term is deliberately weaker so yaw cannot compensate for vy.
            straight_lateral_drift = -32.0
            straight_yaw_error = -24.0
            straight_path_lateral_velocity = -8.0

            # Prevent the coordinated solution from increasing action energy.
            policy_action_magnitude = -0.13
            policy_action_saturation = -1.55


class FanfanOmniDiagonalCoordCfgPPO(FanfanOmniYawDriftCleanCfgPPO):
    class algorithm(FanfanOmniYawDriftCleanCfgPPO.algorithm):
        entropy_coef = 0.00015

    class runner(FanfanOmniYawDriftCleanCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "diagonal_coord_from_yaw_clean_5100"


class FanfanOmniCoordinatedStraightCfg(FanfanOmniDiagonalCoordCfg):
    """Remove residual side slip without allowing compensating body yaw."""

    class commands(FanfanOmniDiagonalCoordCfg.commands):
        stand_probability = 0.08
        pure_yaw_probability = 0.02
        pure_lateral_probability = 0.04
        pure_sagittal_probability = 0.82
        resampling_time = 6.0

        omni_curriculum_stages = [
            {
                "until_iteration": 200,
                "lin_vel_x": [0.15, 0.38],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniDiagonalCoordCfg.rewards):
        class scales(FanfanOmniDiagonalCoordCfg.rewards.scales):
            # Absolute-value terms retain useful gradient at the measured
            # 0.02-0.03 m/s side-slip level where exp/square terms stagnated.
            straight_lateral_speed = -18.0
            straight_heading_error = -6.0
            straight_lateral_drift = -36.0
            straight_yaw_error = -20.0
            straight_path_lateral_velocity = -4.0

            # Keep joint-level coordination, with a little more pressure on
            # physical targets than on raw torque equality.
            straight_diagonal_target_mirror = -1.40
            straight_diagonal_joint_mirror = -0.90
            straight_diagonal_torque_mirror = -0.10

            policy_action_magnitude = -0.14
            policy_action_saturation = -1.80


class FanfanOmniCoordinatedStraightCfgPPO(FanfanOmniDiagonalCoordCfgPPO):
    class algorithm(FanfanOmniDiagonalCoordCfgPPO.algorithm):
        entropy_coef = 0.00010

    class runner(FanfanOmniDiagonalCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "coordinated_straight_from_diagonal_5150"


class FanfanOmniProjectedCoordCfg(FanfanOmniDiagonalCoordCfg):
    """Structurally guarantee straight diagonal action coordination."""

    class control(FanfanOmniDiagonalCoordCfg.control):
        project_straight_diagonal_actions = True

    class domain_rand(FanfanOmniDiagonalCoordCfg.domain_rand):
        # Preserve robustness without injecting an artificial left/right
        # actuator mismatch that a deliberately symmetric controller cannot
        # and should not cancel with asymmetric actions.
        pair_diagonal_motor_strength = True

    class commands(FanfanOmniDiagonalCoordCfg.commands):
        stand_probability = 0.10
        pure_yaw_probability = 0.06
        pure_lateral_probability = 0.12
        pure_sagittal_probability = 0.66
        omni_curriculum_stages = [
            {
                "until_iteration": 200,
                "lin_vel_x": [0.10, 0.38],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniDiagonalCoordCfg.rewards):
        class scales(FanfanOmniDiagonalCoordCfg.rewards.scales):
            straight_lateral_drift = -48.0
            straight_yaw_error = -30.0
            straight_path_lateral_velocity = -8.0
            policy_action_magnitude = -0.14
            policy_action_saturation = -1.80


class FanfanOmniProjectedCoordCfgPPO(FanfanOmniDiagonalCoordCfgPPO):
    class algorithm(FanfanOmniDiagonalCoordCfgPPO.algorithm):
        entropy_coef = 0.00010

    class runner(FanfanOmniDiagonalCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "projected_coord_from_yaw_clean_5100"


class FanfanOmniStrongSymmetryCfg(FanfanOmniDiagonalCoordCfg):
    """High-signal soft symmetry training without rigid action projection."""

    class commands(FanfanOmniDiagonalCoordCfg.commands):
        stand_probability = 0.06
        pure_yaw_probability = 0.02
        pure_lateral_probability = 0.04
        pure_sagittal_probability = 0.86
        resampling_time = 6.0
        omni_curriculum_stages = [
            {
                "until_iteration": 150,
                "lin_vel_x": [0.15, 0.38],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniDiagonalCoordCfg.rewards):
        class scales(FanfanOmniDiagonalCoordCfg.rewards.scales):
            # Calibrated so the measured 0.02-0.03 m/s residual contributes
            # several reward units per second instead of less than one.
            straight_lateral_speed = -200.0
            straight_heading_error = -40.0
            straight_lateral_drift = -80.0
            straight_yaw_error = -40.0
            straight_path_lateral_velocity = -10.0

            straight_diagonal_target_mirror = -10.0
            straight_diagonal_joint_mirror = -6.0
            straight_diagonal_torque_mirror = -0.50

            policy_action_magnitude = -0.15
            policy_action_saturation = -2.00


class FanfanOmniStrongSymmetryCfgPPO(FanfanOmniDiagonalCoordCfgPPO):
    class algorithm(FanfanOmniDiagonalCoordCfgPPO.algorithm):
        entropy_coef = 0.00010

    class runner(FanfanOmniDiagonalCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "strong_symmetry_from_yaw_clean_5100"


class FanfanOmniNoCompSymmetryCfg(FanfanOmniDiagonalCoordCfg):
    """Symmetry polish that forbids the yaw/side-slip compensation shortcut."""

    class commands(FanfanOmniDiagonalCoordCfg.commands):
        stand_probability = 0.06
        pure_yaw_probability = 0.02
        pure_lateral_probability = 0.04
        pure_sagittal_probability = 0.86
        resampling_time = 6.0
        omni_curriculum_stages = [
            {
                "until_iteration": 150,
                "lin_vel_x": [0.15, 0.38],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniDiagonalCoordCfg.rewards):
        class scales(FanfanOmniDiagonalCoordCfg.rewards.scales):
            # Do not reward world-path cancellation. Require each underlying
            # error to independently approach zero.
            straight_path_lateral_velocity = 0.0
            straight_lateral_speed = -250.0
            straight_heading_error = -120.0
            straight_lateral_drift = -80.0
            straight_yaw_error = -80.0

            straight_diagonal_target_mirror = -5.0
            straight_diagonal_joint_mirror = -3.0
            straight_diagonal_torque_mirror = -0.20
            policy_action_magnitude = -0.14
            policy_action_saturation = -1.80


class FanfanOmniNoCompSymmetryCfgPPO(FanfanOmniDiagonalCoordCfgPPO):
    class algorithm(FanfanOmniDiagonalCoordCfgPPO.algorithm):
        entropy_coef = 0.0
        learning_rate = 1.0e-4
        desired_kl = 0.003

    class runner(FanfanOmniDiagonalCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "no_comp_symmetry_from_yaw_clean_5100"


class FanfanOmniHeadingBoundSymmetryCfg(FanfanOmniNoCompSymmetryCfg):
    """Reject compensating-yaw policies during straight symmetry training."""

    class rewards(FanfanOmniNoCompSymmetryCfg.rewards):
        terminate_straight_heading_error = 0.08

        class scales(FanfanOmniNoCompSymmetryCfg.rewards.scales):
            termination = -20.0
            straight_lateral_speed = -250.0
            straight_heading_error = -160.0
            straight_path_lateral_velocity = 0.0


class FanfanOmniHeadingBoundSymmetryCfgPPO(FanfanOmniNoCompSymmetryCfgPPO):
    class runner(FanfanOmniNoCompSymmetryCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "heading_bound_symmetry_from_yaw_clean_5100"


class FanfanOmniForceCoordCfg(FanfanOmniHeadingBoundSymmetryCfg):
    """Coordinate straight gait through ground reaction forces, not action equality."""

    class domain_rand(FanfanOmniHeadingBoundSymmetryCfg.domain_rand):
        # Independent, unobserved actuator asymmetry obscures whether the gait
        # itself is balanced. Validate actuator robustness separately.
        randomize_motor_strength = False

    class commands(FanfanOmniHeadingBoundSymmetryCfg.commands):
        stand_probability = 0.06
        pure_yaw_probability = 0.03
        pure_lateral_probability = 0.08
        pure_sagittal_probability = 0.78
        resampling_time = 6.0
        omni_curriculum_stages = [
            {
                "until_iteration": 120,
                "lin_vel_x": [0.12, 0.40],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniHeadingBoundSymmetryCfg.rewards):
        class scales(FanfanOmniHeadingBoundSymmetryCfg.rewards.scales):
            # The hard-projection experiment proved that equal targets can
            # produce unequal forces on this front/rear-asymmetric mechanism.
            straight_diagonal_target_mirror = -0.40
            straight_diagonal_joint_mirror = -0.25
            straight_diagonal_torque_mirror = -0.08

            straight_contact_lateral_force = -18.0
            straight_contact_yaw_moment = -24.0
            straight_contact_side_load_balance = -6.0
            straight_diagonal_contact_sync = -1.2

            straight_lateral_speed = -180.0
            straight_heading_error = -120.0
            straight_lateral_drift = -60.0
            straight_yaw_error = -60.0
            policy_action_magnitude = -0.15
            policy_action_saturation = -2.0


class FanfanOmniForceCoordCfgPPO(FanfanOmniHeadingBoundSymmetryCfgPPO):
    class algorithm(FanfanOmniHeadingBoundSymmetryCfgPPO.algorithm):
        learning_rate = 7.5e-5
        desired_kl = 0.0025

    class runner(FanfanOmniHeadingBoundSymmetryCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "force_coord_from_heading_bound_5100"


class FanfanOmniForceDesatCfg(FanfanOmniForceCoordCfg):
    """Retain force coordination while reducing normalized action saturation."""

    class control(FanfanOmniForceCoordCfg.control):
        action_scale = 0.232
        rear_action_scale = 0.254
        hip_action_scale = 0.100

    class commands(FanfanOmniForceCoordCfg.commands):
        omni_curriculum_stages = [
            {
                "until_iteration": 80,
                "lin_vel_x": [0.12, 0.40],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.46],
                "lin_vel_y": [-0.12, 0.12],
                "ang_vel_yaw": [-0.85, 0.85],
            },
        ]

    class rewards(FanfanOmniForceCoordCfg.rewards):
        action_saturation_threshold = 0.70

        class scales(FanfanOmniForceCoordCfg.rewards.scales):
            policy_action_magnitude = -0.24
            policy_action_saturation = -5.0


class FanfanOmniForceDesatCfgPPO(FanfanOmniForceCoordCfgPPO):
    class algorithm(FanfanOmniForceCoordCfgPPO.algorithm):
        learning_rate = 6.0e-5
        desired_kl = 0.002

    class runner(FanfanOmniForceCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "force_desat_from_force_coord_5250"


class FanfanOmniHighSpeedTransitionCfg(FanfanOmniForceCoordCfg):
    """Fast omnidirectional tracking with frequent mixed-command transitions."""

    class control(FanfanOmniForceCoordCfg.control):
        action_scale = 0.225
        rear_action_scale = 0.245
        hip_action_scale = 0.098
        policy_action_filter_alpha = 0.32
        policy_action_rate_limits = {
            "hip": 0.95 / 0.098,
            "thigh": 1.75 / 0.225,
            "calf": 2.70 / 0.225,
        }
        policy_action_accel_limits = {
            "hip": 34.0 / 0.098,
            "thigh": 58.0 / 0.225,
            "calf": 96.0 / 0.225,
        }

    class domain_rand(FanfanOmniForceCoordCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.65, 1.30]
        randomize_base_mass = True
        added_mass_range = [-0.30, 0.30]
        randomize_base_com = True
        base_com_x_range = [-0.010, 0.010]
        base_com_y_range = [-0.018, 0.018]
        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.15]
        pair_diagonal_motor_strength = False
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.20

    class commands(FanfanOmniForceCoordCfg.commands):
        resampling_time = 2.0
        stand_probability = 0.08
        pure_yaw_probability = 0.10
        pure_lateral_probability = 0.14
        pure_sagittal_probability = 0.18
        hard_transition_probability = 0.28
        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 180,
                "lin_vel_x": [-0.16, 0.50],
                "lin_vel_y": [-0.16, 0.16],
                "ang_vel_yaw": [-1.00, 1.00],
            },
            {
                "until_iteration": 420,
                "lin_vel_x": [-0.22, 0.56],
                "lin_vel_y": [-0.22, 0.22],
                "ang_vel_yaw": [-1.18, 1.18],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.25, 0.60],
                "lin_vel_y": [-0.26, 0.26],
                "ang_vel_yaw": [-1.30, 1.30],
            },
        ]

        class ranges(FanfanOmniForceCoordCfg.commands.ranges):
            lin_vel_x = [-0.25, 0.60]
            lin_vel_y = [-0.26, 0.26]
            ang_vel_yaw = [-1.30, 1.30]

    class rewards(FanfanOmniForceCoordCfg.rewards):
        tracking_sigma = 0.010
        longitudinal_tracking_sigma = 0.0010
        lateral_tracking_sigma = 0.00018
        command_transition_duration = 1.2
        terminate_straight_heading_error = 0.10

        class scales(FanfanOmniForceCoordCfg.rewards.scales):
            tracking_lin_vel = 18.0
            tracking_longitudinal_vel = 15.0
            tracking_lateral_vel = 18.0
            tracking_ang_vel = 12.0
            heading_tracking = 5.0
            planar_direction_error = -8.0
            command_transition_tracking = 7.0
            absolute_longitudinal_tracking_error = -12.0
            absolute_lateral_tracking_error = -14.0
            absolute_yaw_tracking_error = -8.0

            translation_yaw_error = -15.0
            lateral_yaw_error = -17.0
            diagonal_yaw_error = -13.0
            lateral_forward_drift = -14.0
            straight_lateral_speed = -110.0
            straight_heading_error = -80.0
            straight_lateral_drift = -45.0
            straight_yaw_error = -45.0

            diagonal_contact_sync_all = -1.5
            straight_diagonal_target_mirror = -0.30
            straight_diagonal_joint_mirror = -0.18
            straight_diagonal_torque_mirror = -0.06

            action_magnitude = -0.018
            policy_action_magnitude = -0.055
            action_rate = -0.16
            policy_action_rate = -0.24
            action_saturation = -0.25
            policy_action_saturation = -0.65
            policy_filter_gap = -0.25

            torque_near_limit = -0.18
            peak_torque = -0.18
            sustained_torque = -0.24
            torque_clip = -0.40
            torques = -7.0e-6


class FanfanOmniHighSpeedTransitionCfgPPO(FanfanOmniForceCoordCfgPPO):
    class algorithm(FanfanOmniForceCoordCfgPPO.algorithm):
        entropy_coef = 0.0007
        learning_rate = 1.0e-4
        desired_kl = 0.004

    class runner(FanfanOmniForceCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "high_speed_transition_from_force_coord_5280"


class FanfanOmniHighAuthorityTransitionCfg(FanfanOmniHighSpeedTransitionCfg):
    """Add physical stride authority after the high-speed actor saturates."""

    class control(FanfanOmniHighSpeedTransitionCfg.control):
        action_scale = 0.260
        rear_action_scale = 0.280
        hip_action_scale = 0.130
        policy_action_filter_alpha = 0.36
        policy_action_rate_limits = {
            "hip": 1.25 / 0.130,
            "thigh": 2.10 / 0.260,
            "calf": 3.10 / 0.260,
        }
        policy_action_accel_limits = {
            "hip": 46.0 / 0.130,
            "thigh": 72.0 / 0.260,
            "calf": 116.0 / 0.260,
        }

    class commands(FanfanOmniHighSpeedTransitionCfg.commands):
        resampling_time = 2.5
        hard_transition_probability = 0.25
        omni_curriculum_stages = [
            {
                "until_iteration": 140,
                "lin_vel_x": [-0.20, 0.54],
                "lin_vel_y": [-0.20, 0.20],
                "ang_vel_yaw": [-1.10, 1.10],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.25, 0.60],
                "lin_vel_y": [-0.26, 0.26],
                "ang_vel_yaw": [-1.30, 1.30],
            },
        ]

    class rewards(FanfanOmniHighSpeedTransitionCfg.rewards):
        class scales(FanfanOmniHighSpeedTransitionCfg.rewards.scales):
            absolute_longitudinal_tracking_error = -18.0
            absolute_lateral_tracking_error = -20.0
            absolute_yaw_tracking_error = -10.0
            planar_direction_error = -10.0
            command_transition_tracking = 8.0

            action_magnitude = -0.014
            policy_action_magnitude = -0.040
            action_saturation = -0.30
            policy_action_saturation = -0.90
            policy_filter_gap = -0.22

            torque_near_limit = -0.24
            peak_torque = -0.24
            sustained_torque = -0.30
            torque_clip = -0.50


class FanfanOmniHighAuthorityTransitionCfgPPO(
        FanfanOmniHighSpeedTransitionCfgPPO):
    class algorithm(FanfanOmniHighSpeedTransitionCfgPPO.algorithm):
        entropy_coef = 0.0015
        learning_rate = 2.0e-4
        desired_kl = 0.008

    class runner(FanfanOmniHighSpeedTransitionCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "high_authority_transition"


class FanfanOmniHighAuthorityDirectionCfg(
        FanfanOmniHighAuthorityTransitionCfg):
    """Retain high-speed authority while rejecting directional drift."""

    class rewards(FanfanOmniHighAuthorityTransitionCfg.rewards):
        terminate_straight_heading_error = 0.06
        terminate_translation_heading_error = 0.075

        class scales(FanfanOmniHighAuthorityTransitionCfg.rewards.scales):
            termination = -100.0
            translation_heading_error_abs = -220.0
            pure_lateral_forward_speed = -120.0
            planar_direction_error = -16.0

            straight_lateral_speed = -180.0
            straight_heading_error = -180.0
            straight_lateral_drift = -90.0
            straight_yaw_error = -120.0
            translation_yaw_error = -25.0
            lateral_yaw_error = -28.0
            diagonal_yaw_error = -22.0
            lateral_forward_drift = -22.0


class FanfanOmniHighAuthorityDirectionCfgPPO(
        FanfanOmniHighAuthorityTransitionCfgPPO):
    class algorithm(FanfanOmniHighAuthorityTransitionCfgPPO.algorithm):
        entropy_coef = 0.0002
        learning_rate = 8.0e-5
        desired_kl = 0.003

    class runner(FanfanOmniHighAuthorityTransitionCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "high_authority_direction"


class FanfanOmniHighAuthorityClosedLoopCfg(
        FanfanOmniHighAuthorityDirectionCfg):
    """Deployment-oriented outer velocity and heading feedback."""

    class control(FanfanOmniHighAuthorityDirectionCfg.control):
        command_feedback_longitudinal_gain = 0.9
        command_feedback_lateral_gain = 1.2
        command_feedback_yaw_gain = 0.45
        command_feedback_heading_gain = 1.8
        command_feedback_heading_damping = 0.65


class FanfanOmniHighAuthorityClosedLoopCfgPPO(
        FanfanOmniHighAuthorityDirectionCfgPPO):
    class runner(FanfanOmniHighAuthorityDirectionCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "high_authority_closed_loop"


class FanfanOmniHighCadenceCfg(FanfanOmniForceCoordCfg):
    """Probe a faster cadence without changing the selected actor weights."""

    class commands(FanfanOmniForceCoordCfg.commands):
        omni_curriculum = False

        class ranges(FanfanOmniForceCoordCfg.commands.ranges):
            lin_vel_x = [-0.25, 0.60]
            lin_vel_y = [-0.26, 0.26]
            ang_vel_yaw = [-1.30, 1.30]

    class rewards(FanfanOmniForceCoordCfg.rewards):
        gait_period = 0.45


class FanfanOmniHighCadenceCfgPPO(FanfanOmniForceCoordCfgPPO):
    class runner(FanfanOmniForceCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "high_cadence_probe"


class FanfanOmniSymmetricTransitionCfg(FanfanOmniForceCoordCfg):
    """High-cadence all-motion task with transition and hardware robustness."""

    class control(FanfanOmniForceCoordCfg.control):
        # Preserve the authority of the selected 5280 actor. The smaller
        # high-speed envelope was the main cause of forward under-tracking.
        action_scale = 0.215
        rear_action_scale = 0.235
        hip_action_scale = 0.115
        policy_action_filter_alpha = 0.30
        policy_action_rate_limits = {
            "hip": 1.05 / 0.115,
            "thigh": 1.70 / 0.215,
            "calf": 2.65 / 0.215,
        }
        policy_action_accel_limits = {
            "hip": 40.0 / 0.115,
            "thigh": 58.0 / 0.215,
            "calf": 94.0 / 0.215,
        }
        command_feedback_longitudinal_gain = 0.40
        command_feedback_lateral_gain = 0.80
        command_feedback_yaw_gain = 0.25
        command_feedback_heading_gain = 4.00
        command_feedback_heading_damping = 1.00
        command_feedback_diagonal_longitudinal_scale = 0.60
        enforce_policy_symmetry = True

    class domain_rand(FanfanOmniForceCoordCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.75, 1.20]
        randomize_base_mass = True
        added_mass_range = [-0.20, 0.20]
        randomize_base_com = True
        base_com_x_range = [-0.006, 0.006]
        base_com_y_range = [-0.010, 0.010]
        randomize_motor_strength = True
        motor_strength_range = [0.90, 1.10]
        pair_diagonal_motor_strength = False
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.14

    class commands(FanfanOmniForceCoordCfg.commands):
        resampling_time = 2.5
        stand_probability = 0.04
        pure_yaw_probability = 0.08
        pure_lateral_probability = 0.22
        pure_sagittal_probability = 0.14
        hard_transition_probability = 0.30
        omni_curriculum = False
        omni_curriculum_stages = [
            {
                "until_iteration": 100,
                "lin_vel_x": [-0.18, 0.50],
                "lin_vel_y": [-0.18, 0.18],
                "ang_vel_yaw": [-1.00, 1.00],
            },
            {
                "until_iteration": 240,
                "lin_vel_x": [-0.22, 0.56],
                "lin_vel_y": [-0.22, 0.22],
                "ang_vel_yaw": [-1.18, 1.18],
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.25, 0.60],
                "lin_vel_y": [-0.26, 0.26],
                "ang_vel_yaw": [-1.30, 1.30],
            },
        ]

        class ranges(FanfanOmniForceCoordCfg.commands.ranges):
            lin_vel_x = [-0.25, 0.60]
            lin_vel_y = [-0.26, 0.26]
            ang_vel_yaw = [-1.30, 1.30]

    class rewards(FanfanOmniForceCoordCfg.rewards):
        gait_period = 0.45
        tracking_sigma = 0.012
        longitudinal_tracking_sigma = 0.0015
        lateral_tracking_sigma = 0.00025
        command_transition_duration = 1.2
        terminate_straight_heading_error = 0.10

        class scales(FanfanOmniForceCoordCfg.rewards.scales):
            tracking_lin_vel = 18.0
            tracking_longitudinal_vel = 15.0
            tracking_lateral_vel = 25.0
            tracking_ang_vel = 11.0
            heading_tracking = 5.0
            absolute_longitudinal_tracking_error = -15.0
            absolute_lateral_tracking_error = -40.0
            absolute_yaw_tracking_error = -9.0
            planar_direction_error = -25.0
            command_transition_tracking = 8.0

            translation_heading_error_abs = -80.0
            pure_lateral_forward_speed = -80.0
            translation_yaw_error = -20.0
            lateral_yaw_error = -80.0
            diagonal_yaw_error = -17.0
            lateral_forward_drift = -18.0
            straight_lateral_speed = -180.0
            straight_heading_error = -140.0
            straight_lateral_drift = -75.0
            straight_yaw_error = -85.0
            diagonal_contact_sync_all = -1.5

            action_magnitude = -0.020
            policy_action_magnitude = -0.070
            action_rate = -0.18
            policy_action_rate = -0.25
            action_saturation = -0.30
            policy_action_saturation = -0.80
            policy_filter_gap = -0.28
            torque_near_limit = -0.20
            peak_torque = -0.20
            sustained_torque = -0.26
            torque_clip = -0.42


class FanfanOmniSymmetricTransitionCfgPPO(FanfanOmniForceCoordCfgPPO):
    class algorithm(FanfanOmniForceCoordCfgPPO.algorithm):
        entropy_coef = 0.0008
        learning_rate = 1.0e-4
        desired_kl = 0.0035

    class runner(FanfanOmniForceCoordCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "symmetric_transition_from_force_coord_5280"
        # The deployable actor is exactly symmetrized; this term keeps each
        # branch close while leaving PPO enough freedom to learn fast diagonals.
        symmetry_coef = 1.0


class FanfanOmniHardwareBalance5530Cfg(FanfanOmniSymmetricTransitionCfg):
    """Conservative all-motion continuation from the selected model_5530.

    Keep its forward gait and calf Kp=70, while replacing fast lateral throws
    and deep backward rear-leg folds with small, level and symmetric steps.
    """

    class control(FanfanOmniSymmetricTransitionCfg.control):
        # These are deliberately unchanged from model_5530. In particular,
        # calf stiffness stays at 70 and forward physical authority is kept.
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        action_scale = 0.215
        rear_action_scale = 0.235
        hip_action_scale = 0.115

        # Slightly stronger filtering/rate bounds soften command transitions
        # without changing the learned forward target scale.
        policy_action_filter_alpha = 0.27
        policy_action_rate_limits = {
            "hip": 0.78 / 0.115,
            "thigh": 1.35 / 0.215,
            "calf": 2.20 / 0.215,
        }
        policy_action_accel_limits = {
            "hip": 28.0 / 0.115,
            "thigh": 44.0 / 0.215,
            "calf": 72.0 / 0.215,
        }

        # Do not amplify the already-aggressive lateral command. Heading hold
        # remains active to reject drift in every translation direction.
        command_feedback_longitudinal_gain = 0.30
        command_feedback_lateral_gain = 0.20
        command_feedback_yaw_gain = 0.18
        command_feedback_heading_gain = 4.00
        command_feedback_heading_damping = 1.20
        command_feedback_diagonal_longitudinal_scale = 0.70
        enforce_policy_symmetry = True
        # Final target-layer guard for weak/lagged real rear calves. It is
        # command-gated and therefore leaves the selected forward gait intact.
        backward_rear_calf_target_min = -1.38

    class domain_rand(FanfanOmniSymmetricTransitionCfg.domain_rand):
        # Retain moderate sim-to-real robustness without forcing a violent gait.
        friction_range = [0.75, 1.20]
        added_mass_range = [-0.20, 0.20]
        base_com_x_range = [-0.006, 0.006]
        base_com_y_range = [-0.010, 0.010]
        motor_strength_range = [0.90, 1.10]
        push_interval_s = 7
        max_push_vel_xy = 0.12

    class commands(FanfanOmniSymmetricTransitionCfg.commands):
        # Forward 0.40 m/s stays in the replay set. Lateral and backward are
        # intentionally slow because real-hardware safety outranks raw speed.
        resampling_time = 3.0
        stand_probability = 0.08
        pure_yaw_probability = 0.10
        pure_lateral_probability = 0.28
        pure_sagittal_probability = 0.28
        hard_transition_probability = 0.12
        omni_curriculum = False

        class ranges(FanfanOmniSymmetricTransitionCfg.commands.ranges):
            lin_vel_x = [-0.12, 0.45]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.80, 0.80]

    class rewards(FanfanOmniSymmetricTransitionCfg.rewards):
        gait_period = 0.45
        swing_height_target = 0.050
        swing_height_sigma = 0.00035
        backward_rear_calf_soft_limit = -1.55
        rear_calf_fold_limit = -1.80
        transition_smooth_duration = 0.60
        command_transition_duration = 1.2
        terminate_straight_heading_error = 0.10
        terminate_translation_heading_error = 0.14

        class scales(FanfanOmniSymmetricTransitionCfg.rewards.scales):
            # Preserve forward tracking but remove the incentive to lunge
            # sideways or backwards merely to eliminate a small speed error.
            tracking_lin_vel = 15.0
            tracking_longitudinal_vel = 14.0
            tracking_lateral_vel = 12.0
            tracking_ang_vel = 9.0
            absolute_longitudinal_tracking_error = -10.0
            absolute_lateral_tracking_error = -10.0
            absolute_yaw_tracking_error = -6.0
            planar_direction_error = -18.0
            command_transition_tracking = 3.0

            # Direction and drift control in every translational mode.
            translation_heading_error_abs = -100.0
            pure_lateral_forward_speed = -80.0
            translation_yaw_error = -24.0
            lateral_yaw_error = -45.0
            diagonal_yaw_error = -20.0
            lateral_forward_drift = -24.0
            straight_lateral_speed = -180.0
            straight_heading_error = -140.0
            straight_lateral_drift = -75.0
            straight_yaw_error = -85.0

            # Balance, diagonal timing and sufficient toe clearance produce a
            # visibly coordinated trot instead of four independent legs.
            orientation = -8.0
            ang_vel_xy = -1.8
            base_height = -14.0
            low_base_height = -24.0
            rear_sit = -24.0
            rear_calf_fold = -12.0
            rear_load_bias = -3.0
            rear_leg_posture = -3.0
            diagonal_gait = 7.0
            swing_height = 0.8
            diagonal_joint_sync = -0.9
            diagonal_contact_sync_all = -3.0
            diagonal_foot_height_sync_all = -120.0
            translation_roll = -35.0
            lateral_roll = -80.0
            lateral_action_magnitude = -0.30
            backward_rear_calf_fold = -80.0
            backward_rear_action = -0.12
            transition_action_rate = -0.45
            feet_contact_forces = -0.001

            # A conservative, recoverable motion envelope for real hardware.
            action_magnitude = -0.030
            policy_action_magnitude = -0.095
            action_rate = -0.25
            policy_action_rate = -0.36
            action_saturation = -0.45
            policy_action_saturation = -1.10
            policy_filter_gap = -0.38
            dof_acc = -4.0e-7
            torque_near_limit = -0.24
            peak_torque = -0.24
            sustained_torque = -0.30
            torque_clip = -0.50


class FanfanOmniHardwareBalance5530CfgPPO(
        FanfanOmniSymmetricTransitionCfgPPO):
    class algorithm(FanfanOmniSymmetricTransitionCfgPPO.algorithm):
        entropy_coef = 0.0
        learning_rate = 3.0e-5
        desired_kl = 0.0015

    class runner(FanfanOmniSymmetricTransitionCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "hardware_balance_safe_from_5530"
        symmetry_coef = 1.5


class FanfanOmniHardwareBalance5530V2Cfg(FanfanOmniHardwareBalance5530Cfg):
    """Final rear-knee recovery polish from the best first-stage checkpoint."""

    class rewards(FanfanOmniHardwareBalance5530Cfg.rewards):
        # model_5800 reaches about -1.42 rad in simulation. Start penalizing
        # before that point so real actuator lag cannot turn it into a squat.
        backward_rear_target_soft_limit = -1.36

        class scales(FanfanOmniHardwareBalance5530Cfg.rewards.scales):
            backward_rear_target_fold = -140.0
            backward_rear_calf_fold = -100.0
            backward_rear_action = -0.16


class FanfanOmniHardwareBalance5530V2CfgPPO(
        FanfanOmniHardwareBalance5530CfgPPO):
    class algorithm(FanfanOmniHardwareBalance5530CfgPPO.algorithm):
        learning_rate = 1.5e-5
        desired_kl = 0.001

    class runner(FanfanOmniHardwareBalance5530CfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "hardware_balance_rear_recovery_from_5800"
        symmetry_coef = 1.5


class FanfanOmniRealDataCurriculumCfg(FanfanOmniHardwareBalance5530V2Cfg):
    """Real-CSV-driven actuator, estimator and command curriculum."""

    class control(FanfanOmniHardwareBalance5530V2Cfg.control):
        # The real and simulated controller are both exactly 50 Hz with the
        # deployed PD and strict per-joint hardware ceilings.
        decimation = 4
        stiffness = {"hip": 60.0, "thigh": 70.0, "calf": 70.0}
        damping = {"hip": 1.2, "thigh": 1.6, "calf": 1.6}
        torque_limits_by_joint = {"hip": 10.0, "thigh": 10.0, "calf": 13.0}
        action_scale = 0.215
        rear_action_scale = 0.235
        hip_action_scale = 0.115

        filter_policy_actions = True
        policy_action_filter_alpha = 0.26
        policy_action_filter_alpha_range = [0.22, 0.30]

        # Gait amplitude is now genuinely small at low speed and grows
        # continuously. Values are physical calf radians, not action units.
        use_continuous_gait_scaling = True
        gait_equivalent_speed_weights = [1.0, 1.5, 0.18]
        gait_speed_knots = [0.0, 0.01, 0.02, 0.05, 0.12, 0.20, 0.30]
        gait_calf_amplitude_knots = [
            0.0, 0.0, 0.020, 0.040, 0.085, 0.140, 0.200
        ]

        # Limit default + filtered actor + gait, never the actor alone.
        final_target_rate_limits_initial = {
            "hip": 0.60, "thigh": 0.85, "calf": 1.20
        }
        final_target_accel_limits_initial = {
            "hip": 16.0, "thigh": 24.0, "calf": 34.0
        }
        final_target_rate_limits_final = {
            "hip": 0.85, "thigh": 1.20, "calf": 1.80
        }
        final_target_accel_limits_final = {
            "hip": 25.0, "thigh": 38.0, "calf": 55.0
        }
        rear_calf_target_rate_scale = 0.90
        final_target_limit_open_start_iteration = 250
        final_target_limit_open_end_iteration = 1000

        # A 5 ms internal actuator update reproduces the measured 80-100 ms
        # response while policy commands still arrive only every 20 ms.
        use_real_actuator_model = True
        actuator_time_constant_ranges = {
            "hip": [0.035, 0.070],
            "thigh": [0.045, 0.090],
            "calf": [0.055, 0.110],
        }
        command_delay_range_s = [0.005, 0.015]
        command_delay_slow_probability = 0.10
        command_delay_slow_range_s = [0.015, 0.030]
        command_delay_max_s = 0.030

        # Training authority is sometimes lower, but can never exceed the
        # actual 10/10/13 Nm contract.
        training_torque_limit_ranges = {
            "hip": [8.5, 10.0],
            "thigh": [8.5, 10.0],
            "calf": [11.0, 13.0],
        }

    class domain_rand(FanfanOmniHardwareBalance5530V2Cfg.domain_rand):
        randomize_friction = True
        friction_range = [0.55, 1.20]
        low_friction_probability = 0.10
        low_friction_range = [0.35, 0.55]
        randomize_foot_friction_independent = True
        independent_shape_friction_fraction = 0.15

        randomize_base_mass = False
        base_mass_fraction_range = [0.95, 1.05]
        randomize_base_com = True
        base_com_x_range = [-0.015, 0.015]
        base_com_y_range = [-0.012, 0.012]
        base_com_z_range = [-0.008, 0.008]

        randomize_motor_strength = True
        motor_strength_range = [0.85, 1.05]
        rear_calf_strength_range = [0.80, 1.03]
        pair_diagonal_motor_strength = False
        kp_multiplier_range = [0.80, 1.05]
        kd_multiplier_range = [0.75, 1.25]
        joint_zero_offset_ranges = {
            "hip": [-0.012, 0.012],
            "thigh": [-0.015, 0.015],
            "calf": [-0.025, 0.025],
        }
        joint_backlash_ranges = {
            "hip": [0.005, 0.015],
            "thigh": [0.005, 0.015],
            "calf": [0.010, 0.025],
        }
        randomize_joint_friction_damping = True
        joint_friction_multiplier_range = [0.70, 1.30]
        joint_damping_multiplier_range = [0.70, 1.30]

        gait_calf_amplitude_max_range = [0.17, 0.22]
        gait_stance_ratio_range = [0.60, 0.68]
        gait_low_speed_period_range = [0.65, 0.85]
        gait_high_speed_period_range = [0.48, 0.60]
        gait_backward_scale_range = [0.75, 0.90]

        randomize_initial_tilt = True
        initial_tilt_range_rad = [-0.05, 0.05]
        initial_velocity_max = 0.08
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.14

    class noise(FanfanOmniHardwareBalance5530V2Cfg.noise):
        # Disable the old broad uniform noise and use the measured stochastic
        # model below. Observation dimensionality remains exactly 52.
        add_noise = False
        use_real_observation_model = True
        joint_position_noise_sigma = 0.0015
        joint_velocity_noise_sigma = 0.10
        joint_velocity_noise_clip = 0.30
        gravity_xy_noise_sigma = 0.007
        imu_ang_vel_noise_sigma = 0.015
        imu_install_bias_max_rad = 0.05

        observation_delay_range_s = [0.015, 0.030]
        observation_delay_slow_probability = 0.10
        observation_delay_slow_range_s = [0.030, 0.050]
        observation_delay_rare_probability = 0.02
        observation_delay_rare_range_s = [0.040, 0.060]
        observation_delay_max_s = 0.060

        lin_vel_white_noise_sigma = 0.03
        lin_vel_episode_bias_max = 0.04
        lin_vel_random_walk_sigma_per_s = 0.02
        lin_vel_random_walk_clip = 0.06
        velocity_slip_event_probability_per_step = 0.0015
        velocity_slip_error_range = [0.10, 0.25]
        velocity_slip_duration_steps = [1, 3]
        velocity_hold_event_probability_per_step = 0.003
        velocity_hold_duration_steps = [1, 3]
        velocity_zero_fraction = 0.35
        contact_misclassification_probability = 0.025
        contact_velocity_error_scale = 0.12

    class commands(FanfanOmniHardwareBalance5530V2Cfg.commands):
        resampling_time = 4.0
        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 150,
                "lin_vel_x": [0.0, 0.12],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "stand_probability": 0.50,
                "pure_yaw_probability": 0.0,
                "pure_lateral_probability": 0.0,
                "pure_sagittal_probability": 0.50,
                "hard_transition_probability": 0.0,
            },
            {
                "until_iteration": 400,
                "lin_vel_x": [0.05, 0.30],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "stand_probability": 0.25,
                "pure_yaw_probability": 0.0,
                "pure_lateral_probability": 0.0,
                "pure_sagittal_probability": 0.75,
                "hard_transition_probability": 0.0,
            },
            {
                "until_iteration": 700,
                "lin_vel_x": [-0.08, 0.35],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-0.50, 0.50],
                "stand_probability": 0.15,
                "pure_yaw_probability": 0.20,
                "pure_lateral_probability": 0.0,
                "pure_sagittal_probability": 0.55,
                "hard_transition_probability": 0.05,
            },
            {
                "until_iteration": 1050,
                "lin_vel_x": [-0.08, 0.35],
                "lin_vel_y": [-0.04, 0.04],
                "ang_vel_yaw": [-0.50, 0.50],
                "stand_probability": 0.12,
                "pure_yaw_probability": 0.16,
                "pure_lateral_probability": 0.25,
                "pure_sagittal_probability": 0.32,
                "hard_transition_probability": 0.10,
            },
            {
                "until_iteration": 1350,
                "lin_vel_x": [-0.10, 0.40],
                "lin_vel_y": [-0.06, 0.06],
                "ang_vel_yaw": [-0.65, 0.65],
                "stand_probability": 0.10,
                "pure_yaw_probability": 0.15,
                "pure_lateral_probability": 0.25,
                "pure_sagittal_probability": 0.28,
                "hard_transition_probability": 0.18,
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.45],
                "lin_vel_y": [-0.08, 0.08],
                "ang_vel_yaw": [-0.80, 0.80],
                "stand_probability": 0.08,
                "pure_yaw_probability": 0.14,
                "pure_lateral_probability": 0.25,
                "pure_sagittal_probability": 0.25,
                "hard_transition_probability": 0.30,
            },
        ]

        class ranges(FanfanOmniHardwareBalance5530V2Cfg.commands.ranges):
            lin_vel_x = [-0.12, 0.45]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.80, 0.80]

    class rewards(FanfanOmniHardwareBalance5530V2Cfg.rewards):
        gait_period = 0.54
        gait_stance_ratio = 0.64
        gait_calf_amplitude = -0.20
        torque_curriculum = False
        torque_near_limit_ratio = 0.68
        peak_torque_soft_ratio = 0.82
        sustained_torque_ratio = 0.60
        torque_ema_alpha = 0.99
        pd_pos_err_soft_limit = {
            "hip": 0.08, "thigh": 0.10, "calf": 0.12
        }

        enable_actuator_safety_termination = True
        actuator_safety_grace_steps = 12
        raw_torque_termination_ratio = 1.15
        raw_torque_termination_steps = 2
        torque_saturation_window_steps = 25
        torque_saturation_window_ratio = 0.08
        calf_error_termination_rad = 0.25
        calf_error_termination_steps = 3

        class scales(FanfanOmniHardwareBalance5530V2Cfg.rewards.scales):
            termination = -120.0
            torques = -1.5e-5
            torque_near_limit = -1.00
            peak_torque = -1.50
            sustained_torque = -1.30
            sustained_torque_max = -1.80
            torque_clip = -2.50
            mechanical_power = -0.002
            pd_position_error_over_limit = -1.20
            motor_target_tracking_error = -0.30

            final_target_velocity = -0.55
            final_target_acceleration = -0.22
            action_rate = -0.35
            policy_action_rate = -0.45
            dof_acc = -7.0e-7
            feet_contact_forces = -0.002


class FanfanOmniRealDataCurriculumCfgPPO(
        FanfanOmniHardwareBalance5530V2CfgPPO):
    class algorithm(FanfanOmniHardwareBalance5530V2CfgPPO.algorithm):
        entropy_coef = 0.0002
        learning_rate = 2.0e-5
        desired_kl = 0.0012

    class runner(FanfanOmniHardwareBalance5530V2CfgPPO.runner):
        # Keep the source experiment root so TaskRegistry can resume the
        # selected 5550 checkpoint without copying or relabelling it.
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "realdata_curriculum_from_best_5550"
        max_iterations = 1800
        save_interval = 50
        symmetry_coef = 1.5


class FanfanOmniRealDataSpeedPolishCfg(FanfanOmniRealDataCurriculumCfg):
    """All-direction speed recovery under the unchanged real-data contract."""

    class control(FanfanOmniRealDataCurriculumCfg.control):
        # The deployment filter is fully open in this continuation. All
        # hardware authority, bandwidth and 10/10/13 Nm limits stay unchanged.
        final_target_rate_limits_initial = {
            "hip": 0.85, "thigh": 1.20, "calf": 2.00
        }
        final_target_accel_limits_initial = {
            "hip": 25.0, "thigh": 38.0, "calf": 60.0
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_final = final_target_accel_limits_initial
        rear_calf_target_rate_scale = 0.95
        final_target_limit_open_start_iteration = 0
        final_target_limit_open_end_iteration = 1

    class commands(FanfanOmniRealDataCurriculumCfg.commands):
        omni_curriculum = False
        resampling_time = 3.0
        stand_probability = 0.08
        pure_yaw_probability = 0.16
        pure_lateral_probability = 0.26
        pure_sagittal_probability = 0.30
        hard_transition_probability = 0.30

    class rewards(FanfanOmniRealDataCurriculumCfg.rewards):
        class scales(FanfanOmniRealDataCurriculumCfg.rewards.scales):
            # Reject the safe-but-nearly-stationary local optimum while keeping
            # every direction, symmetry, posture and saturation penalty.
            tracking_lin_vel = 28.0
            tracking_longitudinal_vel = 26.0
            tracking_lateral_vel = 24.0
            tracking_ang_vel = 20.0
            absolute_longitudinal_tracking_error = -24.0
            absolute_lateral_tracking_error = -22.0
            absolute_yaw_tracking_error = -14.0
            command_transition_tracking = 6.0

            # Permit the small additional residual motion needed to track, but
            # retain the complete final-target velocity/acceleration limiter.
            action_magnitude = -0.020
            policy_action_magnitude = -0.060
            action_rate = -0.28
            policy_action_rate = -0.36


class FanfanOmniRealDataSpeedPolishCfgPPO(
        FanfanOmniRealDataCurriculumCfgPPO):
    class algorithm(FanfanOmniRealDataCurriculumCfgPPO.algorithm):
        learning_rate = 1.5e-5
        desired_kl = 0.0010

    class runner(FanfanOmniRealDataCurriculumCfgPPO.runner):
        run_name = "realdata_speed_polish_from_7330"
        max_iterations = 500
        save_interval = 50
        symmetry_coef = 1.5


class FanfanOmniRealDataCoordinatedCfg(FanfanOmniRealDataCurriculumCfg):
    """Recover useful, high-clearance motion without weakening real safety."""

    class control(FanfanOmniRealDataCurriculumCfg.control):
        # Start and finish at the measured deployable bandwidth.  The previous
        # curriculum had already opened these limits before this continuation.
        final_target_rate_limits_initial = {
            "hip": 0.85, "thigh": 1.20, "calf": 1.80
        }
        final_target_accel_limits_initial = {
            "hip": 25.0, "thigh": 38.0, "calf": 55.0
        }
        final_target_limit_open_start_iteration = 0
        final_target_limit_open_end_iteration = 1

        # Keep 11.md's continuous calf envelope, but give the trot a small
        # coordinated thigh sweep instead of asking four independent policy
        # outputs to invent the entire stride under delayed observations.
        gait_calf_amplitude_knots = [
            0.0, 0.0, 0.022, 0.045, 0.095, 0.155, 0.220
        ]
        enforce_swing_calf_reference = True
        # Start the target before nominal lift-off so the measured command
        # delay + first-order actuator lag puts physical toe lift in mid-swing.
        gait_target_phase_lead = 0.14
        # This is pre-limiter reference shaping, not extra motor authority.
        swing_calf_reference_scale = 2.5
        front_swing_calf_reference_scale = 4.0
        rear_swing_calf_reference_scale = 2.5
        enforce_stance_leg_extension = True
        preserve_forward_gait = True
        stance_calf_extension = 0.06
        stance_thigh_extension = -0.02

    class domain_rand(FanfanOmniRealDataCurriculumCfg.domain_rand):
        gait_calf_amplitude_max_range = [0.20, 0.22]
        gait_stance_ratio_range = [0.58, 0.62]
        gait_low_speed_period_range = [0.62, 0.72]
        gait_high_speed_period_range = [0.56, 0.60]

    class commands(FanfanOmniRealDataCurriculumCfg.commands):
        resampling_time = 4.0
        omni_curriculum = True
        omni_curriculum_stages = [
            {
                "until_iteration": 250,
                "lin_vel_x": [0.05, 0.40],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
                "stand_probability": 0.12,
                "pure_yaw_probability": 0.0,
                "pure_lateral_probability": 0.0,
                "pure_sagittal_probability": 0.88,
                "hard_transition_probability": 0.05,
            },
            {
                "until_iteration": 550,
                "lin_vel_x": [-0.10, 0.42],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [-0.60, 0.60],
                "stand_probability": 0.10,
                "pure_yaw_probability": 0.24,
                "pure_lateral_probability": 0.0,
                "pure_sagittal_probability": 0.58,
                "hard_transition_probability": 0.12,
            },
            {
                "until_iteration": 900,
                "lin_vel_x": [-0.10, 0.45],
                "lin_vel_y": [-0.06, 0.06],
                "ang_vel_yaw": [-0.75, 0.75],
                "stand_probability": 0.08,
                "pure_yaw_probability": 0.18,
                "pure_lateral_probability": 0.28,
                "pure_sagittal_probability": 0.30,
                "hard_transition_probability": 0.22,
            },
            {
                "until_iteration": 1.0e12,
                "lin_vel_x": [-0.12, 0.45],
                "lin_vel_y": [-0.08, 0.08],
                "ang_vel_yaw": [-0.80, 0.80],
                "stand_probability": 0.08,
                "pure_yaw_probability": 0.16,
                "pure_lateral_probability": 0.28,
                "pure_sagittal_probability": 0.28,
                "hard_transition_probability": 0.30,
            },
        ]

    class rewards(FanfanOmniRealDataCurriculumCfg.rewards):
        # Preserve the selected policy's thigh trajectory.  Clearance is
        # learned through the phase-aware reward rather than a hard reference
        # that destabilizes the source gait during continuation.
        gait_thigh_amplitude = 0.0
        swing_height_target = 0.052
        swing_height_sigma = 0.00022
        swing_clearance_minimum = 0.045

        class scales(FanfanOmniRealDataCurriculumCfg.rewards.scales):
            # Dense progress eliminates the nearly-stationary local optimum.
            command_velocity_progress = 12.0
            normalized_command_tracking = 10.0
            tracking_lin_vel = 22.0
            tracking_longitudinal_vel = 20.0
            tracking_lateral_vel = 18.0
            tracking_ang_vel = 15.0
            absolute_longitudinal_tracking_error = -15.0
            absolute_lateral_tracking_error = -14.0
            absolute_yaw_tracking_error = -9.0
            command_transition_tracking = 5.0

            # Clearance and diagonal timing are explicit acceptance targets.
            swing_height = 8.0
            swing_clearance_shortfall = -320.0
            diagonal_gait = 10.0
            diagonal_contact_sync_all = -5.0
            diagonal_foot_height_sync_all = -160.0
            diagonal_joint_sync = -1.2

            # Avoid suppressing useful motion; the final-target hard limits and
            # strict torque guards below remain fully active.
            action_magnitude = -0.016
            policy_action_magnitude = -0.045
            action_rate = -0.22
            policy_action_rate = -0.30
            final_target_velocity = -0.38
            final_target_acceleration = -0.16
            dof_acc = -4.5e-7

            torque_near_limit = -1.00
            peak_torque = -1.50
            sustained_torque = -1.30
            sustained_torque_max = -1.80
            torque_clip = -2.50


class FanfanOmniRealDataCoordinatedCfgPPO(
        FanfanOmniRealDataCurriculumCfgPPO):
    class algorithm(FanfanOmniRealDataCurriculumCfgPPO.algorithm):
        entropy_coef = 0.0020
        learning_rate = 3.0e-5
        desired_kl = 0.0015

    class runner(FanfanOmniRealDataCurriculumCfgPPO.runner):
        run_name = "realdata_coordinated_clearance_v2_from_7500"
        max_iterations = 1200
        save_interval = 50
        symmetry_coef = 2.0


class FanfanOmniRealDataClearancePolishCfg(
        FanfanOmniRealDataCoordinatedCfg):
    """Full-envelope polish with a guaranteed coordinated swing-lift floor."""

    class commands(FanfanOmniRealDataCoordinatedCfg.commands):
        omni_curriculum = False
        resampling_time = 3.5
        stand_probability = 0.08
        pure_yaw_probability = 0.16
        pure_lateral_probability = 0.28
        pure_sagittal_probability = 0.28
        hard_transition_probability = 0.30

    class rewards(FanfanOmniRealDataCoordinatedCfg.rewards):
        base_height_target = 0.292
        min_base_height_soft = 0.268
        gait_thigh_amplitude = 0.0

        class scales(FanfanOmniRealDataCoordinatedCfg.rewards.scales):
            swing_height = 14.0
            swing_clearance_shortfall = -700.0
            swing_contact = -10.0
            diagonal_gait = 12.0
            diagonal_contact_sync_all = -7.0
            base_height = -30.0
            low_base_height = -60.0
            stand_height = 3.0


class FanfanOmniRealDataClearancePolishCfgPPO(
        FanfanOmniRealDataCoordinatedCfgPPO):
    class algorithm(FanfanOmniRealDataCoordinatedCfgPPO.algorithm):
        entropy_coef = 0.0010
        learning_rate = 2.0e-5
        desired_kl = 0.0012

    class runner(FanfanOmniRealDataCoordinatedCfgPPO.runner):
        run_name = "realdata_clearance_polish_from_coordinated"
        max_iterations = 400
        save_interval = 50
        symmetry_coef = 2.5


class FanfanOmniRealDataDirectionalPolishCfg(
        FanfanOmniRealDataClearancePolishCfg):
    """Recover lateral/yaw authority while retaining coordinated clearance."""

    class commands(FanfanOmniRealDataClearancePolishCfg.commands):
        omni_curriculum = False
        resampling_time = 3.0
        stand_probability = 0.05
        pure_yaw_probability = 0.18
        pure_lateral_probability = 0.42
        pure_sagittal_probability = 0.22
        hard_transition_probability = 0.32

    class rewards(FanfanOmniRealDataClearancePolishCfg.rewards):
        class scales(FanfanOmniRealDataClearancePolishCfg.rewards.scales):
            command_velocity_progress = 28.0
            normalized_command_tracking = 18.0
            tracking_lin_vel = 26.0
            tracking_longitudinal_vel = 22.0
            tracking_lateral_vel = 40.0
            tracking_ang_vel = 24.0
            absolute_longitudinal_tracking_error = -18.0
            absolute_lateral_tracking_error = -28.0
            absolute_yaw_tracking_error = -14.0
            command_transition_tracking = 8.0

            # Lateral hip work is necessary for real translation; balance is
            # enforced by roll, contact timing and torque constraints instead.
            lateral_action_magnitude = -0.08
            action_magnitude = -0.012
            policy_action_magnitude = -0.035


class FanfanOmniRealDataDirectionalPolishCfgPPO(
        FanfanOmniRealDataClearancePolishCfgPPO):
    class algorithm(FanfanOmniRealDataClearancePolishCfgPPO.algorithm):
        entropy_coef = 0.0015
        learning_rate = 3.0e-5
        desired_kl = 0.0015

    class runner(FanfanOmniRealDataClearancePolishCfgPPO.runner):
        run_name = "realdata_directional_polish_from_8630"
        max_iterations = 500
        save_interval = 50
        symmetry_coef = 2.5


class FanfanOmniRealDataPerformanceRecoveryCfg(
        FanfanOmniRealDataCurriculumCfg):
    """Retain 5530's useful gait while adapting it to the real actuator chain.

    The long conservative continuation from 5530 converged to a nearly static
    solution.  This branch restarts from the last policy with genuine
    all-direction authority, then learns the measured delay/randomization and
    strict torque contract without a stance clamp that fights translation.
    """

    class control(FanfanOmniRealDataCurriculumCfg.control):
        final_target_rate_limits_initial = {
            "hip": 0.90, "thigh": 1.30, "calf": 3.50
        }
        final_target_accel_limits_initial = {
            "hip": 28.0, "thigh": 42.0, "calf": 80.0
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_final = final_target_accel_limits_initial
        rear_calf_target_rate_scale = 0.92
        final_target_limit_open_start_iteration = 0
        final_target_limit_open_end_iteration = 1

        # A moderate phase-leading calf floor compensates measured actuator
        # lag, while leaving hip/thigh placement to the proven 5530 actor.
        gait_calf_amplitude_knots = [
            0.0, 0.0, 0.020, 0.042, 0.090, 0.150, 0.215
        ]
        gait_target_phase_lead = 0.12
        enforce_swing_calf_reference = True
        swing_calf_reference_scale = 3.50
        front_swing_calf_reference_scale = 3.50
        rear_swing_calf_reference_scale = 4.00
        enforce_stance_leg_extension = False
        preserve_forward_gait = False

        # Training and export must see the same command observation.  Legacy
        # post-export feedback was never present inside PPO and caused a
        # policy/deployment mismatch.
        command_feedback_longitudinal_gain = 0.0
        command_feedback_lateral_gain = 0.0
        command_feedback_yaw_gain = 0.0

    class domain_rand(FanfanOmniRealDataCurriculumCfg.domain_rand):
        gait_calf_amplitude_max_range = [0.19, 0.22]
        gait_stance_ratio_range = [0.57, 0.63]
        gait_low_speed_period_range = [0.58, 0.70]
        gait_high_speed_period_range = [0.48, 0.56]

    class commands(FanfanOmniRealDataCurriculumCfg.commands):
        omni_curriculum = False
        resampling_time = 3.5
        stand_probability = 0.06
        pure_yaw_probability = 0.17
        pure_lateral_probability = 0.32
        pure_sagittal_probability = 0.28
        hard_transition_probability = 0.30

    class rewards(FanfanOmniRealDataCurriculumCfg.rewards):
        gait_period = 0.52
        gait_stance_ratio = 0.60
        # Positive phase-synchronous thigh sweep and calf flexion form one
        # diagonal-leg trajectory.  This recovers the clean 5530 forward
        # stride and lifts the toe instead of folding the knee in place.
        gait_thigh_amplitude = 0.15
        gait_lateral_hip_amplitude = -0.15
        gait_lateral_command_scale = 0.08
        gait_lateral_diagonal_scale = 0.35
        gait_thigh_lateral_scale = 2.0
        swing_height_target = 0.050
        swing_height_sigma = 0.00025
        swing_clearance_minimum = 0.042
        base_height_target = 0.290
        min_base_height_soft = 0.260

        class scales(FanfanOmniRealDataCurriculumCfg.rewards.scales):
            command_velocity_progress = 36.0
            normalized_command_tracking = 24.0
            tracking_lin_vel = 34.0
            tracking_longitudinal_vel = 30.0
            tracking_lateral_vel = 48.0
            tracking_ang_vel = 30.0
            absolute_longitudinal_tracking_error = -24.0
            absolute_lateral_tracking_error = -34.0
            absolute_yaw_tracking_error = -18.0
            command_transition_tracking = 9.0

            swing_height = 12.0
            swing_clearance_shortfall = -420.0
            swing_contact = -8.0
            diagonal_gait = 11.0
            diagonal_contact_sync_all = -6.0
            diagonal_foot_height_sync_all = -140.0
            diagonal_joint_sync = -1.0

            # Keep enough residual authority to adapt the working 5530 gait;
            # hard target bandwidth, torque clipping and saturation penalties
            # remain the physical safety boundary.
            action_magnitude = -0.014
            policy_action_magnitude = -0.040
            lateral_action_magnitude = -0.10
            action_rate = -0.24
            policy_action_rate = -0.32
            final_target_velocity = -0.34
            final_target_acceleration = -0.14
            dof_acc = -4.5e-7

            torque_near_limit = -1.20
            peak_torque = -1.70
            sustained_torque = -1.60
            sustained_torque_max = -2.00
            torque_clip = -2.80


class FanfanOmniRealDataPerformanceRecoveryCfgPPO(
        FanfanOmniRealDataCurriculumCfgPPO):
    class algorithm(FanfanOmniRealDataCurriculumCfgPPO.algorithm):
        entropy_coef = 0.0012
        learning_rate = 2.5e-5
        desired_kl = 0.0013

    class runner(FanfanOmniRealDataCurriculumCfgPPO.runner):
        run_name = "realdata_performance_recovery_from_5530"
        max_iterations = 900
        save_interval = 50
        symmetry_coef = 2.0


class FanfanOmniCalibratedSymmetryCfg(FanfanOmniHeadingBoundSymmetryCfg):
    """Selected 5100 plus straight-only cross-simulator action calibration."""

    class control(FanfanOmniHeadingBoundSymmetryCfg.control):
        straight_action_bias_by_joint = {
            "FL_hip_joint": 0.07,
            "FR_hip_joint": -0.07,
            "RL_hip_joint": 0.07,
            "RR_hip_joint": -0.07,
            "FL_thigh_joint": -0.04,
            "FR_thigh_joint": -0.04,
            "RL_thigh_joint": 0.04,
            "RR_thigh_joint": 0.04,
            "FL_calf_joint": -0.01,
            "FR_calf_joint": 0.01,
            "RL_calf_joint": 0.01,
            "RR_calf_joint": -0.01,
        }


class FanfanOmniCalibratedSymmetryCfgPPO(FanfanOmniHeadingBoundSymmetryCfgPPO):
    class runner(FanfanOmniHeadingBoundSymmetryCfgPPO.runner):
        experiment_name = "rough_fanfan_omni_desat_torque"
        run_name = "calibrated_symmetry_5100"
