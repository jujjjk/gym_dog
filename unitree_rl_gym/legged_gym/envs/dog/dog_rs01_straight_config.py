"""RS01 straight-walking retraining tasks: flat-ground stand, then low-speed trot.

Two fresh tasks trained from scratch. They deliberately do not resume any
earlier checkpoint and do not touch the existing ``dog_rs01_*`` runs.

Scope, by design:

* flat ground only (``mesh_type = "plane"``, no height scanner);
* forward command only (``lin_vel_y`` and ``ang_vel_yaw`` pinned to zero);
* one diagonal ``FL+RR`` / ``FR+RL`` gait with a high duty factor;
* the identified RS01 actuator chain plus a 10 N·m project torque envelope;
* URDF joint semantics used directly (no real-motor id/sign/offset mapping).

Torque layering, per the RS01 manual and this project's working envelope:

======================  =========  ====================================
Layer                   Value      Meaning
======================  =========  ====================================
rated                   6 N·m      reported/shaped only, not a hard cap
project soft envelope    10 N·m     hard applied clip during training
URDF/hardware peak      17 N·m     mechanical ceiling, never requested
======================  =========  ====================================

``audit_dog_joints.py`` measures 6.79 N·m of static calf torque in diagonal
two-leg support, so the 10 N·m envelope leaves about 1.5x headroom. Raise
``RS01_SOFT_TORQUE_NM`` if the stand stage cannot hold height.
"""

from .dog_cpg_fixed_config import (
    DogRs01TrotCfg as _Direct12Cfg,
    DogRs01TrotCfgPPO as _Direct12CfgPPO,
)

# RS01 manual values. Only the soft envelope is applied as a hard clip.
RS01_RATED_TORQUE_NM = 6.0
# Raised from 10.0. At 10 N·m the position PD had no budget to swing a leg:
# holding the default pose alone peaked at 9.67 N·m, and any calf offset large
# enough to lift a toe asked for more than the clip, which truncated the command
# and flung the leg instead of swinging it. 14 N·m is still well under the
# 17 N·m motor peak and is only reached transiently at lift-off.
RS01_SOFT_TORQUE_NM = 14.0
RS01_PEAK_TORQUE_NM = 17.0

# High duty factor keeps the heavy trunk in four-foot support for most of the
# cycle, which is where the static calf demand halves.
#
# The period must then be long enough that the remaining 24% is still a usable
# swing. At the first 0.48 s the swing was 0.115 s, i.e. 5.8 control steps at
# 50 Hz, against an identified 39-55 ms actuator delay: the calf target reached
# its peak for a single step and the toe never left the floor. 0.80 s gives a
# 0.192 s swing, about 9.6 steps, so the delay is a quarter of the swing.
STRAIGHT_GAIT_PERIOD_S = 0.80
# 0.76 left only a 0.192 s swing. Under the 10 N·m clip that was not enough
# time for a loaded toe to unload and rise: any amplitude large enough to lift
# saturated the clip and bounced the robot backwards, while any amplitude inside
# the clip never left the floor. 0.65 gives a 0.28 s swing, so the toe rises
# slowly on a small tracking error instead of being flung by a truncated one.
STRAIGHT_STANCE_RATIO = 0.65

# Single fixed forward speed for the walk stage. A sampled 0.03-0.12 m/s band
# let a motionless robot sit near the bottom of its own command range and still
# collect tracking reward; one fixed target removes that entirely.
STRAIGHT_WALK_SPEED_MS = 0.10


class DogRs01StraightBaseCfg(_Direct12Cfg):
    """Shared flat-ground, straight-only, RS01-limited environment."""

    class env(_Direct12Cfg.env):
        num_envs = 4096
        # Existing 52-wide layout, unchanged and documented in
        # artifacts/rs01_straight/README_RUN_MANUALLY.md:
        # lin_vel 3 | ang_vel 3 | gravity 3 | commands 3 | dof_pos_error 12
        # | dof_vel 12 | previous_actions 12 | phase_sin_cos 2 | heading 2
        num_observations = 52
        num_actions = 12
        episode_length_s = 20
        observe_actuator_state = False

    class terrain(_Direct12Cfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class control(_Direct12Cfg.control):
        # PD gains are inherited unchanged: kp 60/70/70, kd 1.2/1.6/1.6.
        # Raising kd to 1.2/1.8/2.0 was measured with validate_dog_straight.py
        # and made things clearly worse. The derivative term acts on the true
        # joint velocity while the target is delayed and lagged, so the extra
        # gain turned the stance into a permanently saturated limit cycle
        # (calf error 0.58 rad, 10 N·m for the whole rollout) instead of
        # settling to the 3.8-6.0 N·m stance these gains reach.

        # Applied clip is the project soft envelope, not the 17 N·m peak.
        torque_limits_by_joint = {
            "hip": RS01_SOFT_TORQUE_NM,
            "thigh": RS01_SOFT_TORQUE_NM,
            "calf": RS01_SOFT_TORQUE_NM,
        }
        # Do not randomize the safety envelope itself.
        training_torque_limit_ranges = {
            "hip": [RS01_SOFT_TORQUE_NM, RS01_SOFT_TORQUE_NM],
            "thigh": [RS01_SOFT_TORQUE_NM, RS01_SOFT_TORQUE_NM],
            "calf": [RS01_SOFT_TORQUE_NM, RS01_SOFT_TORQUE_NM],
        }
        # A fresh policy must not also fight a shrinking thermal limit.
        apply_continuous_torque_derating = False

        # Small stride and low clearance need less authority than the old
        # 0.22 rad sweep, and less authority is the cheapest way to keep the
        # raw PD request inside the envelope.
        action_scale = 0.18
        rear_action_scale = 0.18
        hip_action_scale = 0.10
        thigh_action_scale = 0.18
        calf_action_scale = 0.18

        # These limit the *summed* target (policy residual plus open-loop
        # reference), so they have to clear the reference on their own.
        #
        # Swing lasts (1 - 0.76) * 0.48 = 0.115 s. A sinusoidal calf swing of
        # amplitude A peaks at A*pi/T_swing rad/s and A*(pi/T_swing)^2 rad/s^2,
        # which is 4.4 rad/s and 119 rad/s^2 for A = 0.19. The first values here
        # were 3.0 and 80, and they silently clipped the swing to roughly two
        # thirds of its required speed: the toe never left the floor. Leave
        # headroom above the reference so the policy residual still has room.
        final_target_rate_limits_initial = {
            "hip": 4.0, "thigh": 6.0, "calf": 8.0,
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_initial = {
            "hip": 120.0, "thigh": 200.0, "calf": 260.0,
        }
        final_target_accel_limits_final = final_target_accel_limits_initial

    class commands(_Direct12Cfg.commands):
        heading_command = False
        observe_heading_error = True
        # Straight only: no lateral and no yaw command is ever sampled.
        pure_sagittal_probability = 1.0
        pure_lateral_probability = 0.0
        pure_yaw_probability = 0.0
        hard_transition_probability = 0.0

        class ranges(_Direct12Cfg.commands.ranges):
            lin_vel_x = [0.03, 0.12]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(_Direct12Cfg.domain_rand):
        # Keep the identified RS01 gain/tau/friction/backlash bands from
        # dog_config; only the coarse platform randomization is narrowed so a
        # from-scratch policy sees a consistent robot first.
        randomize_friction = True
        friction_range = [0.80, 1.10]
        randomize_base_mass = True
        added_mass_range = [-0.20, 0.40]
        randomize_base_com = True
        base_com_x_range = [-0.006, 0.006]
        base_com_y_range = [-0.005, 0.005]
        base_com_z_range = [-0.004, 0.004]
        randomize_motor_strength = True
        motor_strength_range = [0.97, 1.03]
        rear_calf_strength_range = [0.97, 1.03]
        kp_multiplier_range = [0.97, 1.03]
        kd_multiplier_range = [0.95, 1.05]
        randomize_gait_phase_on_reset = True
        gait_stance_ratio_range = [STRAIGHT_STANCE_RATIO, STRAIGHT_STANCE_RATIO]
        gait_low_speed_period_range = [
            STRAIGHT_GAIT_PERIOD_S, STRAIGHT_GAIT_PERIOD_S
        ]
        gait_high_speed_period_range = [
            STRAIGHT_GAIT_PERIOD_S, STRAIGHT_GAIT_PERIOD_S
        ]
        gait_calf_amplitude_max_range = [0.0, 0.0]
        push_robots = False

    class rewards(_Direct12Cfg.rewards):
        gait_period = STRAIGHT_GAIT_PERIOD_S
        gait_stance_ratio = STRAIGHT_STANCE_RATIO
        # Direct-12 has no open-loop leg reference; these stay at zero.
        gait_thigh_amplitude = 0.0
        gait_swing_thigh_lift_amplitude = 0.0
        gait_calf_amplitude = 0.0
        gait_lateral_hip_amplitude = 0.0

        # The geometric default pose puts the toe 0.300 m below the trunk with
        # a 0.016 m sphere, so the unloaded stance is 0.316 m. Under load the
        # measured stance settles at 0.309 m, and asking for the unloaded
        # height would permanently penalize a healthy stance and push the
        # policy to extend the knee for no gain.
        base_height_target = 0.3095
        min_base_height = 0.250
        min_base_height_soft = 0.290
        # One second of spawn settling is a simulation artifact, not a fall.
        min_base_height_grace_steps = 50
        # Low clearance: a scuffing floor, not an instruction to lift higher.
        swing_height_target = 0.020
        swing_clearance_minimum = 0.015
        diagonal_pair_lift_start_height = 0.010
        diagonal_pair_lift_target_height = 0.020

        # One contact definition for reward, termination, statistics and CSV.
        foot_contact_force_threshold = 3.0
        foot_contact_release_force_threshold = 1.5

        # 6 N·m is the reported rating; the 10 N·m clip is the hard envelope.
        continuous_torque_limits_by_joint = {
            "hip": RS01_RATED_TORQUE_NM,
            "thigh": RS01_RATED_TORQUE_NM,
            "calf": RS01_RATED_TORQUE_NM,
        }
        # Fixed torque penalties: no curriculum that silently rescales them.
        torque_curriculum = False
        torque_near_limit_ratio = 0.80
        peak_torque_soft_ratio = 0.92
        sustained_torque_ratio = 0.60
        # URDF velocity limit is 32.99 rad/s; low-speed walking stays far below.
        soft_dof_pos_limit = 0.92
        soft_dof_vel_limit = 0.55
        only_positive_rewards = True
        tracking_sigma = 0.02
        longitudinal_tracking_sigma = 0.002

        enable_actuator_safety_termination = False
        terminate_on_calf_angle = False


class DogRs01StraightBaseCfgPPO(_Direct12CfgPPO):
    """Fresh from-scratch PPO. No checkpoint is inherited."""

    class policy(_Direct12CfgPPO.policy):
        init_noise_std = 0.5

    class algorithm(_Direct12CfgPPO.algorithm):
        learning_rate = 1.0e-3
        entropy_coef = 0.005
        schedule = "adaptive"

    class runner(_Direct12CfgPPO.runner):
        experiment_name = "rs01_straight"
        resume = False
        load_optimizer = False
        actor_output_init_scale = None
        symmetry_coef = 0.0
        reference_policy_coef = 0.0
        phase_residual_policy = False
        save_interval = 50


class DogRs01StraightStandCfg(DogRs01StraightBaseCfg):
    """Stage 1: hold a stable flat-ground stance under small perturbations."""

    class commands(DogRs01StraightBaseCfg.commands):
        # Mostly zero command; a small forward sample keeps the same policy
        # usable as the warm start for Stage 2.
        stand_probability = 0.80
        resampling_time = 6.0

        class ranges(DogRs01StraightBaseCfg.commands.ranges):
            lin_vel_x = [0.0, 0.03]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(DogRs01StraightBaseCfg.domain_rand):
        # "Small perturbations" for standing, not locomotion disturbance.
        push_robots = True
        push_interval_s = 6.0
        max_push_vel_xy = 0.10

    class rewards(DogRs01StraightBaseCfg.rewards):
        # A standing robot has all four feet planted, so the diagonal-swing
        # and flight terminations have nothing legal to measure.
        enable_non_diagonal_swing_termination = False
        enable_rear_pair_air_termination = False
        enable_overlapping_diagonal_termination = False
        enable_flight_termination = False

        class scales:
            termination = -12.0

            # Posture: hold the measured default pose at the nominal height.
            # `only_positive_rewards` clips the shaped sum at zero, so the
            # positive terms have to stay clearly larger than the penalties or
            # an untrained policy sees a flat zero reward and learns nothing.
            stand_height = 8.0
            stand_posture = 4.0
            stand_still = -0.5
            stand_dof_velocity = -0.005
            tracking_lin_vel = 3.0

            # Body attitude.
            orientation = -4.0
            base_height = -10.0
            low_base_height = -10.0
            lin_vel_z = -2.0
            ang_vel_xy = -1.5
            yaw_rate = -1.0

            # Joint and actuator feasibility.
            dof_pos_limits = -5.0
            dof_vel_limits = -0.5
            dof_acc = -2.5e-7
            action_rate = -0.05
            torques = -1.0e-5
            torque_clip = -1.5
            sustained_torque = -0.6
            motor_continuous_usage = -0.5
            mechanical_power = -0.002

            # Contact quality.
            collision = -5.0
            stumble = -1.0
            feet_contact_forces = -0.02
            flight = -20.0


class DogRs01StraightStandCfgPPO(DogRs01StraightBaseCfgPPO):
    class runner(DogRs01StraightBaseCfgPPO.runner):
        run_name = "rs01_straight_stand"
        max_iterations = 600


class DogRs01StraightWalkCfg(DogRs01StraightBaseCfg):
    """Stage 2: low-speed straight diagonal walking at a fixed 0.10 m/s."""

    class commands(DogRs01StraightBaseCfg.commands):
        # No stand samples and no speed band. Every environment, every step, is
        # asked for the same 0.10 m/s. Standing is then unambiguously wrong
        # instead of being the easy end of a sampled range.
        stand_probability = 0.0
        resampling_time = 5.0

        class ranges(DogRs01StraightBaseCfg.commands.ranges):
            lin_vel_x = [STRAIGHT_WALK_SPEED_MS, STRAIGHT_WALK_SPEED_MS]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class control(DogRs01StraightBaseCfg.control):
        # The first attempt left every open-loop amplitude at zero, so the
        # policy had to invent a whole trot from scratch. It never did: at 3000
        # iterations it stood on four feet with 0.0038 m/s of drift.
        #
        # Re-enable the joint-space diagonal reference that the environment
        # already supports (`target = action_scale * action + gait_offset +
        # default`). Zero policy output now *is* a low trot, and PPO only has
        # to refine it. The reference is speed-scaled, so a stand command still
        # produces exactly zero offset and the stand behaviour is unchanged.
        # The speed command is a single fixed value, so a speed-scaled envelope
        # would only ever evaluate to one point. Use the fixed amplitudes below.
        use_continuous_gait_scaling = False
        # Start the target ahead of nominal lift-off. This has to cover far more
        # than the 39-55 ms command delay: the PD lag and the time the swing leg
        # needs to unload before it can rise add up. Measured with a 0.06 lead,
        # physical toe lift still trailed the scheduled swing window by 0.165 of
        # a cycle, which put the leg into its stance sweep while still airborne
        # and walked the robot backwards at -0.009 m/s. 0.22 realigns lift-off
        # with the schedule that the contact rewards score against.
        gait_target_phase_lead = 0.22

        # Softening the gains to 45 fitted the swing inside a 10 N·m clip at
        # 6.75 N·m, but that torque could not fold a loaded calf at all: the toe
        # never left the floor. The gains stay at the inherited values and the
        # torque envelope is what was raised instead.

        # The first torque-feasibility continuation cut the residual authority
        # to 0.10/0.09. It reduced mean torque, but by model_800 the normalized
        # policy output was already near its tanh boundary (P95 0.942 thigh,
        # 0.831 calf) while speed and exact diagonal contact had regressed.
        # Restore only enough authority for stance placement and phase
        # correction. Keep calf authority below thigh authority because the
        # remaining raw-torque and joint-speed peaks are calf-dominated.
        thigh_action_scale = 0.12
        calf_action_scale = 0.10

        # Sized to the open-loop reference and nothing more. A 0.17 rad calf
        # swing over the 0.28 s swing needs 1.9 rad/s and 21 rad/s^2 at its
        # peak, so these pass the reference through while denying the policy the
        # high-rate target steps that produced the raw torque spikes.
        final_target_rate_limits_initial = {
            "hip": 1.6, "thigh": 2.2, "calf": 2.4,
        }
        final_target_rate_limits_final = final_target_rate_limits_initial
        final_target_accel_limits_initial = {
            "hip": 40.0, "thigh": 50.0, "calf": 55.0,
        }
        final_target_accel_limits_final = final_target_accel_limits_initial

    class domain_rand(DogRs01StraightBaseCfg.domain_rand):
        # Nominal robot only, for this stage. The pilot's actuator numbers have
        # to be brought inside the envelope on one consistent machine before
        # randomization widens the distribution again; otherwise a marginal
        # policy and a marginal sample are indistinguishable.
        randomize_friction = False
        randomize_base_mass = False
        randomize_base_com = False
        randomize_motor_strength = False
        kp_multiplier_range = [1.0, 1.0]
        kd_multiplier_range = [1.0, 1.0]
        randomize_gait_phase_on_reset = False
        push_robots = False

    class rewards(DogRs01StraightBaseCfg.rewards):
        # Open-loop diagonal reference amplitudes, scaled by the knot curve.
        #
        # `gait_thigh_amplitude` is the fore/aft stance sweep. The toe sits
        # 0.300 m below the thigh axis, so a +/-A rad sweep retracts the toe by
        # 2*A*0.300 m per stance. The 0.608 s stance at 0.10 m/s must retract
        # 6.1 cm, which needs A = 0.101 rad.
        #
        # `gait_calf_amplitude` sets toe clearance: -0.17 rad lifts the toe
        # 2.2 cm by the default-pose Jacobian, matching `swing_height_target`.
        #
        # `gait_swing_thigh_lift_amplitude` cancels part of the forward toe
        # drift that calf flexion produces, so the toe rises closer to
        # vertically instead of scuffing forward and low.
        # Amplitudes are capped by the torque clip, not by kinematics. The
        # reference is tracked by a position PD, so an offset of A rad asks for
        # A * kp N·m the instant it appears: at kp = 70 the 0.17 rad calf swing
        # demanded 11.9 N·m against a 10 N·m clip. The clip then truncated the
        # command, the leg was flung, and the robot bounced backwards. 10/70 =
        # 0.143 rad is the hard ceiling, so stay clearly under it.
        # Sweep sign and magnitude were both measured, not assumed. Flipping the
        # sign made the robot go backwards faster and wrecked the footfall
        # rhythm, so the positive sense is correct. Magnitude scales the residual
        # backward drift (0.101 -> -0.045 m/s, 0.170 -> -0.081 m/s) while 0.170
        # gave the cleanest rhythm; 0.101 is kept as the lower-drift, lower-
        # torque starting point for the policy to correct.
        gait_thigh_amplitude = 0.101
        gait_swing_thigh_lift_amplitude = 0.055
        gait_calf_amplitude = -0.170
        gait_lateral_hip_amplitude = 0.0

        # At the old 0.08 m/s command, sigma 0.02 gave a motionless robot
        # exp(-0.0056/0.02) = 0.76 of the full tracking reward: standing was
        # worth 76% of walking. At the fixed 0.10 m/s command 0.0008 makes a
        # motionless robot score exp(-12.5) = 0, while a 0.02 m/s tracking
        # error still scores 0.61.
        tracking_sigma = 0.0008

        # Positive speed and gait terms are scaled by remaining actuator
        # headroom, which is what removes the pilot's trade of saturation for
        # gait reward. The gate is 1 - saturation/0.25, so the pilot's 22.6%
        # keeps only 10% of the positive terms, the 15% target keeps 40%, and
        # 10% keeps 60%. A tighter 0.12 zeroed the gate at the untrained 17%
        # saturation and left early training with no gradient at all.
        headroom_gate_saturation_ratio = 0.25

        # Debounced diagonal legality. Deployment/eval is stricter than the
        # early training curriculum, which is what the curriculum below does.
        enable_non_diagonal_swing_termination = True
        enable_rear_pair_air_termination = False
        enable_overlapping_diagonal_termination = False
        enable_flight_termination = True
        non_diagonal_swing_grace_steps = 15
        non_diagonal_swing_termination_steps = 3
        non_diagonal_swing_termination_steps_test = 3
        non_diagonal_termination_curriculum = [
            {"until_iteration": 800, "steps": 6},
            {"until_iteration": 1800, "steps": 4},
            {"until_iteration": 1.0e12, "steps": 3},
        ]
        flight_termination_grace_steps = 15
        flight_termination_steps = 4

        class scales:
            termination = -12.0

            # Speed. `tracking_sigma` above is what makes this term actually
            # require motion. `normalized_command_tracking` was removed: its
            # error floors are hardcoded at (0.10, 0.055, 0.30) m/s in the
            # environment, so at a 0.08 m/s command a motionless robot scored
            # 0.56 of the maximum. That is structurally unusable below
            # 0.1 m/s and it was the single largest standing subsidy.
            tracking_lin_vel = 20.0
            absolute_longitudinal_tracking_error = -8.0
            backward_velocity = -15.0
            commanded_smooth_straight_progress = 6.0

            # Diagonal gait structure and load handoff.
            diagonal_gait = 8.0
            exact_diagonal_swing = 20.0
            scheduled_diagonal_pair_lift = 12.0
            touchdown_pair_support = 4.0
            diagonal_load_transfer = 6.0
            phase_contact_mismatch = -6.0
            non_diagonal_swing = -20.0
            flight = -30.0

            # Dense, phase-resolved load and stride errors. Binary contact only
            # changes after lift-off, so a four-foot policy gets no gradient
            # from it at all; normalized vertical force and body-frame toe
            # velocity both move while the swing diagonal is still unloading.
            phase_foot_force_tracking = -8.0
            phase_foot_velocity_tracking = -2.0

            # Standing under a walk command has to cost more than it pays.
            # `all_feet_contact` saturates 0.16 s after four-foot support
            # begins, and `excessive_foot_contact_time` sums over all four
            # feet, so a planted robot pays roughly -10 and -12 per step.
            all_feet_contact = -10.0
            excessive_foot_contact_time = -3.0
            swing_clearance_shortfall = -3.0
            swing_height = 0.5

            # The pilot reached 0.128 m/s against a 0.10 command. The symmetric
            # tracking error prices overshoot and undershoot alike, and
            # overshoot is the cheaper one when it also buys stride reward.
            forward_overspeed = -25.0

            # Straightness. Closed-loop recovery targets, not a blanket
            # penalty on every corrective lateral/yaw motion. Raised because the
            # pilot drifted -0.183 m sideways in 10 s: forward progress must not
            # be bought with lateral displacement.
            straight_lateral_speed = -18.0
            straight_heading_error = -8.0
            straight_path_lateral_displacement = -10.0
            straight_path_recovery_velocity = -1.5
            straight_heading_recovery_rate = -4.0

            # Body attitude. `stand_height` is deliberately absent here: it is
            # exp(-height_error) with no command gate, so it paid a motionless
            # robot its full value. Height is regulated by the two penalties
            # below instead, which cost nothing to a robot at the right height.
            orientation = -4.0
            base_height = -8.0
            low_base_height = -10.0
            lin_vel_z = -2.0
            ang_vel_xy = -1.0
            body_angular_velocity = -1.0

            # Joint and actuator feasibility.
            dof_pos_limits = -5.0
            dof_vel_limits = -0.5
            dof_acc = -2.5e-7
            action_rate = -0.05
            # The pilot's calf raw P95 was 58-64 N·m with 19% of steps demanding
            # more than the 17 N·m motor peak and 22.6% at the clip. These are
            # the terms that price that demand, and at -1.5 and -0.6 they were
            # an order of magnitude too cheap next to the gait reward. Short
            # 12-14 N·m handoff transients remain affordable; sustained
            # saturation and raw demand beyond the motor peak do not.
            raw_torque_rate = -1.2
            torques = -1.0e-4
            torque_clip = -20.0
            peak_torque = -12.0
            sustained_torque = -8.0
            motor_continuous_usage = -3.0
            mechanical_power = -0.008

            # Contact quality.
            collision = -5.0
            stumble = -1.0
            feet_contact_forces = -0.02


class DogRs01StraightWalkCfgPPO(DogRs01StraightBaseCfgPPO):
    class policy(DogRs01StraightBaseCfgPPO.policy):
        # Half the base exploration. The open-loop reference is already a valid
        # gait; noise wide enough to destroy it costs more than it buys.
        init_noise_std = 0.25

    class runner(DogRs01StraightBaseCfgPPO.runner):
        run_name = "rs01_straight_walk"
        max_iterations = 3000
        # Start the actor's output layer near zero so the very first rollout is
        # the open-loop diagonal reference rather than random joint noise on top
        # of it. Without this the seed is buried before PPO can exploit it.
        actor_output_init_scale = 0.01
