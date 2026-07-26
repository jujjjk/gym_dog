"""Stage-0 validation for the RS01 straight-walking tasks.

Builds the real Isaac Gym environment with a handful of robots and checks the
things that must be true before any training is worth starting:

* task registration and observation/action dimensions;
* the runtime Isaac Gym DOF order versus ``cfg.control.policy_joint_order``;
* per-leg joint index and foot slot resolution;
* default joint angles and the resulting stance height;
* contact sensors reporting four planted feet at rest;
* torque staying inside the configured RS01 envelope;
* no NaN/Inf in observations, actions, torques or rewards.

Usage:
    python3 legged_gym/scripts/validate_dog_straight.py \
        --task dog_rs01_straight_stand --headless
"""

import math
import os

import isaacgym  # noqa: F401 - must precede torch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers the tasks
from legged_gym.envs.base.terminal_snapshot import RESET_REASON_BITS
from legged_gym.utils import get_args, task_registry

LEGS = ("FL", "FR", "RL", "RR")
JOINT_TYPES = ("hip", "thigh", "calf")
DIAGONALS = (("FL", "RR"), ("FR", "RL"))


def _finite(name, tensor, problems):
    if not torch.isfinite(tensor).all():
        problems.append(f"{name} contains NaN or Inf")
        return False
    return True


def seed_report(env, env_cfg, zero_actions, emit, problems):
    """Does the open-loop gait reference walk on its own?

    The first straight-walk attempt left every open-loop amplitude at zero, so a
    from-scratch policy had to invent an entire trot against a reward landscape
    where standing still paid most of the tracking reward. It never did. This
    check answers the question directly: hold the policy output at zero, command
    a forward speed, and see whether the robot steps.
    """
    command = env_cfg.commands.ranges.lin_vel_x[1]
    if command <= 0.0:
        emit("## Open-loop gait seed")
        emit()
        emit("Skipped: this task commands no forward motion.")
        emit()
        return

    emit("## Open-loop gait seed (zero policy output)")
    emit()
    emit(f"Commanding {command:.3f} m/s forward with `action = 0`, so the only "
         "thing driving the legs is the joint-space diagonal reference.")
    emit()

    emit(f"- `use_continuous_gait_scaling`: "
         f"{getattr(env, 'use_continuous_gait_scaling', None)}")
    env.commands[:, 0] = command
    env.commands[:, 1:4] = 0.0
    if getattr(env, "use_continuous_gait_scaling", False):
        amplitude = float(env._continuous_gait_amplitude()[0])
        emit(f"- Speed-scaled calf amplitude at {command:.3f} m/s: "
             f"**{amplitude:.4f} rad**")
    else:
        emit(f"- Fixed calf amplitude: "
             f"**{env_cfg.rewards.gait_calf_amplitude} rad**")
    emit(f"- `gait_thigh_amplitude`: {env_cfg.rewards.gait_thigh_amplitude}, "
         f"`gait_swing_thigh_lift_amplitude`: "
         f"{env_cfg.rewards.gait_swing_thigh_lift_amplitude}")
    emit()

    env.reset()
    steps = 400
    settle = 100
    reference_excursion = torch.zeros(env.num_actions, device=env.device)
    target_excursion = torch.zeros(env.num_actions, device=env.device)
    actual_excursion = torch.zeros(env.num_actions, device=env.device)
    forward = []
    contact_fraction = torch.zeros(4, device=env.device)
    footfalls = torch.zeros(4, device=env.device)
    previous_contact = None
    peak_motor = torch.zeros(env.num_actions, device=env.device)
    samples = 0
    cycle_trace = []
    trunk_min = float("inf")
    trunk_max = float("-inf")
    roll_sum = 0.0
    pitch_sum = 0.0
    fl_slot = env.foot_slot_by_leg["FL"]
    fl_calf = env.leg_dof_indices["FL"]["calf"]
    trace_steps = int(round(env_cfg.rewards.gait_period / env.dt))
    for step in range(steps):
        env.commands[:, 0] = command
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0
        env.step(zero_actions)
        contact = (
            env.contact_forces[:, env.feet_indices, 2]
            > env_cfg.rewards.foot_contact_force_threshold
        )
        if step < settle:
            previous_contact = contact
            continue
        samples += 1
        for source, accumulator in (
            ("raw_target_dof_pos", "reference"),
            ("limited_target_dof_pos", "target"),
            ("dof_pos", "actual"),
        ):
            tensor = getattr(env, source, None)
            if tensor is None:
                continue
            peak = torch.abs(tensor - env.default_dof_pos).amax(dim=0)
            if accumulator == "reference":
                reference_excursion = torch.maximum(reference_excursion, peak)
            elif accumulator == "target":
                target_excursion = torch.maximum(target_excursion, peak)
            else:
                actual_excursion = torch.maximum(actual_excursion, peak)
        trunk = float(env.root_states[0, 2])
        trunk_min = min(trunk_min, trunk)
        trunk_max = max(trunk_max, trunk)
        # Horizontal projected gravity is a wrap-free tilt measure: it is
        # sin(roll) and -sin(pitch) for a level-spawned trunk.
        roll_sum += float(env.projected_gravity[0, 1]) ** 2
        pitch_sum += float(env.projected_gravity[0, 0]) ** 2
        if len(cycle_trace) < trace_steps:
            cycle_trace.append((
                str(samples),
                f"{float(env.gait_phase[0]):.3f}",
                f"{float(env.limited_target_dof_pos[0, fl_calf] - env.default_dof_pos[0, fl_calf]):+.4f}",
                f"{float(env.dof_pos[0, fl_calf] - env.default_dof_pos[0, fl_calf]):+.4f}",
                f"{float(env.feet_state[0, fl_slot, 2]):.4f}",
                f"{trunk:.4f}",
                "yes" if bool(contact[0, fl_slot]) else "no",
            ))
        forward.append(float(env.base_lin_vel[:, 0].mean()))
        contact_fraction += contact.float().mean(dim=0)
        if previous_contact is not None:
            footfalls += (contact & ~previous_contact).float().mean(dim=0)
        previous_contact = contact
        peak_motor = torch.maximum(
            peak_motor,
            torch.abs(env.motor_electromagnetic_torques).amax(dim=0),
        )

    contact_fraction /= max(samples, 1)
    roll_rms = math.sqrt(roll_sum / max(samples, 1))
    pitch_rms = math.sqrt(pitch_sum / max(samples, 1))
    mean_forward = sum(forward) / max(len(forward), 1)
    duty = env_cfg.rewards.gait_stance_ratio
    seconds = samples * env.dt

    emit(f"- Mean forward velocity after settling: **{mean_forward:.4f} m/s** "
         f"(commanded {command:.3f})")
    emit(f"- Per-foot contact fraction (FL, FR, RL, RR): "
         f"{[round(float(value), 3) for value in contact_fraction]} "
         f"(scheduled duty factor {duty:.2f})")
    emit(f"- Footfalls per foot over {seconds:.1f} s: "
         f"{[round(float(value), 1) for value in footfalls]} "
         f"(one per {env_cfg.rewards.gait_period:.2f} s cycle "
         f"= {seconds / env_cfg.rewards.gait_period:.1f} expected)")
    emit(f"- Peak motor torque: **{float(peak_motor.max()):.2f} N·m** "
         f"of a {float(env.torque_limits.max()):.1f} N·m clip")
    emit()
    emit("Peak excursion from the default pose at each stage of the target "
         "chain. A large reference with a small target means the rate or "
         "acceleration limiter is clipping the swing; a large target with a "
         "small actual means the PD or the torque clip cannot follow it.")
    emit()
    emit("| Joint | Reference (rad) | After limiter (rad) | Achieved (rad) |")
    emit("|---|---:|---:|---:|")
    for index in range(env.num_actions):
        emit(
            f"| {env.dof_names[index]} | "
            f"{float(reference_excursion[index]):.4f} | "
            f"{float(target_excursion[index]):.4f} | "
            f"{float(actual_excursion[index]):.4f} |"
        )
    emit()
    emit("One FL cycle, sampled every control step. `toe z` is the world height "
         "of the toe centre, so a planted toe sits at its 0.016 m radius.")
    emit()
    emit("| Step | phase | calf target off (rad) | calf actual off (rad) | "
         "FL toe z (m) | trunk z (m) | contact |")
    emit("|---:|---:|---:|---:|---:|---:|:-:|")
    for row in cycle_trace:
        emit("| " + " | ".join(row) + " |")
    emit()
    emit(f"- Trunk height over the seed rollout: "
         f"**{trunk_min:.4f} - {trunk_max:.4f} m**")
    emit(f"- Tilt RMS (roll, pitch as projected gravity): "
         f"{roll_rms:.4f} / {pitch_rms:.4f}")
    emit()
    if trunk_min < env_cfg.rewards.base_height_target - 0.03:
        problems.append(
            f"the trunk sinks to {trunk_min:.4f} m under the open-loop "
            "reference, so the two stance legs cannot carry the body while the "
            "diagonal swings; the swing never clears the floor as a result"
        )

    if float(contact_fraction.min()) > 0.98:
        problems.append(
            "the open-loop gait reference does not lift any foot, so a "
            "from-scratch policy has no gait to refine; check "
            "use_continuous_gait_scaling, gait_calf_amplitude_knots and "
            "domain_rand.gait_calf_amplitude_max_range"
        )
    if mean_forward < 0.3 * command:
        problems.append(
            f"the open-loop gait reference only reaches {mean_forward:.4f} m/s "
            f"against a {command:.3f} m/s command; increase "
            "gait_thigh_amplitude or lengthen the stance sweep"
        )
    expected_falls = seconds / env_cfg.rewards.gait_period
    if float(footfalls.min()) < 0.5 * expected_falls:
        problems.append(
            "at least one foot barely leaves the floor under the open-loop "
            "reference, so the diagonal pairs are not symmetric"
        )


def validate(args):
    steps = int(os.environ.get("RS01_VALIDATE_STEPS", "200"))
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 16
    env_cfg.env.test = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gait_phase_on_reset = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    lines = []
    problems = []

    def emit(text=""):
        lines.append(text)

    emit(f"# RS01 straight-walk validation ({args.task})")
    emit()
    emit(f"- Environments: {env.num_envs}")
    emit(f"- Observations: {env.num_obs} (cfg {env_cfg.env.num_observations})")
    emit(f"- Actions: {env.num_actions} (cfg {env_cfg.env.num_actions})")
    emit(f"- Control dt: {env.dt:.4f} s ({1.0 / env.dt:.1f} Hz)")
    emit(f"- Terrain: {env_cfg.terrain.mesh_type}, "
         f"measure_heights={env_cfg.terrain.measure_heights}")
    emit()
    if env.num_obs != env_cfg.env.num_observations:
        problems.append("runtime observation width differs from the config")
    if env.num_actions != env_cfg.env.num_actions:
        problems.append("runtime action width differs from the config")

    runtime_order = list(env.dof_names)
    declared_order = list(env_cfg.control.policy_joint_order)

    emit("## Joint order")
    emit()
    emit("| DOF index | Isaac Gym runtime order | cfg.policy_joint_order | Match |")
    emit("|---:|---|---|:-:|")
    for index, name in enumerate(runtime_order):
        declared = declared_order[index]
        emit(f"| {index} | {name} | {declared} | "
             f"{'yes' if name == declared else 'NO'} |")
    emit()
    if runtime_order != declared_order:
        problems.append(
            "cfg.control.policy_joint_order does not match the runtime Isaac "
            "Gym DOF order; observations, actions and torques all use the "
            "runtime order, so policy_joint_order must be corrected to "
            f"{runtime_order}"
        )
        emit("> Every environment tensor (`dof_pos`, `actions`, `torques`) is "
             "indexed by the runtime order above. `policy_joint_order` is only "
             "used by export tooling, so it must be set to the runtime order.")
        emit()

    emit("## Per-leg index resolution")
    emit()
    emit("| Leg | hip | thigh | calf | foot slot | foot body index |")
    emit("|---|---:|---:|---:|---:|---:|")
    for leg in LEGS:
        indices = env.leg_dof_indices[leg]
        slot = env.foot_slot_by_leg[leg]
        emit(f"| {leg} | {indices['hip']} | {indices['thigh']} | "
             f"{indices['calf']} | {slot} | "
             f"{int(env.feet_indices[slot])} |")
    emit()
    resolved = sorted(
        indices
        for leg in LEGS
        for indices in env.leg_dof_indices[leg].values()
    )
    if resolved != list(range(12)):
        problems.append("per-leg joint indices do not cover all 12 DOFs exactly once")
    if sorted(env.foot_slot_by_leg.values()) != [0, 1, 2, 3]:
        problems.append("foot slots do not cover the four feet exactly once")
    emit(f"- `hip_dof_indices` (FL, FR, RL, RR order): "
         f"{env.hip_dof_indices.tolist()}")
    emit(f"- Diagonal groups used by the gait: {DIAGONALS}")
    emit()

    emit("## Default pose and limits")
    emit()
    emit("| Joint | Default (rad) | cfg default (rad) | Lower | Upper | "
         "Torque limit (N·m) |")
    emit("|---|---:|---:|---:|---:|---:|")
    default = env.default_dof_pos[0]
    for index, name in enumerate(runtime_order):
        expected = env_cfg.init_state.default_joint_angles[name]
        actual = float(default[index])
        if abs(actual - expected) > 1.0e-6:
            problems.append(f"default angle mismatch for {name}")
        emit(
            f"| {name} | {actual:+.8f} | {expected:+.8f} | "
            f"{float(env.dof_pos_limits[index, 0]):+.5f} | "
            f"{float(env.dof_pos_limits[index, 1]):+.5f} | "
            f"{float(env.torque_limits[index]):.2f} |"
        )
    emit()

    emit("## Zero-action rollout")
    emit()
    emit(f"Sending `action = 0` for {steps} steps, i.e. holding the default "
         "pose through the identified RS01 actuator chain. The first 100 steps "
         "are reported separately as the spawn transient.")
    emit()
    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )
    # reset() is what initializes the delayed actuator target buffers to the
    # default pose. Stepping without it drives every joint towards zero.
    env.reset()
    obs = env.get_observations()
    _finite("initial observations", obs, problems)

    # Phase A: spawn transient with the height termination relaxed, so the true
    # settling depth is measured instead of a post-reset value.
    settle_steps = 100
    configured_min_height = getattr(env.cfg.rewards, "min_base_height", None)
    env.cfg.rewards.min_base_height = 0.05
    trace = []
    settle_min_height = float("inf")
    for step in range(settle_steps):
        # Pin a stand command: this phase measures holding the default pose, so
        # the speed-scaled open-loop gait reference must stay out of it.
        env.commands[:, :4] = 0.0
        obs, _, _, _, _ = env.step(zero_actions)
        height = float(env.root_states[:, 2].min())
        settle_min_height = min(settle_min_height, height)
        if step % 10 == 0 or step == settle_steps - 1:
            error = torch.abs(env.dof_pos - env.default_dof_pos)
            trace.append((
                step,
                float(env.root_states[:, 2].mean()),
                float(error[:, [1, 4, 7, 10]].max()),
                float(error[:, [2, 5, 8, 11]].max()),
                float(torch.abs(env.motor_electromagnetic_torques).max()),
            ))
    env.cfg.rewards.min_base_height = configured_min_height

    emit("| Step | Mean trunk height (m) | max thigh err (rad) | "
         "max calf err (rad) | max motor torque (N·m) |")
    emit("|---:|---:|---:|---:|---:|")
    for step, height, thigh, calf, torque in trace:
        emit(f"| {step} | {height:.4f} | {thigh:.4f} | {calf:.4f} | "
             f"{torque:.3f} |")
    emit()
    emit(f"- Deepest trunk height during the spawn transient: "
         f"**{settle_min_height:.4f} m** "
         f"(`min_base_height` termination at {configured_min_height})")
    emit()
    if configured_min_height is not None and settle_min_height < configured_min_height:
        problems.append(
            f"the spawn transient sinks to {settle_min_height:.4f} m, below the "
            f"{configured_min_height} m height termination, so every episode is "
            "reset before it starts"
        )

    peak_raw = torch.zeros(env.num_actions, device=env.device)
    peak_motor = torch.zeros(env.num_actions, device=env.device)
    peak_applied = torch.zeros(env.num_actions, device=env.device)
    peak_error = torch.zeros(env.num_actions, device=env.device)
    height_min = float("inf")
    height_max = float("-inf")
    contact_fraction = torch.zeros(4, device=env.device)
    reset_reasons = {}
    resets = 0
    for _ in range(steps):
        env.commands[:, :4] = 0.0
        obs, _, rewards, dones, _ = env.step(zero_actions)
        if not _finite("observations", obs, problems):
            break
        if not _finite("rewards", rewards, problems):
            break
        if not _finite("torques", env.torques, problems):
            break
        peak_applied = torch.maximum(
            peak_applied, torch.abs(env.torques).amax(dim=0)
        )
        for attribute, accumulator in (
            ("raw_pd_torques", "raw"),
            ("motor_electromagnetic_torques", "motor"),
        ):
            tensor = getattr(env, attribute, None)
            if tensor is None:
                continue
            peak = torch.abs(tensor).amax(dim=0)
            if accumulator == "raw":
                peak_raw = torch.maximum(peak_raw, peak)
            else:
                peak_motor = torch.maximum(peak_motor, peak)
        peak_error = torch.maximum(
            peak_error,
            torch.abs(env.dof_pos - env.default_dof_pos).amax(dim=0),
        )
        if torch.any(dones):
            bits = env.reset_reason_bits[dones]
            for name, bit in RESET_REASON_BITS.items():
                count = int(((bits & bit) != 0).sum())
                if count:
                    reset_reasons[name] = reset_reasons.get(name, 0) + count
        height = env.root_states[:, 2]
        height_min = min(height_min, float(height.min()))
        height_max = max(height_max, float(height.max()))
        contact = (
            env.contact_forces[:, env.feet_indices, 2]
            > env_cfg.rewards.foot_contact_force_threshold
        )
        contact_fraction += contact.float().mean(dim=0)
        resets += int(dones.sum())
    contact_fraction /= steps

    emit(f"- Resets during the rollout: **{resets}**")
    emit(f"- Reset reasons: "
         f"{reset_reasons if reset_reasons else 'none'}")
    emit(f"- Trunk height range: **{height_min:.4f} - {height_max:.4f} m** "
         f"(target {env_cfg.rewards.base_height_target:.4f} m)")
    emit(f"- Per-foot contact fraction (FL, FR, RL, RR slots): "
         f"{[round(float(value), 3) for value in contact_fraction]}")
    emit()
    if resets > 0:
        problems.append(
            f"the default pose could not be held: {resets} resets under zero action"
        )
    if height_min < env_cfg.rewards.base_height_target - 0.05:
        problems.append(
            f"trunk sagged to {height_min:.4f} m under zero action; the RS01 "
            "torque envelope or the PD gains cannot hold the stance"
        )
    if float(contact_fraction.min()) < 0.5:
        problems.append(
            "at least one foot was airborne for most of the zero-action "
            "rollout; check contact thresholds or the default pose"
        )

    emit("### Torque envelope")
    emit()
    emit("`raw PD` is the unclipped PD request, `motor` is the clipped "
         "electromagnetic torque the RS01 must produce, and `applied` also "
         "contains the passive Coulomb friction term, so it may sit slightly "
         "above the clip.")
    emit()
    emit("| Joint | Peak |q - q_default| (rad) | Peak raw PD (N·m) | "
         "Peak motor (N·m) | Peak applied (N·m) | Limit (N·m) | Motor/limit |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for index, name in enumerate(runtime_order):
        limit = float(env.torque_limits[index])
        motor = float(peak_motor[index])
        emit(
            f"| {name} | {float(peak_error[index]):.4f} | "
            f"{float(peak_raw[index]):.3f} | {motor:.3f} | "
            f"{float(peak_applied[index]):.3f} | {limit:.2f} | "
            f"{motor / max(limit, 1e-6):.2f} |"
        )
    emit()
    over = [
        runtime_order[index]
        for index in range(env.num_actions)
        if float(peak_motor[index]) > float(env.torque_limits[index]) + 1.0e-4
    ]
    if over:
        problems.append(f"motor torque exceeded its clip on {over}")
    saturated = [
        runtime_order[index]
        for index in range(env.num_actions)
        if float(peak_motor[index]) > 0.95 * float(env.torque_limits[index])
    ]
    if saturated:
        problems.append(
            "the RS01 envelope is already saturated while merely holding the "
            f"default pose on {saturated}; raise RS01_SOFT_TORQUE_NM or soften "
            "the PD gains before training"
        )
    emit()

    seed_report(env, env_cfg, zero_actions, emit, problems)

    emit("## Findings")
    emit()
    if problems:
        for problem in problems:
            emit(f"- ISSUE: {problem}")
    else:
        emit("- No blocking issue found. The task is ready for manual training.")
    emit()

    report = "\n".join(lines)
    output = os.path.join(
        LEGGED_GYM_ROOT_DIR, "..", "artifacts", "rs01_straight",
        f"validation_{args.task}.md",
    )
    output = os.path.realpath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(report)
    print(f"\n[validate] wrote {output}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(validate(get_args()))
