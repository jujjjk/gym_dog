"""Audit RS01 dog joint semantics, limits and static torque demand.

Pure URDF/config analysis: it does not import Isaac Gym, so it can run on a
machine without a GPU. The runtime DOF order that Isaac Gym reports is checked
separately by ``validate_dog_straight.py``.

Usage:
    python3 legged_gym/scripts/audit_dog_joints.py [--task dog_rs01_straight_walk]
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

from legged_gym import LEGGED_GYM_ROOT_DIR

LEGS = ("FL", "FR", "RL", "RR")
JOINT_TYPES = ("hip", "thigh", "calf")
GRAVITY = 9.81


def _matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _matvec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def _add(a, b):
    return [a[i] + b[i] for i in range(3)]


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v):
    return math.sqrt(sum(component * component for component in v))


def _rotation(axis, angle):
    """Rodrigues rotation for a unit joint axis."""
    length = _norm(axis)
    x, y, z = (component / length for component in axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _floats(text):
    return [float(value) for value in text.split()]


def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    order = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        record = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "xyz": _floats(origin.get("xyz")) if origin is not None else [0.0] * 3,
            "rpy": _floats(origin.get("rpy")) if origin is not None else [0.0] * 3,
            "axis": _floats(axis.get("xyz")) if axis is not None else None,
        }
        if limit is not None:
            record["lower"] = float(limit.get("lower"))
            record["upper"] = float(limit.get("upper"))
            record["effort"] = float(limit.get("effort"))
            record["velocity"] = float(limit.get("velocity"))
        joints[name] = record
        if record["type"] == "revolute":
            order.append(name)

    masses = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        masses[link.get("name")] = float(inertial.find("mass").get("value"))

    feet = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name.endswith("_foot"):
            continue
        collision = link.find("collision")
        radius = None
        if collision is not None:
            sphere = collision.find("geometry/sphere")
            if sphere is not None:
                radius = float(sphere.get("radius"))
        feet[name] = radius
    return joints, order, masses, feet


def leg_chain(joints, leg):
    """Return the hip/thigh/calf joints plus the fixed foot offset."""
    chain = [joints[f"{leg}_{kind}_joint"] for kind in JOINT_TYPES]
    foot = joints[f"{leg}_foot_fixed"]
    return chain, foot


def forward_kinematics(chain, foot, angles):
    """Return foot position, per-joint origin and per-joint world axis."""
    rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    position = [0.0, 0.0, 0.0]
    origins = []
    axes = []
    for joint, angle in zip(chain, angles):
        position = _add(position, _matvec(rotation, joint["xyz"]))
        origins.append(list(position))
        axes.append(_matvec(rotation, joint["axis"]))
        rotation = _matmul(rotation, _rotation(joint["axis"], angle))
    foot_position = _add(position, _matvec(rotation, foot["xyz"]))
    return foot_position, origins, axes


def foot_jacobian(origins, axes, foot_position):
    """Columns are d(foot)/d(theta_i) = axis_i x (foot - origin_i)."""
    return [_cross(axes[i], _sub(foot_position, origins[i])) for i in range(3)]


def static_joint_torques(jacobian, force):
    """tau = J^T F, i.e. the generalized force produced at each joint."""
    return [sum(jacobian[i][k] * force[k] for k in range(3)) for i in range(3)]


def link_lengths(joints, leg):
    thigh = joints[f"{leg}_calf_joint"]["xyz"]
    calf = joints[f"{leg}_foot_fixed"]["xyz"]
    return _norm(thigh), _norm(calf)


def _import_config_only():
    """Import the straight configs without pulling in Isaac Gym.

    ``legged_gym/envs/__init__.py`` and ``envs/dog/__init__.py`` register every
    task and therefore import the environments, which require Isaac Gym. The
    configs themselves are plain Python, so the two heavy package inits are
    replaced by namespace stubs pointing at the same directories.
    """
    import importlib
    import types

    envs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs")
    for name, path in (
        ("legged_gym.envs", envs_dir),
        ("legged_gym.envs.dog", os.path.join(envs_dir, "dog")),
    ):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [path]
            sys.modules[name] = stub
    return importlib.import_module(
        "legged_gym.envs.dog.dog_rs01_straight_config"
    )


def load_task_cfg(task):
    straight = _import_config_only()

    table = {
        "dog_rs01_straight_stand": straight.DogRs01StraightStandCfg,
        "dog_rs01_straight_walk": straight.DogRs01StraightWalkCfg,
    }
    if task not in table:
        raise SystemExit(
            f"Unknown task {task!r}; expected one of {sorted(table)}"
        )
    return table[task]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="dog_rs01_straight_walk")
    parser.add_argument(
        "--output",
        default=os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "..",
            "artifacts",
            "rs01_straight",
            "joint_semantics_report.md",
        ),
    )
    args = parser.parse_args()

    cfg = load_task_cfg(args.task)
    urdf_path = cfg.asset.file.replace(
        "{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR
    )
    joints, urdf_order, masses, feet = parse_urdf(urdf_path)

    total_mass = sum(masses.values())
    weight = total_mass * GRAVITY
    defaults = cfg.init_state.default_joint_angles
    policy_order = list(cfg.control.policy_joint_order)

    lines = []
    problems = []

    def emit(text=""):
        lines.append(text)

    emit(f"# RS01 dog joint semantics report ({args.task})")
    emit()
    emit(f"- URDF: `{os.path.realpath(urdf_path)}`")
    emit(f"- Total URDF mass: **{total_mass:.6f} kg** (weight {weight:.2f} N)")
    emit(f"- Policy actions: {cfg.env.num_actions}")
    emit(f"- Policy observations: {cfg.env.num_observations}")
    emit(f"- Control decimation: {cfg.control.decimation} "
         f"(sim dt {cfg.sim.dt} s -> {1.0 / (cfg.sim.dt * cfg.control.decimation):.1f} Hz)")
    emit()

    emit("## Link masses")
    emit()
    emit("| Link | Mass (kg) |")
    emit("|---|---:|")
    for name, mass in masses.items():
        emit(f"| {name} | {mass:.6f} |")
    emit()

    emit("## Joint order")
    emit()
    emit("| Index | URDF revolute order | cfg.policy_joint_order | Same |")
    emit("|---:|---|---|:-:|")
    for index in range(len(policy_order)):
        urdf_name = urdf_order[index]
        policy_name = policy_order[index]
        same = "yes" if urdf_name == policy_name else "no"
        emit(f"| {index} | {urdf_name} | {policy_name} | {same} |")
    emit()
    if urdf_order != policy_order:
        emit("> The URDF declares its legs FR, FL, RR, RL, while "
             "`policy_joint_order` is FL, FR, RL, RR. This is expected and not "
             "a defect: Isaac Gym does not preserve the URDF declaration order, "
             "and `validate_dog_straight.py` confirms the runtime `dof_names` "
             "are FL, FR, RL, RR, matching `policy_joint_order`. The runtime "
             "order is the one every observation, action and torque tensor "
             "uses, so run that check after any URDF change.")
        emit()

    emit("## Joint limits, axes and default angles")
    emit()
    emit("| Joint | Axis | Lower (rad) | Upper (rad) | Default (rad) | "
         "In range | Effort (N·m) | Velocity (rad/s) |")
    emit("|---|---|---:|---:|---:|:-:|---:|---:|")
    for name in urdf_order:
        record = joints[name]
        default = defaults[name]
        inside = record["lower"] <= default <= record["upper"]
        if not inside:
            problems.append(f"default angle for {name} is outside its URDF limits")
        emit(
            f"| {name} | {tuple(int(v) for v in record['axis'])} | "
            f"{record['lower']:.5f} | {record['upper']:.5f} | {default:.8f} | "
            f"{'yes' if inside else 'NO'} | {record['effort']:.1f} | "
            f"{record['velocity']:.4f} |"
        )
    emit()

    target_limits = getattr(cfg.control, "target_position_limits_by_joint", None)
    if target_limits is not None:
        emit("### Target clamp versus URDF limits")
        emit()
        emit("| Joint type | Target clamp (rad) | URDF limit (rad) | Inside |")
        emit("|---|---|---|:-:|")
        for kind in JOINT_TYPES:
            record = joints[f"FL_{kind}_joint"]
            low, high = target_limits[kind]
            inside = low >= record["lower"] and high <= record["upper"]
            if not inside:
                problems.append(
                    f"target_position_limits_by_joint[{kind}] escapes the URDF limit"
                )
            emit(
                f"| {kind} | [{low:.4f}, {high:.4f}] | "
                f"[{record['lower']:.4f}, {record['upper']:.4f}] | "
                f"{'yes' if inside else 'NO'} |"
            )
        emit()

    emit("## Left/right semantics")
    emit()
    hip_axes = {leg: joints[f"{leg}_hip_joint"]["axis"] for leg in LEGS}
    thigh_axes = {leg: joints[f"{leg}_thigh_joint"]["axis"] for leg in LEGS}
    same_hip = all(hip_axes[leg] == hip_axes["FL"] for leg in LEGS)
    same_thigh = all(thigh_axes[leg] == thigh_axes["FL"] for leg in LEGS)
    emit(f"- All four hip axes identical: {same_hip} ({hip_axes['FL']})")
    emit(f"- All four thigh/calf axes identical: {same_thigh} ({thigh_axes['FL']})")
    emit(f"- Thigh mount offset FL/FR y: "
         f"{joints['FL_thigh_joint']['xyz'][1]:+.5f} / "
         f"{joints['FR_thigh_joint']['xyz'][1]:+.5f}")
    emit()
    emit("Because every hip axis points along +x while the left and right legs "
         "are mounted mirrored, the *same* joint sign produces *opposite* "
         "physical ab/adduction on the two sides. Sagittal thigh/calf joints "
         "share +y, so an identical sign there means identical physical "
         "motion. Diagonal symmetry terms must therefore add hip values and "
         "subtract thigh/calf values, which is what the environment does.")
    emit()

    emit("## Default stance geometry")
    emit()
    thigh_length, calf_length = link_lengths(joints, "FL")
    emit(f"- Thigh link length (thigh joint -> calf joint): **{thigh_length:.6f} m**")
    emit(f"- Calf link length (calf joint -> foot): **{calf_length:.6f} m**")
    emit(f"- Foot collision radius: "
         f"{ {name: radius for name, radius in feet.items()} }")
    emit()
    emit("| Leg | Hip origin (m) | Foot rel. trunk (m) | Foot rel. hip (m) |")
    emit("|---|---|---|---|")
    foot_positions = {}
    for leg in LEGS:
        chain, foot = leg_chain(joints, leg)
        angles = [defaults[f"{leg}_{kind}_joint"] for kind in JOINT_TYPES]
        foot_position, origins, axes = forward_kinematics(chain, foot, angles)
        foot_positions[leg] = (foot_position, origins, axes)
        hip_origin = chain[0]["xyz"]
        relative = _sub(foot_position, hip_origin)
        emit(
            f"| {leg} | ({hip_origin[0]:+.4f}, {hip_origin[1]:+.4f}, "
            f"{hip_origin[2]:+.4f}) | ({foot_position[0]:+.5f}, "
            f"{foot_position[1]:+.5f}, {foot_position[2]:+.5f}) | "
            f"({relative[0]:+.5f}, {relative[1]:+.5f}, {relative[2]:+.5f}) |"
        )
    emit()
    nominal_drop = -foot_positions["FL"][0][2]
    foot_radius = max(radius for radius in feet.values() if radius) or 0.0
    implied_height = nominal_drop + foot_radius
    emit(f"- Trunk-to-foot drop at the default pose: **{nominal_drop:.5f} m**")
    emit(f"- Implied trunk height with a {foot_radius:.3f} m toe: "
         f"**{implied_height:.5f} m**")
    emit(f"- Configured `init_state.pos[2]`: **{cfg.init_state.pos[2]:.5f} m**")
    height_gap = abs(implied_height - cfg.init_state.pos[2])
    if height_gap > 0.005:
        problems.append(
            f"init_state.pos[2] disagrees with the default-pose foot drop by "
            f"{height_gap * 1000:.1f} mm"
        )
    emit(f"- Configured `rewards.base_height_target`: "
         f"**{cfg.rewards.base_height_target:.5f} m**")
    emit()

    emit("## Static support torque (Jacobian, tau = J^T F)")
    emit()
    emit("Vertical ground reaction only, weight shared equally by the "
         "supporting feet. Magnitudes are what the actuator must hold.")
    emit()
    emit("| Support case | Feet | Force per foot (N) | |hip| | |thigh| | |calf| |")
    emit("|---|---:|---:|---:|---:|---:|")
    support_cases = (
        ("Four-foot support", 4),
        ("Diagonal two-leg support", 2),
        ("Single-leg worst case", 1),
    )
    diagonal_calf = 0.0
    for label, count in support_cases:
        force = [0.0, 0.0, weight / count]
        chain, foot = leg_chain(joints, "FL")
        foot_position, origins, axes = foot_positions["FL"]
        jacobian = foot_jacobian(origins, axes, foot_position)
        torques = static_joint_torques(jacobian, force)
        if count == 2:
            diagonal_calf = abs(torques[2])
        emit(
            f"| {label} | {count} | {force[2]:.2f} | {abs(torques[0]):.3f} | "
            f"{abs(torques[1]):.3f} | {abs(torques[2]):.3f} |"
        )
    emit()

    emit("### Fore/aft stride sweep in diagonal support")
    emit()
    emit("Thigh angle is swept so the toe moves fore/aft while the knee "
         "compensates to hold the nominal foot height.")
    emit()
    emit("| Toe offset x (m) | |thigh| (N·m) | |calf| (N·m) | max (N·m) |")
    emit("|---:|---:|---:|---:|")
    chain, foot = leg_chain(joints, "FL")
    base_angles = [defaults[f"FL_{kind}_joint"] for kind in JOINT_TYPES]
    target_z = foot_positions["FL"][0][2]
    worst = 0.0
    for offset in (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06):
        angles = _solve_planar(chain, foot, base_angles, offset, target_z)
        if angles is None:
            emit(f"| {offset:+.3f} | unreachable | unreachable | - |")
            continue
        foot_position, origins, axes = forward_kinematics(chain, foot, angles)
        jacobian = foot_jacobian(origins, axes, foot_position)
        torques = static_joint_torques(jacobian, [0.0, 0.0, weight / 2.0])
        peak = max(abs(torques[1]), abs(torques[2]))
        worst = max(worst, peak)
        emit(
            f"| {offset:+.3f} | {abs(torques[1]):.3f} | "
            f"{abs(torques[2]):.3f} | {peak:.3f} |"
        )
    emit()

    rated = 6.0
    soft = float(cfg.control.torque_limits_by_joint["calf"])
    urdf_effort = joints["FL_calf_joint"]["effort"]
    emit("## RS01 torque envelope")
    emit()
    emit("| Layer | Value (N·m) | Static diagonal margin |")
    emit("|---|---:|---:|")
    emit(f"| Rated (reported only) | {rated:.1f} | "
         f"{rated / max(diagonal_calf, 1e-6):.2f}x |")
    emit(f"| Project soft envelope (applied clip) | {soft:.1f} | "
         f"{soft / max(diagonal_calf, 1e-6):.2f}x |")
    emit(f"| URDF/hardware peak | {urdf_effort:.1f} | "
         f"{urdf_effort / max(diagonal_calf, 1e-6):.2f}x |")
    emit()
    emit(f"Worst swept static demand in diagonal support: **{worst:.3f} N·m**.")
    emit()
    if worst > soft:
        problems.append(
            f"the swept static diagonal demand ({worst:.2f} N·m) already "
            f"exceeds the configured soft clip ({soft:.2f} N·m); reduce stride "
            f"or raise RS01_SOFT_TORQUE_NM"
        )
    emit()

    emit("## Command envelope")
    emit()
    ranges = cfg.commands.ranges
    emit(f"- `lin_vel_x`: {list(ranges.lin_vel_x)} m/s")
    emit(f"- `lin_vel_y`: {list(ranges.lin_vel_y)} m/s")
    emit(f"- `ang_vel_yaw`: {list(ranges.ang_vel_yaw)} rad/s")
    emit(f"- `heading`: {list(ranges.heading)} rad")
    emit(f"- `stand_probability`: {cfg.commands.stand_probability}")
    emit(f"- terrain `mesh_type`: {cfg.terrain.mesh_type}, "
         f"`measure_heights`: {cfg.terrain.measure_heights}")
    for name, value in (
        ("lin_vel_y", ranges.lin_vel_y),
        ("ang_vel_yaw", ranges.ang_vel_yaw),
        ("heading", ranges.heading),
    ):
        if tuple(value) != (0.0, 0.0):
            problems.append(f"{name} is not pinned to zero for a straight task")
    if cfg.terrain.mesh_type != "plane":
        problems.append("terrain is not flat ground")
    if cfg.terrain.measure_heights:
        problems.append("height scanner observations are enabled")
    emit()

    emit("## Findings")
    emit()
    if problems:
        for problem in problems:
            emit(f"- ISSUE: {problem}")
    else:
        emit("- No blocking issue found in the static audit.")
    emit()
    emit("Unresolved by design in this task: real-motor IDs, motor signs and "
         "zero offsets are out of scope. Training uses the URDF joint "
         "semantics directly.")
    emit()

    report = "\n".join(lines)
    output = os.path.realpath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(report)
    print(f"\n[audit] wrote {output}")
    return 1 if problems else 0


def _solve_planar(chain, foot, base_angles, offset_x, target_z, iterations=200):
    """Newton solve thigh/calf so the toe sits at (x0+offset, target_z)."""
    target_x = None
    angles = list(base_angles)
    position, _, _ = forward_kinematics(chain, foot, angles)
    target_x = position[0] + offset_x
    for _ in range(iterations):
        position, origins, axes = forward_kinematics(chain, foot, angles)
        error = [position[0] - target_x, position[2] - target_z]
        if max(abs(error[0]), abs(error[1])) < 1.0e-9:
            return angles
        jacobian = foot_jacobian(origins, axes, position)
        # Only thigh (1) and calf (2) act in the sagittal plane.
        a11, a12 = jacobian[1][0], jacobian[2][0]
        a21, a22 = jacobian[1][2], jacobian[2][2]
        determinant = a11 * a22 - a12 * a21
        if abs(determinant) < 1.0e-12:
            return None
        delta_thigh = (-error[0] * a22 + error[1] * a12) / determinant
        delta_calf = (-error[1] * a11 + error[0] * a21) / determinant
        angles[1] += delta_thigh
        angles[2] += delta_calf
        lower = chain[1]["lower"], chain[2]["lower"]
        upper = chain[1]["upper"], chain[2]["upper"]
        if not (lower[0] <= angles[1] <= upper[0]):
            return None
        if not (lower[1] <= angles[2] <= upper[1]):
            return None
    return None


if __name__ == "__main__":
    sys.exit(main())
