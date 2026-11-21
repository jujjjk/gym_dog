"""Audit a slow, single-foot step climb on the battery-corrected URDF4."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy.interpolate import interp1d

from design_v2 import IK, LEGS, NAMES, ROOT, setup
from physical_v3 import crossmat, rot, rpy

PLANT = ROOT / 'artifacts/urdf4_amp_design_20261116_v3/plant'
URDF = ROOT / 'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf'
RADIUS = .020
RATE = np.tile([2., 2.6, 3.2], 4)
SCALE = np.tile([.15, .25, .25], 4)
CONTROL_DT = .02
# V4 reset/configured target center, ordered FL/FR/RL/RR as in the runtime.
DEFAULT_Q = np.array([0., -.41428, 1.28031, 0., .41428, -1.28031]*2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_scene(out, height):
    meta = json.loads((PLANT / 'manifest.json').read_text())
    assert meta['urdf_sha256'] == sha(URDF) and meta['scene_sha256'] == sha(PLANT / 'scene.xml')
    tree = ET.parse(PLANT / 'scene.xml')
    if height > 0:
        ET.SubElement(tree.find('worldbody'), 'geom', name='step', type='box',
                      pos=f'.8 0 {height/2}', size=f'.8 .7 {height/2}', rgba='.36 .48 .57 1',
                      contype='1', conaffinity='2', friction='.8 0 0',
                      solref='.0065 1', solimp='.99 .999 .0001 .5 2', condim='3')
    path = out / 'scene.xml'
    with path.open('wb') as file:
        tree.write(file)
    return mj.MjModel.from_xml_path(str(path))


def rotation(pitch):
    quat = np.array([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
    return quat, rot(quat)


def smooth(x):
    return x*x*(3-2*x)


def swing_points(old, new, u, clearance):
    result = old.copy()
    result[0] += (new[0]-old[0])*smooth(np.clip((u-.28)/.44, 0, 1))
    high = max(old[2], new[2]) + clearance
    if u < .28:
        result[2] += (high-old[2])*smooth(u/.28)
    else:
        result[2] = high + (new[2]-high)*smooth(np.clip((u-.72)/.28, 0, 1))
    return result


class Planner:
    def __init__(self, model):
        self.m = model
        self.ik = IK(model)
        self.d = mj.MjData(model)
        self.qa, self.va, self.fb, self.limits = setup(model)
        self.seed = self.ik.q.copy()

    def solve(self, base, pitch, feet, qstart=None):
        quat, matrix = rotation(pitch)
        self.ik.q = self.seed.copy() if qstart is None else qstart.copy()
        q, error = self.ik.solve((feet-base) @ matrix)
        self.d.qpos[:7] = np.r_[base, quat]
        self.d.qpos[self.qa] = q
        mj.mj_forward(self.m, self.d)
        limb_collision = 0.
        for contact in self.d.contact:
            bodies = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
            if (0 in bodies and max(bodies) not in self.fb) or (0 not in bodies):
                limb_collision = max(limb_collision, -contact.dist)
        joint_margin = float(np.minimum(q-self.limits[:, 0], self.limits[:, 1]-q).min())
        command_room = float(np.min(4*SCALE - abs(q-DEFAULT_Q)/.928))
        return q, np.array([error, limb_collision, joint_margin, command_room])

    def find_swing_pose(self, feet, leg, destination, clearance):
        stance = np.arange(4) != leg
        support = feet[stance, :2]
        hull = ConvexHull(support)
        seed_xy = feet[:, :2].mean(0)
        # Sample well inside the triangular support area, including positions
        # near the previous body center and towards either supporting pair.
        options = []
        interpolated = []
        for t in (0., .27, .50, .73, 1.):
            test = feet.copy()
            test[leg] = swing_points(test[leg], destination, t, clearance)
            interpolated.append(test)
        for wx in (.13, .23, .33, .43, .53, .63, .73):
            for wy in (.13, .23, .33, .43, .53, .63, .73):
                if wx+wy > .88:
                    continue
                xy = np.array([wx, wy, 1-wx-wy]) @ support
                options.append(xy)
        options.sort(key=lambda xy: np.linalg.norm(xy-seed_xy))
        # Gravity-only support is feasible if the actual center of mass is
        # inside the three-point polygon. Check joint room during swing too.
        local_height = feet[:, 2].mean()-RADIUS
        pitch_hint = -.55*(feet[:2, 2].mean()-feet[2:, 2].mean())/.43
        best = None
        for xy in options:
            for z in local_height + np.array([.27,.29,.31,.33,.35,.25,.37]):
                for pitch in (pitch_hint, pitch_hint-.12, pitch_hint+.12, pitch_hint-.23):
                    base = np.r_[xy, z]
                    metrics = []
                    previous = None
                    stability = 1.
                    for foot_set in interpolated:
                        previous, checks = self.solve(base, pitch, foot_set, previous)
                        if checks[0] > .003 or checks[1] > .002 or checks[2] < .015 or checks[3] < -.06:
                            break
                        edge_distance = -(hull.equations[:, :2] @ self.d.subtree_com[1, :2] + hull.equations[:, 2]).max()
                        stability = min(stability, edge_distance)
                        metrics.append(checks)
                    if len(metrics) != len(interpolated) or stability < .009:
                        continue
                    data = np.array(metrics)
                    # Prefer margin at the joint stops and the physical output envelope.
                    score = 2*abs(z-(local_height+.30)) + np.linalg.norm(xy-seed_xy) + .04*abs(pitch-pitch_hint)
                    score += max(0., .055-data[:, 2].min()) + 2*max(0., .025-data[:, 3].min()) + max(0., .023-stability)
                    if best is None or score < best[0]:
                        best = (score, base, pitch, stability)
            if best is not None and best[0] < .08:
                break
        return best


def build_reference(model, height, short_strides=False, same_level_clearance=.028, rear_right_first=False):
    planner = Planner(model)
    feet = np.array([[-.075,.1475,RADIUS],[-.075,-.1475,RADIUS],[-.505,.1475,RADIUS],[-.505,-.1475,RADIUS]])
    base = np.array([-.29, 0., .29])
    pitch = 0.
    values = []
    stage_names = []
    support_margins = []

    def add(name, duration, make):
        index = len(stage_names)
        stage_names.append(name)
        for k in range(round(duration / CONTROL_DT)):
            b, p, f, weights = make((k+1)*CONTROL_DT/duration)
            values.append((b.copy(), float(p), f.copy(), weights.copy(), index))

    add('settle', 2.5, lambda u: (base, pitch, feet, np.ones(4)))
    sequence = [(0,.080),(1,.080),(2,-.255),(3,-.255),(0,.290),(1,.290),
                (2,-.073),(3,-.073),(0,.437),(1,.437),(2,.066),(3,.066)]
    if short_strides:
        sequence = [(0,.070),(1,.070),(2,-.395),(3,-.395),(2,-.275),(3,-.275),
                    (0,.195),(1,.195),(2,-.165),(3,-.165),(0,.320),(1,.320),
                    (2,-.065),(3,-.065),(2,.066),(3,.066)]
    if rear_right_first:
        sequence = [(5-leg if leg >= 2 else leg, x) for leg, x in sequence]
    failure = None
    for leg, x in sequence:
        destination = np.array([x, feet[leg, 1], RADIUS + (height if x>0 else 0.)])
        clearance = same_level_clearance if abs(destination[2]-feet[leg, 2]) < .001 else .028
        pose = planner.find_swing_pose(feet, leg, destination, clearance)
        if pose is None:
            failure = dict(leg=LEGS[leg], x_m=x, reason='No sampled collision-free pose with a statically supported foot swing')
            break
        _, next_base, next_pitch, margin = pose
        support_margins.append(float(margin))
        add(f'{LEGS[leg]} shift to {x:+.3f}', 1.5,
            lambda u: (base + smooth(u)*(next_base-base), pitch+smooth(u)*(next_pitch-pitch), feet, np.ones(4)))
        base, pitch = next_base, next_pitch
        load = np.ones(4)
        load[leg] = 0.
        add(f'{LEGS[leg]} unload', .4, lambda u: (base, pitch, feet, 1-smooth(u)*(1-load)))

        def swing(u):
            xy = feet.copy()
            xy[leg] = swing_points(xy[leg], destination, u, clearance)
            return base, pitch, xy, load
        add(f'{LEGS[leg]} swing to {x:+.3f}', 1.55, swing)
        feet[leg] = destination
        add(f'{LEGS[leg]} accept load', .5, lambda u: (base, pitch, feet, load+smooth(u)*(1-load)))
    if failure is None:
        end = np.r_[feet[:, :2].mean(0), .29 + height]
        add('four feet on top', 2., lambda u: (base+smooth(u)*(end-base), pitch*(1-smooth(u)), feet, np.ones(4)))
        base, pitch = end, 0.
        add('final hold', 2., lambda u: (base, pitch, feet, np.ones(4)))

    joints, checks = [], []
    previous = None
    for b, p, f, _, _ in values:
        previous, diagnostic = planner.solve(b, p, f, previous)
        joints.append(previous)
        checks.append(diagnostic)
    return values, np.array(joints), np.array(checks), stage_names, support_margins, failure


def physical_rollout(model, reference, stages, height, feed=.95, swing_world_weight=1.,
                     unilateral_support=False, planar_feedback=1., attitude_feedback=1.):
    """Free base with physical contacts and the same V4 RS01 target chain."""
    d = mj.MjData(model)
    qa, va, feet_ids, limits = setup(model)
    planner = Planner(model)
    plans = reference['joints']; bases = reference['base']; pitches = reference['pitch']
    planned_feet = reference['feet']; loads = reference['load']; stage = reference['stage']
    dt = model.opt.timestep
    substeps = round(CONTROL_DT/dt)
    mass = float(model.body_mass.sum())
    default = DEFAULT_Q
    initial = plans[0]
    d.qpos[:7] = np.r_[bases[0]+[0., 0., .003], rotation(pitches[0])[0]]
    d.qpos[qa] = initial
    mj.mj_forward(model, d)
    target = default + (initial-default)/.928
    response = initial.copy()
    delay = np.tile(target, (9, 1))
    rates = np.zeros(12)
    jacobian = np.zeros((3, model.nv))
    gravity = np.array([0., 0., 9.81])
    filtered_ff = np.zeros(12)
    motor_squared = np.zeros(12)
    peak_raw = np.zeros(12)
    peak_motor = np.zeros(12)
    seconds_above6 = np.zeros(12)
    streak_above6 = np.zeros(12)
    longest_above6 = np.zeros(12)
    seconds_saturated = np.zeros(12)
    peak_velocity = np.zeros(12)
    joint_margin = np.full(12, np.inf)
    frames = []
    first_nonfoot_contact = None
    first_front_top = None
    first_four_top = None
    top_hold_s = 0.
    minimum_swing_clearance = np.inf
    reason = 'trajectory ended'
    u0 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'step')
    for k, q_plan in enumerate(plans):
        relative = d.qpos[:3] - bases[k]
        R = rot(d.qpos[3:7])
        attitude = rpy(d.qpos[3:7]) - [0., pitches[k], 0.]
        ref_velocity = (bases[k]-bases[max(0, k-1)])/CONTROL_DT
        # A bounded centroidal wrench is used only to select joint commands.
        feedback = -np.array([32., 32., 105.])*relative + np.array([9., 9., 15.])*(ref_velocity-d.qvel[:3])
        feedback[:2] *= planar_feedback
        force = mass * (gravity + feedback.clip(-5., 5.))
        moment = attitude_feedback * R @ (-np.array([15., 18., 10.])*attitude - np.array([1.7, 1.7, 1.])*d.qvel[3:6])
        desired = np.r_[force, moment]
        contacts = d.xpos[feet_ids].copy()
        matrix = np.vstack((np.tile(np.eye(3), (1, 4)), np.hstack([crossmat(p-d.subtree_com[1]) for p in contacts])))
        w = np.repeat(loads[k], 3)
        for _ in range(4):
            gains = (matrix*w) @ matrix.T
            load = (w*(matrix.T @ np.linalg.solve(gains + 1.e-4*np.eye(6), desired))).reshape(4, 3)
            negative = load[:, 2] < -1.e-7
            if not unilateral_support or not negative.any() or np.count_nonzero(w[::3]) <= 2:
                break
            least = int(load[:, 2].argmin())
            w[3*least:3*least+3] = 0.
        load[:, 2] = load[:, 2].clip(0, 130)
        load[:, :2] = load[:, :2].clip(-.65*load[:, 2:3], .65*load[:, 2:3])
        ff = d.qfrc_bias[va].copy()
        for foot, contact_force in zip(feet_ids, load):
            mj.mj_jacBody(model, d, jacobian, None, foot)
            ff -= jacobian[:, va].T @ contact_force
        filtered_ff += .35*(ff-filtered_ff)

        q = q_plan.copy()
        swinging = loads[k] == 0
        if swinging.any() and ' swing ' in stages[stage[k]]:
            # Keep stance feet on the nominal body path. Correct the swinger's
            # world target for residual body pose error.
            _, planned_matrix = rotation(pitches[k])
            q_target = (planned_feet[k]-bases[k])@planned_matrix
            actual_target = (planned_feet[k]-d.qpos[:3])@R
            correction = (actual_target-q_target).clip(-.055,.055)
            q_target[swinging] += swing_world_weight * correction[swinging]
            planner.ik.q = q.copy()
            swing_solution, _ = planner.ik.solve(q_target)
            q[np.repeat(swinging, 3)] = swing_solution[np.repeat(swinging, 3)]
        desired = default+(q-default+feed*filtered_ff.clip(-14,14)/40.)/.928
        desired = desired.clip(default-4*SCALE, default+4*SCALE).clip(limits[:, 0], limits[:, 1])
        want = ((desired-target)/CONTROL_DT).clip(-RATE,RATE)
        rates = (rates+(want-rates).clip(-RATE*30*CONTROL_DT,RATE*30*CONTROL_DT)).clip(-RATE,RATE)
        advanced = target+rates*CONTROL_DT
        arrived = (desired-target)*(desired-advanced) <= 0
        target = np.where(arrived, desired, advanced)
        rates[arrived] = 0.
        stage_raw = 0.
        stage_name = stages[stage[k]]
        for tick in range(substeps):
            if tick % 2 == 0:
                delay[1:] = delay[:-1]
                delay[0] = target
                response += (1-np.exp(-.005/.0319))*(default+.928*(delay[-1]-default)-response)
            velocity = d.qvel[va]
            raw = 40*(response-d.qpos[qa])-velocity
            motor = raw.clip(-14., 14.)
            d.qfrc_applied[:] = 0.
            d.qfrc_applied[va] = motor-.156*np.tanh(velocity/.05)
            peak_raw = np.maximum(peak_raw, abs(raw)); peak_motor = np.maximum(peak_motor, abs(motor))
            peak_velocity = np.maximum(peak_velocity, abs(velocity))
            seconds_above6 += (abs(motor) > 6.)*dt
            streak_above6 = np.where(abs(motor) > 6., streak_above6 + dt, 0.)
            longest_above6 = np.maximum(longest_above6, streak_above6)
            seconds_saturated += (abs(raw) > 14.)*dt
            motor_squared += motor**2*dt
            joint_margin = np.minimum(joint_margin, np.minimum(d.qpos[qa]-limits[:, 0], limits[:, 1]-d.qpos[qa]))
            stage_raw = max(stage_raw, float(abs(raw).max()))
            mj.mj_step(model, d)
            if first_nonfoot_contact is None:
                for contact_id, contact in enumerate(d.contact):
                    bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                    if 0 in bodies and max(bodies) not in feet_ids and contact.dist <= 0.:
                        wrench = np.zeros(6)
                        mj.mj_contactForce(model, d, contact_id, wrench)
                        if wrench[0] >= 2.:
                            first_nonfoot_contact = dict(time_s=(k*substeps+tick+1)*dt,
                                stage=stage_name, body=model.body(int(max(bodies))).name)
        mj.mj_forward(model, d)
        time_s = (k+1)*CONTROL_DT
        top = np.zeros(4, dtype=bool)
        for ji, contact in enumerate(d.contact):
            geom = [contact.geom1, contact.geom2]
            body = model.geom_bodyid[geom]
            if 0 not in body or contact.dist > 0:
                continue
            wrench = np.zeros(6)
            mj.mj_contactForce(model, d, ji, wrench)
            if wrench[0] < 2.:
                continue
            foot = np.flatnonzero(feet_ids == max(body))
            if len(foot) and u0 >= 0 and u0 in geom and abs(contact.frame[2]) > .75 and contact.pos[2] > height-.011:
                top[foot[0]] = True
            elif not len(foot) and first_nonfoot_contact is None:
                first_nonfoot_contact = dict(time_s=time_s, stage=stages[stage[k]], body=model.body(int(max(body))).name)
        if top[:2].all() and first_front_top is None:
            first_front_top = time_s
        if top.all() and first_four_top is None:
            first_four_top = time_s
        top_hold_s = top_hold_s + CONTROL_DT if top.all() and np.linalg.norm(d.qvel[:3]) < .06 else 0.
        near_edge = swinging & (abs(d.xpos[feet_ids, 0]) < RADIUS+.01)
        if height and near_edge.any():
            minimum_swing_clearance = min(minimum_swing_clearance, float((d.xpos[feet_ids[near_edge], 2]-RADIUS-height).min()))
        frames.append((time_s, d.qpos.copy(), d.qvel.copy(), top.copy(), stage_raw, stage[k]))
        if d.qpos[2] < .13 or np.max(abs(rpy(d.qpos[3:7])[:2])) > .65 or not np.isfinite(d.qpos).all():
            reason = 'fall or nonfinite state'
            break
    report = dict(physically_executed=True, applied_only_joint_torques=True, height_m=height, duration_s=time_s,
        feed=feed, swing_world_weight=swing_world_weight, unilateral_support=unilateral_support,
        planar_feedback=planar_feedback, attitude_feedback=attitude_feedback,
        stop_reason=reason, last_stage=stages[stage[k]],
        first_front_pair_on_top_s=first_front_top, first_four_feet_on_top_s=first_four_top,
        final_four_foot_hold_s=top_hold_s, final_base_position_m=d.qpos[:3].tolist(),
        first_nonfoot_contact=first_nonfoot_contact, min_edge_clearance_m=None if np.isinf(minimum_swing_clearance) else minimum_swing_clearance,
        joint_order=NAMES, raw_torque_peak_Nm=peak_raw.tolist(), motor_peak_Nm=peak_motor.tolist(),
        motor_rms_Nm=np.sqrt(motor_squared/time_s).tolist(), motor_seconds_over_6Nm=seconds_above6.tolist(),
        motor_longest_continuous_over_6Nm_s=longest_above6.tolist(),
        saturation_seconds=seconds_saturated.tolist(), peak_joint_velocity_rad_s=peak_velocity.tolist(),
        min_joint_margin_deg=np.rad2deg(joint_margin).tolist(), actuator_cap_Nm=14.,
        climbed_and_held=bool(top_hold_s >= 1. and reason == 'trajectory ended'),
        passed=bool(top_hold_s >= 1. and reason == 'trajectory ended' and first_nonfoot_contact is None and peak_raw.max() <= 17.))
    return report, dict(time_s=[f[0] for f in frames], qpos=[f[1] for f in frames], qvel=[f[2] for f in frames],
                        top=[f[3] for f in frames], raw_substep_peak=[f[4] for f in frames], stage=[f[5] for f in frames])


def static_load_audit(model, reference, stages):
    """Best-case static support with unilateral friction and minimum max torque."""
    data = mj.MjData(model)
    qa, va, feet, _ = setup(model)
    jac = np.zeros((3, model.nv))
    rows = []
    cost = np.r_[np.zeros(12), 1.]
    rhs = np.array([0., 0., model.body_mass.sum()*9.81, 0., 0., 0.])
    for k in range(0, len(reference['stage']), 10):
        data.qpos[:7] = np.r_[reference['base'][k], rotation(reference['pitch'][k])[0]]
        data.qpos[qa] = reference['joints'][k]
        mj.mj_forward(model, data)
        points = data.xpos[feet] - [0., 0., RADIUS]
        equality = np.vstack((np.tile(np.eye(3), (1, 4)), np.hstack([crossmat(p-data.subtree_com[1]) for p in points])))
        equality = np.c_[equality, np.zeros(6)]
        tau_from_force = np.zeros((12, 12))
        for j, body in enumerate(feet):
            mj.mj_jac(model, data, jac, None, points[j], int(body))
            tau_from_force[:, j*3:(j+1)*3] = -jac[:, va].T
        upper = [*np.c_[tau_from_force, -np.ones(12)], *np.c_[-tau_from_force, -np.ones(12)]]
        bound = np.r_[-data.qfrc_bias[va], data.qfrc_bias[va], np.zeros(16)]
        for j in range(4):
            # Inscribed square within the Coulomb friction cone.
            for axis in (0, 1):
                for sign in (-1., 1.):
                    row = np.zeros(13)
                    row[3*j+axis] = sign
                    row[3*j+2] = -.65/np.sqrt(2.)
                    upper.append(row)
        limits = []
        for w in reference['load'][k]:
            limits.extend([(None, None), (None, None), (0, 130*w)])
        solution = linprog(cost, A_ub=upper, b_ub=bound, A_eq=equality, b_eq=rhs,
                           bounds=limits+[(0, None)], method='highs')
        rows.append(dict(time_s=(k+1)*CONTROL_DT, stage=stages[reference['stage'][k]],
                         min_peak_Nm=float(solution.fun) if solution.success else None))
    peaks = [row for row in rows if row['min_peak_Nm'] is not None]
    return dict(static_only=True, samples=len(rows), feasible_samples=len(peaks),
                worst=max(peaks, key=lambda row: row['min_peak_Nm']) if peaks else None, cases=rows)


def retime(reference, duration_scale):
    if duration_scale == 1.:
        return reference
    times = np.arange(len(reference['stage']))
    grid = np.minimum(np.arange(round(times[-1]*duration_scale)+1)/duration_scale, times[-1])
    return {key: (data[grid.astype(int)] if key == 'stage' else interp1d(times, data, axis=0)(grid))
            for key, data in reference.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=float, default=.14)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulate', action='store_true', help='Run the generated reference on the free-base plant')
    parser.add_argument('--existing-reference', action='store_true')
    parser.add_argument('--short-strides', action='store_true')
    parser.add_argument('--same-level-clearance', type=float, default=.028)
    parser.add_argument('--rear-right-first', action='store_true')
    parser.add_argument('--static-load', action='store_true')
    parser.add_argument('--duration-scale', type=float, default=1.)
    parser.add_argument('--feed', type=float, default=.95)
    parser.add_argument('--swing-world-weight', type=float, default=1.)
    parser.add_argument('--unilateral-support', action='store_true')
    parser.add_argument('--planar-feedback', type=float, default=1.)
    parser.add_argument('--attitude-feedback', type=float, default=1.)
    parser.add_argument('--physical-output', type=Path, help='Write verification trace in a separate directory')
    args = parser.parse_args()
    if args.existing_reference:
        if not args.simulate:
            parser.error('--existing-reference requires --simulate')
        kin = json.loads((args.output / 'kinematics.json').read_text())
        assert kin['height_m'] == args.height and kin['urdf_sha256'] == sha(URDF)
        model = mj.MjModel.from_xml_path(str(args.output / 'scene.xml'))
        run_physics(args, model, kin)
        if args.static_load:
            audit = static_load_audit(model, np.load(args.output / 'reference.npz'), kin['stages'])
            (args.output / 'static_load.json').write_text(json.dumps(audit, indent=2)+'\n')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    model = make_scene(args.output, args.height)
    values, q, check, stages, margins, failure = build_reference(model, args.height, args.short_strides,
                                                               args.same_level_clearance, args.rear_right_first)
    report = dict(height_m=args.height, mass_kg=float(model.body_mass.sum()), urdf_sha256=sha(URDF),
                  geometric_search_failure=failure, planned_seconds=len(values)*CONTROL_DT, short_strides=args.short_strides,
                  same_level_clearance_m=args.same_level_clearance,
                  rear_right_first=args.rear_right_first,
                  max_ik_error_m=float(check[:, 0].max()), nonfoot_penetration_m=float(check[:, 1].max()),
                  min_joint_margin_deg=float(np.rad2deg(check[:, 2].min())), min_uncompensated_command_room_rad=float(check[:, 3].min()),
                  min_planned_support_margin_m=min(margins, default=0.), stages=stages)
    (args.output / 'kinematics.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output / 'reference.npz', joints=q, checks=check, base=[a[0] for a in values],
                        pitch=[a[1] for a in values], feet=[a[2] for a in values], load=[a[3] for a in values], stage=[a[4] for a in values])
    print(json.dumps(report, indent=2), flush=True)
    if args.simulate:
        run_physics(args, model, report)
    if args.static_load:
        audit = static_load_audit(model, np.load(args.output / 'reference.npz'), stages)
        (args.output / 'static_load.json').write_text(json.dumps(audit, indent=2)+'\n')


def run_physics(args, model, kin):
    ref = retime(np.load(args.output / 'reference.npz'), args.duration_scale)
    report, data = physical_rollout(model, ref, kin['stages'], args.height, args.feed, args.swing_world_weight,
                                   args.unilateral_support, args.planar_feedback, args.attitude_feedback)
    report.update(duration_scale=args.duration_scale, scene_sha256=sha(args.output/'scene.xml'),
                  reference_sha256=sha(args.output/'reference.npz'), urdf_sha256=sha(URDF))
    out = args.physical_output or args.output
    if out != args.output:
        out.mkdir(parents=True, exist_ok=False)
    (out / 'physical.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(out / 'rollout.npz', **data)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
