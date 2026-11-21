"""Per-leg timing calibration for a shared diagonal clock, with unchanged geometry.

Asymmetric joint axes and actuator response need slightly different commanded
timing. Acceptance is always based on the actual contacts, never this clock.
"""
import numpy as np
from design_v2 import IK, feet_target


def design(m, cmd, T, duty, clear, height, width, dt=.02, xoffset=0.,
           front_clear=None, rear_clear=None, leg_clearance=None,
           leg_stance_ratio=None, stride_scale=None):
    ik = IK(m)
    n = round(T / dt)
    dt = T / n
    duties = np.full(4, duty) if leg_stance_ratio is None else np.asarray(leg_stance_ratio)
    scales = np.ones(3) if stride_scale is None else np.asarray(stride_scale)
    q, feet, phases, stance, error = [], [], [], [], []
    for k in range(n):
        f = np.zeros((4, 3)); p = np.zeros(4); support = np.zeros(4, bool)
        for j in range(4):
            fj, pj, sj = feet_target(k/n, np.asarray(cmd)*scales, T, duties[j],
                clear, height, width, xoffset, front_clear, rear_clear, leg_clearance)
            f[j], p[j], support[j] = fj[j], pj[j], sj[j]
        solution, residual = ik.solve(f)
        q.append(solution); feet.append(f); phases.append(p); stance.append(support); error.append(residual)
    q = np.asarray(q)
    return dict(q=q, dq=(np.roll(q, -1, axis=0)-np.roll(q, 1, axis=0))/(2*dt),
        feet=np.asarray(feet), phase=np.asarray(phases), stance=np.asarray(stance),
        max_ik_error_m=max(error), dt=dt)
