"""Bounded lateral position hold for march/straight gait.

The V22 observation carries lateral VELOCITY error (obs 53) but no lateral
POSITION error (obs 52 is a literal zero in training, see assemble61). On
hardware the leg odometry stance selection is left/right asymmetric and the
robot drifts left ~0.03 m/s while marching even though the policy keeps
nulling its own estimated velocity. This hold integrates the estimated
lateral velocity in the heading frame into a session-relative offset and
injects a small bounded correcting vy into the direction-mixer target.

Operator authority always wins: standing, intentional turns and lateral
commands re-reference the offset to zero, so the hold only ever acts on
drift accumulated during the current pure march/straight episode. The raw
command observation channels (obs 57:60) are never modified; the correction
only enters the mixed target, exactly like the trained planar correction.
"""
import math


class LateralPositionHold:
    deadband_m = .02
    gain_per_s = 1.2
    correction_limit_mps = .05
    offset_limit_m = .25
    operator_vy_gate_mps = .02
    max_dt_sec = .05

    def __init__(self):
        self.reset()

    def reset(self, reason='reset'):
        self.offset = 0.
        self.correction = 0.
        self.active = False
        self.reason = reason

    def update(self, dt, velocity, confidence, command, turning, gait,
               heading_error):
        """Advance one policy tick; velocity is the estimated body [vx,vy,vz]."""
        try:
            finite = (math.isfinite(float(dt)) and math.isfinite(float(confidence))
                      and math.isfinite(float(heading_error))
                      and all(math.isfinite(float(v)) for v in command)
                      and (velocity is None
                           or all(math.isfinite(float(v)) for v in velocity)))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            self.reset('nonfinite_input')
            return
        if gait != 1. or bool(turning):
            self.reset('not_pure_march')
            return
        if abs(float(command[1])) >= self.operator_vy_gate_mps:
            self.reset('operator_lateral_command')
            return
        if float(confidence) <= 0. or velocity is None:
            # Odometry has no legal support pair: freeze offset and correction.
            # The observation-quality guard soft-holds if this persists.
            self.active = False
            self.reason = 'odometry_unconfident'
            return
        if not 0. < float(dt) <= self.max_dt_sec:
            self.active = False
            self.reason = 'dt_out_of_range'
            return
        error = float(heading_error)
        vx, vy = float(velocity[0]), float(velocity[1])
        lateral = vy*math.cos(error)-vx*math.sin(error)
        self.offset = max(-self.offset_limit_m,
                          min(self.offset_limit_m, self.offset+lateral*float(dt)))
        excess = abs(self.offset)-self.deadband_m
        if excess <= 0.:
            self.correction = 0.
        else:
            self.correction = -math.copysign(
                min(self.gain_per_s*excess, self.correction_limit_mps),
                self.offset)
        self.active = True
        self.reason = 'holding'

    def state(self):
        return dict(lateral_hold_active=bool(self.active),
                    lateral_hold_offset_m=float(self.offset),
                    lateral_hold_correction_mps=float(self.correction),
                    lateral_hold_reason=str(self.reason))
