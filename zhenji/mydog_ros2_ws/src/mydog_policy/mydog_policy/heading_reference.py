"""Heading for the direction controller: gyro integration, magnetometer-gated.

Every B23500 walk capture since 2026-09-20 shows the same picture at vx 0.4:
the bias-corrected gyro integrates to 20-50 deg of right turn in 5-7 s, the
vendor's fused yaw reports a few degrees of left drift, and the magnetometer
norm swings from a steady 80 uT at stand to 170-200 uT while the motors pull
current. The operator confirmed the robot turns right. The fused yaw is being
dragged by the motor field, so steering on it turns the robot the wrong way.

This estimator integrates the calibrated gyro while walking and only lets the
fused yaw pull the heading when the field norm is back near the standing
baseline. Standing still with a quiet field it tracks the fused yaw closely,
so the heading target defined at walk start is the same value as before.
"""
import math


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class GyroHeadingReference:
    def __init__(self, mag_tolerance_ut=8., walking_pull_tau_sec=5.,
                 standing_pull_tau_sec=.5, baseline_tau_sec=2., max_dt_sec=.1):
        self.mag_tolerance_ut = float(mag_tolerance_ut)
        self.walking_pull_tau_sec = float(walking_pull_tau_sec)
        self.standing_pull_tau_sec = float(standing_pull_tau_sec)
        self.baseline_tau_sec = float(baseline_tau_sec)
        self.max_dt_sec = float(max_dt_sec)
        self.reset()

    def reset(self, heading=None, now=None):
        self.heading = None if heading is None else float(heading)
        self.last_time = None if now is None else float(now)
        self.baseline_ut = None
        self.diagnostics = {}

    def update(self, now, fused_yaw, gyro_z, mag_norm_ut, walking):
        now, fused_yaw, gyro_z = float(now), float(fused_yaw), float(gyro_z)
        if not (math.isfinite(now) and math.isfinite(fused_yaw) and math.isfinite(gyro_z)):
            raise ValueError('non-finite heading input')
        mag_ok = mag_norm_ut is not None and math.isfinite(float(mag_norm_ut))
        mag_norm_ut = float(mag_norm_ut) if mag_ok else float('nan')
        if self.heading is None or self.last_time is None:
            self.heading, self.last_time = fused_yaw, now
            dt = 0.
        else:
            dt = now - self.last_time
            self.last_time = now
            if not 0. <= dt <= self.max_dt_sec:
                # Clock jump or long stall: keep the heading, skip integration.
                dt = 0.
        self.heading = wrap_pi(self.heading + gyro_z*dt)
        quiet = bool(mag_ok and self.baseline_ut is not None
                     and abs(mag_norm_ut-self.baseline_ut) <= self.mag_tolerance_ut)
        if not walking and mag_ok:
            # The standing field (motors holding stance) is the baseline by
            # definition; follow it slowly and freeze it during walk.
            if self.baseline_ut is None:
                self.baseline_ut = mag_norm_ut
                quiet = True
            elif dt > 0.:
                alpha = 1.-math.exp(-dt/self.baseline_tau_sec)
                self.baseline_ut += alpha*(mag_norm_ut-self.baseline_ut)
                quiet = abs(mag_norm_ut-self.baseline_ut) <= self.mag_tolerance_ut
        tau = None
        if quiet:
            tau = self.walking_pull_tau_sec if walking else self.standing_pull_tau_sec
        if tau is not None and dt > 0.:
            alpha = 1.-math.exp(-dt/tau)
            self.heading = wrap_pi(self.heading + alpha*wrap_pi(fused_yaw-self.heading))
        source = ('gyro_integrated' if tau is None else
                  ('gyro_with_slow_fused_pull' if walking else 'fused_tracking'))
        self.diagnostics = dict(
            heading_reference_rad=self.heading, heading_fused_yaw_rad=fused_yaw,
            heading_reference_minus_fused_rad=wrap_pi(self.heading-fused_yaw),
            heading_source=source, mag_norm_ut=mag_norm_ut,
            mag_norm_baseline_ut=self.baseline_ut, mag_field_quiet=quiet)
        return self.heading
