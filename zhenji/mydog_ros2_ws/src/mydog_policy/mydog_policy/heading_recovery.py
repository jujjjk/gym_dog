"""B23500 heading monitor. Disagreement is corrected, not a reason to stand."""
import math


class HeadingRecovery:
    # These limits only classify the yaw/gyro monitor. They do not scale the
    # heading correction down and they do not request a soft stand.
    severe_mean_rad_s = .12
    severe_abs_rad_s = .30

    def __init__(self):
        self.weight = 1.

    def update(self, now, monitor, walking):
        mean = float(monitor.get('mean_error_rad_s', 0.))
        absolute = float(monitor.get('abs_error_rad_s', 0.))
        finite = math.isfinite(mean) and math.isfinite(absolute)
        severe = not finite or abs(mean) >= self.severe_mean_rad_s or absolute >= self.severe_abs_rad_s
        healthy = finite and bool(monitor.get('ready', False)) and bool(monitor.get('healthy', False)) and not severe
        # Keep the full heading error in the direction controller.
        self.weight = 1.
        if not walking:
            state = 'idle'
        else:
            state = 'severe' if severe else ('normal' if healthy else 'degraded')
        return dict(heading_recovery_state=state, heading_correction_weight=self.weight,
                    heading_severe=severe, heading_monitor_healthy=healthy,
                    heading_severe_mean_limit_rad_s=self.severe_mean_rad_s,
                    heading_severe_abs_limit_rad_s=self.severe_abs_rad_s)
