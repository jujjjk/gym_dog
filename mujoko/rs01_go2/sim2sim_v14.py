"""Same dual-clock, non-ideal RS01 actuator as the V14 PhysX task."""
import numpy as np
import mujoco
from sim2sim import compute_rs01_torques
from sim2sim_v13 import V13Sim, run, get_parser


class V14Sim(V13Sim):
    task_name = 'rs01_omni_v14_actuator_parity'
    def __init__(self, scene, policy, command):
        super().__init__(scene, policy, command)
        c = self.cfg.get('v14')
        if not c or self.cfg['task'] != self.task_name:
            raise ValueError('Use export_v14.py; V13 is a different feedback contract')
        self.response_dt = float(c['response_update_dt_s'])
        self.feedback_dt = float(c['feedback_update_dt_s'])
        ratio = self.response_dt / self.feedback_dt
        self.response_substeps = round(ratio)
        if self.response_substeps < 1 or abs(ratio-self.response_substeps)>1e-6:
            raise ValueError('Inconsistent RS01 response clock')
        if abs(self.feedback_dt-self.physics_dt)>1e-9 or self.integration_substeps != 1:
            raise ValueError('PD must be recomputed on every physical integration step')
        if self.decimation % self.response_substeps:
            raise ValueError('Policy and response clocks must align')
        if c['speed_limit_semantics'] != 'terminate_outside_validated_domain_no_velocity_projection':
            raise ValueError('Unsupported speed-domain semantics')
        self.speed_limit = float(c['speed_validity_limit_rad_s'])
        delay = np.asarray(self.cfg['control']['observed_closed_loop_delay_s'])
        self.delay_steps = np.rint(delay/self.response_dt).astype(int)
        self.max_delay_steps = int(self.delay_steps.max())
        self.response_alpha = 1-np.exp(-self.response_dt/self.time_constant)
        self.reset()

    def reset(self):
        super().reset()
        self.feedback_tick = 0
        self.step_overspeed = False
        self.step_max_speed = 0.

    def _observe_speed_domain(self):
        speed = float(np.max(np.abs(self.data.qvel[self.qvel_indices])))
        self.step_max_speed = max(self.step_max_speed, speed)
        self.step_overspeed |= not np.isfinite(speed) or speed >= self.speed_limit

    def physics_step(self):
        self._observe_speed_domain()
        if self.feedback_tick % self.response_substeps == 0:
            self.target_history = np.roll(self.target_history, 1, axis=0)
            self.target_history[0] = self.limited_target
            delayed = self.target_history[self.delay_steps, np.arange(len(self.names))]
            self.last_delayed_target = delayed.copy()
            self.response_target += self.response_alpha * (
                self.default+self.response_gain*(delayed-self.default)-self.response_target)
        self.feedback_tick += 1
        raw, motor, applied = compute_rs01_torques(
            self.response_target, self.data.qpos[self.qpos_indices], self.data.qvel[self.qvel_indices],
            self.kp, self.kd, self.peak_limit, self.friction, self.friction_smoothing)
        self.data.ctrl[self.actuator_indices] = applied
        mujoco.mj_step(self.model, self.data)
        self.last_raw_torque = raw; self.last_motor_torque = motor; self.last_applied_torque = applied
        self._observe_speed_domain()

    def control_step(self):
        self.step_overspeed = False; self.step_max_speed = 0.
        super().control_step()


if __name__ == '__main__':
    p=get_parser();args=p.parse_args()
    if args.duration<=0 or args.settle_seconds<0 or not 0<=args.phase<1:
        p.error('Require duration>0, settle>=0, 0<=phase<1')
    run(args, sim_class=V14Sim)
