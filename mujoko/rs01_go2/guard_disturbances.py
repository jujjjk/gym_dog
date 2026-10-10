"""Deterministic external wrenches; never edits robot qpos/qvel or motor ctrl."""
import math
import numpy as np
import mujoco


class DisturbanceMixin:
    payload_kg = 0.
    pushes = False
    disturbance_seed = 20261014
    latest = None

    def __init__(self, *args):
        super().__init__(*args)
        type(self).latest = self
        self.push_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'Trunk')
        if self.push_body < 1:
            raise ValueError('Missing RS01 Trunk')
        self.original_mass = float(self.model.body_mass.sum())
        if self.payload_kg:
            before_q, before_dq = self.data.qpos.copy(), self.data.qvel.copy()
            mass = self.model.body_mass[self.push_body]
            # Equivalent distributed trunk mass, same COM and radius of gyration.
            self.model.body_inertia[self.push_body] *= (mass+self.payload_kg)/mass
            self.model.body_mass[self.push_body] += self.payload_kg
            # mj_setConst uses qpos0 and can change the supplied scratch data.
            # Never pass the live robot state here.
            mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
            mujoco.mj_forward(self.model, self.data)
            if not (np.array_equal(before_q, self.data.qpos) and
                    np.array_equal(before_dq, self.data.qvel)):
                raise RuntimeError('Payload initialization changed physical state')
        if not np.isclose(self.model.body_mass.sum()-self.original_mass, self.payload_kg):
            raise RuntimeError('Payload mass mismatch')
        self.push_rng = np.random.default_rng(self.disturbance_seed)
        self.next_push = self.push_rng.uniform(4., 7.)
        self.push_end = -1.
        self.push_events = []
        self.push_impulse = 0.
        self.current_force = np.zeros(3)
        self.illegal_names = set()

    def contact_diagnostics(self):
        result = super().contact_diagnostics()
        if hasattr(self, 'illegal_names'):
            self.illegal_names.update(result[2])
        return result

    def physics_step(self):
        now = float(self.data.time)
        if self.pushes and now >= self.next_push:
            angle = self.push_rng.uniform(0., 2.*math.pi)
            peak = self.push_rng.uniform(6., 18.)
            self.push_duration = self.push_rng.uniform(.1, .2)
            self.push_start = now
            self.push_end = now+self.push_duration
            self.push_vector = peak*np.array([math.cos(angle), math.sin(angle), 0.])
            self.next_push = self.push_end+self.push_rng.uniform(6., 10.)
            self.push_events.append(dict(time_s=now, peak_n=peak,
                                         duration_s=self.push_duration, angle_rad=angle))
            print(f'PUSH t={now:.2f}s peak={peak:.1f}N duration={self.push_duration:.3f}s', flush=True)
        self.current_force[:] = 0.
        midpoint = now+.5*self.physics_dt
        if self.pushes and midpoint < self.push_end:
            phase = (midpoint-self.push_start)/self.push_duration
            self.current_force[:] = self.push_vector*math.sin(math.pi*phase)**2
        # xfrc_applied: world-frame force then torque at body COM.
        self.data.xfrc_applied[self.push_body, :3] = self.current_force
        self.data.xfrc_applied[self.push_body, 3:] = np.cross([0., 0., .06], self.current_force)
        self.push_impulse += float(np.linalg.norm(self.current_force))*self.physics_dt
        super().physics_step()

    def disturbance_report(self):
        return dict(payload_kg=self.payload_kg, total_mass_kg=float(self.model.body_mass.sum()),
                    pushes_enabled=self.pushes, disturbance_seed=self.disturbance_seed,
                    push_events=self.push_events, total_impulse_ns=self.push_impulse,
                    illegal_geom_names=sorted(self.illegal_names),
                    payload_model='distributed trunk mass; unchanged COM; inertia scaled with mass')
