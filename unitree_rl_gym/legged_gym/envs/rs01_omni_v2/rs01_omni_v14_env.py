"""Same physical validity domain as MuJoCo; no instantaneous velocity clipping."""
import torch
from .rs01_omni_v13_env import Rs01OmniV13Robot


class Rs01OmniV14Robot(Rs01OmniV13Robot):
    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props.copy(), env_id)
        if env_id == 0:
            limit = self.cfg.rs01_actuator.speed_validity_limit_rad_s
            if not torch.allclose(self.dof_vel_limits, torch.full_like(self.dof_vel_limits, limit)):
                raise ValueError("RS01 URDF speed and exported validity domain differ")
        # Keep dof_vel_limits at the physical audit value. Remove PhysX's
        # artificial wall from the valid operating domain; overspeed fails.
        props['velocity'] = self.cfg.rs01_actuator.solver_velocity_ceiling_rad_s
        return props

    def step(self, actions):
        self.rs01_step_overspeed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rs01_step_max_speed = torch.zeros(self.num_envs, device=self.device)
        return super().step(actions)

    def _observe_speed_domain(self):
        if not hasattr(self, 'rs01_step_overspeed'):
            return
        speed = self.dof_vel.abs().amax(dim=1)
        self.rs01_step_max_speed = torch.maximum(self.rs01_step_max_speed, speed)
        self.rs01_step_overspeed |= (~torch.isfinite(speed)) | (
            speed >= self.cfg.rs01_actuator.speed_validity_limit_rad_s)

    def _compute_torques(self, actions):
        self._observe_speed_domain()
        return super()._compute_torques(actions)

    def check_termination(self):
        super().check_termination()
        self._observe_speed_domain()  # Includes final feedback step before reset.
        if hasattr(self, 'rs01_step_overspeed'):
            self.reset_buf |= self.rs01_step_overspeed
