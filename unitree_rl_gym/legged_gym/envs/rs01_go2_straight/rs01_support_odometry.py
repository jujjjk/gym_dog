"""V15 sensor-only support estimator; identical Torch math in both simulators.

No commanded velocity, simulator contacts, or root velocity is consumed.
Support inference is not a no-slip guarantee; confidence is a heuristic.
"""
import torch


class GravitySupportOdometry:
    def __init__(self, kinematics):
        self.core = kinematics
        self.gravity = None

    def reset(self, env_ids=None):
        self.core.reset(env_ids)

    def estimate(self, q, dq, omega):
        c = self.core
        if self.gravity is None:
            raise ValueError('Projected IMU gravity is required')
        tensor = lambda x: torch.as_tensor(x, device=c.device, dtype=c.dtype)
        p, j = c.foot_position_and_jacobian(tensor(q).reshape(c.num_envs, 12))
        joint_v = torch.einsum('nlij,nlj->nli', j, tensor(dq).reshape(c.num_envs, 4, 3))
        relative_v = joint_v + torch.linalg.cross(tensor(omega).reshape(-1, 1, 3).expand_as(p), p)
        by_foot = -relative_v
        down = torch.nn.functional.normalize(tensor(self.gravity).reshape(-1, 3), dim=1)
        height = (p * down[:, None]).sum(-1) + c.foot_radius
        gap = height.amax(1, keepdim=True) - height
        vertical = (relative_v * down[:, None]).sum(-1).abs()
        valid = (gap <= c.height_margin) & (vertical <= c.vertical_speed_threshold)
        valid &= (height - c.nominal_base_height).abs() <= .10
        quality = (1 - gap / c.height_margin).clamp(0, 1)
        quality *= (1 - vertical / c.vertical_speed_threshold).clamp(0, 1)
        # Estimate support independently of the desired diagonal gait: at
        # handoff/stand, real support need not consist of exactly two diagonals.
        residual = torch.linalg.vector_norm(by_foot[:, :, None, :2] - by_foot[:, None, :, :2], dim=-1)
        agreement = (1 - residual / c.velocity_residual_threshold).clamp(0, 1)
        weights = agreement * quality[:, None] * valid[:, None]
        score = weights.sum(-1) * valid
        anchor = score.argmax(1)
        selected = weights[torch.arange(c.num_envs, device=c.device), anchor]
        stance = (selected > 0) & valid
        supported = stance.sum(1) >= 2
        selected = selected * supported[:, None]
        total = selected.sum(1)
        raw = (by_foot * selected[:, :, None]).sum(1) / total.clamp_min(1e-6)[:, None]
        raw[:, :2] = raw[:, :2].clamp(-1, 1)
        raw[:, 2] = 0
        confidence = (total / 2).clamp(0, 1) * supported
        alpha = c.filter_alpha * confidence
        c.filtered = torch.where(supported[:, None],
                                 (1-alpha[:, None])*c.filtered + alpha[:, None]*raw,
                                 c.filtered*c.no_contact_decay)
        c.last_stance = stance & supported[:, None]
        return dict(base_linear_velocity=c.filtered.clone(), raw_base_linear_velocity=raw,
                    confidence=confidence, stance_mask=c.last_stance.clone(),
                    velocity_by_foot=by_foot, foot_position=p, foot_velocity=joint_v,
                    base_height_proxy=height)
