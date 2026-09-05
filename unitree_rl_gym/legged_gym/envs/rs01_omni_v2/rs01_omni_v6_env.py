"""Short-lived correlated exploration for scratch RS01 omni policies."""

import torch

from .rs01_omni_v5_env import Rs01OmniV5Robot


class Rs01OmniV6Robot(Rs01OmniV5Robot):
    def __init__(self, *args, **kwargs):
        self._structured_exploration_ready = False
        super().__init__(*args, **kwargs)
        slot_by_name = {
            name: index for index, name in enumerate(self.dof_names)
        }
        self.dof_slot_by_leg = {}
        for leg in ("FL", "FR", "RL", "RR"):
            expected = {
                joint: f"{leg}_{joint}_joint"
                for joint in ("hip", "thigh", "calf")
            }
            if any(name not in slot_by_name for name in expected.values()):
                raise ValueError(
                    f"Cannot construct structured exploration slots for {leg}"
                )
            self.dof_slot_by_leg[leg] = {
                joint: slot_by_name[name]
                for joint, name in expected.items()
            }
        self._structured_exploration_ready = True

    def _structured_exploration_fraction(self):
        if bool(getattr(self.cfg.env, "test", False)):
            return 0.0
        decay_steps = max(
            int(self.cfg.control.structured_exploration_decay_steps), 1
        )
        return max(1.0 - float(self.common_step_counter) / decay_steps, 0.0)

    def _structured_exploration_action(self):
        phase_a = torch.remainder(
            self._gait_phase()
            + float(self.cfg.control.structured_exploration_phase_lead),
            1.0,
        )
        phase_b = torch.remainder(phase_a + 0.5, 1.0)
        stance_ratio = float(self.cfg.rewards.gait_stance_ratio)

        def profiles(phase):
            swing_progress = torch.clamp(
                (phase - stance_ratio) / max(1.0 - stance_ratio, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            smooth_swing = swing_progress * swing_progress * (
                3.0 - 2.0 * swing_progress
            )
            profile = str(
                self.cfg.control.structured_exploration_profile
            )
            if profile == "sine":
                swing = torch.sin(torch.pi * smooth_swing)
            elif profile == "plateau":
                lift_fraction = float(
                    self.cfg.control.structured_exploration_lift_fraction
                )
                lower_start = float(
                    self.cfg.control
                    .structured_exploration_lower_start_fraction
                )
                rise_progress = torch.clamp(
                    swing_progress / lift_fraction, min=0.0, max=1.0
                )
                rise = rise_progress * rise_progress * (
                    3.0 - 2.0 * rise_progress
                )
                fall_progress = torch.clamp(
                    (swing_progress - lower_start)
                    / max(1.0 - lower_start, 1.0e-6),
                    min=0.0,
                    max=1.0,
                )
                fall = 1.0 - fall_progress * fall_progress * (
                    3.0 - 2.0 * fall_progress
                )
                swing = torch.minimum(rise, fall)
            else:
                raise ValueError(
                    f"Unknown structured exploration profile: {profile}"
                )
            swing *= (phase >= stance_ratio).to(dtype=torch.float)
            stance_progress = torch.clamp(
                phase / max(stance_ratio, 1.0e-6), min=0.0, max=1.0
            )
            stride = torch.where(
                phase < stance_ratio,
                -1.0 + 2.0 * stance_progress,
                1.0 - 2.0 * smooth_swing,
            )
            return swing, stride

        swing_a, stride_a = profiles(phase_a)
        swing_b, stride_b = profiles(phase_b)
        amplitude = float(
            self.cfg.control.structured_exploration_amplitude
        ) * self._structured_exploration_fraction()
        gait_gate = self._walking_command_gate()
        forward_fraction = torch.clamp(
            self.commands[:, 0]
            / max(
                float(
                    self.cfg.control
                    .structured_exploration_full_stride_speed_m_s
                ),
                1.0e-6,
            ),
            min=-1.0,
            max=1.0,
        )
        seed = torch.zeros_like(self.actions)
        for leg in ("FL", "RR", "FR", "RL"):
            in_diagonal_a = leg in ("FL", "RR")
            swing = swing_a if in_diagonal_a else swing_b
            stride = stride_a if in_diagonal_a else stride_b
            thigh = self.dof_slot_by_leg[leg]["thigh"]
            calf = self.dof_slot_by_leg[leg]["calf"]
            seed[:, thigh] = amplitude * (
                float(
                    self.cfg.control
                    .structured_exploration_stride_thigh_action
                )
                * forward_fraction
                * stride
                + float(
                    self.cfg.control
                    .structured_exploration_swing_thigh_action
                )
                * swing
            )
            seed[:, calf] = (
                amplitude
                * float(
                    self.cfg.control
                    .structured_exploration_calf_action
                )
                * swing
            )
        return seed * gait_gate.unsqueeze(1)

    def step(self, actions):
        self.unseeded_policy_actions = actions.to(self.device).clone()
        if getattr(self, "_structured_exploration_ready", False) and bool(
            getattr(
                self.cfg.control, "structured_exploration_enabled", False
            )
        ):
            actions = actions + self._structured_exploration_action()
        return super().step(actions)

    def _reward_action_saturation(self):
        """Do not charge the actor for the temporary external exploration."""
        if not hasattr(self, "unseeded_policy_actions"):
            return super()._reward_action_saturation()
        soft_limit = float(
            self.cfg.rewards.action_saturation_soft_limit
        )
        excess = torch.clamp(
            torch.abs(self.unseeded_policy_actions) - soft_limit,
            min=0.0,
        )
        return torch.mean(torch.square(excess), dim=1)
