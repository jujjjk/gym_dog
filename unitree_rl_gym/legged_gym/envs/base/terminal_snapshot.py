"""Terminal-frame storage kept across the automatic environment reset."""

import torch


RESET_REASONS = (
    "trunk_contact",
    "orientation",
    "timeout",
    "flight",
    "illegal_contact",
    "straight_heading",
    "translation_heading",
    "low_height",
    "rear_sit",
    "calf_angle",
    "actuator_safety",
)
RESET_REASON_BITS = {
    name: 1 << index for index, name in enumerate(RESET_REASONS)
}


def format_reset_reason(reason_bits):
    names = [
        name for name, bit in RESET_REASON_BITS.items()
        if int(reason_bits) & bit
    ]
    return "|".join(names) if names else "none"


class TerminalSnapshot:
    """Device-resident snapshot of the exact frame that triggers reset."""

    def __init__(self, num_envs, num_feet, num_dof, device):
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset_reason_bits = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.illegal_contact_count = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self.contact_mask = torch.zeros(
            num_envs, num_feet, dtype=torch.bool, device=device
        )
        self.desired_contact_mask = torch.zeros(
            num_envs, num_feet, dtype=torch.bool, device=device
        )
        self.phase = torch.zeros(num_envs, device=device)
        self.rpy = torch.zeros(num_envs, 3, device=device)
        self.yaw_rate = torch.zeros(num_envs, device=device)
        self.raw_pd_torques = torch.zeros(num_envs, num_dof, device=device)
        self.motor_electromagnetic_torques = torch.zeros(
            num_envs, num_dof, device=device
        )
        self.applied_joint_torques = torch.zeros(
            num_envs, num_dof, device=device
        )
        self.peak_torque_limits = torch.zeros(
            num_envs, num_dof, device=device
        )
        self.active_torque_limits = torch.zeros(
            num_envs, num_dof, device=device
        )

    def capture(
        self,
        reset_mask,
        reset_reason_bits,
        illegal_contact_count,
        contact_mask,
        desired_contact_mask,
        phase,
        rpy,
        yaw_rate,
        raw_pd_torques,
        motor_electromagnetic_torques,
        applied_joint_torques,
        peak_torque_limits,
        active_torque_limits,
    ):
        self.valid.copy_(reset_mask)
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        values = (
            (self.reset_reason_bits, reset_reason_bits),
            (self.illegal_contact_count, illegal_contact_count),
            (self.contact_mask, contact_mask),
            (self.desired_contact_mask, desired_contact_mask),
            (self.phase, phase),
            (self.rpy, rpy),
            (self.yaw_rate, yaw_rate),
            (self.raw_pd_torques, raw_pd_torques),
            (
                self.motor_electromagnetic_torques,
                motor_electromagnetic_torques,
            ),
            (self.applied_joint_torques, applied_joint_torques),
            (self.peak_torque_limits, peak_torque_limits),
            (self.active_torque_limits, active_torque_limits),
        )
        for destination, source in values:
            destination.index_copy_(0, ids, source.index_select(0, ids))
