"""Feasible URDF-derived CPG settings for the RS01 dog.

This module intentionally subclasses the first RS01 task configuration instead
of duplicating it.  The original task used a 0.38 s cycle whose Cartesian
trajectory required joint speeds far above the configured 2.6/3.2 rad/s target
limits.  These overrides keep the a 72 ms diagonal load-transfer overlap
while lengthening swing time enough for the measured RS01 actuator chain.
"""

from .dog_config import (
    DogRs01TrotCfg as _LegacyDogRs01TrotCfg,
    DogRs01TrotCfgPPO as _LegacyDogRs01TrotCfgPPO,
)


class DogRs01TrotCfg(_LegacyDogRs01TrotCfg):
    """URDF-consistent, rate-feasible diagonal trot configuration."""

    class asset(_LegacyDogRs01TrotCfg.asset):
        # This is the actual file tracked by the dog_urdf subrepository.
        file = (
            "{LEGGED_GYM_ROOT_DIR}/../dog_urdf/urdf/"
            "URDFzhuangpei.SLDASM.urdf"
        )

    class control(_LegacyDogRs01TrotCfg.control):
        # Symmetric nominal footprint.  Do not add empirical load-bias offsets
        # until the geometry-only CPG is clean and repeatable.
        cpg_nominal_foot_x_m = 0.0
        cpg_nominal_foot_z_m = -0.300

        # At vx=0.15 m/s this produces a 44 mm stride.  The 55 mm hard cap also
        # keeps the 0.30 m/s training command inside the configured joint-rate
        # envelope after URDF IK.
        cpg_stride_gain = 0.70
        cpg_max_stride_m = 0.055

        # A 20 mm toe lift is sufficient for the 16 mm foot sphere on a plane.
        # Quintic rise/fall in rs01_cpg.py removes the abrupt acceleration of
        # the former 18%-of-swing lift.
        cpg_swing_clearance_m = 0.020
        cpg_full_clearance_speed_m_s = 0.15
        cpg_lift_fraction = 0.40
        cpg_lower_start_fraction = 0.60

        # 0.10 * 0.72 s = 72 ms, matching the measured command delay plus
        # most of the identified motor time constant without advancing a full
        # quarter-cycle as the failed 0.25 setting did.
        gait_target_phase_lead = 0.10
        gait_transition_ramp_s = 0.80

        # Keep the deterministic geometry preview deterministic.  Per-motor
        # gain compensation belongs to residual training, not to the nominal
        # physical diagonal trajectory.
        compensate_identified_position_gain_in_gait = False

        # First validate an exact periodic trot.  Contact gates and phase holds
        # change oscillator timing and were masking the infeasible trajectory
        # with irregular pauses.  They can be reintroduced after a trained
        # residual policy is available.
        gate_swing_on_opposite_diagonal_support = False
        use_contact_aware_phase_transfer = False

        # Remove millimetre-scale empirical biases from the base trajectory.
        cpg_force_balance_gain_m_per_weight = 0.0
        cpg_front_rear_load_bias_m = 0.0
        cpg_diagonal_load_bias_m = 0.0
        cpg_vertical_velocity_damping_s = 0.0

    class domain_rand(_LegacyDogRs01TrotCfg.domain_rand):
        # 1.39 Hz cycle.  A 0.60 stance ratio gives:
        #   swing: 0.72 * 0.40 = 288 ms
        #   double support: 0.72 * (0.60 - 0.50) = 72 ms
        # The old 114 ms swing could not be followed by the configured target
        # rate limits after the measured 39-60 ms command delay.
        gait_stance_ratio_range = [0.60, 0.60]
        gait_low_speed_period_range = [0.72, 0.72]
        gait_high_speed_period_range = [0.72, 0.72]
        randomize_gait_phase_on_reset = False

    class rewards(_LegacyDogRs01TrotCfg.rewards):
        gait_period = 0.72
        gait_stance_ratio = 0.60
        max_all_feet_contact_time_s = 0.10
        all_feet_contact_penalty_saturation_s = 0.10


class DogRs01TrotCfgPPO(_LegacyDogRs01TrotCfgPPO):
    """PPO settings are unchanged; only the deterministic base gait changes."""

    pass
