"""Resolved feedback with unchanged identified RS01 dynamics and V13 rewards."""
from .rs01_omni_v13_config import Rs01OmniV13DirectionCfg, Rs01OmniV13DirectionCfgPPO


class Rs01OmniV14ActuatorCfg(Rs01OmniV13DirectionCfg):
    class asset(Rs01OmniV13DirectionCfg.asset):
        name = "rs01_omni_v14_actuator_parity"

    class sim(Rs01OmniV13DirectionCfg.sim):
        dt = 0.0025
        substeps = 1

    class control(Rs01OmniV13DirectionCfg.control):
        decimation = 8  # Policy/target limiting remains 50 Hz.

    class rs01_actuator(Rs01OmniV13DirectionCfg.rs01_actuator):
        response_update_dt_s = 0.005  # Preserve measured delay quantization/filter.
        # Manual's 315 rpm at nominal voltage: validation boundary, NOT an
        # ideal brake, protocol range, or measured torque-speed envelope.
        speed_validity_limit_rad_s = 32.9867
        speed_limit_semantics = "terminate_outside_validated_domain_no_velocity_projection"
        solver_velocity_ceiling_rad_s = 1000.0  # Numerical emergency ceiling only.


class Rs01OmniV14ActuatorCfgPPO(Rs01OmniV13DirectionCfgPPO):
    class runner(Rs01OmniV13DirectionCfgPPO.runner):
        experiment_name = "rs01_omni_v14_actuator_parity"
