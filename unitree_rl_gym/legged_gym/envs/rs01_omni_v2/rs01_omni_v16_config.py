"""Hardware-guarded RS01 support experiment; A/B differ only in initialization."""
from .rs01_omni_v15_config import Rs01OmniV15StandCfg, Rs01OmniV15StandCfgPPO


class Rs01OmniV16Cfg(Rs01OmniV15StandCfg):
    class asset(Rs01OmniV15StandCfg.asset):
        name = 'rs01_omni_v16_guarded_support'

    class commands(Rs01OmniV15StandCfg.commands):
        # March is the zero-command transition; no randomly selected stand.
        mode_probabilities = [0.0, 0.40, 0.192, 0.084, 0.12, 0.108, 0.096]
        moving_to_stand_probability = 0.0

    class rs01_actuator(Rs01OmniV15StandCfg.rs01_actuator):
        # Deployed operating cap, NOT a revision to the motor's 17 Nm rating.
        peak_torque_limit_nm = 14.0
        guard_derate_full_rms_nm = 8.0
        guard_time_constant_s = 2.0

    class rewards(Rs01OmniV15StandCfg.rewards):
        gait_stance_ratio = 0.72
        # Longest requested handoff: (.72-.5)*.60=.132 s, plus two ticks.
        all_feet_contact_grace_s = 0.172
        footprint_straight_min_width_m = 0.20
        footprint_omni_min_width_m = 0.14
        footprint_soft_scale_m = 0.04

        class scales(Rs01OmniV15StandCfg.rewards.scales):
            # One dense structure reward replaces binary two-foot reward AND
            # the duplicated phase error. Tracking is never contact-gated.
            phase_contact_error = 0.0


class Rs01OmniV16CfgPPO(Rs01OmniV15StandCfgPPO):
    class runner(Rs01OmniV15StandCfgPPO.runner):
        experiment_name = 'rs01_omni_v16_guarded_support'
        resume = False
        load_optimizer = False
