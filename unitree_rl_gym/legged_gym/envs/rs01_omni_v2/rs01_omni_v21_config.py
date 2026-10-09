"""Replace mean sagittal bias with half-cycle coordination; modest clearance."""
from .rs01_omni_v20_config import Rs01OmniV20BCfg, Rs01OmniV20BCfgPPO


class Rs01OmniV21Cfg(Rs01OmniV20BCfg):
    class asset(Rs01OmniV20BCfg.asset):
        name='rs01_omni_v21_phase_coord'

    class rewards(Rs01OmniV20BCfg.rewards):
        swing_clearance_m=.025
        coordination_deadband_x_m=.015
        coordination_deadband_z_m=.006
        coordination_scale_x_m=.030
        coordination_scale_z_m=.015
        coordination_weight=1.0


class Rs01OmniV21CfgPPO(Rs01OmniV20BCfgPPO):
    class runner(Rs01OmniV20BCfgPPO.runner):
        experiment_name='rs01_omni_v21_phase_coord'
