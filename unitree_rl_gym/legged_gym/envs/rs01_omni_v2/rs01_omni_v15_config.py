from .rs01_omni_v14_config import Rs01OmniV14ActuatorCfg, Rs01OmniV14ActuatorCfgPPO


class Rs01OmniV15StandCfg(Rs01OmniV14ActuatorCfg):
    class asset(Rs01OmniV14ActuatorCfg.asset):
        name = 'rs01_omni_v15_stand_phase'


class Rs01OmniV15StandCfgPPO(Rs01OmniV14ActuatorCfgPPO):
    class runner(Rs01OmniV14ActuatorCfgPPO.runner):
        experiment_name = 'rs01_omni_v15_stand_phase'


class Rs01OmniV15SupportCfg(Rs01OmniV14ActuatorCfg):
    class asset(Rs01OmniV14ActuatorCfg.asset):
        name = 'rs01_omni_v15_support'


class Rs01OmniV15SupportCfgPPO(Rs01OmniV14ActuatorCfgPPO):
    class runner(Rs01OmniV14ActuatorCfgPPO.runner):
        experiment_name = 'rs01_omni_v15_support'
