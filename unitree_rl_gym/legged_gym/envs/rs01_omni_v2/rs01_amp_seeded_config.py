from .rs01_amp_core_config import Rs01AMPCoreCfg,Rs01AMPCoreSlowCfg,Rs01AMPCorePPO,Rs01AMPCoreSlowPPO
from .rs01_amp_config import REFERENCE
from .rs01_amp_scratch_config import SLOW_REFERENCE


class Rs01AMPSeededCfg(Rs01AMPCoreCfg):
    class amp(Rs01AMPCoreCfg.amp):
        reset_reference=REFERENCE
        reference_reset_probability=.7


class Rs01AMPSeededSlowCfg(Rs01AMPCoreSlowCfg):
    class amp(Rs01AMPCoreSlowCfg.amp):
        reset_reference=SLOW_REFERENCE
        reference_reset_probability=.7


class Rs01AMPSeededPPO(Rs01AMPCorePPO):
    class runner(Rs01AMPCorePPO.runner):
        experiment_name='rs01_amp_seeded_original'


class Rs01AMPSeededSlowPPO(Rs01AMPCoreSlowPPO):
    class runner(Rs01AMPCoreSlowPPO.runner):
        experiment_name='rs01_amp_seeded_slow'
