"""AMP on a reference the RS01 plant actually executed; commands are the measured speeds."""
import json
from pathlib import Path
from .rs01_amp_core_config import Rs01AMPCoreCfg,Rs01AMPCorePPO
from .rs01_amp_seeded_config import Rs01AMPSeededCfg,Rs01AMPSeededPPO

PHYS_REFERENCE = str(Path(__file__).resolve().parents[4] / 'artifacts/rs01_reference/amp_physics_20261007')
# Read the cadence from the dataset: the runner rejects any reward clock that drifts from it.
REFERENCE_FREQUENCY = json.loads((Path(PHYS_REFERENCE)/'manifest.json').read_text())['settings']['frequency']
# Per-clip mean base velocity measured in the unchanged plant (fresh_rs01_10s_v1/*_physics reports).
PHYS_COMMANDS = [[0.,0.,0.], [.0532,0.,0.], [-.071,0.,0.], [0.,.0348,0.], [0.,-.0327,0.],
                 [0.,0.,.1096], [0.,0.,-.1157], [.043,.0224,0.], [.0421,-.0292,0.],
                 [.028,.0201,.1122]]


class Rs01AMPPhysCfg(Rs01AMPCoreCfg):
    class amp(Rs01AMPCoreCfg.amp):
        commands = PHYS_COMMANDS
    class rewards(Rs01AMPCoreCfg.rewards):
        gait_period_s = 1. / REFERENCE_FREQUENCY


class Rs01AMPPhysSeededCfg(Rs01AMPSeededCfg):
    class amp(Rs01AMPSeededCfg.amp):
        commands = PHYS_COMMANDS
        reset_reference = PHYS_REFERENCE
    class rewards(Rs01AMPPhysCfg.rewards):
        pass


class Rs01AMPPhysPPO(Rs01AMPCorePPO):
    class runner(Rs01AMPCorePPO.runner):
        experiment_name = 'rs01_amp_phys_original'
        # Recorded plant states, but still an experimental style reference: approval gates are open.
        amp_allow_kinematic = True
        amp_reference = PHYS_REFERENCE


class Rs01AMPPhysSeededPPO(Rs01AMPPhysPPO):
    class runner(Rs01AMPPhysPPO.runner):
        experiment_name = 'rs01_amp_phys_seeded'
