"""AMP on a style the IsaacGym RS01 plant demonstrably executed, clock included."""
import json
from pathlib import Path
from .rs01_amp_core_config import Rs01AMPCoreCfg, Rs01AMPCorePPO
from .rs01_amp_seeded_config import Rs01AMPSeededCfg, Rs01AMPSeededPPO
from .rs01_omni_v30_config import Rs01OmniV30Cfg

INPLANT_REFERENCE = str(Path(__file__).resolve().parents[4]
                        / 'artifacts/rs01_reference/amp_isaacgym_v2_20261007')
MANIFEST = json.loads((Path(INPLANT_REFERENCE)/'manifest.json').read_text())
# The commands are the speeds the demonstrator actually drove and the clock is the map that
# produced the recorded per-band tempo, so both have to be read from the dataset itself.
INPLANT_COMMANDS = [[float(v) for v in clip['command']] for clip in MANIFEST['clips']]


class Rs01AMPInPlantCfg(Rs01AMPCoreCfg):
    class control(Rs01AMPCoreCfg.control):
        # The clips were executed through the demonstrator's own output range; widening it here
        # would make every warm-started action command more joint travel than it was calibrated
        # for. The core experiments keep the wide range for random exploration from scratch.
        action_scale_by_joint = Rs01OmniV30Cfg.control.action_scale_by_joint
    class amp(Rs01AMPCoreCfg.amp):
        commands = INPLANT_COMMANDS
        command_dependent_clock = True
    class rewards(Rs01AMPCoreCfg.rewards):
        gait_period_s = MANIFEST['settings']['period_s']
        gait_stance_ratio = MANIFEST['settings']['duty']


class Rs01AMPInPlantSeededCfg(Rs01AMPSeededCfg):
    class control(Rs01AMPInPlantCfg.control):
        pass
    # Both branches have to be in the bases: the in-plant one carries the dataset's commands and
    # clock, the seeded one carries reference_reset_probability. A nested class silently shadows
    # the parent's same-named one, so leaving either out loses its fields.
    class amp(Rs01AMPInPlantCfg.amp, Rs01AMPSeededCfg.amp):
        reset_reference = INPLANT_REFERENCE
    class rewards(Rs01AMPInPlantCfg.rewards):
        pass


class Rs01AMPInPlantPPO(Rs01AMPCorePPO):
    class runner(Rs01AMPCorePPO.runner):
        experiment_name = 'rs01_amp_inplant_original'
        amp_reference = INPLANT_REFERENCE
        # Plant recordings, not a kinematic wish: no opt-in for an unverified style.
        amp_allow_kinematic = False


class Rs01AMPInPlantSeededPPO(Rs01AMPInPlantPPO):
    class runner(Rs01AMPInPlantPPO.runner):
        experiment_name = 'rs01_amp_inplant_seeded'
