"""Stage 1: mincing lateral/turn cadence at 3.0 Hz; V28 command envelope unchanged."""
from .rs01_omni_v28_config import Rs01OmniV28Cfg, Rs01OmniV28CfgPPO


class Rs01OmniV29Cfg(Rs01OmniV28Cfg):
    class rewards(Rs01OmniV28Cfg.rewards):
        # The V25 floor saturated lateral/turn cadence at 2.0 Hz; raise the ceiling
        # so full-blend commands step at mincing cadence (stride = vy / f).
        lateral_frequency_floor_hz = 3.0


class Rs01OmniV29CfgPPO(Rs01OmniV28CfgPPO):
    class runner(Rs01OmniV28CfgPPO.runner):
        experiment_name = 'rs01_omni_v29_mince'
