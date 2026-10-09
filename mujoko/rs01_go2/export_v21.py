"""V21 changes rewards only; retain V20 RS01 plant/observation/action contract."""
import export_v20
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_config import Rs01OmniV21Cfg, Rs01OmniV21CfgPPO


if __name__ == '__main__':
    legacy = export_v20.export_v14.export_v13.legacy
    legacy.TASKS = {'rs01_omni_v21_phase_coord': (Rs01OmniV21Cfg, Rs01OmniV21CfgPPO)}
    legacy.build_contract = export_v20.build_contract
    legacy.main()
