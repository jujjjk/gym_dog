import export_v14
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_config import Rs01OmniV15SupportCfg, Rs01OmniV15SupportCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_config import Rs01OmniV15StandCfg, Rs01OmniV15StandCfgPPO


def build_contract(task, cfg, checkpoint, output):
    c = export_v14.build_contract(task, cfg, checkpoint, output)
    c['v15'] = dict(stand_phase='freeze_and_resume', odometry=(
        'legacy' if task == 'rs01_omni_v15_stand_phase' else 'gravity_support_v1'))
    return c


if __name__ == '__main__':
    legacy = export_v14.export_v13.legacy
    legacy.TASKS = {'rs01_omni_v15_support': (Rs01OmniV15SupportCfg, Rs01OmniV15SupportCfgPPO)}
    legacy.TASKS['rs01_omni_v15_stand_phase'] = (Rs01OmniV15StandCfg, Rs01OmniV15StandCfgPPO)
    legacy.build_contract = build_contract
    legacy.main()
