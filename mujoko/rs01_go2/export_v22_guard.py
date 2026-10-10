"""Export Guard5 only, with explicit replacement of the V20 action mapping."""
import export_v22
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_guard import (
    Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard5PPO)
from legged_gym.envs.rs01_omni_v2.rs01_v22_guard_robust import (
    Rs01V22GuardRobustCfg, Rs01V22GuardRobustPPO)


def build_contract(task, cfg, checkpoint, output):
    c = export_v22.build_contract(task, cfg, checkpoint, output)
    c['v20']['mapping'] = 'omnidirectional_inward_scale_v1'
    c['v20']['inward_target_rad'] = cfg.inward_guard.target_limit_rad
    c['v20']['mapping_note'] = 'Replace V20 map; all commands; before rate/acceleration limiter; no physical-state clamp'
    return c


if __name__ == '__main__':
    legacy = export_v22.export_v20.export_v14.export_v13.legacy
    legacy.TASKS = {
        'rs01_v22_phase_guard5': (Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard5PPO),
        'rs01_v22_guard5_robust': (Rs01V22GuardRobustCfg, Rs01V22GuardRobustPPO),
    }
    legacy.build_contract = build_contract
    legacy.main()
