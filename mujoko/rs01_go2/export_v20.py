"""Export V20 actor plus mandatory target mapping and policy-boundary guard."""
import export_v14
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_config import Rs01OmniV20BCfg, Rs01OmniV20BCfgPPO


def build_contract(task, cfg, checkpoint, output):
    c = export_v14.build_contract(task, cfg, checkpoint, output)
    a = cfg.rs01_actuator
    c['v20'] = dict(mapping='conditional_inward_scale_v1',
        inward_target_rad=cfg.control.straight_inward_target_rad,
        vy_gate=.08, wz_gate=.20, stand_phase='freeze_and_resume', odometry='legacy',
        guard=dict(continuous=a.continuous_torque_nm, peak=a.peak_torque_limit_nm,
                   full=a.guard_derate_full_rms_nm, tau=a.guard_time_constant_s))
    c['v14']['hardware_model'] = 'Identified RS01 plant; 14 Nm operating cap and V16 policy-boundary guard'
    return c


if __name__ == '__main__':
    legacy = export_v14.export_v13.legacy
    legacy.TASKS = {'rs01_omni_v20_bounded_hip': (Rs01OmniV20BCfg, Rs01OmniV20BCfgPPO)}
    legacy.build_contract = build_contract
    legacy.main()
