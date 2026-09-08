"""Export the resolved RS01 feedback contract; old V13 exports stay unchanged."""
import export_v13
from legged_gym.envs.rs01_omni_v2.rs01_omni_v14_config import (
    Rs01OmniV14ActuatorCfg, Rs01OmniV14ActuatorCfgPPO,
)


def build_contract(task, cfg, checkpoint, output):
    c = export_v13.build_contract(task, cfg, checkpoint, output)
    a = cfg.rs01_actuator
    c['v14'] = dict(
        response_update_dt_s=a.response_update_dt_s,
        feedback_update_dt_s=cfg.sim.dt,
        speed_validity_limit_rad_s=a.speed_validity_limit_rad_s,
        speed_limit_semantics=a.speed_limit_semantics,
        speed_source='dog_urdf/config/rs01_motor_limits.yaml: 315 rpm no-load at 36 V',
        hardware_model='per-motor identified gain/FOPDT/delay/Coulomb friction; 17 Nm electromagnetic cap',
        unresolved='full torque-speed/voltage/thermal envelope and firmware feedback bandwidth not validated',
    )
    c['control']['response_update_dt_s'] = a.response_update_dt_s
    c['control']['feedback_update_dt_s'] = cfg.sim.dt
    c['simulator']['mujoco']['integration_timestep_s'] = cfg.sim.dt
    c['simulator']['mujoco']['integration_substeps_per_motor_step'] = 1
    return c


if __name__ == '__main__':
    legacy = export_v13.legacy
    legacy.TASKS = {'rs01_omni_v14_actuator_parity': (Rs01OmniV14ActuatorCfg, Rs01OmniV14ActuatorCfgPPO)}
    legacy.build_contract = build_contract
    legacy.main()
