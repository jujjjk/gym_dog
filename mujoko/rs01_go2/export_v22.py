"""Export V22 sensor model and unchanged RS01 plant alongside actor."""
import export_v20
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg, Rs01OmniV22CfgPPO
from legged_gym.utils.helpers import class_to_dict


def build_contract(task,cfg,checkpoint,output):
    c=export_v20.build_contract(task,cfg,checkpoint,output)
    c['v22']=dict(sensor=class_to_dict(cfg.sensor),odometry=class_to_dict(cfg.rs01_odometry),
                  host_guard='policy_sensor_snapshot',observation='sensor_snapshot_61')
    return c


if __name__=='__main__':
    legacy=export_v20.export_v14.export_v13.legacy
    legacy.TASKS={'rs01_omni_v22_sensor':(Rs01OmniV22Cfg,Rs01OmniV22CfgPPO)}
    legacy.build_contract=build_contract
    legacy.main()
