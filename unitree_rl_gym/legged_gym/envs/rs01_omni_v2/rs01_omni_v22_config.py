from .rs01_omni_v21_config import Rs01OmniV21Cfg, Rs01OmniV21CfgPPO


class Rs01OmniV22Cfg(Rs01OmniV21Cfg):
    class sensor:
        enabled=True
        clean_fraction=.25
        robust_fraction=.25  # Of non-clean environments; 18.75% total.
        motor_period_s=.005
        imu_period_s=.010
        motor_normal_max_s=.010
        motor_robust_max_s=.015
        imu_min_s=.010
        imu_normal_max_s=.030
        imu_robust_max_s=.040
        q_sigma=.0005
        q_clip=.0015
        dq_sigma=.08
        dq_clip=.25
        dq_robust_clip=.30
        gyro_sigma=.004
        gyro_clip=.015
        gyro_bias_rad_s=.005
        mount_normal_deg=2.
        mount_robust_deg=3.


class Rs01OmniV22CleanCfg(Rs01OmniV22Cfg):
    class sensor(Rs01OmniV22Cfg.sensor):
        enabled=False


class Rs01OmniV22CfgPPO(Rs01OmniV21CfgPPO):
    class runner(Rs01OmniV21CfgPPO.runner):
        experiment_name='rs01_omni_v22_sensor'


class Rs01OmniV22CleanCfgPPO(Rs01OmniV22CfgPPO):
    class runner(Rs01OmniV22CfgPPO.runner):
        experiment_name='rs01_omni_v22_clean'
