from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.fanfan.fanfan_config import FanfanRoughCfg, FanfanRoughCfgPPO
from legged_gym.envs.fanfan.fanfan_env import FanfanRobot
from legged_gym.envs.fanfan.fanfan_omni_safe_config import (
    FanfanOmniSafeCfg, FanfanOmniSafeCfgPPO,
    FanfanOmniFastCfg, FanfanOmniFastCfgPPO,
    FanfanOmniSmoothRealCfg, FanfanOmniSmoothRealCfgPPO,
    FanfanOmniFilteredCfg, FanfanOmniFilteredCfgPPO,
    FanfanOmniVelTrackV3Cfg, FanfanOmniVelTrackV3CfgPPO,
    FanfanOmniLateralFixCfg, FanfanOmniLateralFixCfgPPO,
    FanfanOmniLateralSpeedCleanCfg, FanfanOmniLateralSpeedCleanCfgPPO,
    FanfanOmniDesatTorqueCfg, FanfanOmniDesatTorqueCfgPPO,
    FanfanOmniYawDriftCleanCfg, FanfanOmniYawDriftCleanCfgPPO,
    FanfanOmniYawSymmetryCfg, FanfanOmniYawSymmetryCfgPPO,
    FanfanOmniYawPathFixCfg, FanfanOmniYawPathFixCfgPPO,
    FanfanOmniDiagonalCoordCfg, FanfanOmniDiagonalCoordCfgPPO,
    FanfanOmniCoordinatedStraightCfg, FanfanOmniCoordinatedStraightCfgPPO,
    FanfanOmniProjectedCoordCfg, FanfanOmniProjectedCoordCfgPPO,
    FanfanOmniStrongSymmetryCfg, FanfanOmniStrongSymmetryCfgPPO,
    FanfanOmniNoCompSymmetryCfg, FanfanOmniNoCompSymmetryCfgPPO,
    FanfanOmniHeadingBoundSymmetryCfg, FanfanOmniHeadingBoundSymmetryCfgPPO,
    FanfanOmniForceCoordCfg, FanfanOmniForceCoordCfgPPO,
    FanfanOmniForceDesatCfg, FanfanOmniForceDesatCfgPPO,
    FanfanOmniHighSpeedTransitionCfg, FanfanOmniHighSpeedTransitionCfgPPO,
    FanfanOmniHighAuthorityTransitionCfg,
    FanfanOmniHighAuthorityTransitionCfgPPO,
    FanfanOmniHighAuthorityDirectionCfg,
    FanfanOmniHighAuthorityDirectionCfgPPO,
    FanfanOmniHighAuthorityClosedLoopCfg,
    FanfanOmniHighAuthorityClosedLoopCfgPPO,
    FanfanOmniHighCadenceCfg, FanfanOmniHighCadenceCfgPPO,
    FanfanOmniSymmetricTransitionCfg,
    FanfanOmniSymmetricTransitionCfgPPO,
    FanfanOmniHardwareBalance5530Cfg,
    FanfanOmniHardwareBalance5530CfgPPO,
    FanfanOmniHardwareBalance5530V2Cfg,
    FanfanOmniHardwareBalance5530V2CfgPPO,
    FanfanOmniRealDataCurriculumCfg,
    FanfanOmniRealDataCurriculumCfgPPO,
    FanfanOmniRealDataSpeedPolishCfg,
    FanfanOmniRealDataSpeedPolishCfgPPO,
    FanfanOmniRealDataCoordinatedCfg,
    FanfanOmniRealDataCoordinatedCfgPPO,
    FanfanOmniRealDataClearancePolishCfg,
    FanfanOmniRealDataClearancePolishCfgPPO,
    FanfanOmniRealDataDirectionalPolishCfg,
    FanfanOmniRealDataDirectionalPolishCfgPPO,
    FanfanOmniRealDataPerformanceRecoveryCfg,
    FanfanOmniRealDataPerformanceRecoveryCfgPPO,
    FanfanOmniCalibratedSymmetryCfg, FanfanOmniCalibratedSymmetryCfgPPO,
)
from legged_gym.envs.fanfan_rouhe.fanfan_config import FanfanRouheRoughCfg, FanfanRouheRoughCfgPPO
from legged_gym.envs.fanfan_rouhe.fanfan_env import FanfanRouheRobot
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "fanfan", FanfanRobot, FanfanRoughCfg(), FanfanRoughCfgPPO())
task_registry.register(
    "fanfan_omni_safe", FanfanRobot,
    FanfanOmniSafeCfg(), FanfanOmniSafeCfgPPO(),
)
task_registry.register(
    "fanfan_omni_fast", FanfanRobot,
    FanfanOmniFastCfg(), FanfanOmniFastCfgPPO(),
)
task_registry.register(
    "fanfan_omni_smooth_real", FanfanRobot,
    FanfanOmniSmoothRealCfg(), FanfanOmniSmoothRealCfgPPO(),
)
task_registry.register(
    "fanfan_omni_filtered", FanfanRobot,
    FanfanOmniFilteredCfg(), FanfanOmniFilteredCfgPPO(),
)
task_registry.register(
    "fanfan_omni_veltrack_v3", FanfanRobot,
    FanfanOmniVelTrackV3Cfg(), FanfanOmniVelTrackV3CfgPPO(),
)
task_registry.register(
    "fanfan_omni_lateral_fix", FanfanRobot,
    FanfanOmniLateralFixCfg(), FanfanOmniLateralFixCfgPPO(),
)
task_registry.register(
    "fanfan_omni_lateral_speed_clean", FanfanRobot,
    FanfanOmniLateralSpeedCleanCfg(), FanfanOmniLateralSpeedCleanCfgPPO(),
)
task_registry.register(
    "fanfan_omni_desat_torque", FanfanRobot,
    FanfanOmniDesatTorqueCfg(), FanfanOmniDesatTorqueCfgPPO(),
)
task_registry.register( "fanfan_rouhe", FanfanRouheRobot, FanfanRouheRoughCfg(), FanfanRouheRoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register(
    "fanfan_omni_yaw_drift_clean",
    FanfanRobot,
    FanfanOmniYawDriftCleanCfg(),
    FanfanOmniYawDriftCleanCfgPPO(),
)
task_registry.register(
    "fanfan_omni_yaw_symmetry",
    FanfanRobot,
    FanfanOmniYawSymmetryCfg(),
    FanfanOmniYawSymmetryCfgPPO(),
)
task_registry.register(
    "fanfan_omni_yaw_path_fix",
    FanfanRobot,
    FanfanOmniYawPathFixCfg(),
    FanfanOmniYawPathFixCfgPPO(),
)
task_registry.register(
    "fanfan_omni_diagonal_coord",
    FanfanRobot,
    FanfanOmniDiagonalCoordCfg(),
    FanfanOmniDiagonalCoordCfgPPO(),
)
task_registry.register(
    "fanfan_omni_coordinated_straight",
    FanfanRobot,
    FanfanOmniCoordinatedStraightCfg(),
    FanfanOmniCoordinatedStraightCfgPPO(),
)
task_registry.register(
    "fanfan_omni_projected_coord",
    FanfanRobot,
    FanfanOmniProjectedCoordCfg(),
    FanfanOmniProjectedCoordCfgPPO(),
)
task_registry.register(
    "fanfan_omni_strong_symmetry",
    FanfanRobot,
    FanfanOmniStrongSymmetryCfg(),
    FanfanOmniStrongSymmetryCfgPPO(),
)
task_registry.register(
    "fanfan_omni_no_comp_symmetry",
    FanfanRobot,
    FanfanOmniNoCompSymmetryCfg(),
    FanfanOmniNoCompSymmetryCfgPPO(),
)
task_registry.register(
    "fanfan_omni_heading_bound_symmetry",
    FanfanRobot,
    FanfanOmniHeadingBoundSymmetryCfg(),
    FanfanOmniHeadingBoundSymmetryCfgPPO(),
)
task_registry.register(
    "fanfan_omni_force_coord",
    FanfanRobot,
    FanfanOmniForceCoordCfg(),
    FanfanOmniForceCoordCfgPPO(),
)
task_registry.register(
    "fanfan_omni_force_desat",
    FanfanRobot,
    FanfanOmniForceDesatCfg(),
    FanfanOmniForceDesatCfgPPO(),
)
task_registry.register(
    "fanfan_omni_high_speed_transition",
    FanfanRobot,
    FanfanOmniHighSpeedTransitionCfg(),
    FanfanOmniHighSpeedTransitionCfgPPO(),
)
task_registry.register(
    "fanfan_omni_high_authority_transition",
    FanfanRobot,
    FanfanOmniHighAuthorityTransitionCfg(),
    FanfanOmniHighAuthorityTransitionCfgPPO(),
)
task_registry.register(
    "fanfan_omni_high_authority_direction",
    FanfanRobot,
    FanfanOmniHighAuthorityDirectionCfg(),
    FanfanOmniHighAuthorityDirectionCfgPPO(),
)
task_registry.register(
    "fanfan_omni_high_authority_closed_loop",
    FanfanRobot,
    FanfanOmniHighAuthorityClosedLoopCfg(),
    FanfanOmniHighAuthorityClosedLoopCfgPPO(),
)
task_registry.register(
    "fanfan_omni_high_cadence",
    FanfanRobot,
    FanfanOmniHighCadenceCfg(),
    FanfanOmniHighCadenceCfgPPO(),
)
task_registry.register(
    "fanfan_omni_symmetric_transition",
    FanfanRobot,
    FanfanOmniSymmetricTransitionCfg(),
    FanfanOmniSymmetricTransitionCfgPPO(),
)
task_registry.register(
    "fanfan_omni_hardware_balance_5530",
    FanfanRobot,
    FanfanOmniHardwareBalance5530Cfg(),
    FanfanOmniHardwareBalance5530CfgPPO(),
)
task_registry.register(
    "fanfan_omni_hardware_balance_5530_v2",
    FanfanRobot,
    FanfanOmniHardwareBalance5530V2Cfg(),
    FanfanOmniHardwareBalance5530V2CfgPPO(),
)
task_registry.register(
    "fanfan_omni_realdata_curriculum",
    FanfanRobot,
    FanfanOmniRealDataCurriculumCfg(),
    FanfanOmniRealDataCurriculumCfgPPO(),
)
task_registry.register(
    "fanfan_omni_realdata_speed_polish",
    FanfanRobot,
    FanfanOmniRealDataSpeedPolishCfg(),
    FanfanOmniRealDataSpeedPolishCfgPPO(),
)
task_registry.register(
    "fanfan_omni_realdata_coordinated",
    FanfanRobot,
    FanfanOmniRealDataCoordinatedCfg(),
    FanfanOmniRealDataCoordinatedCfgPPO(),
)
task_registry.register(
    "fanfan_omni_realdata_clearance_polish",
    FanfanRobot,
    FanfanOmniRealDataClearancePolishCfg(),
    FanfanOmniRealDataClearancePolishCfgPPO(),
)
task_registry.register(
    "fanfan_omni_realdata_directional_polish",
    FanfanRobot,
    FanfanOmniRealDataDirectionalPolishCfg(),
    FanfanOmniRealDataDirectionalPolishCfgPPO(),
)
task_registry.register(
    "fanfan_omni_realdata_performance_recovery",
    FanfanRobot,
    FanfanOmniRealDataPerformanceRecoveryCfg(),
    FanfanOmniRealDataPerformanceRecoveryCfgPPO(),
)
task_registry.register(
    "fanfan_omni_calibrated_symmetry",
    FanfanRobot,
    FanfanOmniCalibratedSymmetryCfg(),
    FanfanOmniCalibratedSymmetryCfgPPO(),
)
