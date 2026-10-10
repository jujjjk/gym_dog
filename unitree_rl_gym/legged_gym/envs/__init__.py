from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.rs01_go2_straight import (
    Rs01Go2Kp40Cfg,
    Rs01Go2Kp40CfgPPO,
    Rs01Go2Kp40PolishCfg,
    Rs01Go2Kp40PolishCfgPPO,
    Rs01Go2PathPolishCfg,
    Rs01Go2PathPolishCfgPPO,
    Rs01Go2RearCoordCfg,
    Rs01Go2RearCoordCfgPPO,
    Rs01Go2Sim2SimAdaptCfg,
    Rs01Go2Sim2SimAdaptCfgPPO,
    Rs01Go2Sim2SimCalfRepairCfg,
    Rs01Go2Sim2SimCalfRepairCfgPPO,
    Rs01Go2Sim2SimKd050Cfg,
    Rs01Go2Sim2SimKd050CfgPPO,
    Rs01Go2Sim2SimRobustCfg,
    Rs01Go2Sim2SimRobustCfgPPO,
    Rs01Go2MatchedTransferCfg,
    Rs01Go2MatchedTransferCfgPPO,
    Rs01Go2Heading52Cfg,
    Rs01Go2Heading52CfgPPO,
    Rs01Go2Model930DriftRepairCfg,
    Rs01Go2Model930DriftRepairCfgPPO,
    Rs01Go2Model1425Path54Cfg,
    Rs01Go2Model1425Path54CfgPPO,
    Rs01Go2Path54Sim2SimTransferCfg,
    Rs01Go2Path54Sim2SimTransferCfgPPO,
    Rs01Go2EstimatorParityCfg,
    Rs01Go2EstimatorParityCfgPPO,
    Rs01Go2OmniDiagonalCfg,
    Rs01Go2OmniDiagonalCfgPPO,
    Rs01Go2OmniDiagonalRobot,
    Rs01Go2StraightCfg,
    Rs01Go2StraightCfgPPO,
    Rs01Go2StraightRobot,
)
from legged_gym.envs.rs01_omni_v2 import (
    Rs01OmniV2Cfg,
    Rs01OmniV2CfgPPO,
    Rs01OmniV2Robot,
    Rs01OmniV3Contact1Cfg,
    Rs01OmniV3Contact1CfgPPO,
    Rs01OmniV3Contact15Cfg,
    Rs01OmniV3Contact15CfgPPO,
    Rs01OmniV3Robot,
    Rs01OmniV4Contact1Cfg,
    Rs01OmniV4Contact1CfgPPO,
    Rs01OmniV4Contact15Cfg,
    Rs01OmniV4Contact15CfgPPO,
    Rs01OmniV4Robot,
    Rs01OmniV5Odd05Cfg,
    Rs01OmniV5Odd05CfgPPO,
    Rs01OmniV5Odd10Cfg,
    Rs01OmniV5Odd10CfgPPO,
    Rs01OmniV5Robot,
    Rs01OmniV6Seed08Cfg,
    Rs01OmniV6Seed08CfgPPO,
    Rs01OmniV6Seed11Cfg,
    Rs01OmniV6Seed11CfgPPO,
    Rs01OmniV6Robot,
    Rs01OmniV7Seed14Cfg,
    Rs01OmniV7Seed14CfgPPO,
    Rs01OmniV7Seed18Cfg,
    Rs01OmniV7Seed18CfgPPO,
    Rs01OmniV8Clearance2Cfg,
    Rs01OmniV8Clearance2CfgPPO,
    Rs01OmniV8Clearance4Cfg,
    Rs01OmniV8Clearance4CfgPPO,
    Rs01OmniV9Speed10Cfg,
    Rs01OmniV9Speed10CfgPPO,
    Rs01OmniV9Speed14Cfg,
    Rs01OmniV9Speed14CfgPPO,
)
from legged_gym.envs.rs01_omni_v2.rs01_omni_v10_config import (
    Rs01OmniV10RecoveryCfg, Rs01OmniV10RecoveryCfgPPO,
    Rs01OmniV10RecoveryStrongCfg, Rs01OmniV10RecoveryStrongCfgPPO,
)
from legged_gym.envs.rs01_omni_v2.rs01_omni_v10_env import Rs01OmniV10Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v11_config import (
    Rs01OmniV11HardGateCfg, Rs01OmniV11HardGateCfgPPO,
    Rs01OmniV11ContinuousCfg, Rs01OmniV11ContinuousCfgPPO,
)
from legged_gym.envs.rs01_omni_v2.rs01_omni_v11_env import Rs01OmniV11Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_config import Rs01OmniV12WideCfg, Rs01OmniV12WideCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v12_env import Rs01OmniV12Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_config import Rs01OmniV13DirectionCfg, Rs01OmniV13DirectionCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v13_env import Rs01OmniV13Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v14_config import Rs01OmniV14ActuatorCfg, Rs01OmniV14ActuatorCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v14_env import Rs01OmniV14Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_config import Rs01OmniV15SupportCfg, Rs01OmniV15SupportCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_env import Rs01OmniV15Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_env import Rs01OmniV15StandRobot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v15_config import Rs01OmniV15StandCfg, Rs01OmniV15StandCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_config import Rs01OmniV16Cfg, Rs01OmniV16CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v16_env import Rs01OmniV16Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v17_config import Rs01OmniV17Cfg, Rs01OmniV17CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v17_env import Rs01OmniV17Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_config import Rs01OmniV18Cfg, Rs01OmniV18LowCfg, Rs01OmniV18CfgPPO, Rs01OmniV18LowCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_env import Rs01OmniV18Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v19_config import Rs01OmniV19Cfg, Rs01OmniV19SoftCfg, Rs01OmniV19CfgPPO, Rs01OmniV19SoftCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v19_env import Rs01OmniV19Robot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v18_config import Rs01OmniV18SoftCfg, Rs01OmniV18SoftCfgPPO
from legged_gym.envs.dog import (
    DogRs01Robot,
    DogRs01TrotCfg,
    DogRs01TrotCfgPPO,
    DogRs01BalanceCfg,
    DogRs01BalanceCfgPPO,
    DogRs01BodyStableCfg,
    DogRs01BodyStableCfgPPO,
    DogRs01LowTwistCfg,
    DogRs01LowTwistCfgPPO,
    DogRs01HipTorqueCfg,
    DogRs01HipTorqueCfgPPO,
    DogRs01StraightBalanceCfg,
    DogRs01StraightBalanceCfgPPO,
    DogRs01CompactHipCfg,
    DogRs01CompactHipCfgPPO,
    DogRs01SafeTorquePathCfg,
    DogRs01SafeTorquePathCfgPPO,
    DogRs01SafeTorquePathV2Cfg,
    DogRs01SafeTorquePathV2CfgPPO,
    DogRs01SmoothStraightCfg,
    DogRs01SmoothStraightCfgPPO,
    DogRs01SmoothStraightV2Cfg,
    DogRs01SmoothStraightV2CfgPPO,
    DogRs01StraightGuardedCfg,
    DogRs01StraightGuardedCfgPPO,
    DogRs01TorqueStraightV5Cfg,
    DogRs01TorqueStraightV5CfgPPO,
    DogRs01Stage2ActuatorACfg,
    DogRs01Stage2ActuatorACfgPPO,
    DogRs01Stage2ActuatorBCfg,
    DogRs01Stage2ActuatorBCfgPPO,
    DogRs01StraightStandCfg,
    DogRs01StraightStandCfgPPO,
    DogRs01StraightWalkCfg,
    DogRs01StraightWalkCfgPPO,
)
from legged_gym.envs.fanfan.fanfan_config import FanfanRoughCfg, FanfanRoughCfgPPO
from legged_gym.envs.fanfan.fanfan_env import FanfanRobot
from legged_gym.envs.fanfan.fanfan_omni_safe_config import (
    FanfanOmniSafeCfg,
    FanfanOmniSafeCfgPPO,
    FanfanOmniFastCfg,
    FanfanOmniFastCfgPPO,
    FanfanOmniSmoothRealCfg,
    FanfanOmniSmoothRealCfgPPO,
    FanfanOmniFilteredCfg,
    FanfanOmniFilteredCfgPPO,
    FanfanOmniVelTrackV3Cfg,
    FanfanOmniVelTrackV3CfgPPO,
    FanfanOmniLateralFixCfg,
    FanfanOmniLateralFixCfgPPO,
    FanfanOmniLateralSpeedCleanCfg,
    FanfanOmniLateralSpeedCleanCfgPPO,
    FanfanOmniDesatTorqueCfg,
    FanfanOmniDesatTorqueCfgPPO,
    FanfanOmniYawDriftCleanCfg,
    FanfanOmniYawDriftCleanCfgPPO,
    FanfanOmniYawSymmetryCfg,
    FanfanOmniYawSymmetryCfgPPO,
    FanfanOmniYawPathFixCfg,
    FanfanOmniYawPathFixCfgPPO,
    FanfanOmniDiagonalCoordCfg,
    FanfanOmniDiagonalCoordCfgPPO,
    FanfanOmniCoordinatedStraightCfg,
    FanfanOmniCoordinatedStraightCfgPPO,
    FanfanOmniProjectedCoordCfg,
    FanfanOmniProjectedCoordCfgPPO,
    FanfanOmniStrongSymmetryCfg,
    FanfanOmniStrongSymmetryCfgPPO,
    FanfanOmniNoCompSymmetryCfg,
    FanfanOmniNoCompSymmetryCfgPPO,
    FanfanOmniHeadingBoundSymmetryCfg,
    FanfanOmniHeadingBoundSymmetryCfgPPO,
    FanfanOmniForceCoordCfg,
    FanfanOmniForceCoordCfgPPO,
    FanfanOmniForceDesatCfg,
    FanfanOmniForceDesatCfgPPO,
    FanfanOmniHighSpeedTransitionCfg,
    FanfanOmniHighSpeedTransitionCfgPPO,
    FanfanOmniHighAuthorityTransitionCfg,
    FanfanOmniHighAuthorityTransitionCfgPPO,
    FanfanOmniHighAuthorityDirectionCfg,
    FanfanOmniHighAuthorityDirectionCfgPPO,
    FanfanOmniHighAuthorityClosedLoopCfg,
    FanfanOmniHighAuthorityClosedLoopCfgPPO,
    FanfanOmniHighCadenceCfg,
    FanfanOmniHighCadenceCfgPPO,
    FanfanOmniSymmetricTransitionCfg,
    FanfanOmniSymmetricTransitionCfgPPO,
    FanfanOmniTiltRecovery5530Cfg,
    FanfanOmniTiltRecovery5530CfgPPO,
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
    FanfanOmniCalibratedSymmetryCfg,
    FanfanOmniCalibratedSymmetryCfgPPO,
)
from legged_gym.envs.fanfan_rouhe.fanfan_config import (
    FanfanRouheRoughCfg,
    FanfanRouheRoughCfgPPO,
)
from legged_gym.envs.fanfan_rouhe.fanfan_env import FanfanRouheRobot
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry as task_registry
task_registry.register('rs01_omni_v15_support', Rs01OmniV15Robot, Rs01OmniV15SupportCfg(), Rs01OmniV15SupportCfgPPO())
task_registry.register('rs01_omni_v15_stand_phase', Rs01OmniV15StandRobot, Rs01OmniV15StandCfg(), Rs01OmniV15StandCfgPPO())
task_registry.register('rs01_omni_v16_guarded_support', Rs01OmniV16Robot, Rs01OmniV16Cfg(), Rs01OmniV16CfgPPO())
task_registry.register('rs01_omni_v17_balance', Rs01OmniV17Robot, Rs01OmniV17Cfg(), Rs01OmniV17CfgPPO())
task_registry.register('rs01_omni_v18_balance18', Rs01OmniV18Robot, Rs01OmniV18Cfg(), Rs01OmniV18CfgPPO())
task_registry.register('rs01_omni_v18_balance14', Rs01OmniV18Robot, Rs01OmniV18LowCfg(), Rs01OmniV18LowCfgPPO())
task_registry.register('rs01_omni_v18_balance_soft', Rs01OmniV18Robot, Rs01OmniV18SoftCfg(), Rs01OmniV18SoftCfgPPO())
task_registry.register('rs01_omni_v19_placement', Rs01OmniV19Robot, Rs01OmniV19Cfg(), Rs01OmniV19CfgPPO())
task_registry.register('rs01_omni_v19_placement_soft', Rs01OmniV19Robot, Rs01OmniV19SoftCfg(), Rs01OmniV19SoftCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_config import Rs01OmniV20ACfg, Rs01OmniV20BCfg, Rs01OmniV20ACfgPPO, Rs01OmniV20BCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_env import Rs01OmniV20GeometryRobot, Rs01OmniV20BoundedRobot
task_registry.register('rs01_omni_v20_geometry', Rs01OmniV20GeometryRobot, Rs01OmniV20ACfg(), Rs01OmniV20ACfgPPO())
task_registry.register('rs01_omni_v20_bounded_hip', Rs01OmniV20BoundedRobot, Rs01OmniV20BCfg(), Rs01OmniV20BCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_config import Rs01OmniV21Cfg, Rs01OmniV21CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v21_env import Rs01OmniV21Robot
task_registry.register('rs01_omni_v21_phase_coord', Rs01OmniV21Robot, Rs01OmniV21Cfg(), Rs01OmniV21CfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_config import Rs01OmniV22Cfg, Rs01OmniV22CfgPPO, Rs01OmniV22CleanCfg, Rs01OmniV22CleanCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v22_env import Rs01OmniV22Robot
task_registry.register('rs01_omni_v22_sensor', Rs01OmniV22Robot, Rs01OmniV22Cfg(), Rs01OmniV22CfgPPO())
task_registry.register('rs01_omni_v22_clean', Rs01OmniV22Robot, Rs01OmniV22CleanCfg(), Rs01OmniV22CleanCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_scratch import Rs01V22PhaseScratchRobot, Rs01V22PhaseScratchCfg, Rs01V22PhaseScratchPPO
task_registry.register('rs01_v22_phase_scratch', Rs01V22PhaseScratchRobot, Rs01V22PhaseScratchCfg(), Rs01V22PhaseScratchPPO())
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_lift import Rs01V22PhaseLiftRobot, Rs01V22PhaseLiftCfg, Rs01V22PhaseLiftPPO
task_registry.register('rs01_v22_phase_lift', Rs01V22PhaseLiftRobot, Rs01V22PhaseLiftCfg(), Rs01V22PhaseLiftPPO())
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_guard import (Rs01V22PhaseGuardRobot, Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard3Cfg, Rs01V22PhaseGuard5PPO, Rs01V22PhaseGuard3PPO)
task_registry.register('rs01_v22_phase_guard5', Rs01V22PhaseGuardRobot, Rs01V22PhaseGuard5Cfg(), Rs01V22PhaseGuard5PPO())
from legged_gym.envs.rs01_omni_v2.rs01_v22_guard_robust import Rs01V22GuardRobustRobot, Rs01V22GuardRobustCfg, Rs01V22GuardRobustPPO
task_registry.register('rs01_v22_guard5_robust', Rs01V22GuardRobustRobot, Rs01V22GuardRobustCfg(), Rs01V22GuardRobustPPO())
task_registry.register('rs01_v22_phase_guard3', Rs01V22PhaseGuardRobot, Rs01V22PhaseGuard3Cfg(), Rs01V22PhaseGuard3PPO())
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_lateral import Rs01V22PhaseLateralRobot, Rs01V22PhaseLateralCfg, Rs01V22PhaseLateralPPO
task_registry.register('rs01_v22_phase_lateral', Rs01V22PhaseLateralRobot, Rs01V22PhaseLateralCfg(), Rs01V22PhaseLateralPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v23_config import Rs01OmniV23Cfg, Rs01OmniV23WideCfg, Rs01OmniV23CfgPPO, Rs01OmniV23WideCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v23_env import Rs01OmniV23Robot
task_registry.register('rs01_omni_v23_balance18', Rs01OmniV23Robot, Rs01OmniV23Cfg(), Rs01OmniV23CfgPPO())
task_registry.register('rs01_omni_v23_balance20', Rs01OmniV23Robot, Rs01OmniV23WideCfg(), Rs01OmniV23WideCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v24_config import Rs01OmniV24Cfg, Rs01OmniV24WideCfg, Rs01OmniV24CfgPPO, Rs01OmniV24WideCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v24_env import Rs01OmniV24Robot
task_registry.register('rs01_omni_v24_support10', Rs01OmniV24Robot, Rs01OmniV24Cfg(), Rs01OmniV24CfgPPO())
task_registry.register('rs01_omni_v24_support11', Rs01OmniV24Robot, Rs01OmniV24WideCfg(), Rs01OmniV24WideCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_config import Rs01OmniV25Cfg, Rs01OmniV25FastCfg, Rs01OmniV25CfgPPO, Rs01OmniV25FastCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v25_env import Rs01OmniV25Robot
task_registry.register('rs01_omni_v25_cadence20', Rs01OmniV25Robot, Rs01OmniV25Cfg(), Rs01OmniV25CfgPPO())
task_registry.register('rs01_omni_v25_cadence22', Rs01OmniV25Robot, Rs01OmniV25FastCfg(), Rs01OmniV25FastCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v26_config import Rs01OmniV26Cfg, Rs01OmniV26SoftCfg, Rs01OmniV26CfgPPO, Rs01OmniV26SoftCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v26_env import Rs01OmniV26Robot
task_registry.register('rs01_omni_v26_inward3', Rs01OmniV26Robot, Rs01OmniV26Cfg(), Rs01OmniV26CfgPPO())
task_registry.register('rs01_omni_v26_inward5', Rs01OmniV26Robot, Rs01OmniV26SoftCfg(), Rs01OmniV26SoftCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v27_config import Rs01OmniV27Cfg,Rs01OmniV27StrongCfg,Rs01OmniV27CfgPPO,Rs01OmniV27StrongCfgPPO
task_registry.register('rs01_omni_v27_foot15', Rs01OmniV25Robot, Rs01OmniV27Cfg(), Rs01OmniV27CfgPPO())
task_registry.register('rs01_omni_v27_foot10', Rs01OmniV25Robot, Rs01OmniV27StrongCfg(), Rs01OmniV27StrongCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_config import Rs01OmniV28Cfg,Rs01OmniV28CfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_omni_v28_env import Rs01OmniV28Robot
task_registry.register('rs01_omni_v28_curriculum', Rs01OmniV28Robot, Rs01OmniV28Cfg(), Rs01OmniV28CfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v29_config import Rs01OmniV29Cfg,Rs01OmniV29CfgPPO
task_registry.register('rs01_omni_v29_mince', Rs01OmniV28Robot, Rs01OmniV29Cfg(), Rs01OmniV29CfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_omni_v30_config import Rs01OmniV30Cfg,Rs01OmniV30CfgPPO
task_registry.register('rs01_omni_v30_clearance', Rs01OmniV28Robot, Rs01OmniV30Cfg(), Rs01OmniV30CfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import Rs01AMPCfg,Rs01AMPCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_amp_env import Rs01AMPRobot
task_registry.register('rs01_amp_style', Rs01AMPRobot, Rs01AMPCfg(), Rs01AMPCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import Rs01AMPRetimeCfg, Rs01AMPControlCfgPPO, Rs01AMPRetimeCfgPPO
task_registry.register('rs01_amp_control', Rs01AMPRobot, Rs01AMPCfg(), Rs01AMPControlCfgPPO())
task_registry.register('rs01_amp_retime', Rs01AMPRobot, Rs01AMPRetimeCfg(), Rs01AMPRetimeCfgPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_scratch_config import Rs01AMPScratchCfg, Rs01AMPScratchPPO, Rs01AMPScratchSlowCfg, Rs01AMPScratchSlowPPO
from legged_gym.envs.rs01_omni_v2.rs01_amp_scratch_env import Rs01AMPScratchRobot
task_registry.register('rs01_amp_scratch_original', Rs01AMPScratchRobot, Rs01AMPScratchCfg(), Rs01AMPScratchPPO())
task_registry.register('rs01_amp_scratch_slow', Rs01AMPScratchRobot, Rs01AMPScratchSlowCfg(), Rs01AMPScratchSlowPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_config import Rs01AMPCoreCfg, Rs01AMPCoreSlowCfg, Rs01AMPCorePPO, Rs01AMPCoreSlowPPO
from legged_gym.envs.rs01_omni_v2.rs01_amp_core_env import Rs01AMPCoreRobot
task_registry.register('rs01_amp_core_original', Rs01AMPCoreRobot, Rs01AMPCoreCfg(), Rs01AMPCorePPO())
task_registry.register('rs01_amp_core_slow', Rs01AMPCoreRobot, Rs01AMPCoreSlowCfg(), Rs01AMPCoreSlowPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_seeded_env import Rs01AMPSeededRobot
from legged_gym.envs.rs01_omni_v2.rs01_amp_seeded_config import Rs01AMPSeededCfg, Rs01AMPSeededSlowCfg, Rs01AMPSeededPPO, Rs01AMPSeededSlowPPO
task_registry.register('rs01_amp_seeded_original', Rs01AMPSeededRobot, Rs01AMPSeededCfg(), Rs01AMPSeededPPO())
task_registry.register('rs01_amp_seeded_slow', Rs01AMPSeededRobot, Rs01AMPSeededSlowCfg(), Rs01AMPSeededSlowPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_phys_config import (
    Rs01AMPPhysCfg, Rs01AMPPhysSeededCfg, Rs01AMPPhysPPO, Rs01AMPPhysSeededPPO)
task_registry.register('rs01_amp_phys_original', Rs01AMPCoreRobot, Rs01AMPPhysCfg(), Rs01AMPPhysPPO())
task_registry.register('rs01_amp_phys_seeded', Rs01AMPSeededRobot, Rs01AMPPhysSeededCfg(), Rs01AMPPhysSeededPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_inplant_config import (
    Rs01AMPInPlantCfg, Rs01AMPInPlantSeededCfg, Rs01AMPInPlantPPO, Rs01AMPInPlantSeededPPO)
task_registry.register('rs01_amp_inplant_original', Rs01AMPCoreRobot, Rs01AMPInPlantCfg(), Rs01AMPInPlantPPO())
task_registry.register('rs01_amp_inplant_seeded', Rs01AMPSeededRobot, Rs01AMPInPlantSeededCfg(), Rs01AMPInPlantSeededPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_style_first import (
    Rs01AMPStyleFirstRobot, Rs01AMPStyleFirstCfg, Rs01AMPStyleFirstSlowCfg,
    Rs01AMPStyleFirstPPO, Rs01AMPStyleFirstSlowPPO)
task_registry.register('rs01_amp_style_first', Rs01AMPStyleFirstRobot, Rs01AMPStyleFirstCfg(), Rs01AMPStyleFirstPPO())
task_registry.register('rs01_amp_style_first_slow', Rs01AMPStyleFirstRobot, Rs01AMPStyleFirstSlowCfg(), Rs01AMPStyleFirstSlowPPO())
from legged_gym.envs.rs01_omni_v2.rs01_amp_bootstrap import (
    Rs01AMPBootstrapRobot, Rs01AMPBootstrapCfg, Rs01AMPBootstrapSlowCfg,
    Rs01AMPBootstrapPPO, Rs01AMPBootstrapSlowPPO)
task_registry.register('rs01_amp_bootstrap', Rs01AMPBootstrapRobot, Rs01AMPBootstrapCfg(), Rs01AMPBootstrapPPO())
task_registry.register('rs01_amp_bootstrap_slow', Rs01AMPBootstrapRobot, Rs01AMPBootstrapSlowCfg(), Rs01AMPBootstrapSlowPPO())
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase import Rs01StablePhaseRobot, Rs01StablePhaseCfg, Rs01StablePhasePPO
task_registry.register('rs01_stable_phase', Rs01StablePhaseRobot, Rs01StablePhaseCfg(), Rs01StablePhasePPO())
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase_v2 import Rs01StablePhaseV2Robot,Rs01StablePhaseV2Cfg,Rs01StablePhaseV2PPO
task_registry.register('rs01_stable_phase_v2', Rs01StablePhaseV2Robot,Rs01StablePhaseV2Cfg(),Rs01StablePhaseV2PPO())
from legged_gym.envs.rs01_omni_v2.rs01_stable_phase_v3 import Rs01StablePhaseV3Cfg,Rs01StablePhaseV3PPO
task_registry.register('rs01_stable_phase_v3', Rs01StablePhaseV2Robot,Rs01StablePhaseV3Cfg(),Rs01StablePhaseV3PPO())

task_registry.register("go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register(
    "rs01_go2_straight",
    Rs01Go2StraightRobot,
    Rs01Go2StraightCfg(),
    Rs01Go2StraightCfgPPO(),
)
task_registry.register(
    "rs01_go2_straight_rear_coord",
    Rs01Go2StraightRobot,
    Rs01Go2RearCoordCfg(),
    Rs01Go2RearCoordCfgPPO(),
)
task_registry.register(
    "rs01_go2_straight_path_polish",
    Rs01Go2StraightRobot,
    Rs01Go2PathPolishCfg(),
    Rs01Go2PathPolishCfgPPO(),
)
task_registry.register(
    "rs01_go2_straight_kp40",
    Rs01Go2StraightRobot,
    Rs01Go2Kp40Cfg(),
    Rs01Go2Kp40CfgPPO(),
)
task_registry.register(
    "rs01_go2_straight_kp40_polish",
    Rs01Go2StraightRobot,
    Rs01Go2Kp40PolishCfg(),
    Rs01Go2Kp40PolishCfgPPO(),
)
task_registry.register(
    "rs01_go2_sim2sim_adapt",
    Rs01Go2StraightRobot,
    Rs01Go2Sim2SimAdaptCfg(),
    Rs01Go2Sim2SimAdaptCfgPPO(),
)
task_registry.register(
    "rs01_go2_sim2sim_calf_repair",
    Rs01Go2StraightRobot,
    Rs01Go2Sim2SimCalfRepairCfg(),
    Rs01Go2Sim2SimCalfRepairCfgPPO(),
)
task_registry.register(
    "rs01_go2_sim2sim_kd050",
    Rs01Go2StraightRobot,
    Rs01Go2Sim2SimKd050Cfg(),
    Rs01Go2Sim2SimKd050CfgPPO(),
)
task_registry.register(
    "rs01_go2_sim2sim_robust",
    Rs01Go2StraightRobot,
    Rs01Go2Sim2SimRobustCfg(),
    Rs01Go2Sim2SimRobustCfgPPO(),
)
task_registry.register(
    "rs01_go2_sim2sim_matched_transfer",
    Rs01Go2StraightRobot,
    Rs01Go2MatchedTransferCfg(),
    Rs01Go2MatchedTransferCfgPPO(),
)
task_registry.register(
    "rs01_go2_sim2sim_heading52",
    Rs01Go2StraightRobot,
    Rs01Go2Heading52Cfg(),
    Rs01Go2Heading52CfgPPO(),
)
task_registry.register(
    "rs01_go2_model930_drift_repair",
    Rs01Go2StraightRobot,
    Rs01Go2Model930DriftRepairCfg(),
    Rs01Go2Model930DriftRepairCfgPPO(),
)
task_registry.register(
    "rs01_go2_model1425_path54",
    Rs01Go2StraightRobot,
    Rs01Go2Model1425Path54Cfg(),
    Rs01Go2Model1425Path54CfgPPO(),
)
task_registry.register(
    "rs01_go2_path54_sim2sim_transfer",
    Rs01Go2StraightRobot,
    Rs01Go2Path54Sim2SimTransferCfg(),
    Rs01Go2Path54Sim2SimTransferCfgPPO(),
)
task_registry.register(
    "rs01_go2_estimator_parity",
    Rs01Go2StraightRobot,
    Rs01Go2EstimatorParityCfg(),
    Rs01Go2EstimatorParityCfgPPO(),
)
task_registry.register(
    "rs01_go2_omni_diagonal",
    Rs01Go2OmniDiagonalRobot,
    Rs01Go2OmniDiagonalCfg(),
    Rs01Go2OmniDiagonalCfgPPO(),
)
task_registry.register(
    "rs01_omni_v2",
    Rs01OmniV2Robot,
    Rs01OmniV2Cfg(),
    Rs01OmniV2CfgPPO(),
)
task_registry.register(
    "rs01_omni_v3_contact1",
    Rs01OmniV3Robot,
    Rs01OmniV3Contact1Cfg(),
    Rs01OmniV3Contact1CfgPPO(),
)
task_registry.register(
    "rs01_omni_v3_contact15",
    Rs01OmniV3Robot,
    Rs01OmniV3Contact15Cfg(),
    Rs01OmniV3Contact15CfgPPO(),
)
task_registry.register(
    "rs01_omni_v4_contact1",
    Rs01OmniV4Robot,
    Rs01OmniV4Contact1Cfg(),
    Rs01OmniV4Contact1CfgPPO(),
)
task_registry.register(
    "rs01_omni_v4_contact15",
    Rs01OmniV4Robot,
    Rs01OmniV4Contact15Cfg(),
    Rs01OmniV4Contact15CfgPPO(),
)
task_registry.register(
    "rs01_omni_v5_odd05",
    Rs01OmniV5Robot,
    Rs01OmniV5Odd05Cfg(),
    Rs01OmniV5Odd05CfgPPO(),
)
task_registry.register(
    "rs01_omni_v5_odd10",
    Rs01OmniV5Robot,
    Rs01OmniV5Odd10Cfg(),
    Rs01OmniV5Odd10CfgPPO(),
)
task_registry.register(
    "rs01_omni_v6_seed08",
    Rs01OmniV6Robot,
    Rs01OmniV6Seed08Cfg(),
    Rs01OmniV6Seed08CfgPPO(),
)
task_registry.register(
    "rs01_omni_v6_seed11",
    Rs01OmniV6Robot,
    Rs01OmniV6Seed11Cfg(),
    Rs01OmniV6Seed11CfgPPO(),
)
task_registry.register(
    "rs01_omni_v7_seed14",
    Rs01OmniV6Robot,
    Rs01OmniV7Seed14Cfg(),
    Rs01OmniV7Seed14CfgPPO(),
)
task_registry.register(
    "rs01_omni_v7_seed18",
    Rs01OmniV6Robot,
    Rs01OmniV7Seed18Cfg(),
    Rs01OmniV7Seed18CfgPPO(),
)
task_registry.register(
    "rs01_omni_v8_clearance2",
    Rs01OmniV5Robot,
    Rs01OmniV8Clearance2Cfg(),
    Rs01OmniV8Clearance2CfgPPO(),
)
task_registry.register(
    "rs01_omni_v8_clearance4",
    Rs01OmniV5Robot,
    Rs01OmniV8Clearance4Cfg(),
    Rs01OmniV8Clearance4CfgPPO(),
)
task_registry.register(
    "rs01_omni_v9_speed10",
    Rs01OmniV5Robot,
    Rs01OmniV9Speed10Cfg(),
    Rs01OmniV9Speed10CfgPPO(),
)
task_registry.register(
    "rs01_omni_v9_speed14",
    Rs01OmniV5Robot,
    Rs01OmniV9Speed14Cfg(),
    Rs01OmniV9Speed14CfgPPO(),
)
task_registry.register(
    "dog_rs01_trot", DogRs01Robot, DogRs01TrotCfg(), DogRs01TrotCfgPPO()
)
task_registry.register(
    "rs01_omni_v10_recovery", Rs01OmniV10Robot,
    Rs01OmniV10RecoveryCfg(), Rs01OmniV10RecoveryCfgPPO(),
)
task_registry.register(
    "rs01_omni_v10_recovery_strong", Rs01OmniV10Robot,
    Rs01OmniV10RecoveryStrongCfg(), Rs01OmniV10RecoveryStrongCfgPPO(),
)
task_registry.register(
    "rs01_omni_v11_hard_gate", Rs01OmniV11Robot,
    Rs01OmniV11HardGateCfg(), Rs01OmniV11HardGateCfgPPO(),
)
task_registry.register(
    "rs01_omni_v11_continuous", Rs01OmniV11Robot,
    Rs01OmniV11ContinuousCfg(), Rs01OmniV11ContinuousCfgPPO(),
)
task_registry.register(
    "rs01_omni_v12_wide", Rs01OmniV12Robot,
    Rs01OmniV12WideCfg(), Rs01OmniV12WideCfgPPO(),
)
task_registry.register(
    "rs01_omni_v13_direction", Rs01OmniV13Robot,
    Rs01OmniV13DirectionCfg(), Rs01OmniV13DirectionCfgPPO(),
)
task_registry.register(
    "rs01_omni_v14_actuator_parity", Rs01OmniV14Robot,
    Rs01OmniV14ActuatorCfg(), Rs01OmniV14ActuatorCfgPPO(),
)
task_registry.register(
    "dog_rs01_balance", DogRs01Robot, DogRs01BalanceCfg(), DogRs01BalanceCfgPPO()
)
task_registry.register(
    "dog_rs01_body_stable",
    DogRs01Robot,
    DogRs01BodyStableCfg(),
    DogRs01BodyStableCfgPPO(),
)
task_registry.register(
    "dog_rs01_low_twist", DogRs01Robot, DogRs01LowTwistCfg(), DogRs01LowTwistCfgPPO()
)
task_registry.register(
    "dog_rs01_hip_torque", DogRs01Robot, DogRs01HipTorqueCfg(), DogRs01HipTorqueCfgPPO()
)
task_registry.register(
    "dog_rs01_straight_balance",
    DogRs01Robot,
    DogRs01StraightBalanceCfg(),
    DogRs01StraightBalanceCfgPPO(),
)
task_registry.register(
    "dog_rs01_compact_hip",
    DogRs01Robot,
    DogRs01CompactHipCfg(),
    DogRs01CompactHipCfgPPO(),
)
task_registry.register(
    "dog_rs01_safe6nm",
    DogRs01Robot,
    DogRs01SafeTorquePathCfg(),
    DogRs01SafeTorquePathCfgPPO(),
)
task_registry.register(
    "dog_rs01_safe6nm_v2",
    DogRs01Robot,
    DogRs01SafeTorquePathV2Cfg(),
    DogRs01SafeTorquePathV2CfgPPO(),
)
task_registry.register(
    "dog_rs01_smooth_straight",
    DogRs01Robot,
    DogRs01SmoothStraightCfg(),
    DogRs01SmoothStraightCfgPPO(),
)
task_registry.register(
    "dog_rs01_smooth_straight_v2",
    DogRs01Robot,
    DogRs01SmoothStraightV2Cfg(),
    DogRs01SmoothStraightV2CfgPPO(),
)
task_registry.register(
    "dog_rs01_straight_guarded",
    DogRs01Robot,
    DogRs01StraightGuardedCfg(),
    DogRs01StraightGuardedCfgPPO(),
)
task_registry.register(
    "dog_rs01_torque_straight_v5",
    DogRs01Robot,
    DogRs01TorqueStraightV5Cfg(),
    DogRs01TorqueStraightV5CfgPPO(),
)
task_registry.register(
    "dog_rs01_stage2_actuator_a",
    DogRs01Robot,
    DogRs01Stage2ActuatorACfg(),
    DogRs01Stage2ActuatorACfgPPO(),
)
task_registry.register(
    "dog_rs01_stage2_actuator_b",
    DogRs01Robot,
    DogRs01Stage2ActuatorBCfg(),
    DogRs01Stage2ActuatorBCfgPPO(),
)
task_registry.register(
    "dog_rs01_straight_stand",
    DogRs01Robot,
    DogRs01StraightStandCfg(),
    DogRs01StraightStandCfgPPO(),
)
task_registry.register(
    "dog_rs01_straight_walk",
    DogRs01Robot,
    DogRs01StraightWalkCfg(),
    DogRs01StraightWalkCfgPPO(),
)
task_registry.register("fanfan", FanfanRobot, FanfanRoughCfg(), FanfanRoughCfgPPO())
task_registry.register(
    "fanfan_omni_safe", FanfanRobot, FanfanOmniSafeCfg(), FanfanOmniSafeCfgPPO()
)
task_registry.register(
    "fanfan_omni_fast", FanfanRobot, FanfanOmniFastCfg(), FanfanOmniFastCfgPPO()
)
task_registry.register(
    "fanfan_omni_smooth_real",
    FanfanRobot,
    FanfanOmniSmoothRealCfg(),
    FanfanOmniSmoothRealCfgPPO(),
)
task_registry.register(
    "fanfan_omni_filtered",
    FanfanRobot,
    FanfanOmniFilteredCfg(),
    FanfanOmniFilteredCfgPPO(),
)
task_registry.register(
    "fanfan_omni_veltrack_v3",
    FanfanRobot,
    FanfanOmniVelTrackV3Cfg(),
    FanfanOmniVelTrackV3CfgPPO(),
)
task_registry.register(
    "fanfan_omni_lateral_fix",
    FanfanRobot,
    FanfanOmniLateralFixCfg(),
    FanfanOmniLateralFixCfgPPO(),
)
task_registry.register(
    "fanfan_omni_lateral_speed_clean",
    FanfanRobot,
    FanfanOmniLateralSpeedCleanCfg(),
    FanfanOmniLateralSpeedCleanCfgPPO(),
)
task_registry.register(
    "fanfan_omni_desat_torque",
    FanfanRobot,
    FanfanOmniDesatTorqueCfg(),
    FanfanOmniDesatTorqueCfgPPO(),
)
task_registry.register(
    "fanfan_rouhe", FanfanRouheRobot, FanfanRouheRoughCfg(), FanfanRouheRoughCfgPPO()
)
task_registry.register("h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register("h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register("g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
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
    "fanfan_omni_tilt_recovery_5530",
    FanfanRobot,
    FanfanOmniTiltRecovery5530Cfg(),
    FanfanOmniTiltRecovery5530CfgPPO(),
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
