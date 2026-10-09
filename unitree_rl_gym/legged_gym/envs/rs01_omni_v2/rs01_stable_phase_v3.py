from .rs01_stable_phase_v2 import Rs01StablePhaseV2Cfg,Rs01StablePhaseV2PPO


class Rs01StablePhaseV3Cfg(Rs01StablePhaseV2Cfg):
    pass


class Rs01StablePhaseV3PPO(Rs01StablePhaseV2PPO):
    class runner(Rs01StablePhaseV2PPO.runner):
        experiment_name='rs01_stable_phase_v3'
        phase_seed_initialization=True
