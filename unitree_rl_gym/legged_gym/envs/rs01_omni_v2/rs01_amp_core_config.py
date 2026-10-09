"""Style-first scratch experiments, no change to identified physical RS01 plant."""
from .rs01_amp_scratch_config import Rs01AMPScratchCfg,Rs01AMPScratchSlowCfg,Rs01AMPScratchPPO,Rs01AMPScratchSlowPPO


class Rs01AMPCoreCfg(Rs01AMPScratchCfg):
    class control(Rs01AMPScratchCfg.control):
        # POLICY range, not motor speed/acceleration/torque limits. New deployment contract.
        action_scale_by_joint={'hip':.22,'thigh':.40,'calf':.55}
    class rewards(Rs01AMPScratchCfg.rewards):
        class scales:
            tracking_command_velocity=2.
            orientation=-1.
            illegal_support=-.5
            prolonged_all_feet_contact=-1.
            collision=-1.
            dof_pos_limits=-2.
            raw_torque_over_peak=-.5
            action_rate=-.002
            termination=-50.


class Rs01AMPCoreSlowCfg(Rs01AMPCoreCfg):
    class amp(Rs01AMPScratchSlowCfg.amp):pass
    class rewards(Rs01AMPCoreCfg.rewards):
        gait_period_s=1/1.5


class Rs01AMPCorePPO(Rs01AMPScratchPPO):
    class runner(Rs01AMPScratchPPO.runner):
        experiment_name='rs01_amp_core_original'
        freeze_action_std=True
        action_std_value=.35
        amp_style_weight=12.
        amp_warmup_iterations=50
        amp_disc_steps=1
        amp_disc_lr=5e-5
        # Limb geometry/velocity describes style; ideal root truth is not a discriminator shortcut.
        amp_feature_weights=[1.]*36+[0.]*10


class Rs01AMPCoreSlowPPO(Rs01AMPCorePPO):
    class runner(Rs01AMPCorePPO.runner):
        experiment_name='rs01_amp_core_slow'
        amp_reference=Rs01AMPScratchSlowPPO.runner.amp_reference
