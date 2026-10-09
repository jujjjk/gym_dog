"""Independent scratch AMP experiments; kinematic style is explicitly opt-in."""
from pathlib import Path
from .rs01_amp_config import Rs01AMPCfg, Rs01AMPCfgPPO, COMMANDS

SLOW_REFERENCE=str(Path(__file__).resolve().parents[4]/'artifacts/rs01_reference/amp_scratch_slow_20261007')


class Rs01AMPScratchCfg(Rs01AMPCfg):
    class rewards(Rs01AMPCfg.rewards):
        # No inheritance: old phase/placement/clearance rewards cannot leak in.
        class scales:
            tracking_command_velocity=4.
            alive=1.
            orientation=-1.
            lin_vel_z=-.5
            ang_vel_xy=-.03
            illegal_support=-1.
            prolonged_all_feet_contact=-1.
            stance_foot_slip=-.05
            collision=-1.
            dof_pos_limits=-2.
            raw_torque_over_peak=-.5
            torques=-.00005
            action_rate=-.005
            action_saturation=-.5
            termination=-50.


class Rs01AMPScratchSlowCfg(Rs01AMPScratchCfg):
    class amp(Rs01AMPScratchCfg.amp):
        commands=[[v*.75 for v in cmd] for cmd in COMMANDS]
    class rewards(Rs01AMPScratchCfg.rewards):
        gait_period_s=1/1.5


class Rs01AMPScratchPPO(Rs01AMPCfgPPO):
    class runner(Rs01AMPCfgPPO.runner):
        experiment_name='rs01_amp_scratch_original'
        resume=False
        load_run=-1
        checkpoint=-1
        resume_path=None
        amp_allow_reference_warmstart=False
        freeze_action_std=False
        action_std_value=.5
        amp_style_weight=4.
        max_iterations=10000


class Rs01AMPScratchSlowPPO(Rs01AMPScratchPPO):
    class runner(Rs01AMPScratchPPO.runner):
        experiment_name='rs01_amp_scratch_slow'
        amp_reference=SLOW_REFERENCE
