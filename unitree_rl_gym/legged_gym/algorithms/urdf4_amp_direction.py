"""AMP+CTS direction continuation with a bound, reviewable preflight gate."""
import hashlib,inspect,json
from pathlib import Path
import torch
from .urdf4_amp_expand import Urdf4AMPExpandRunner,CoveredStyle
from .urdf4_amp_cts import Urdf4AMPCTSRunner
from legged_gym.envs.urdf4.path_tracking import PlanarPathController


class DirectionStyle(CoveredStyle):
    def rollout_reward(self,a):
        return super().rollout_reward(dict(a,command=a['style_command']))


class Urdf4AMPDirectionRunner(Urdf4AMPExpandRunner):
    style_class=DirectionStyle
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        source=Path(inspect.getsourcefile(PlanarPathController))
        self.experiment['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        for cls in (Urdf4AMPExpandRunner,Urdf4AMPCTSRunner):
            source=Path(inspect.getsourcefile(cls))
            self.experiment['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        if self.log_dir:
            (Path(self.log_dir)/'experiment_contract.json').write_text(json.dumps(self.experiment,indent=2,default=str)+'\n')
        self.direction_status={}
        self.direction_updates_since_load=0

    def _amp_weights(self,a,style,valid):
        return super()._amp_weights(dict(a,command=a['style_command']),style,valid)

    def _tracking_statistics(self,a):
        return super()._tracking_statistics(dict(a,command=a['style_command']))

    def load(self,path,load_optimizer=True):
        saved=torch.load(path,map_location=self.device)
        if saved.get('contract',{}).get('schema')==self.contract['schema']:
            result=super().load(path,load_optimizer)
            self.env.direction_meter.zero_()
            return result
        expected=dict(self.contract,schema=saved.get('contract',{}).get('schema'))
        if expected!=saved.get('contract'):raise ValueError('Direction seed public/plant/data contract mismatch')
        if expected['schema']!='v4_continuous_robust_bounded_xonly':raise ValueError('Use the selected x-expansion checkpoint')
        self.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        self.style.load_state_dict(saved['amp_state']);self.style.clear_history()
        self._freeze_seed_policy()  # Anchor this selected gait, not an older ancestor.
        self.current_learning_iteration=0
        self.loaded_checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.warm_start_sha256=self.loaded_checkpoint_sha256
        self.fresh_stage_warmstart=True
        self.env.reset()
        print('DIRECTION WARM START: selected weights retained; fresh PPO/encoder optimizers; new iteration 0.',flush=True)
        return saved.get('infos')

    def _validate_training_request(self,iterations):
        super()._validate_training_request(iterations)
        if self.env.cfg.rewards.heading_gain!=0:raise ValueError('Duplicate heading correction in reward')
        if iterations>500:
            path=Path(self.cfg['direction_preflight'])
            receipt=json.loads(path.read_text())
            if (not receipt.get('approved_for_training') or receipt['checkpoint_sha256']!=self.loaded_checkpoint_sha256
                or receipt['contract']!=self.contract or receipt['source_sha256']!=self.experiment['source_sha256']):
                raise ValueError('Direction preflight does not match checkpoint, contract and current code')

    def _after_update(self,metrics):
        self.direction_updates_since_load+=1
        if self.direction_updates_since_load>20 and metrics['amp_valid_fraction']<.40:
            raise RuntimeError('AMP moving pairs unexpectedly filtered; refusing ineffective direction training')
        env=self.env;cfg=env.cfg.expand
        old_stage=env.curriculum_stage
        if self.current_learning_iteration%cfg.curriculum_window==0:
            meter=env.direction_meter.clone();env.direction_meter.zero_()
            counts=meter[:,2];mean=meter[:,:2]/counts[:,None].clamp(min=1)
            ready=bool((counts>=cfg.curriculum_min_samples_per_direction).all())
            passed=ready and bool((mean[:,0]<=env.cfg.direction.curriculum_cross_m).all()
                and (mean[:,1]<=env.cfg.direction.curriculum_heading_rad).all())
            self.direction_status=dict(direction_window_passed=float(passed),
                direction_worst_cross_m=float(mean[:,0].max()),direction_worst_heading_rad=float(mean[:,1].max()))
            # A direction failure breaks consecutive curriculum pass windows.
            if not passed:env.curriculum_pass_windows=0
        result=super()._after_update(metrics)
        if env.curriculum_stage>old_stage and not self.direction_status.get('direction_window_passed',False):
            env.curriculum_stage=old_stage;env.curriculum_pass_windows=0
        return dict(result,**self.direction_status,curriculum_stage=env.curriculum_stage)

    def export_student(self,path):
        super().export_student(path)
        meta_path=Path(path+'.json');meta=json.loads(meta_path.read_text())
        meta.update(training_objective='omni_direction',
            heading_target='Integrate external user yaw only; do not integrate corrected twist yaw',
            deployment_requires_path_controller=True,
            path_controller=dict(gains=self.env.cfg.direction.gains,
                pose='External globally corrected planar position/yaw/velocity; fixed reference frame',
                simulation='Nominal truth or noisy delayed simulated localization; not hardware validation'))
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
