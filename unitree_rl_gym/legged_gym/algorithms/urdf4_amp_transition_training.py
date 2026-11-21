"""V4 continuation with explicit measurement of transition rollouts."""
import hashlib
import json
import copy
from pathlib import Path

import torch

from legged_gym.utils.helpers import class_to_dict
from .urdf4_amp_rough import Urdf4AMPRoughRunner
from .urdf4_transition_style import TransitionStyle


class Urdf4AMPTransitionTrainingRunner(Urdf4AMPRoughRunner):
    style_class = TransitionStyle

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        settings = (env.transition_control or {}).get('coordination')
        if settings is not None:
            train_cfg = copy.deepcopy(train_cfg)
            train_cfg['policy']['coordination_mirror_share'] = settings['mirror_share']
            train_cfg['policy']['coordination_fast_balance'] = settings['fast_balance']
            train_cfg['policy']['coordination_landing'] = tuple(settings.get('landing', (0., -.16, 1.)))
        super().__init__(env, train_cfg, log_dir, device)
        self.contract['transition_training'] = class_to_dict(env.cfg.transition)
        if env.transition_control is not None:
            self.contract['transition_control'] = dict(env.transition_control)
            self.contract['transition_seed_trust'] = {
                'coefficient': self.cfg['seed_trust_coef'],
                'deadband': self.cfg['seed_action_deadband'],
                'sampling_std_max': self.cfg['transition_sampling_std_max'],
            }
        self.contract['fast_transition_amp'] = 'masked until a validated transition reference is available'
        base = Path(__file__).with_name('urdf4_amp_rough.py').resolve()
        self.experiment['source_sha256'][str(base)] = hashlib.sha256(base.read_bytes()).hexdigest()
        if self.style.transition_sha is not None:
            self.contract.update(fast_transition_amp='matched acceleration and braking with independent discriminator',
                                 transition_reference_sha256=self.style.transition_sha,
                                 transition_reference_rows=self.style.transition_rows)
        source = Path(__file__).with_name('urdf4_transition_style.py').resolve()
        self.experiment['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        if self.log_dir:
            file = Path(self.log_dir) / 'experiment_contract.json'
            file.write_text(json.dumps(self.experiment, indent=2) + '\n')

    def _amp_weights(self, amp, style, valid):
        weights, replay = super()._amp_weights(amp, style, valid)
        # Steady fast expert pairs are an invalid target for an acceleration
        # history. The original AMP bank still shapes matched steady motions.
        steady = ~amp['fast_transition']
        if self.style.transition_sha is None:
            return weights * steady, replay & steady
        matched = amp['transition_kind'] != 0.
        # Treat controlled terrain and actuator variation as policy examples,
        # and exclude transitions with an active push or illegal support.
        transition_replay = matched & valid & (amp['style_factor'] > .999) & amp['legal'].bool()
        return weights * torch.where(matched, .35, steady.float()), (replay & steady) | transition_replay

    def _after_update(self, metrics):
        result = super()._after_update(metrics)
        if self.env.transition_control is not None:
            with torch.no_grad():
                self.alg.actor_critic.std.clamp_(max=self.cfg['transition_sampling_std_max'])
            self._update_forward_curriculum()
            result['forward_training_limit_m_s'] = self.env.training_forward_limit
        values = self.env.transition_meter.tolist()
        self.env.transition_meter.zero_()
        count, fast, roll, pitch, single, falls, cross = values
        result.update(transition_samples=count, fast_transition_samples=fast,
            transition_roll_peak_deg=roll, transition_pitch_peak_deg=pitch,
            transition_single_support_fraction=single/max(count, 1.),
            transition_falls=falls, transition_cross_peak_m=cross)
        if self.env.torque_burst is not None:
            exposed, used, peak = self.env.burst_meter.tolist()
            self.env.burst_meter.zero_()
            result.update(transition_burst_exposed_samples=exposed, transition_burst_used_samples=used,
                          transition_measured_motor_peak_nm=peak)
        if self.style.transition_sha is not None:
            accelerate, brake = self.style.transition_replay_counts.tolist()
            result.update(transition_accel_replay=accelerate, transition_brake_replay=brake)
        return result

    def _update_forward_curriculum(self):
        control = self.env.transition_control or {}
        start = control.get('training_start_forward_m_s', .50)
        progress = min(1., max(0., (self.current_learning_iteration-600)/3400))
        self.env.training_forward_limit = start + (.50-start)*progress

    def load(self, path, load_optimizer=True):
        result = super().load(path, load_optimizer=load_optimizer)
        self._update_forward_curriculum()
        return result

    def export_student(self, path):
        super().export_student(path)
        file = Path(str(path) + '.json')
        metadata = json.loads(file.read_text())
        metadata['phase_clock']['selected_from_user_target'] = False
        metadata['synchronized_transition_controller_required'] = True
        if self.env.transition_control is not None:
            metadata['transition_control'] = self.contract['transition_control']
        file.write_text(json.dumps(metadata, indent=2) + '\n')
