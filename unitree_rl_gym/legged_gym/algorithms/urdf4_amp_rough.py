"""Keep the flat AMP reference and train locomotion on a measured mesh."""
import hashlib
import json
from pathlib import Path

from legged_gym.utils.helpers import class_to_dict
from .urdf4_amp_v4 import Urdf4AMPV4Runner


class Urdf4AMPRoughRunner(Urdf4AMPV4Runner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        self.contract['rough_ground'] = class_to_dict(env.cfg.rough)
        self.contract['rough_amp_replay'] = 'learn discriminator from flat rollouts only'
        source = Path(__file__).resolve()
        self.experiment['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        if self.log_dir:
            (Path(self.log_dir) / 'experiment_contract.json').write_text(json.dumps(self.experiment, indent=2) + '\n')

    def warm_start(self, path):
        import torch
        saved = torch.load(path, map_location=self.device)
        for key in ('urdf_sha256', 'public_size', 'feedback_samples_sha256', 'native_prior_sha256', 'balance_archive_sha256'):
            if self.contract[key] != saved['contract'][key]:
                raise ValueError(f'Rough-ground seed is incompatible: {key}')
        self.alg.actor_critic.load_state_dict(saved['model_state_dict'], strict=True)
        self.style.load_state_dict(saved['amp_state'])
        self.style.clear_history()
        self._freeze_seed_policy()
        self.seed_migration = dict(checkpoint=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                   optimizers_fresh=True, terrain_changed=True)

    def _amp_weights(self, amp, style, valid):
        weights, replay = super()._amp_weights(amp, style, valid)
        rough = self.env.rough_env
        weights[rough] *= self.env.cfg.rough.rough_amp_weight
        return weights, replay & ~rough
