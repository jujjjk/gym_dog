"""Separate discriminator for validated faster acceleration and braking pairs."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .urdf4_amp_v3 import V3Style


class TransitionStyle(V3Style):
    def __init__(self, path, env, settings=None, validation_receipt=None):
        super().__init__(path, env, settings, validation_receipt)
        self.transition_sha = None
        self.transition_rows = 0
        directory = (settings or {}).get('transition_reference')
        if not directory:
            return
        directory = Path(directory)
        receipt_file, data_file = directory/'result.json', directory/'qualified_pairs.npz'
        receipt = json.loads(receipt_file.read_text())
        qualified = {trial['trial'] for trial in receipt['trials'] if trial['passed']}
        if (receipt['urdf_sha256'] != env.urdf_sha256 or abs(receipt['dt_s']-env.dt) > 1.e-7
                or not receipt['true_gym_pairs'] or receipt['auto_reset'] or len(qualified) < 2
                or receipt.get('height_reference') != 'local_terrain'):
            raise ValueError('Fast AMP reference lacks compatible, qualified physical trials')
        with np.load(data_file, allow_pickle=False) as reference:
            pair = torch.tensor(reference['pairs'], device=env.device, dtype=torch.float32)
            self.transition_command = torch.tensor(reference['command'], device=env.device, dtype=torch.float32)
            following = torch.tensor(reference['next_command'], device=env.device, dtype=torch.float32)
            self.transition_kind = (following[:, 0] - self.transition_command[:, 0]).sign()
            clips = set(reference['source_trial'].tolist())
        if (clips != qualified or pair.shape != (len(self.transition_command), 92)
                or not torch.isfinite(pair).all() or not torch.isfinite(following).all()
                or not torch.isfinite(self.transition_command).all() or set(self.transition_kind.unique().tolist()) != {-1., 1.}):
            raise ValueError('Invalid fast-transition reference rows')
        self.transition_pair = pair
        self.transition_rows = len(pair)
        self.transition_sha = hashlib.sha256(receipt_file.read_bytes()+data_file.read_bytes()).hexdigest()
        self.transition_net = copy.deepcopy(self.net)
        self.transition_net[0] = nn.Linear(96, self.net[0].out_features, device=env.device)
        self._seed_transition_net()
        self.transition_optimizer = torch.optim.Adam(self.transition_net.parameters(), lr=1.e-4, weight_decay=1.e-4)
        self.transition_buffer = torch.zeros(16384, 96, device=env.device)
        self.transition_replay_counts = torch.zeros(2, dtype=torch.long, device=env.device)
        self.transition_size = self.transition_cursor = self.transition_updates = 0

    def _seed_transition_net(self):
        with torch.no_grad():
            self.transition_net[0].weight[:, :95].copy_(self.net[0].weight)
            self.transition_net[0].weight[:, 95].zero_()
            self.transition_net[0].bias.copy_(self.net[0].bias)
            for index in (2, 4):
                self.transition_net[index].load_state_dict(self.net[index].state_dict())

    def rollout_reward(self, amp):
        score, inputs, valid = super().rollout_reward(amp)
        if self.transition_sha is None:
            return score, inputs, valid
        kind = amp['transition_kind']
        inputs = torch.cat((inputs, kind[:, None]), 1)
        selected = kind != 0.
        if selected.any():
            value = self.transition_net(inputs[selected]).flatten()
            score[selected] = (1.-.25*(value-1.).square()).clamp(0., 1.)
        return score, inputs, valid

    def add(self, inputs):
        if self.transition_sha is None:
            return super().add(inputs)
        steady = inputs[:, 95] == 0.
        super().add(inputs[steady, :95])
        samples = inputs[~steady][-len(self.transition_buffer):]
        self.transition_replay_counts += torch.stack(((samples[:, 95] > 0.).sum(), (samples[:, 95] < 0.).sum()))
        indices = (torch.arange(len(samples), device=inputs.device)+self.transition_cursor) % len(self.transition_buffer)
        self.transition_buffer[indices] = samples
        self.transition_cursor = (self.transition_cursor+len(samples)) % len(self.transition_buffer)
        self.transition_size = min(len(self.transition_buffer), self.transition_size+len(samples))

    def update(self):
        super().update()
        if self.transition_sha is None or self.transition_size < 32:
            return
        device = self.transition_buffer.device
        fake = self.transition_buffer[torch.randint(self.transition_size, (256,), device=device)]
        distance = abs(fake[:, 92:93]*.5 - self.transition_command[None, :, 0])
        distance += (fake[:, 95:96] != self.transition_kind[None]).float()*100.
        nearest = distance.topk(8, largest=False).indices
        sample = nearest[torch.arange(len(fake), device=device), torch.randint(8, (len(fake),), device=device)]
        real = self._encode(self.transition_pair[sample], self.transition_command[sample],
                            fake[:, 92:95]*fake.new_tensor([.5, .2, .8]))
        real = torch.cat((real, fake[:, 95:96]), 1).detach().requires_grad_(True)
        expert_score = self.transition_net(real)
        policy_score = self.transition_net(fake)
        grad = torch.autograd.grad(expert_score.sum(), real, create_graph=True)[0][:, :92]
        loss = .5*((expert_score-1.).square().mean()+(policy_score+1.).square().mean())+10.*grad.square().sum(1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite transition discriminator')
        self.transition_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.transition_net.parameters(), 1.)
        self.transition_optimizer.step()
        self.transition_updates += 1
        self.metrics.update(transition_disc_loss=float(loss.detach()), transition_disc_updates=float(self.transition_updates))

    def state_dict(self):
        state = super().state_dict()
        if self.transition_sha is not None:
            state['transition'] = dict(sha=self.transition_sha, net=self.transition_net.state_dict(),
                optimizer=self.transition_optimizer.state_dict(), updates=self.transition_updates)
        return state

    def load_state_dict(self, state, allow_reference_warmstart=False):
        super().load_state_dict(state, allow_reference_warmstart)
        saved = state.get('transition')
        if saved is not None:
            if self.transition_sha != saved['sha']:
                raise ValueError('Transition AMP reference changed since checkpoint')
            self.transition_net.load_state_dict(saved['net'])
            self.transition_optimizer.load_state_dict(saved['optimizer'])
            self.transition_updates = saved['updates']
        elif self.transition_sha is not None:
            self._seed_transition_net()
