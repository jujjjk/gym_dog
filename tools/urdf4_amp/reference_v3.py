"""Checksum-checked V3 AMP pairs, variable commands/cadences and true stand.

This is a data adapter, not a training runner. It never changes the production
task, its rewards, or its control limits.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


class V3Reference:
    def __init__(self,root,urdf,simulator='mujoco',audit_only=False):
        self.root=Path(root);m=json.loads((self.root/'manifest.json').read_text())
        if m.get('dataset_version')!='V3':raise ValueError('Expected V3 data')
        if hashlib.sha256(Path(urdf).read_bytes()).hexdigest()!=m['urdf_sha256']:raise ValueError('Reference URDF hash mismatch')
        if not audit_only and simulator not in m.get('validated_simulators',[]):raise ValueError('Dataset not qualified for '+simulator+'; choose its qualified subset')
        self.clips=[];previous=[];following=[];commands=[];phases=[];phase_valid=[];cadence=[];labels=[];contacts=[]
        manifests=[(self.root,m)]
        if (self.root/'transitions/manifest.json').exists():manifests.append((self.root/'transitions',json.loads((self.root/'transitions/manifest.json').read_text())))
        for folder,manifest in manifests:
            for c in manifest['clips']:
                file=folder/(c['clip']+'.npz')
                if hashlib.sha256(file.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Clip checksum mismatch: '+c['clip'])
                z=np.load(file,allow_pickle=False);x=z['amp_features'];n=len(x)
                if x.shape!=(n,46) or not np.isfinite(x).all():raise ValueError('Invalid features')
                idx=np.flatnonzero(z['transition_valid'][:-1]);command=z['command'];valid=z['phase_valid'] if 'phase_valid' in z else np.full(n,c['family']!='stand')
                phase=z['leg_phase']
                if c.get('variable_command'):
                    # Standard diagonal clock aligned to measured contacts; keep
                    # original per-leg delayed controller clocks in the source NPZ.
                    clock=(z['time_s'][:,None]/c['design']['period_s']+np.array([0,.5,.5,0]))%1
                    active=valid & (np.linalg.norm(command,axis=1)>.01)
                    _,offset=max((float((z['contact'][active]==((clock[active]+v)%1<c['design']['stance_ratio'])).mean()),float(v)) for v in np.linspace(-.35,.35,141))
                    phase=(clock+offset)%1
                rate=z['cadence_hz'] if 'cadence_hz' in z else np.full(n,c['cadence_hz'])
                previous.append(x[idx]);following.append(x[idx+1]);commands.append((command[idx]+command[idx+1])*.5)
                phases.append(phase[idx]);phase_valid.append(valid[idx]&valid[idx+1]);cadence.append(rate[idx]);labels.append(np.full(len(idx),len(self.clips),np.int32));self.clips.append(c)
                contacts.append(z['contact'][idx])
        self.previous=np.concatenate(previous);self.following=np.concatenate(following);self.commands=np.concatenate(commands)
        self.phases=np.concatenate(phases);self.phase_valid=np.concatenate(phase_valid);self.cadence=np.concatenate(cadence);self.labels=np.concatenate(labels);self.contacts=np.concatenate(contacts)
        self.groups=[np.flatnonzero(self.labels==i) for i in range(len(self.clips))]
        self.scale=np.array([.05,.05,.1]);keys=np.round(self.commands/self.scale).astype(np.int32)
        bins,inverse=np.unique(keys,axis=0,return_inverse=True);self.bins=bins*self.scale;self.buckets=[np.flatnonzero(inverse==i) for i in range(len(bins))]
        self.zero_stand=np.flatnonzero((np.linalg.norm(self.commands,axis=1)<1e-8)&(np.linalg.norm(self.previous[:,36:38],axis=1)<.015)&(abs(self.previous[:,41])<.03)&~self.phase_valid&self.contacts.all(1))

    def sample(self,count,rng=None,command_query=None):
        rng=rng or np.random.default_rng()
        if command_query is None:
            which=rng.integers(len(self.groups),size=count)
            ids=np.array([rng.choice(self.groups[i]) for i in which])
        else:
            query=np.asarray(command_query,float).reshape(-1,3)
            if len(query)!=count:raise ValueError('One command query per sample is required')
            closest=(((query[:,None,:]-self.bins[None,:,:])/np.array([.5,.2,.8]))**2).sum(2).argmin(1)
            ids=np.array([rng.choice(self.zero_stand if np.linalg.norm(q)<1e-8 else self.buckets[b]) for q,b in zip(query,closest)])
            if np.any(abs(self.commands[ids]-query)>np.array([.061,.041,.101])):raise ValueError('Command is outside validated reference coverage')
        return dict(previous=self.previous[ids],next=self.following[ids],command=self.commands[ids],
            phase=self.phases[ids],phase_valid=self.phase_valid[ids],cadence_hz=self.cadence[ids],clip_index=self.labels[ids],pair_index=ids)
