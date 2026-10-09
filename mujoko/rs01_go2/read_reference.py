"""Read explicitly labelled reference transitions without joining clip boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np


class ReferenceDataset:
    def __init__(self, directory, allow_kinematic=False):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if not self.manifest['amp_training_approved'] and not allow_kinematic:
            raise ValueError('Unvalidated kinematic design. Pass allow_kinematic=True explicitly for style experiments.')
        self.clips=[]
        for clip in self.manifest['clips']:
            with np.load(self.directory/(clip['clip']+'.npz'),allow_pickle=False) as f:
                a={key:f[key].copy() for key in f.files}
            if a['amp_features'].shape[1]!=46:raise ValueError('Expected 46D style features')
            if not np.isfinite(a['amp_features']).all():raise ValueError('Nonfinite features')
            self.clips.append(a)

    def sample(self, count, seed=0):
        """Uniform clip sampling. Returns [state,next_state], body command, clip id."""
        rng=np.random.RandomState(seed);ids=rng.randint(len(self.clips),size=count)
        pairs=[];commands=[]
        for ci in ids:
            a=self.clips[ci];indices=np.flatnonzero(a['transition_valid'][:-1]);i=int(rng.choice(indices))
            pairs.append(np.r_[a['amp_features'][i],a['amp_features'][i+1]])
            commands.append(a['command'][i])
        return np.asarray(pairs,dtype=np.float32),np.asarray(commands,dtype=np.float32),ids


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--allow-kinematic',action='store_true');a=p.parse_args()
    d=ReferenceDataset(a.directory,a.allow_kinematic);pairs,cmd,ids=d.sample(32)
    print('Clips:',len(d.clips),'dt:',d.manifest['frame_dt_s'])
    print('AMP transition batch:',pairs.shape,'command:',cmd.shape,'finite:',np.isfinite(pairs).all())
    print('Kinematic style example only; no implied physical approval.')
