"""Regenerate exact time-scaled kinematic style, never pretend physical validation."""
import json
from types import SimpleNamespace
import numpy as np
from reference_gait import ROOT, Kinematics, make_clip, export_clip


def main():
    source=ROOT/'artifacts/rs01_reference/design_10s_20261005'
    out=ROOT/'artifacts/rs01_reference/amp_scratch_slow_20261007'
    original=json.loads((source/'manifest.json').read_text());settings=original['settings']
    cfg=SimpleNamespace(**settings);cfg.frequency*=.75;cfg.seconds=10.;cfg.output=str(out)
    k=Kinematics(cfg.scene,cfg.contract)
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(original);manifest.update(settings=vars(cfg),clips=[],amp_training_approved=False,
        note='Explicit experimental AMP style reference, NOT dynamically validated expert',
        source_reference=str(source),time_scale=.75,old_policy_loaded=False)
    for clip in original['clips']:
        name=clip['clip']
        with np.load(source/(name+'.npz')) as old:command=old['command'][0]*.75
        data=make_clip(k,command,cfg)
        manifest['clips'].append(export_clip(out,name,data,k,cfg))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
