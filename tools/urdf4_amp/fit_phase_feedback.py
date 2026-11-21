"""Fit a bounded phase feedback initializer from actual assisted Gym states."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import torch,numpy as np
sys.path.insert(0,str(ROOT/'unitree_rl_gym'))
from legged_gym.algorithms.urdf4_phase_feedback import balance_features,filtered_balance_features
from legged_gym.algorithms.urdf4_native_prior import conditioned_native_actions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extra',type=Path,action='append',default=[]);parser.add_argument('--output',type=Path,default=ROOT/'artifacts/urdf4_speed_fix_20261117/balance_prior.pt')
    parser.add_argument('--temporal',action='store_true')
    parser.add_argument('--heading',action='store_true',help='Include existing public route-heading error in feedback')
    parser.add_argument('--derivatives',type=Path,help='Local teacher derivatives, never added to AMP expert data')
    parser.add_argument('--derivative-weight',type=float,default=100.)
    args=parser.parse_args();output=args.output
    assert not output.exists(),'Preserve previous balance fit'
    base=ROOT/'artifacts/urdf4_speed_fix_20261117';old=torch.load(base/'corrective1/model_0.pt',map_location='cpu');weights=old['model_state_dict'];cfg=old['experiment']['env']
    path=Path(cfg['amp_cts']['reference']);meta=json.load(open(path/'manifest.json'));c=next(c for c in meta['clips'] if c['command']==[.5,0,0]);rp=Path(c['isaac_gym_validation']['receipt']);r=json.load(open(rp));file=rp.parent/'live_rollout.npz';z=np.load(file);i=next(j for j,v in enumerate(r['cases']) if v['clip']==c['isaac_gym_validation']['case']['clip'])
    names=meta['joint_order'];default=np.array([cfg['init_state']['default_joint_angles'][n] for n in names]);scale=np.array([.15,.25,.25]*4)
    q0=np.load(path/c['source_folder']/'physical.npz')['actuator_default'];correction=(q0-default)*(1/.928-1)
    previous=(z['issued_joint_targets'][:-1,i]+correction-default)/scale
    f=z['amp_features'][:-1,i];time=z['time_s'][1:]-.02;n=round(c['design']['period_s']/.02)
    features=np.c_[f[:,39:42]*.25,f[:,42:44],f[:,:12]-default,f[:,12:24]*.05,previous,f[:,36:38]-np.array(c['command'][:2])]
    if args.heading:
        root=z['root_state_xyzw_world'][:-1,i];qx,qy,qz,qw=root[:,3:7].T
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        features=np.c_[features,-np.sin(yaw),np.cos(yaw)-1]
    steady=(time>=5)&(time<55);bucket=np.rint(time/.02).astype(int)%n
    reference=torch.tensor(np.stack([features[steady&(bucket==j)].mean(0) for j in range(n)]),dtype=torch.float)
    cold=(time>-.5)&(time<-.05);stand=features[cold].mean(0);stand[-2:]=f[cold,36:38].mean(0)
    stand=torch.tensor(stand,dtype=torch.float);scales=torch.tensor([.125]*3+[.05]*2+[.05]*12+[.10]*12+[.2]*12+[.15]*2)
    if args.heading:scales=torch.cat((scales,torch.tensor([.05,.05])))
    data=torch.load(base/'corrective1/corrective_samples.pt',map_location='cpu');expert=torch.load(base/'feedback_data/samples.pt',map_location='cpu');high=expert['observations'][:,1020]>.35
    observations=torch.cat((data['observations'],expert['observations'][high]));targets=torch.cat((data['actions'],expert['actions'][high]))
    for extra in args.extra:
        supplemental=torch.load(extra,map_location='cpu');observations=torch.cat((observations,supplemental['observations']));targets=torch.cat((targets,supplemental['actions']))
    pub=observations[:,:1029];prior=conditioned_native_actions(pub,weights['prior_commands'],weights['native_table'],weights['native_counts'],weights['native_offsets'],weights['native_stands'],weights['stand_action'])
    amplitude=pub[:,1028:1029];keep=amplitude[:,0]>.5;offset=torch.tensor(c['phase_alignment_offset_cycles'])
    feature_function=filtered_balance_features if args.temporal else balance_features
    x=feature_function(pub,reference,stand,scales,offset)[keep].to('cuda:1');delta=(targets-prior)[keep].to('cuda:1');amp=amplitude[keep].to('cuda:1')
    y=torch.atanh((delta/amp).clamp(-.98,.98));held=(torch.arange(len(x),device=x.device)%5)==4;train=~held
    # Normalize by sample count so ridge values have stable meaning.
    xx=x[train].T@x[train]/int(train.sum());xy=x[train].T@y[train]/int(train.sum());gains=[];reports=[]
    derivative_sources={}
    if args.derivatives:
        queries=torch.load(args.derivatives,map_location='cpu')
        assert queries['report']['not_AMP_expert_data'] and queries['report']['maximum_pose_perturbation_FK_foot_error_m']<.0005
        assert queries['report']['urdf_sha256']==meta['urdf_sha256']
        # These labels differentiate the unfiltered feedback function. Apply
        # the runtime temporal filter exactly once, not in this local Jacobian.
        dx=(balance_features(queries['following_public'],reference,stand,scales,offset)-
            balance_features(queries['previous_public'],reference,stand,scales,offset)).to(x.device)
        dy=queries['delta_action'].to(x.device)
        xx+=args.derivative_weight*(dx.T@dx)/len(dx)
        xy+=args.derivative_weight*(dx.T@dy)/len(dx)
        derivative_sources={str(args.derivatives):hashlib.sha256(args.derivatives.read_bytes()).hexdigest()}
    for ridge in [.0001,.001,.01,.1]:
        gain=torch.linalg.solve(xx+torch.eye(x.shape[1],device=x.device)*ridge,xy)
        prediction=amp[held]*torch.tanh(x[held]@gain)
        mse=float((prediction-delta[held]).square().mean());gains.append(gain.cpu());reports.append(dict(ridge=ridge,holdout_MSE=mse))
        print('PHASE FEEDBACK',ridge,mse,flush=True)
    archive=dict(gains=torch.stack(gains),reports=reports,reference=reference,stand=stand,scales=scales,offset=offset,temporal=args.temporal,
        urdf_sha256=hashlib.sha256((ROOT/'unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf').read_bytes()).hexdigest(),source_record_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        corrective_sha256=hashlib.sha256((base/'corrective1/corrective_samples.pt').read_bytes()).hexdigest(),
        supplemental_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.extra},
        derivative_sources=derivative_sources,derivative_weight=args.derivative_weight if args.derivatives else 0.,
        public_heading_features=args.heading,
        features='gyro-yaw-error, gravity xy, q error, dq error, previous target error, measured-velocity error; four Fourier harmonics',
        runtime_inputs='public joints/IMU/commands and the already contracted corrected planar localization velocity')
    torch.save(archive,output);output.with_suffix('.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
