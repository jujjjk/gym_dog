"""Build a public policy experiment from 60s physically qualified Gym records."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym'))
import torch,numpy as np
from legged_gym.algorithms.urdf4_family_warp import FamilyStudent,family_coordinates


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clips',required=True,help='Qualified .40/.45/.50 candidate names, increasing speed')
    p.add_argument('--device',default='cuda:0');p.add_argument('--retain-speed-bias',action='store_true')
    p.add_argument('--align-contacts',action='store_true',help='Match measured contact-event phase before sharing balance feedback')
    p.add_argument('--measured-conditioning',action='store_true',help='Calibrate non-base families by measured steady speed, preserving raw reference records')
    args=p.parse_args();assert not args.output.exists()
    receipt=json.loads((args.reference/'result.json').read_text());assert receipt['seconds']>=60
    assert receipt['teacher_controller_used'] and not receipt['root_or_joint_overwrite_during_rollout'] and not receipt['automatic_resets']
    rows=receipt['cases'];chosen=[next(i for i,c in enumerate(rows) if c['spec']['clip']==name) for name in args.clips.split(',')]
    assert all(rows[i]['passed'] for i in chosen)
    speeds=torch.tensor([rows[i]['spec']['command'][0] for i in chosen]);assert torch.all(speeds[1:]>speeds[:-1]) and speeds[-1]==.5
    assert all(rows[i]['spec']['design']['period_s']==.48 for i in chosen),'Share cadence only after independent qualification'
    z=np.load(args.reference/'rollout.npz');time=z['time_s']-.02;steady=(time>=8)&(time<receipt['seconds']-2)
    frames=[];samples=[];bins=[];labels=[]
    for i in chosen:
        obs=z['observations'][steady,i,:1029];bucket=np.rint(z['phase'][steady,i]*24).astype(int)%24
        frames.append(np.stack([obs[bucket==j,:51].mean(0) for j in range(24)]));samples.append(torch.tensor(obs));bins.append(bucket)
        labels.append(torch.tensor(z['actions'][steady,i]).clamp(-4.,4.))
    requested_speeds=speeds.clone();measured_speeds=torch.stack([row[:,1026].mean() for row in samples])
    if args.measured_conditioning:
        speeds[:-1]=measured_speeds[:-1]
        for i in range(len(samples)-1):
            observations=samples[i].clone();history=observations[:,:1020].reshape(-1,20,51)
            direction=torch.stack((history[:,:,49]+1,history[:,:,48]),2)
            history[:,:,6:8]+=2*direction*(speeds[i]-requested_speeds[i])
            observations[:,1020]=speeds[i];observations[:,1023]=speeds[i]
            samples[i]=observations
    mask=torch.zeros(51);mask[:6]=1;mask[9:45]=1
    phase_shifts=np.zeros(len(chosen));event_phases=[]
    if args.align_contacts:
        for i in chosen:
            contact=z['contact'][:-1,i];phase=z['phase'][1:,i];window=steady[1:]
            events=[]
            for sign in [True,False]:
                values=[]
                for leg,offset in enumerate([0.,.5,.5,0.]):
                    active=contact[:,leg] if sign else ~contact[:,leg]
                    indices=np.flatnonzero(~active[:-1]&active[1:])+1;indices=indices[window[indices]]
                    values.extend(phase[indices]-offset)
                assert len(values)>50
                events.append(float(np.angle(np.exp(4j*np.pi*np.asarray(values)).mean())/(4*np.pi)%.5))
            event_phases.append(events)
        for i in range(len(chosen)):
            phase_shifts[i]=np.mean((np.asarray(event_phases[-1])-event_phases[i]+.25)%.5-.25)
    archive=dict(speeds=speeds,frames=torch.tensor(np.stack(frames)),action_offsets=torch.zeros(len(chosen),24,12),
        phase_offset=torch.tensor(rows[chosen[-1]]['spec']['phase_alignment_offset_cycles']),feature_mask=mask,
        phase_shifts=torch.tensor(phase_shifts,dtype=torch.float))
    policy=torch.jit.load(str(args.policy),map_location=args.device).eval()
    gpu={k:v.to(args.device) for k,v in archive.items()};errors=[]
    with torch.no_grad():
        for i,observations in enumerate(samples[:-1]):
            observations=observations.to(args.device)
            canonical,_=family_coordinates(observations,**gpu)
            # Calibrate feedforward at zero mean speed bias; retain the actual
            # measured velocity and physical labels in the source dataset.
            if not args.retain_speed_bias:
                mean_velocity=observations[:,1026:1028].mean(0)
                canonical[:,1026]+=(float(speeds[i])-mean_velocity[0]);canonical[:,1027]-=mean_velocity[1]
            predictions=torch.cat([policy(row) for row in canonical.split(512)]).cpu()
            residual=labels[i]-predictions;bucket=bins[i]
            for j in range(24):archive['action_offsets'][i,j]=residual[bucket==j].mean(0)
            errors.append(float((predictions+archive['action_offsets'][i,torch.tensor(bucket)]-labels[i]).square().mean()))
    assert archive['action_offsets'].abs().max()<2.5
    args.output.mkdir(parents=True)
    torch.save(archive,args.output/'family.pt')
    wrapped=FamilyStudent(policy,{k:v.to(args.device) for k,v in archive.items()}).eval()
    scripted=torch.jit.script(wrapped);file=args.output/'student.pt';scripted.save(str(file))
    # Exact preservation at the already accepted endpoints and low omni goals.
    with torch.no_grad():
        example=samples[-1][:32].to(args.device);assert torch.equal(scripted(example),policy(example))
        low=example.clone();low[:,1020]=.25;assert torch.equal(scripted(low),policy(low))
        for sample in samples:
            value=scripted(sample[:32].to(args.device));assert torch.isfinite(value).all()
    meta=dict(public_size=1029,urdf_sha256=receipt['urdf_sha256'],pure_public=True,teacher_controller_used=False,
        reference_receipt=str((args.reference/'result.json').resolve()),reference_receipt_sha256=hashlib.sha256((args.reference/'result.json').read_bytes()).hexdigest(),
        reference_rollout_sha256=hashlib.sha256((args.reference/'rollout.npz').read_bytes()).hexdigest(),
        source_policy=str(args.policy.resolve()),source_policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        source_clips=[rows[i]['spec']['clip'] for i in chosen],source_frame_MSE=errors,
        fast_straight_reference=[.5,0.,0.],fast_straight_threshold_m_s=.35,zero_speed_bias_feedforward_calibration=not args.retain_speed_bias,
        measured_contact_phases=event_phases,feedback_phase_shifts_cycles=phase_shifts.tolist(),
        measured_conditioning=args.measured_conditioning,controller_speed_bins=speeds.tolist(),
        requested_reference_speeds=requested_speeds.tolist(),measured_reference_speeds=measured_speeds.tolist(),
        original_policy_exact_at_forward_050_and_low_omni=True,policy_accepted=False,
        helper_sha256=hashlib.sha256((ROOT/'unitree_rl_gym/legged_gym/algorithms/urdf4_family_warp.py').read_bytes()).hexdigest())
    (args.output/'student.pt.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
