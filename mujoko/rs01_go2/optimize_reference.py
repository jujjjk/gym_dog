"""Bounded physical direct shooting over teacher geometry/timing, not RL/MPC.

Evaluates actual RS01 plant rollouts; never auto-approves clips for AMP.
Separate candidate directories preserve baselines and raw evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
KEYS=['frequency','stance-ratio','body-height','half-width',
      'front-x-offset','rear-x-offset','front-lift','rear-lift']
LOW=np.array([.90,.58,.275,.155,-.04,-.04,24.,24.])
HIGH=np.array([1.40,.70,.310,.180,.04,.04,40.,40.])
INITIAL=np.array([1.10,.64,.290,.175,0.,0.,30.,30.])
FIXED={}


def metrics(folder):
    report=json.loads((folder/'forward_physics/physical_report.json').read_text())
    if report['stop_reason']!='completed':return 1e4,report
    with np.load(folder/'forward_physics/physical.npz') as d:
        contact=d['desired_contact'].astype(bool)
        peaks=[]
        for leg in range(4):
            swing=~contact[:,leg]
            starts=np.flatnonzero(~swing[:-1]&swing[1:])+1
            ends=np.flatnonzero(swing[:-1]&~swing[1:])+1
            values=[d['foot_clearance_m'][a:ends[ends>a][0],leg].max()*1000
                    for a in starts if np.any(ends>a)]
            peaks.append(float(min(values)) if values else 0.)
        z=d['qpos'][:,2]
        report['complete_swing_min_peak_mm']=peaks
        report['base_z_std_mm']=float(z.std()*1000)
        report['base_z_peak_to_peak_mm']=float(np.ptp(z)*1000)
    v=np.asarray(report['velocity_rmse'])
    angles=np.asarray(report['roll_pitch_max_deg'])
    # Independent dimensions: tracking, body, contact/clearance, actuator demand.
    loss=float(np.sum((v/np.array([.03,.03,.15]))**2)
        +np.sum((angles/4.)**2)+(report['base_z_std_mm']/3.)**2
        +20*np.mean((np.maximum(20-np.asarray(peaks),0)/20)**2)
        +10*np.mean(np.asarray(report['planned_swing_contact_ratio'])**2)
        +5*(max(report['final_target_rate_ratio']-1,0)**2
            +max(report['final_target_acceleration_ratio']-1,0)**2)
        +200*(report['flight_ratio']+report['illegal_two_foot_ratio']
              +report['single_support_ratio']+report['inward_support_ratio'])
        +100*report['active_over_limit_ratio'])
    report['shooting_loss']=loss
    report['amp_training_approved']=False
    return loss,report


def evaluate(job):
    index,x,output,seconds,seed,speed=job
    folder=output/('candidate_%03d'%index)
    command=[sys.executable,str(ROOT/'mujoko/rs01_go2/reference_fresh.py'),
        '--motion','forward','--seconds',str(seconds),'--forward-speed',str(speed),
        '--physics','--feedback','--actuator-compensation','1',
        '--support-smoothing','.04','--unload-phase','.08','--contact-handoff',
        '--swing-ground-gain','.5','--sensor-seed',str(seed),'--output',str(folder)]
    params=dict(FIXED);params.update(zip(KEYS,x.tolist()))
    for key,value in params.items():
        if not key.startswith('harmonic_'):command+=['--'+key,format(float(value),'.12f')]
    if 'harmonic_00' in params:
        command+=['--control-harmonics']+[format(params['harmonic_%02d'%i],'.12f') for i in range(12)]
    with (output/('candidate_%03d.log'%index)).open('w') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    record={'index':index,'parameters':params,
            'command':command,'folder':str(folder),'amp_training_approved':False}
    if result.returncode:
        record.update(loss=1e5,error='rollout process failed; see log')
    else:
        loss,report=metrics(folder);record.update(loss=loss,metrics=report)
    (output/('candidate_%03d.json'%index)).write_text(json.dumps(record,indent=2)+'\n')
    print(index,round(record['loss'],3),flush=True)
    return record


def main():
    global KEYS,LOW,HIGH,INITIAL,FIXED
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--generations',type=int,default=4)
    p.add_argument('--population',type=int,default=12)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--speed',type=float,default=.15)
    p.add_argument('--seed',type=int,default=20261007)
    p.add_argument('--control-from',type=Path,help='Freeze best geometry from summary and optimize final joint-control harmonics')
    args=p.parse_args()
    if args.control_from:
        FIXED=json.loads(args.control_from.read_text())['best']['parameters']
        KEYS=['harmonic_%02d'%i for i in range(12)]
        LOW=np.full(12,-.04);HIGH=np.full(12,.04);INITIAL=np.zeros(12)
    if not (1<=args.generations<=10 and 4<=args.population<=32 and 1<=args.workers<=4
            and 4<=args.seconds<=30 and 0<args.speed<=.25):p.error('Invalid bounded search')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(args.seed);mean=(INITIAL-LOW)/(HIGH-LOW);std=np.full(len(KEYS),.20)
    records=[];best=None
    with ThreadPoolExecutor(args.workers) as pool:
        for generation in range(args.generations):
            population=np.clip(rng.normal(mean,std,(args.population,len(KEYS))),0,1)
            population[0]=mean if best is None else (np.array([best['parameters'][k] for k in KEYS])-LOW)/(HIGH-LOW)
            start=len(records)
            jobs=[(start+i,LOW+x*(HIGH-LOW),args.output,args.seconds,args.seed,args.speed)
                  for i,x in enumerate(population)]
            batch=list(pool.map(evaluate,jobs));records+=batch
            elite=sorted(batch,key=lambda r:r['loss'])[:max(3,args.population//4)]
            values=np.array([[r['parameters'][k] for k in KEYS] for r in elite])
            normalized=(values-LOW)/(HIGH-LOW)
            mean=.25*mean+.75*normalized.mean(0)
            std=np.maximum(.045,.25*std+.75*normalized.std(0))
            best=min(records,key=lambda r:r['loss'])
            summary={'kind':'reduced_parameter_physical_direct_shooting',
                'speed_m_s':args.speed,'seed':args.seed,'seconds':args.seconds,
                'evaluated':len(records),'amp_training_approved':False,'best':best,
                'ranking':[{'index':r['index'],'loss':r['loss']} for r in sorted(records,key=lambda r:r['loss'])]}
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print('generation',generation,'best',best['index'],best['loss'],flush=True)


if __name__=='__main__':main()
