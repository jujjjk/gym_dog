"""60-second confirmation and bounded repairs; no long RL training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from complete_v3 import BASE, evaluate, mutate


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--generations',type=int,default=10);p.add_argument('--population',type=int,default=16)
    p.add_argument('--clips',default='');a=p.parse_args()
    out=BASE/'confirmation60';out.mkdir(exist_ok=True);rng=np.random.default_rng(3389)
    with ProcessPoolExecutor(a.workers) as pool:
        for file in sorted((BASE/'completion_search').glob('*.json')):
            if a.clips and file.stem not in a.clips.split(','):continue
            prior=json.loads(file.read_text());dest=out/file.name
            if not prior[0]['passed']:continue
            if dest.exists() and json.loads(dest.read_text())['passed']:continue
            best=[json.loads(dest.read_text())] if dest.exists() else []
            history=[]
            for gen in range(a.generations+1):
                candidates=prior[:6] if gen==0 else [mutate(best[k%min(4,len(best))],rng,max(.12,.5*.88**gen),k%2==0) for k in range(a.population)]
                batch=list(pool.map(evaluate,[(c,60,None) for c in candidates]));history.extend(batch)
                best=sorted(best+batch,key=lambda r:r['score'])[:6];b=best[0]
                print('CONFIRM60',file.stem,gen,round(b['score'],5),b['passed'],b['rejection_reasons'],flush=True)
                if b['passed']:
                    b=evaluate((b,60,str(out/b['clip'])));dest.write_text(json.dumps(b,indent=2)+'\n');break
            (out/(file.stem+'.attempts.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in history))
            if not best[0]['passed']:(out/(file.stem+'.failed.json')).write_text(json.dumps(best,indent=2))


if __name__=='__main__':main()
