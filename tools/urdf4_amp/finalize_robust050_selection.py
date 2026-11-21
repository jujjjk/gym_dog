"""Preserve the best evaluated snapshot without changing a running trainer."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_robust050_selection_20261118'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    candidates=[];safety={'fall','raw_torque','flight','support'};gait={'clearance','alternation','diagonal_timing'}
    for iteration in (100,500,10000):
        n=json.load(open(BASE/('m%d_nominal60/result.json'%iteration)));r=json.load(open(BASE/('m%d_robust60/result.json'%iteration)))
        fast=next(c for c in n['cases'] if c['motion']=='fast_forward')
        count=lambda a:sum(c['passed'] for c in a['cases'])
        candidates.append(dict(iteration=iteration,nominal=n,robust=r,nominal_passed=count(n),robust_passed=count(r),fast=fast,
            rank=(not(n['core_passed'] and fast['passed']),sum(bool(safety&set(c['rejection_reasons'])) for c in n['cases']),-count(n),
                sum(bool(safety&set(c['rejection_reasons'])) for c in r['cases']),sum(bool(gait&set(c['rejection_reasons'])) for c in r['cases']),-count(r))))
    candidates.sort(key=lambda x:x['rank']);winner=candidates[0]
    assert winner['nominal']['core_passed'] and winner['fast']['passed'],'No qualified nominal all-direction + .50 candidate; keep old baseline'
    source=Path(winner['nominal']['checkpoint']);best=BASE/'best';best.mkdir(exist_ok=False);iteration=winner['iteration']
    shutil.copy2(source,best/source.name);(best/'exported').mkdir()
    for suffix in ('.pt','.pt.json'):
        p=source.parent/'exported'/('student_%d'%iteration+suffix);shutil.copy2(p,best/'exported'/p.name)
    baseline=json.load(open(BASE/'baseline_robust60/result.json'))
    selection=dict(checkpoint=str((best/source.name).resolve()),checkpoint_sha256=sha(source),source_checkpoint=str(source),iteration=iteration,
        nominal_passed=winner['nominal_passed'],robust_passed=winner['robust_passed'],tested_motions=22,fast_forward=winner['fast'],
        nominal_result=str(BASE/('m%d_nominal60/result.json'%iteration)),robust_result=str(BASE/('m%d_robust60/result.json'%iteration)),
        robust_behavior_fully_passed=winner['robust_passed']==22,baseline_robust_passed=sum(c['passed'] for c in baseline['cases']),
        screening_candidates=12,finalists=3,selection_scope='Best among the explicitly screened and retested saved checkpoints; not exhaustive of all iterations',
        training_interrupted=False,deployment_approved=False)
    assert sha(best/source.name)==sha(source)
    (BASE/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    state=json.load(open(BASE/'repository_state.json'))
    lines=['# Robust050 checkpoint selection — 2026-11-18','',
        '## 1. Repository state',f"Branch {state['branch']}, HEAD {state['head']}. Dirty workspace preserved. See repository_state.json.",
        '## 2. Physical model',f"URDF mass {state['total_mass_kg']:.2f} kg, including the provisional 0.80 kg battery correction. Thigh/calf 0.180/0.186 m. URDF SHA256 {state['urdf_sha256']}. Battery placement/inertia still require measurement.",
        '## 3. Control and actuator', 'Original 50 Hz policy, 400 Hz physics; Kp40/Kd1, RS01 delay/FOPDT, rate/acceleration limits unchanged. Operating motor cap14 Nm, raw acceptance17 Nm, continuous reference6 Nm. No PD or hardware limits changed.',
        '## 4. Baseline evidence',f"Initial accepted V4 seed is tested again under the same capped profile: strict robust gates {selection['baseline_robust_passed']}/22. Original nominal evidence remains in ../urdf4_speed_fix_20261117/REPORT.md.",
        '## 5. Problem and evidence', 'Training reward is not a selection criterion. Coarse snapshots include100–27700 and the original seed. Late checkpoints show fast-forward regression. Compare actual contact/diagonal timing, direction, speed, raw torque and stopping.',
        '## 6. Files and changes','Added select_robust050_models.py and this finalizer; evaluate_v3_policy.py restores saved capped command/curriculum settings before strict loading. Evaluation only; trainer, plant and rewards unchanged.',
        '## 7. Configuration','User envelope x[-0.30,0.50] m/s, y+/-0.20 m/s, yaw+/-0.60 rad/s; no stationary marching. Nominal and full-strength DR/push evaluations use the same sensor/control contract.',
        '## 8. Validation','Coarse screen16s+6sstop,12snapshots,22motions. Finalists60s+6sstop,seed1,3copies/motion nominal and4copies/motion full-strength DR/push. No teacher actions, resets or root/joint overwrites. Individual JSON files record hashes and exact per-motion failures. All torque peaks include startup and stopping.',
        '## 9. Training status','The user-started30000-update trainer remains running. No new training started and no weights were overwritten. This is a saved-snapshot selection.',
        '## 10. Evaluation table','| Iteration | Nominal /22 | Strict DR+push /22 | Nominal x=.50 actual m/s | Raw peak Nm | Fast passed |','|---:|---:|---:|---:|---:|---|']
    for c in candidates:lines.append(f"| {c['iteration']} | {c['nominal_passed']} | {c['robust_passed']} | {c['fast']['mean_velocity'][0]:.2f} | {c['fast']['raw_peak_Nm']:.2f} | {c['fast']['passed']} |")
    lines.extend(['## 11. Remaining risks','Selection is not deployment approval. Strict nominal thresholds are also reported under disturbances; inspect individual failures before claiming complete robustness. Static tested command points do not certify every continuous intermediate speed. Only explicitly listed snapshots were screened.',
        '## 12. Rollback',f"Selected exact copy: best/{source.name}. Source remains {source}. Original baseline remains ../urdf4_speed_fix_20261117/selected/model_50.pt. Native play accepts the absolute best directory with the selected iteration."])
    (BASE/'REPORT.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n');print(json.dumps({k:v for k,v in selection.items() if k!='fast_forward'},indent=2))


if __name__=='__main__':main()
