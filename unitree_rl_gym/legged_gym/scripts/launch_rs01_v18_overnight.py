"""Explicitly authorized overnight A/B run, gated by completed smoke results.

Run once after the documented /tmp/rs01_v18_* tests. Logs and manifest are
durable under artifacts; children survive shell exit via a new session.
"""
import datetime
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
GYM = ROOT / 'unitree_rl_gym'
SOURCE = GYM / 'logs/rs01_omni_v17_balance/Sep12_18-06-28_v17_balance_seed17'


def report(path):
    value = Path(path).read_text()
    return json.JSONDecoder().raw_decode(value[value.index('{\n'):])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('A','C'), required=True)
    group_requested = parser.parse_args().group
    if not (SOURCE / 'model_9000.pt').is_file():
        raise RuntimeError('Missing preserved baseline')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Less than 20 GiB free')
    if '\nOK\n' not in Path('/tmp/rs01_v18_tests.log').read_text():
        raise RuntimeError('Unit tests have not passed')
    running = subprocess.check_output(['ps','-eo','args'], text=True)
    if any('scripts/train.py' in line and '--run_name=v18_' + group_requested + '_' in line
           for line in running.splitlines()):
        raise RuntimeError('V18 training already running; do not duplicate')
    sequences = {}
    for group in (group_requested,):
        fixed = report('/tmp/rs01_v18_' + group + '_wide.log')
        cases = [c for c in fixed['cases'] if c['gait_enable']]
        if (not fixed['finite_observations_and_rewards'] or len(cases) != 23
                or fixed['duration_s'] < 30
                or any(c['resets_total'] or c['flight_ratio'] > 0
                       or c['speed_domain_violations']
                       or c['max_joint_speed_rad_s'] >= 32.9867 for c in cases)):
            raise RuntimeError(group + ': fixed smoke safety gate failed')
        log = Path('/tmp/rs01_v18_' + group + '_sequence.log').read_text()
        end = next(x for x in log.splitlines() if x.startswith('Finished:'))
        path = Path(end.split('; ',1)[1])
        seq = json.loads(path.read_text())
        if (seq['resets'] or not seq['finite'] or seq['duration_s'] != 125.
                or seq['max_joint_speed_rad_s'] >= 32.9867):
            raise RuntimeError(group + ': sequence smoke safety gate failed')
        sequences[group] = path

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'artifacts/rs01_v18_balance' / stamp
    output.mkdir(parents=True, exist_ok=False)
    for path in Path('/tmp').glob('rs01_v18_*.log'):
        shutil.copy2(path, output / path.name)
    for group, path in sequences.items():
        shutil.copy2(path, output / (group + '_sequence.json'))
    for name in ('rs01_omni_v18_config.py','rs01_omni_v18_env.py'):
        shutil.copy2(GYM / 'legged_gym/envs/rs01_omni_v2' / name, output / name)
    manifest = dict(source=str(SOURCE / 'model_9000.pt'), seeds=[16],
                    additional_iterations=3000, expected_final_checkpoint=12000,
                    smoke_is_not_final_gait_acceptance=True, runs=[])
    for gpu, group, clearance, seed in ((0,'A',18,16),(1,'C',18,16)):
        if group != group_requested:
            continue
        task = 'rs01_omni_v18_balance18' if group == 'A' else 'rs01_omni_v18_balance_soft'
        name = 'v18_' + group + '_clearance' + str(clearance) + '_seed' + str(seed) + '_' + stamp
        command = [sys.executable, '-u', 'legged_gym/scripts/train.py',
                   '--task=' + task, '--num_envs=4096', '--max_iterations=3000',
                   '--run_name=' + name, '--seed=' + str(seed), '--resume',
                   '--load_run=' + str(SOURCE), '--checkpoint=9000', '--headless',
                   '--sim_device=cuda:' + str(gpu), '--rl_device=cuda:' + str(gpu)]
        logfile = output / (group + '_long.log')
        with logfile.open('x') as stream:
            child = subprocess.Popen(command, cwd=GYM, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=os.environ.copy())
        manifest['runs'].append(dict(group=group, gpu=gpu, task=task, seed=seed,
            run_name=name, pid=child.pid, log=str(logfile), command=command))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(artifact_dir=str(output), **manifest), indent=2))


if __name__ == '__main__':
    main()
