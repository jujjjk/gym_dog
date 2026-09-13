"""Headless evaluation without wall-clock pacing; physics dt is unchanged."""
import sys
import types
import runpy
from pathlib import Path
import isaacgym  # noqa: F401
from legged_gym.envs.base import legged_robot

if '--headless' not in sys.argv:
    raise SystemExit('Fast wrapper requires --headless')
mode = sys.argv.pop(1)
scripts = Path(__file__).resolve().parent
targets = {'balance': scripts / 'evaluate_rs01_balance.py',
           'wide': scripts / 'evaluate_rs01_go2_omni.py',
           'sequence': scripts.parents[2] / 'artifacts/rs01_v16_selection/play_sequence.py'}
if mode not in targets:
    raise SystemExit('Choose balance, wide or sequence')
legged_robot.time = types.SimpleNamespace(sleep=lambda seconds: None)
runpy.run_path(str(targets[mode]), run_name='__main__')
