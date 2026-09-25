"""Explicit B23500 motion command with an isolated command namespace."""
import sys
from .rs01_model6850_command import main as command_main


def main(args=None):
    args = list(sys.argv[1:] if args is None else args)
    if any(x == '--namespace' or x.startswith('--namespace=') for x in args):
        raise SystemExit('B23500 command namespace cannot be overridden')
    fast = '--fast' in args
    args = [x for x in args if x != '--fast']
    command_main(args + ['--namespace', '/mydog/model23500'],
                 allow_continuous=True, speed_caps=(.40, .30, .60) if fast else (.30, .20, .30))
