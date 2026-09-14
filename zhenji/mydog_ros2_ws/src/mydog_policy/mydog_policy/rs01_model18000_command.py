"""Explicit short B18000 trial; cannot accidentally address the A6850 node."""
import sys
from .rs01_model6850_command import main as command_main


def main(args=None):
    args=list(sys.argv[1:] if args is None else args)
    if any(x=='--namespace' or x.startswith('--namespace=') for x in args):
        raise SystemExit('B18000 command namespace cannot be overridden')
    command_main(args+['--namespace','/mydog/model18000'])
