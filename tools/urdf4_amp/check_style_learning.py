"""Compatibility entry point for bounded PPO plus explicit behavior verification."""
import sys
from pathlib import Path
from evaluate_cts_behavior import main,ROOT
if __name__=='__main__':
 if '--checkpoint' not in sys.argv:
  sys.argv += ['--checkpoint',str(ROOT/'artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/phase_seed_refined/run/model_0.pt')]
 main()
