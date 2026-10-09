"""Guard5 B32750: unconditional 5-degree inward hip compression on the V22 bridge.

The rs01_v22_phase_guard5 task replaces the V20 conditional map with an
all-command inward target compression, applied after action clipping/scaling
and BEFORE rate/acceleration limits, delay and the identified RS01 motor
model. V22Sim.update_policy already follows that order via the exported
contract's v20 block (mapping=omnidirectional_inward_scale_v1).
"""
import json
from pathlib import Path
import sim2sim_v22 as v22  # imports isaacgym before torch
import sim2sim_v20 as bridge
import torch

ROOT = Path(__file__).resolve().parents[2]


class Guard5Sim(v22.V22Sim):
    task_name = 'rs01_v22_phase_guard5'
    supported_mappings = ('omnidirectional_inward_scale_v1',)


if __name__ == '__main__':
    p = bridge.get_parser()
    p.add_argument('--sequence', action='store_true')
    p.add_argument('--fast', action='store_true')
    p.add_argument('--sensor_profile', choices=('clean', 'normal', 'robust', 'mixed'), default='normal')
    p.add_argument('--sensor_seed', type=int, default=20260925)
    p.add_argument('--interval', type=float, default=5.)
    p.add_argument('--direct', action='store_true')
    for action in p._actions:
        if action.dest in ('scene', 'policy', 'output'):
            action.required = False
    p.set_defaults(scene=ROOT/'artifacts/rs01_v20_sim2sim/scene.xml',
                   policy=ROOT/'artifacts/rs01_guard5_sim2sim/guard5_32750.onnx',
                   output=ROOT/'artifacts/rs01_guard5_sim2sim/sequence')
    args = p.parse_args()
    if not 0 <= args.phase < 1:
        p.error('phase must be in [0,1)')
    Guard5Sim.sensor_profile = args.sensor_profile
    Guard5Sim.seed = args.sensor_seed
    if args.sequence:
        bridge.V20Sim = Guard5Sim
        if args.fast:
            bridge.MOVEMENTS = v22.FAST_MOVEMENTS
        with torch.no_grad():
            bridge.sequence_run(args)
        report = Path(str(args.output)+'.json')
        result = json.loads(report.read_text())
        result.update(sensor_profile=args.sensor_profile, sensor_seed=args.sensor_seed,
                      fast=args.fast, sensor_model='shared_training_SensorSnapshot',
                      host_guard='policy_sensor_snapshot',
                      inward_guard='omnidirectional_inward_scale_v1_5deg')
        report.write_text(json.dumps(result, indent=2)+'\n')
    else:
        if args.duration <= 0 or args.settle_seconds < 0:
            p.error('Require duration > 0 and settle >= 0')
        with torch.no_grad():
            bridge.run(args, Guard5Sim)
