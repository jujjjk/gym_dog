"""Guard5 actor with V22 sensors, original cadence and real RS01 motor dynamics."""
import json
import math
from pathlib import Path
from sim2sim_v22 import V22Sim, torch
import sim2sim_v20 as bridge
from guard_disturbances import DisturbanceMixin


class Guard5Sim(DisturbanceMixin, V22Sim):
    task_name = 'rs01_v22_phase_guard5'
    supported_mappings = ('omnidirectional_inward_scale_v1',)


if __name__ == '__main__':
    p = bridge.get_parser()
    p.add_argument('--task', choices=('rs01_v22_phase_guard5', 'rs01_v22_guard5_robust'),
                   default='rs01_v22_phase_guard5')
    p.add_argument('--interval', type=float, default=3.)
    p.add_argument('--speed_scale', type=float, default=1.)
    p.add_argument('--sensor_profile', choices=('clean','normal','robust','mixed'), default='normal')
    p.add_argument('--sensor_seed', type=int, default=20261010)
    p.add_argument('--pushes', action='store_true')
    p.add_argument('--payload_kg', type=float, default=0.)
    p.add_argument('--disturbance_seed', type=int, default=20261014)
    args = p.parse_args()
    if not math.isfinite(args.payload_kg) or not 0 <= args.payload_kg <= 2.:
        p.error('payload_kg must be within the trained 0–2 kg range')
    if not math.isfinite(args.speed_scale) or args.speed_scale <= 0 or not 0 <= args.phase < 1:
        p.error('Require finite positive speed_scale and phase in [0,1)')
    Guard5Sim.sensor_profile = args.sensor_profile
    Guard5Sim.task_name = args.task
    Guard5Sim.seed = args.sensor_seed
    Guard5Sim.pushes = args.pushes
    Guard5Sim.payload_kg = args.payload_kg
    Guard5Sim.disturbance_seed = args.disturbance_seed
    bridge.V20Sim = Guard5Sim
    # Match the Gym 13-stage direct-switch protocol, including initial march.
    order = ['forward','backward','left','right','turn_left','turn_right',
             'forward_left','forward_right','backward_left','backward_right','combined','combined_reverse']
    movements = dict(bridge.MOVEMENTS)
    bridge.MOVEMENTS = [(n,tuple(x*args.speed_scale for x in movements[n])) for n in order]
    args.direct = True
    with torch.no_grad():
        bridge.sequence_run(args)
    path = Path(str(args.output)+'.json')
    result = json.loads(path.read_text())
    result.update(task=Guard5Sim.task_name, sensor_profile=args.sensor_profile,
                  sensor_seed=args.sensor_seed, speed_scale=args.speed_scale,
                  interval_s=args.interval, direct_switch=True)
    result.update(Guard5Sim.latest.disturbance_report())
    path.write_text(json.dumps(result,indent=2)+'\n')
