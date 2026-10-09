"""V21 actor on unchanged V20 identified RS01 actuator and sensor bridge."""
import sim2sim_v20 as bridge


class V21Sim(bridge.V20Sim):
    task_name = 'rs01_omni_v21_phase_coord'


if __name__ == '__main__':
    parser = bridge.get_parser()
    parser.add_argument('--sequence', action='store_true')
    args = parser.parse_args()
    if args.duration <= 0 or not 0 <= args.phase < 1:
        parser.error('Invalid duration or phase')
    if args.sequence:
        bridge.V20Sim = V21Sim
        bridge.sequence_run(args)
    else:
        bridge.run(args, V21Sim)
