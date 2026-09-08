"""V15 bridge reuses the training estimator verbatim, on CPU (50 Hz)."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from sim2sim import quaternion_rotation_matrix
from sim2sim_v14 import V14Sim, get_parser, run


def load_module(name):
    path = Path(__file__).resolve().parents[2] / 'unitree_rl_gym/legged_gym/envs/rs01_go2_straight' / (name+'.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SensorAdapter:
    def __init__(self, sim):
        self.sim = sim
        cfg = sim.cfg['observations']['rs01_leg_odometry']
        mapping = dict(nominal_base_height='nominal_base_height_m', foot_radius='foot_radius_m',
                       height_margin='height_margin_m', vertical_speed_threshold='vertical_speed_threshold_m_s',
                       velocity_residual_threshold='velocity_residual_threshold_m_s',
                       filter_alpha='filter_alpha', no_contact_decay='no_contact_decay',
                       previous_stance_score_bonus='previous_stance_score_bonus',
                       strict_diagonal_pairs='strict_diagonal_pairs')
        core = load_module('rs01_odometry').Rs01TorchLegOdometry(
            1, 'cpu', **{key: cfg[value] for key, value in mapping.items()})
        self.estimator = load_module('rs01_support_odometry').GravitySupportOdometry(core)

    def reset(self):
        self.estimator.reset()

    def estimate(self, q, dq, omega):
        self.estimator.gravity = quaternion_rotation_matrix(self.sim.data.qpos[3:7]).T @ np.array([0., 0., -1.])
        with torch.no_grad():
            result = self.estimator.estimate(q, dq, omega)
        return {key: value[0].numpy() for key, value in result.items()}


class V15Sim(V14Sim):
    task_name = 'rs01_omni_v15_support'

    def __init__(self, *args):
        contract=json.loads(Path(args[1]).with_suffix('.json').read_text())
        self.task_name=contract['task']
        kinds={'rs01_omni_v15_support':'gravity_support_v1','rs01_omni_v15_stand_phase':'legacy'}
        if self.task_name not in kinds:
            raise ValueError('Expected a V15 export')
        super().__init__(*args)
        if self.cfg.get('v15') != dict(stand_phase='freeze_and_resume', odometry=kinds[self.task_name]):
            raise ValueError('Use export_v15.py')
        if kinds[self.task_name] != 'legacy':
            self.leg_odometry = SensorAdapter(self)

    def frequency(self):
        return super().frequency() if self.gait_enable > .5 else 0.


if __name__ == '__main__':
    torch.set_num_threads(1)
    p = get_parser(); args = p.parse_args()
    if args.duration <= 0 or args.settle_seconds < 0 or not 0 <= args.phase < 1:
        p.error('Require duration>0, settle>=0, 0<=phase<1')
    run(args, sim_class=V15Sim)
