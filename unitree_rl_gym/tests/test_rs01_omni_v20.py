import isaacgym
import unittest
import torch
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_env import map_hip_targets, Rs01OmniV20GeometryRobot
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_config import Rs01OmniV20ACfg, Rs01OmniV20BCfg
from legged_gym.envs.rs01_omni_v2.rs01_omni_v19_config import Rs01OmniV19SoftCfg
from legged_gym.utils.helpers import class_to_dict


class V20Tests(unittest.TestCase):
    def test_conditional_weight(self):
        e = Rs01OmniV20GeometryRobot.__new__(Rs01OmniV20GeometryRobot)
        e.cfg = Rs01OmniV20ACfg()
        e.commands = torch.tensor([[0.,0.,0.],[.2,0.,0.],[0.,.2,0.],[0.,0.,.3]])
        torch.testing.assert_close(e._placement_reward_weight(), torch.tensor([1.,1.,.5,.5]))

    def test_mapping(self):
        sides = torch.tensor([1., -1., 1., -1.])
        target = torch.zeros(4, 12)
        ids = [0, 3, 6, 9]
        target[:, ids] = -.22 * sides
        commands = torch.tensor([[0., 0., 0.], [-.2, 0., 0.], [0., .2, 0.], [0., 0., .3]])
        out = map_hip_targets(target, commands, ids, sides, .22, .14)
        torch.testing.assert_close(out[:2, ids], (-.14 * sides).expand(2, -1))
        torch.testing.assert_close(out[2:], target[2:])
        torch.testing.assert_close(map_hip_targets(-target, commands, ids, sides, .22, .14), -target)
        self.assertTrue(torch.equal(target[:, ids], (-.22 * sides).expand(4, -1)))

    def test_no_reward_or_plant_conflict(self):
        old, a, b = [class_to_dict(x()) for x in (Rs01OmniV19SoftCfg, Rs01OmniV20ACfg, Rs01OmniV20BCfg)]
        self.assertEqual(a['rewards'], b['rewards'])
        self.assertEqual(old['rewards']['scales'], a['rewards']['scales'])
        for key in ('rs01_actuator', 'sim', 'commands', 'env', 'normalization'):
            self.assertEqual(old[key], a[key]); self.assertEqual(a[key], b[key])
        self.assertEqual(old['control'], a['control'])
        b['control'].pop('straight_inward_target_rad')
        self.assertEqual(a['control'], b['control'])


if __name__ == '__main__':
    unittest.main()
