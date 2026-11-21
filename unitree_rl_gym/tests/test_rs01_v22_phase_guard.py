import math
import unittest
from types import SimpleNamespace
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_guard import (
    map_inward_targets, Rs01V22PhaseGuardRobot, Rs01V22PhaseGuard5Cfg,
    Rs01V22PhaseGuard3Cfg, Rs01V22PhaseGuard5PPO, Rs01V22PhaseGuard3PPO)
from legged_gym.envs.rs01_omni_v2.rs01_v22_phase_lift import (
    Rs01V22PhaseLiftCfg, Rs01V22PhaseLiftPPO)


class GuardTest(unittest.TestCase):
    def test_only_guard_and_run_name_change(self):
        for cls in (Rs01V22PhaseGuard5Cfg, Rs01V22PhaseGuard3Cfg):
            cfg = class_to_dict(cls())
            cfg.pop('inward_guard')
            self.assertEqual(cfg, class_to_dict(Rs01V22PhaseLiftCfg()))
        for cls in (Rs01V22PhaseGuard5PPO, Rs01V22PhaseGuard3PPO):
            cfg, old = class_to_dict(cls()), class_to_dict(Rs01V22PhaseLiftPPO())
            cfg['runner'].pop('experiment_name'); old['runner'].pop('experiment_name')
            self.assertEqual(cfg, old)

    def test_sign_range_outward_and_other_joints(self):
        ids = [0, 3, 6, 9]
        sides = torch.tensor([1., -1., 1., -1.])
        ranges = torch.tensor([.22, .25, .30, .35])
        target = torch.ones(101, 12) * .12
        target[:, ids] = torch.linspace(-1., 1., 101)[:, None] * ranges * sides
        saved = target.clone()
        for degrees in (3., 5.):
            cap = math.radians(degrees)
            result = map_inward_targets(target, ids, sides, ranges, cap)
            signed = result[:, ids] * sides
            self.assertTrue(bool((signed >= -cap-1e-7).all()))
            torch.testing.assert_close(result[50:, ids], target[50:, ids])
            torch.testing.assert_close(result[:, [1,2,4,5,7,8,10,11]],
                                       target[:, [1,2,4,5,7,8,10,11]])
            torch.testing.assert_close(signed[0], torch.full((4,), -cap))
            self.assertTrue(bool((signed[1:] >= signed[:-1]).all()))
        torch.testing.assert_close(target, saved)

    def test_all_commands_same_single_projection(self):
        names = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
                 for joint in ('hip','thigh','calf')]
        env = SimpleNamespace(dof_names=names, default_dof_pos=torch.zeros(1,12),
            rs01_action_scale_rad=torch.full((12,), .22), cfg=Rs01V22PhaseGuard5Cfg())
        env.cfg.normalization.clip_actions = 1.
        target = torch.zeros(1,12); target[:,[0,3,6,9]] = torch.tensor([-.22,.22,-.22,.22])
        for cmd in ([0.,0.,0.], [.2,0.,0.], [0.,.2,0.], [0.,0.,.3], [-.2,-.1,-.3]):
            env.commands = torch.tensor([cmd])
            out = Rs01V22PhaseGuardRobot._project_rs01_policy_target(env, target)
            torch.testing.assert_close(out[:,[0,3,6,9]],
                torch.tensor([[-1.,1.,-1.,1.]]) * math.radians(5.))

    def test_invalid_default_rejected(self):
        env = SimpleNamespace(dof_names=[f'{leg}_hip_joint' for leg in ('FL','FR','RL','RR')],
                              default_dof_pos=torch.ones(1,4))
        with self.assertRaises(ValueError):
            Rs01V22PhaseGuardRobot._project_rs01_policy_target(env, torch.zeros(1,4))


if __name__ == '__main__':
    unittest.main()
