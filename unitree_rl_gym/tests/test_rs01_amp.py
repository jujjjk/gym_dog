import unittest
import isaacgym
import torch
import legged_gym.envs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.rs01_omni_v2.rs01_amp_config import Rs01AMPCfg,REFERENCE,Rs01AMPRetimeCfg,Rs01AMPRetimeCfgPPO
from legged_gym.envs.rs01_omni_v2.rs01_amp_env import valid_transition
from legged_gym.envs.rs01_omni_v2.rs01_omni_v30_config import Rs01OmniV30Cfg
from legged_gym.algorithms.rs01_amp import StyleLearner


class AMPTest(unittest.TestCase):
    def test_retime_only_changes_clock(self):
        a=class_to_dict(Rs01AMPCfg());b=class_to_dict(Rs01AMPRetimeCfg())
        for key in a:
            if key!='rewards':self.assertEqual(a[key],b[key],key)
        for key in a['rewards']:
            if key not in ('gait_period_s','gait_stance_ratio'):
                self.assertEqual(a['rewards'][key],b['rewards'][key],key)

    def test_reference_warmstart_and_resume(self):
        a=StyleLearner(REFERENCE,allow_kinematic=True)
        b=StyleLearner(Rs01AMPRetimeCfgPPO.runner.amp_reference,allow_kinematic=True)
        with self.assertRaises(ValueError):b.load_state_dict(a.state_dict())
        before=next(b.net.parameters()).detach().clone()
        b.load_state_dict(a.state_dict(),allow_reference_warmstart=True)
        torch.testing.assert_close(before,next(b.net.parameters()))
        self.assertEqual(b.updates,0)
        b.updates=123
        c=StyleLearner(Rs01AMPRetimeCfgPPO.runner.amp_reference,allow_kinematic=True)
        c.load_state_dict(b.state_dict(),allow_reference_warmstart=True)
        self.assertEqual(c.updates,123)
        torch.testing.assert_close(next(b.net.parameters()),next(c.net.parameters()))

    def test_plant_and_actor_unchanged(self):
        a=class_to_dict(Rs01AMPCfg());b=class_to_dict(Rs01OmniV30Cfg())
        for key in a:
            if key not in ('amp','commands','rewards'):self.assertEqual(a[key],b[key],key)
        self.assertEqual(a['rewards']['scales']['phase_swing_clearance'],0)
        self.assertEqual(a['rewards']['coordination_weight'],0)

    def test_no_cross_reset_or_command_change(self):
        command=torch.zeros(4,3);new=command.clone();new[3,0]=.2
        v=valid_transition(torch.tensor([0,1,0,0]),torch.tensor([0,0,1,0]),command,new)
        self.assertEqual(v.tolist(),[True,False,False,False])

    def test_learning_and_state_restore(self):
        torch.set_num_threads(1)
        with self.assertRaises(ValueError):StyleLearner(REFERENCE)
        a=StyleLearner(REFERENCE,allow_kinematic=True)
        pair=a.expert[0,:64];command=a.commands[0].expand(64,-1)
        with torch.inference_mode():
            reward,x=a.reward(pair[:,:46],pair[:,46:],command)
            self.assertTrue(((reward>=0)&(reward<=1)).all())
            a.add(x)
        before=next(a.net.parameters()).detach().clone();a.update()
        self.assertFalse(torch.equal(before,next(a.net.parameters())))
        self.assertEqual(a.updates,1)
        b=StyleLearner(REFERENCE,allow_kinematic=True);b.load_state_dict(a.state_dict())
        with torch.no_grad():torch.testing.assert_close(a.net(x.clone()),b.net(x.clone()))
        self.assertEqual(b.updates,1)


if __name__=='__main__':unittest.main()
